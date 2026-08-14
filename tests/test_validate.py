"""Validating a weight file that arrived from a stranger.

Each rejection code gets a fixture that triggers it **and only it**, because the code is what
a submitter is shown and a wrong one sends them to fix the wrong thing. The order of the
checks is part of the contract too: a file that is both oversize and unreadable is reported as
oversize, since that is the one the submitter can act on.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")

from catan import action_space, contract, encoder            # noqa: E402
from training.validate import checksum, validate             # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "models" / "champion_az.pt"


@pytest.fixture(scope="module")
def good():
    """The reigning champion's checkpoint, as a config-and-weights pair.

    The one file in the repository that is *known* to be a valid submission, so every hostile
    fixture below is this with exactly one thing broken. Building a plausible one by hand
    would risk testing the fixture rather than the validator.
    """
    if not CHAMPION.is_file():
        pytest.skip("no AlphaZero champion in this checkout")
    from training.loading import load_model

    checkpoint = load_model(CHAMPION)
    return {"config": dict(checkpoint["config"]),
            "weights": dict(checkpoint["weights"])}


def write(tmp_path, payload, name="submission.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def assert_rejected(report, code):
    assert not report, f"expected a rejection, got {report!r}"
    assert report.code == code, f"expected {code!r}, got {report.code!r}: {report.reason}"
    assert report.reason, "a rejection with no reason is not actionable"
    assert report.checksum, "a rejected file still needs a name in the audit log"


# =========================================================================== #
# THE ONE FILE KNOWN TO BE VALID                                              #
# =========================================================================== #

def test_the_champion_validates():
    if not CHAMPION.is_file():
        pytest.skip("no AlphaZero champion in this checkout")
    report = validate(CHAMPION)
    assert report, f"the champion was rejected: [{report.code}] {report.reason}"
    assert report.network == "structured-v1"
    assert report.parameters > 0
    assert report.checksum == checksum(CHAMPION)


def test_the_verdict_records_the_contract_it_was_reached_under():
    """A stored verdict is only true for one observation. Without the signature beside it, a
    model validated in July is still "validated" in August against a different encoder —
    which is exactly the failure `catan/contract.py` exists to make visible."""
    if not CHAMPION.is_file():
        pytest.skip("no AlphaZero champion in this checkout")
    assert validate(CHAMPION).signature == contract.signature()


def test_the_report_is_json_ready():
    """It goes into a database row and an API response; a tensor or an enum in there would
    serialise to something only Python can read back."""
    import json

    if not CHAMPION.is_file():
        pytest.skip("no AlphaZero champion in this checkout")
    payload = validate(CHAMPION, probe=False).as_dict()
    assert json.loads(json.dumps(payload))["ok"] is True


# =========================================================================== #
# ONE FIXTURE PER REJECTION CODE                                              #
# =========================================================================== #

def test_a_missing_file(tmp_path):
    assert_rejected_without_checksum(validate(tmp_path / "nothing.pt"), "unreadable")


def test_a_file_that_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "notes.pt"
    path.write_bytes(b"just some bytes")
    assert_rejected(validate(path), "unreadable")


def test_a_file_over_the_size_limit(tmp_path, good):
    path = write(tmp_path, good)
    assert_rejected(validate(path, max_bytes=1024), "oversize")


def test_an_older_observation(tmp_path, good):
    """The failure this exists to catch: a checkpoint from before an encoder change loads
    perfectly well and then plays nonsense. models/champion.pt is a real example, promoted at
    1,868 floats against today's 2,503."""
    payload = {**good, "config": {**good["config"], "obs_size": 1884}}
    assert_rejected(validate(write(tmp_path, payload)), "incompatible_observation")


def test_a_different_action_space(tmp_path, good):
    payload = {**good, "config": {**good["config"], "num_actions": 324}}
    assert_rejected(validate(write(tmp_path, payload)), "incompatible_actions")


def test_a_missing_tensor(tmp_path, good):
    weights = dict(good["weights"])
    weights.pop(next(iter(weights)))
    assert_rejected(validate(write(tmp_path, {**good, "weights": weights})),
                    "missing_tensors")


def test_a_tensor_this_architecture_does_not_have(tmp_path, good):
    """Refused rather than ignored. An unexpected tensor means the file came from something
    other than this architecture, and dropping it silently would run a model that is not the
    one the submitter measured locally."""
    weights = {**good["weights"], "secret_head.weight": torch.zeros(4)}
    assert_rejected(validate(write(tmp_path, {**good, "weights": weights})),
                    "unexpected_tensors")


def test_a_tensor_of_the_wrong_shape(tmp_path, good):
    name = next(n for n, t in good["weights"].items() if t.dim() >= 1)
    weights = {**good["weights"], name: torch.zeros(3, 3)}
    assert_rejected(validate(write(tmp_path, {**good, "weights": weights})),
                    "shape_mismatch")


def test_a_tensor_of_the_wrong_dtype(tmp_path, good):
    """float64 is refused rather than cast: the network is float32 throughout, so accepting
    it would silently change what the submitter measured."""
    name, tensor = next(iter(good["weights"].items()))
    weights = {**good["weights"], name: tensor.double()}
    assert_rejected(validate(write(tmp_path, {**good, "weights": weights})), "bad_dtype")


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_a_tensor_that_is_not_finite(tmp_path, good, poison):
    name, tensor = next(iter(good["weights"].items()))
    broken = tensor.clone()
    broken.view(-1)[0] = poison
    weights = {**good["weights"], name: broken}
    assert_rejected(validate(write(tmp_path, {**good, "weights": weights})), "non_finite")


def test_a_config_edited_to_describe_a_smaller_network(tmp_path, good):
    """The config is the submitter's claim about the weights, and the weights are checked
    against a network built from that claim rather than against the claim itself. So shrinking
    the config does not shrink what has to be there — it is caught on the first tensor."""
    payload = {"config": {**good["config"], "width": 7, "context": 5, "trunk": 3},
               "weights": good["weights"]}
    report = validate(write(tmp_path, payload))
    assert_rejected(report, "shape_mismatch")
    assert "context_mlp" in report.reason


def test_a_config_carrying_a_key_the_architecture_does_not_take(tmp_path, good):
    """`check_config` bounds the keys it knows and says nothing about ones it does not, so an
    invented key survives it and reaches `build`, where `from_config` passes it through as a
    keyword argument. Rejected there, with the code that says the config is at fault rather
    than the weights."""
    payload = {"config": {**good["config"], "attention_heads": 8},
               "weights": good["weights"]}
    assert_rejected(validate(write(tmp_path, payload)), "unbuildable")


def test_weights_that_are_finite_and_still_produce_nan(tmp_path, good):
    """The check that runs the file instead of describing it. Every tensor here is the right
    name, shape and dtype and every value is finite — and float32 overflows between a large
    weight matrix and a real activation, so the logits come back `nan`. The place that must
    not happen is mid-game."""
    weights = {name: tensor.clone() for name, tensor in good["weights"].items()}
    for name, tensor in weights.items():
        if tensor.dim() == 2:
            tensor.fill_(3.0e38)                 # finite in float32; its products are not
    report = validate(write(tmp_path, {**good, "weights": weights}))
    assert not report, "a model that produces nan logits was accepted"
    assert report.code == "probe_failed"


def test_the_probe_can_be_skipped(tmp_path, good):
    """The worker runs it; a bulk re-check of ten thousand stored artifacts after an encoder
    change does not need to, and it is the only check that costs a forward pass."""
    assert validate(write(tmp_path, good), probe=False)


# =========================================================================== #
# ORDER, AND WHAT SURVIVES A REJECTION                                        #
# =========================================================================== #

def test_the_checksum_is_returned_for_a_rejected_file(tmp_path, good):
    """"The same bad file has been uploaded forty times" is a question only a hash can
    answer, and it is asked about files that never passed."""
    payload = {**good, "config": {**good["config"], "obs_size": 1884}}
    path = write(tmp_path, payload)
    assert validate(path).checksum == checksum(path)


def test_size_is_checked_before_the_file_is_parsed(tmp_path):
    """A 40 GB upload should be refused on its size, not by torch failing to read it — the
    ceiling exists to stop the parse, so it has to come first."""
    path = tmp_path / "huge.pt"
    path.write_bytes(b"\x00" * 4096)
    assert_rejected(validate(path, max_bytes=64), "oversize")


def test_nothing_is_written_beside_the_file(tmp_path, good):
    """It runs in a worker with no credentials and no outbound network. A validator that
    wrote a sidecar would also need somewhere to write it."""
    path = write(tmp_path, good)
    before = sorted(p.name for p in tmp_path.iterdir())
    validate(path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_the_current_engine_shapes_are_what_is_demanded(tmp_path, good):
    """Pinned against the contract rather than against literals, so this test keeps meaning
    the right thing after the next observation change."""
    assert good["config"]["obs_size"] == encoder.SIZE
    assert good["config"]["num_actions"] == action_space.NUM_ACTIONS


def assert_rejected_without_checksum(report, code):
    """A file that does not exist has no bytes to hash, which is the one case with no name."""
    assert not report
    assert report.code == code
    assert report.checksum == ""
