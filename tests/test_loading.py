"""One door for checkpoints, and it does not execute what it is given.

``torch.load`` unpickles, and unpickling runs code, so loading a checkpoint somebody sent you
is running a program they sent you. Nine call sites in this repository passed
``weights_only=False``, one of them inside a process-pool initializer and one of them in the
promotion gate's ``_install``, which unpickled the *candidate* after it had been validated
safely at match time. See ``docs/audit-2026-08-05-public-arena.md`` §B4.

The tests here are in three groups, and the first is the one that matters longest:
:func:`test_no_module_outside_the_loader_calls_torch_load` is a structural test. Converting
nine sites is a commit; keeping the tenth from appearing is a test.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")

from catan import action_space, contract, encoder            # noqa: E402
from training.loading import (                               # noqa: E402
    MAX_PARAMETERS,
    CheckpointError,
    check_config,
    load_model,
    load_training_state,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: The module allowed to call it, relative to the repository root.
THE_LOADER = pathlib.Path("training") / "loading.py"

STRUCTURED = {"kind": "structured", "obs_size": encoder.SIZE,
              "num_actions": action_space.NUM_ACTIONS}


def write(tmp_path, payload, name="checkpoint.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


# =========================================================================== #
# THE STRUCTURAL RULE                                                         #
# =========================================================================== #

def test_no_module_outside_the_loader_calls_torch_load():
    """``torch.load`` appears in exactly one module, and nowhere else in the tree.

    ⚠️ If this fails, do not add the file to an exclusion list. The point of one door is
    that a reviewer looking at a new checkpoint-reading feature has one place to look. Call
    ``training.loading.load_model`` for a file that plays, or
    ``training.loading.load_training_state`` for your own run resuming.

    Tests are excluded because a test that constructs a hostile file has to be able to read
    one back without going through the code it is testing.
    """
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative == THE_LOADER or relative.parts[0] in ("tests", ".git"):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "torch.load(" not in stripped:
                continue
            offenders.append(f"{relative}:{number}: {stripped}")
    assert not offenders, (
        "torch.load outside training/loading.py:\n  " + "\n  ".join(offenders))


def test_nothing_in_the_tree_asks_for_the_unrestricted_unpickler():
    """``weights_only=False`` is what makes a checkpoint executable, and after this change
    the only occurrence is the documented, warned fallback in the loader itself."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative == THE_LOADER or relative.parts[0] in ("tests", ".git"):
            continue
        if "weights_only=False" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative))
    assert not offenders, f"weights_only=False in {offenders}"


# =========================================================================== #
# THE SHIPPED CHAMPIONS STILL LOAD                                            #
# =========================================================================== #

@pytest.mark.parametrize("name", ["champion.pt", "champion_az.pt"])
def test_the_shipped_champions_load_under_the_restricted_unpickler(name):
    """The audit's claim, re-checked here rather than trusted: ``weights_only=True`` loads
    both champions unchanged, so this was one word per site and not a migration."""
    path = ROOT / "models" / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this checkout")
    checkpoint = load_model(path)
    assert set(checkpoint) >= {"config", "weights"}
    assert checkpoint["weights"]


def test_the_alphazero_champion_is_at_the_current_contract():
    """Not a property of the loader — a property of what is installed. If this fails, the
    interfaces have silently fallen back to the heuristic, which has happened before and
    raised nothing."""
    path = ROOT / "models" / "champion_az.pt"
    if not path.is_file():
        pytest.skip("no AlphaZero champion in this checkout")
    config = load_model(path)["config"]
    assert contract.accepts(config["obs_size"], config["num_actions"])


# =========================================================================== #
# WHAT IS REFUSED, AND WHY                                                    #
# =========================================================================== #

def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path):
    assert_refuses(write(tmp_path, torch.zeros(4)), "not a checkpoint")


def test_a_checkpoint_without_weights_is_refused(tmp_path):
    assert_refuses(write(tmp_path, {"config": dict(STRUCTURED)}), "weights")


def test_a_checkpoint_without_a_config_is_refused(tmp_path):
    assert_refuses(write(tmp_path, {"weights": {}}), "config")


def test_a_missing_file_is_refused_rather_than_returning_none(tmp_path):
    """``champion.load`` documents itself as never raising, which is right for *is one
    installed* and wrong for *someone uploaded this*. The loader raises so the two can be
    told apart; a caller wanting the old behaviour catches it."""
    assert_refuses(tmp_path / "absent.pt", "not a file")


def test_a_file_over_the_size_limit_is_refused_before_it_is_opened(tmp_path):
    path = write(tmp_path, {"config": dict(STRUCTURED), "weights": {"a": torch.zeros(64)}})
    assert_refuses(path, "limit", max_bytes=16)


def test_an_unknown_network_kind_is_refused(tmp_path):
    assert_refuses(write(tmp_path, {"config": {**STRUCTURED, "kind": "convnet"},
                                    "weights": {}}), "convnet")


@pytest.mark.parametrize("key, value", [
    ("width", 2 ** 30),
    ("context", -1),
    ("trunk", 10 ** 9),
    ("depth", 4096),
    ("hops", 9),
    ("obs_size", 10 ** 9),
])
def test_a_dimension_outside_its_range_is_refused(tmp_path, key, value):
    """The allocation bomb. ``network.py`` does ``build(dict(checkpoint["config"]))``, so a
    hostile dimension is an out-of-memory kill *before* any tensor is inspected — and
    ``champion.load``'s bare ``except Exception`` would report that as "no champion"."""
    assert_refuses(write(tmp_path, {"config": {**STRUCTURED, key: value}, "weights": {}}),
                   key)


def test_a_boolean_is_not_accepted_as_a_dimension():
    """``bool`` is an ``int`` in Python, so ``width=True`` builds a one-wide layer instead of
    being refused. Cheap to get wrong, silent when wrong."""
    with pytest.raises(CheckpointError, match="width"):
        check_config({**STRUCTURED, "width": True})


def test_dimensions_that_are_individually_fine_and_jointly_enormous_are_refused():
    """Every bound is survivable on its own and their product is not: sixteen 16,384-wide
    layers is inside every per-key range and is four billion parameters."""
    with pytest.raises(CheckpointError, match="parameters"):
        check_config({"kind": "flat", "obs_size": encoder.SIZE,
                      "num_actions": action_space.NUM_ACTIONS,
                      "hidden": [16384] * 16})


def test_a_realistic_config_is_well_under_the_parameter_limit():
    """The limit is a refusal threshold, not a budget. The reigning champion is ~200k
    parameters and the current defaults ~374k, so there is three orders of magnitude of
    room before anything legitimate is caught."""
    check_config({**STRUCTURED, "width": 128, "road_width": 64, "context": 192,
                  "hops": 1, "depth": 3, "trunk": 256})
    assert MAX_PARAMETERS > 100_000_000


def test_an_unknown_value_activation_is_refused():
    with pytest.raises(CheckpointError, match="value_activation"):
        check_config({**STRUCTURED, "value_activation": "relu"})


def test_the_flat_default_needs_no_kind():
    """The flat network predates the ``kind`` field, so its absence means flat. Old
    checkpoints keep loading — see :func:`training.net.build`."""
    check_config({"obs_size": encoder.SIZE, "num_actions": action_space.NUM_ACTIONS,
                  "hidden": [512, 512]})


# =========================================================================== #
# THE SECOND DOOR                                                             #
# =========================================================================== #

def test_a_training_checkpoint_loads_under_the_restricted_unpickler(tmp_path):
    """Worth checking rather than assuming, and it is why ``weights_only=False`` is gone from
    the tree entirely: the optimiser state, ``pool.snapshot()``'s list of
    ``(iteration, weights)`` tuples and a history of plain dicts are all permitted types."""
    path = write(tmp_path, {
        "config": dict(STRUCTURED),
        "weights": {"a": torch.zeros(3)},
        "optimizer": {"state": {0: {"step": torch.tensor(1.0)}},
                      "param_groups": [{"lr": 3e-4, "params": [0]}]},
        "pool": {"config": dict(STRUCTURED), "frozen": [(1, {"a": torch.zeros(3)})]},
        "iteration": 7,
        "history": [{"iteration": 1, "loss": 0.5}],
    })
    restored = load_training_state(path)
    assert restored["iteration"] == 7
    assert isinstance(restored["pool"]["frozen"][0], tuple)


def test_the_training_door_does_not_require_a_playable_shape(tmp_path):
    """It is a different shape, not a weaker rule: a resume checkpoint has no reason to
    satisfy ``check_config``, and demanding it would break ``--resume``."""
    path = write(tmp_path, {"iteration": 3, "history": []})
    assert load_training_state(path)["iteration"] == 3


def test_the_training_door_can_be_told_not_to_fall_back(tmp_path):
    """``trusted=False`` is what a path reachable by a file you did not write must pass. The
    fallback exists for an older checkpoint of your own carrying something the restricted
    unpickler will not take."""
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a torch file at all")
    with pytest.raises(CheckpointError):
        load_training_state(path, trusted=False)


# =========================================================================== #

def assert_refuses(path, fragment, **kwargs):
    with pytest.raises(CheckpointError) as raised:
        load_model(path, **kwargs)
    assert fragment in str(raised.value), f"expected {fragment!r} in {raised.value!r}"
