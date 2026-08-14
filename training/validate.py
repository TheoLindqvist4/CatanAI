"""Is this pile of numbers a model this engine can play?

    python -m training.validate models/champion_az.pt

The platform's answer to an upload, and it lives here rather than in the web repository for
the same reason every rule lives in `catan/rules.py`: the authority on what a valid model *is*
belongs beside the architecture that defines it. A second implementation in a service would
be a second opinion, and the two would diverge on the first observation change.

**A pure function.** No network, no database, no environment, and nothing written to disk. It
reads one file and returns a :class:`Report`. That is what lets it run in a worker with no
production credentials and no outbound access — the shape
``docs/audit-2026-08-05-public-arena.md`` argues for and the go-live plan requires.

What it is not
--------------
It does not decide whether a model is any *good*. It answers the narrow question of whether
the weights can be loaded into this architecture and asked for a move without failing — the
question that has to be answered before a benchmark can even be run. Strength is measured by
playing, which costs 8.9 seconds a game; this costs milliseconds and refuses most of what
would waste them.

It also does not decide whether the file is *safe*. :mod:`training.loading` does that, and
this module goes through it, because a validator that opened the file itself would be a tenth
``torch.load`` call site — which is the failure ``tests/test_loading.py`` exists to prevent.

Every rejection has a code
--------------------------
A submitter needs to know which of the nine things was wrong with their file, and a support
conversation that begins *"validation failed"* is one nobody can act on. The codes are stable
strings, meant to be stored and displayed:

    unreadable · oversize · incompatible_observation · incompatible_actions
    missing_tensors · unexpected_tensors · shape_mismatch · bad_dtype
    non_finite · unbuildable · probe_failed

The checksum is returned either way. A rejected file still needs a name in an audit log, and
"the same bad file has now been uploaded forty times" is a question only a hash can answer.
"""

import argparse
import hashlib
import pathlib

from catan import action_space, contract, encoder
from training.loading import CheckpointError, load_model

#: How much of the file is read at a time while hashing. The champions are ~1.5 MB; this is
#: about bounding memory for a file that is larger than it should be, not about speed.
CHUNK = 1 << 20

#: Tensor dtypes a submitted checkpoint may use. ``float64`` is refused rather than cast:
#: the network is float32 throughout, so accepting it would mean silently changing what the
#: submitter measured locally.
ALLOWED_DTYPES = ("torch.float32",)


class Report:
    """What validation found. Truthy when the model may play.

    Attributes:
        ok: whether it passed everything.
        code: the stable rejection code, or ``None``.
        reason: one sentence a submitter can act on, or ``None``.
        checksum: SHA-256 of the file, always present.
        size_bytes: as it was on disk.
        signature: the contract this was validated against — so a stored verdict can be
            re-checked for staleness rather than trusted forever. A model validated under one
            observation is not validated under the next.
        config: the checkpoint's network config, when it was readable.
        network: the architecture id, e.g. ``structured-v1``.
        parameters: how many numbers it carries, when it was buildable.
    """

    __slots__ = ("ok", "code", "reason", "checksum", "size_bytes", "signature",
                 "config", "network", "parameters")

    def __init__(self, checksum, size_bytes, ok=False, code=None, reason=None,
                 config=None, network=None, parameters=None):
        self.ok = ok
        self.code = code
        self.reason = reason
        self.checksum = checksum
        self.size_bytes = size_bytes
        self.signature = contract.signature()
        self.config = config
        self.network = network
        self.parameters = parameters

    def __bool__(self):
        return self.ok

    def as_dict(self):
        """JSON-ready, for a database row or an API response."""
        return {name: getattr(self, name) for name in self.__slots__}

    def __repr__(self):
        verdict = "ok" if self.ok else f"rejected: {self.code}"
        return f"Report({verdict}, {self.checksum[:12]}…, {self.size_bytes} bytes)"


def checksum(path):
    """SHA-256 of a file, streamed. The identity of an accepted artifact, forever."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path, *, probe=True, max_bytes=contract.MAX_MODEL_BYTES):
    """Check a submitted weight file against the current contract.

    Args:
        path: the file. Read, never written.
        probe: also ask the loaded network for one move on a real opening position. Costs a
            few milliseconds and is the only check that exercises the weights rather than
            describing them — a tensor can be the right name, shape, dtype and finite and
            still produce ``nan`` logits through an overflow.
        max_bytes: the size ceiling, from the contract.

    Returns:
        Report: truthy when the model may be offered a game.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return Report("", 0, code="unreadable", reason=f"{path.name} is not a file")

    size = path.stat().st_size
    digest = checksum(path)

    if size > max_bytes:
        return Report(digest, size, code="oversize",
                      reason=f"{size} bytes, over the {max_bytes}-byte limit")

    # Through the loader, not around it: `weights_only=True`, and a config whose dimensions
    # are bounded before anything is allocated from them.
    try:
        checkpoint = load_model(path, max_bytes=max_bytes)
    except CheckpointError as error:
        return Report(digest, size, code="unreadable", reason=str(error))

    config = dict(checkpoint["config"])
    weights = checkpoint["weights"]
    network = contract.NETWORKS[config.get("kind", "flat")]

    observed = config.get("obs_size")
    if observed != encoder.SIZE:
        return Report(digest, size, code="incompatible_observation", config=config,
                      network=network,
                      reason=f"built for a {observed}-float observation; this engine "
                             f"encodes {encoder.SIZE}")
    if config.get("num_actions") != action_space.NUM_ACTIONS:
        return Report(digest, size, code="incompatible_actions", config=config,
                      network=network,
                      reason=f"built for {config.get('num_actions')} actions; this engine "
                             f"has {action_space.NUM_ACTIONS}")

    from training.net import build

    try:
        net = build(config)
    except Exception as error:
        return Report(digest, size, code="unbuildable", config=config, network=network,
                      reason=f"the config does not describe a network: {error}")

    reference = net.state_dict()
    parameters = sum(tensor.numel() for tensor in reference.values())

    missing = sorted(set(reference) - set(weights))
    if missing:
        return Report(digest, size, code="missing_tensors", config=config, network=network,
                      parameters=parameters,
                      reason=f"{len(missing)} tensor(s) absent, starting with "
                             f"{missing[0]!r}")

    extra = sorted(set(weights) - set(reference))
    if extra:
        # Refused rather than ignored. An unexpected tensor means the file was produced by
        # something other than this architecture, and quietly dropping it would run a model
        # that is not the one the submitter measured.
        return Report(digest, size, code="unexpected_tensors", config=config,
                      network=network, parameters=parameters,
                      reason=f"{len(extra)} tensor(s) this architecture does not have, "
                             f"starting with {extra[0]!r}")

    for name, expected in reference.items():
        actual = weights[name]
        if tuple(actual.shape) != tuple(expected.shape):
            return Report(digest, size, code="shape_mismatch", config=config,
                          network=network, parameters=parameters,
                          reason=f"{name} is {tuple(actual.shape)}, expected "
                                 f"{tuple(expected.shape)}")
        if str(actual.dtype) not in ALLOWED_DTYPES:
            return Report(digest, size, code="bad_dtype", config=config, network=network,
                          parameters=parameters,
                          reason=f"{name} is {actual.dtype}; expected "
                                 f"{' or '.join(ALLOWED_DTYPES)}")
        if not _finite(actual):
            return Report(digest, size, code="non_finite", config=config, network=network,
                          parameters=parameters,
                          reason=f"{name} contains NaN or infinity")

    net.load_state_dict(weights)

    if probe:
        failure = _probe(net)
        if failure is not None:
            return Report(digest, size, code="probe_failed", config=config, network=network,
                          parameters=parameters, reason=failure)

    return Report(digest, size, ok=True, config=config, network=network,
                  parameters=parameters)


def _finite(tensor):
    import torch

    return bool(torch.isfinite(tensor).all())


def _probe(net):
    """Ask the network for one move on a real position. ``None`` if it behaved.

    Every check above describes the file. This one *runs* it, which is the difference between
    "the tensors are the right shape" and "this can be asked for a move". A network of finite
    weights can still produce ``nan`` logits — a large weight matrix and a large activation
    overflow float32 between them — and the place that must not happen is mid-game.

    The position is a fresh opening at a fixed seed rather than a vector of zeros, because
    zeros are not a state the encoder ever produces and a bug that only shows on real inputs
    would pass.
    """
    import numpy as np
    import torch

    from catan.env import CatanEnv

    env = CatanEnv(num_players=2)
    observation, info = env.reset(seed=1)
    try:
        with torch.no_grad():
            batch = torch.as_tensor(np.asarray([observation], dtype=np.float32))
            logits, value = net(batch)
    except Exception as error:
        return f"the network raised on a real position: {error}"

    if logits.shape[-1] != action_space.NUM_ACTIONS:
        return (f"produced {logits.shape[-1]} logits, expected "
                f"{action_space.NUM_ACTIONS}")
    if not bool(torch.isfinite(logits).all()):
        return "produced non-finite logits on a real position"
    if not bool(torch.isfinite(value).all()):
        return "produced a non-finite value on a real position"

    # And that the legal move it would pick is actually legal — the one end-to-end assertion
    # here, and the cheapest possible proof that the masking lines up with the action space.
    masked = logits[0].numpy().copy()
    masked[~np.asarray(info["mask"], dtype=bool)] = -np.inf
    if int(np.argmax(masked)) not in info["legal"]:
        return "its best legal move was not legal"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m training.validate",
        description="Check a weight file against the CATANIA-1 contract.",
    )
    parser.add_argument("path", help="the checkpoint to validate")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the forward pass on a real position")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    arguments = parser.parse_args(argv)

    report = validate(arguments.path, probe=not arguments.no_probe)

    if arguments.json:
        import json

        print(json.dumps(report.as_dict(), indent=2))
        return 0 if report.ok else 1

    print(f"{arguments.path}")
    print(f"  sha256      {report.checksum}")
    print(f"  size        {report.size_bytes:,} bytes")
    print(f"  contract    {report.signature}")
    if report.network:
        print(f"  network     {report.network}"
              + (f", {report.parameters:,} parameters" if report.parameters else ""))
    if report.ok:
        # ASCII: this prints on a Windows console under cp1252, where an em dash is mojibake.
        print("  VALID       this model can be offered a game")
        return 0
    print(f"  REJECTED    [{report.code}] {report.reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
