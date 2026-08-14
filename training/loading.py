"""The only module in this tree allowed to call ``torch.load``.

``torch.load`` with its default settings unpickles, and unpickling runs code. So loading a
checkpoint somebody sent you is running a program they sent you — with your user's
permissions, before a single tensor has been looked at. That is not a subtle property; it is
what pickle is for.

It matters here because of where the call sites sit. ``training/alphazero/arena.py`` loads a
model inside the process-pool **initializer**, so a hostile file executes once per worker
before a single game is played; and ``champion.py::_install`` unpickles the *candidate* and
re-saves it, so a file validated safely at match time was unpickled unsafely at promotion
time. Nine sites called it with ``weights_only=False``. See
``docs/audit-2026-08-05-public-arena.md`` §B4.

Two doors, and the difference between them is who wrote the file
---------------------------------------------------------------
:func:`load_model` — a file that plays. Loaded with ``weights_only=True``, which restricts
the unpickler to tensors and plain containers, and then checked before anything is built.
Every path a *player* can reach goes through here: the champions, the agents, the arena, the
promotion gate. It is safe for a file that arrived over the internet.

:func:`load_training_state` — your own run, resuming. Carries the optimiser state, the frozen
opponent pool and the run history, and is read from ``checkpoints/`` by a person who named the
path. This one does execute what it is given, which is stated rather than hidden, and no
player-reachable code calls it.

``weights_only=True`` is not the whole story
--------------------------------------------
It stops the *pickle* executing. It does not stop the *config*: :func:`training.net.build`
takes widths and depths straight out of the checkpoint and allocates, so a file claiming a
2-billion-wide layer is an out-of-memory kill before any tensor is inspected — and
``champion.load``'s ``except Exception: return None`` would report that as "no champion
present". :func:`check_config` is the other half, and this module **raises** rather than
returning ``None``, so a refusal cannot be mistaken for an absence.

The audit checked one more thing worth writing down: ``weights_only=True`` **already loads
both shipped champions unchanged** on torch 2.13.0+cpu. There was no migration to do here and
no metadata sidecar to invent — the permitted types cover ``dict``, ``OrderedDict``, ``int``
and ``str``, which is all a checkpoint of this shape holds.
"""

import pathlib

import torch

from catan import contract

#: Config keys whose value is a positive integer, and the largest each may be.
#:
#: The ceilings are not guesses at what is sensible — a sensible network is a few hundred
#: wide. They are the point past which the number stops describing a network and starts
#: describing an allocation, and they exist so the refusal happens before the allocation.
INTEGER_BOUNDS = {
    "obs_size": (1, 1_000_000),
    "num_actions": (1, 100_000),
    "width": (1, 8192),
    "road_width": (1, 8192),
    "context": (1, 8192),
    "trunk": (1, 16384),
    "hops": (0, 2),
    "depth": (1, 32),
    "rounds": (0, 16),
}

#: Layer widths for a flat network, which arrive as a sequence rather than as one number.
MAX_HIDDEN_LAYERS = 16

#: Nothing this project builds comes near it; a checkpoint that does is not a mistake.
MAX_PARAMETERS = 200_000_000


class CheckpointError(ValueError):
    """A checkpoint was refused, and the message says why.

    Deliberately an exception rather than a ``None``. ``champion.load`` documents itself as
    never raising, which is right for *"is there a champion installed?"* and wrong for
    *"someone uploaded this"* — the two failures need to be told apart, and a caller that
    wants the old behaviour can catch this.
    """


def load_model(path, *, map_location="cpu", max_bytes=contract.MAX_MODEL_BYTES):
    """A playable checkpoint, loaded without executing anything it contains.

    Args:
        path: the file.
        map_location: as ``torch.load``.
        max_bytes: refuse anything larger before opening it. The champions are under 1.6 MB.

    Returns:
        dict: the checkpoint, with ``config`` and ``weights`` present and the config within
        bounds. Whether the *weights* match the config is a separate question and belongs to
        :mod:`training.validate`, which answers it by building the network.

    Raises:
        CheckpointError: for anything wrong with it, with the reason in the message.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise CheckpointError(f"{path} is not a file")

    size = path.stat().st_size
    if size > max_bytes:
        raise CheckpointError(f"{path} is {size} bytes, over the {max_bytes}-byte limit")

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        # Includes the case that matters most: a pickle carrying something executable, which
        # `weights_only=True` refuses to unpickle rather than running.
        raise CheckpointError(f"{path} could not be read as a weights-only "
                              f"checkpoint: {error}") from error

    if not isinstance(checkpoint, dict):
        raise CheckpointError(f"{path} is a {type(checkpoint).__name__}, not a checkpoint")
    for key in ("config", "weights"):
        if key not in checkpoint:
            raise CheckpointError(f"{path} has no {key!r}")

    check_config(checkpoint["config"])

    if not isinstance(checkpoint["weights"], dict):
        raise CheckpointError("weights must be a mapping of name to tensor")
    return checkpoint


def check_config(config):
    """Refuse a config that would allocate rather than build. Raises, or returns the config.

    Called by :func:`load_model` before anything is constructed, because
    :func:`training.net.build` allocates from these numbers directly. The network classes do
    validate *some* of this — ``StructuredPolicyValueNet`` checks ``obs_size``,
    ``num_actions`` and ``hops`` — but the checks that would stop a memory bomb (``width``,
    ``context``, ``trunk``, ``depth``) are the ones it does not have, and by the time its
    constructor runs the allocation has already been attempted.
    """
    if not isinstance(config, dict):
        raise CheckpointError(f"config is a {type(config).__name__}, not a mapping")

    kind = config.get("kind", "flat")
    if kind not in contract.NETWORKS:
        raise CheckpointError(f"unknown network kind {kind!r}; "
                              f"expected one of {sorted(contract.NETWORKS)}")

    for key, (low, high) in INTEGER_BOUNDS.items():
        if key not in config:
            continue
        value = config[key]
        # `bool` is an `int` in Python, and `width=True` would build a 1-wide layer rather
        # than being refused.
        if not isinstance(value, int) or isinstance(value, bool):
            raise CheckpointError(f"{key} must be an integer, got {value!r}")
        if not low <= value <= high:
            raise CheckpointError(f"{key} is {value}, outside [{low}, {high}]")

    hidden = config.get("hidden")
    if hidden is not None:
        if not isinstance(hidden, (list, tuple)):
            raise CheckpointError(f"hidden must be a sequence, got {type(hidden).__name__}")
        if len(hidden) > MAX_HIDDEN_LAYERS:
            raise CheckpointError(f"hidden has {len(hidden)} layers, over "
                                  f"{MAX_HIDDEN_LAYERS}")
        for width in hidden:
            if not isinstance(width, int) or isinstance(width, bool) or not 1 <= width <= 16384:
                raise CheckpointError(f"hidden layer width {width!r} is not in [1, 16384]")

    activation = config.get("value_activation", "linear")
    if activation not in ("linear", "tanh"):
        raise CheckpointError(f"unknown value_activation {activation!r}")

    estimate = _parameter_estimate(config, kind)
    if estimate > MAX_PARAMETERS:
        raise CheckpointError(f"config describes roughly {estimate:,} parameters, over "
                              f"the {MAX_PARAMETERS:,} limit")
    return config


def _parameter_estimate(config, kind):
    """Roughly how much this config would allocate.

    Every bound above is individually survivable and their product is not — a flat network
    with sixteen 16,384-wide layers is inside every one of them and is 4.3 billion
    parameters. This is the check that closes over the combination. Deliberately an
    over-estimate of the dominant terms rather than an exact count: it is a refusal
    threshold, not a report, and being wrong low is the only direction that matters.
    """
    obs = config.get("obs_size", 0)
    actions = config.get("num_actions", 0)
    if kind == "flat":
        widths = [obs, *config.get("hidden", ()), actions]
        return sum(a * b for a, b in zip(widths, widths[1:]))

    width = config.get("width", 0)
    context = config.get("context", 0)
    trunk = config.get("trunk", 0)
    depth = config.get("depth", 1)
    rounds = config.get("rounds", 0)
    biggest = max(width, config.get("road_width", 0), context, trunk)
    # Every block is at most `biggest x biggest`, there are `depth` of them per position
    # type, `rounds` message-passing passes, and a handful of heads.
    return (obs * context + biggest * biggest * (3 * depth + 2 * rounds + 8)
            + trunk * actions)


def load_training_state(path, *, map_location="cpu", trusted=True):
    """A checkpoint you wrote yourself, with everything needed to resume the run.

    Separate from :func:`load_model` because it is a different shape, not because it is a
    weaker rule: a run's checkpoint carries the optimiser state, the frozen opponent pool and
    the history, has no reason to satisfy :func:`check_config`, and is read by a person
    naming a path to their own ``checkpoints/`` directory.

    It is loaded with ``weights_only=True`` as well. That was worth checking rather than
    assuming, and it holds: the optimiser's ``state_dict``, ``pool.snapshot()``'s list of
    ``(iteration, weights)`` tuples and a history of plain dicts are all inside the
    restricted unpickler's permitted types. So **no call site in this repository uses
    ``weights_only=False``**, and ``tests/test_loading.py`` is what keeps it that way.

    ⚠️ ``trusted`` is the escape hatch, and it executes what the file contains. It exists for
    one case: an *older* checkpoint on your own disk carrying something the restricted
    unpickler will not take — a numpy scalar in the history is the likely one. It warns
    loudly and names the file, because a fallback nobody notices is the same hole with extra
    steps. Pass ``trusted=False`` on any path that a file you did not write can reach; a
    submitted model must never come through this function at all.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:
        if not trusted:
            raise CheckpointError(f"{path} could not be read as a weights-only "
                                  f"checkpoint: {error}") from error
        import warnings
        warnings.warn(
            f"{path} would not load under weights_only=True ({error}); re-reading it with "
            f"the unrestricted unpickler, which executes what the file contains. That is "
            f"acceptable for a checkpoint you wrote and is never acceptable for one you did "
            f"not — see training/loading.py.",
            RuntimeWarning, stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)
