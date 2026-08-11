"""What the observation looked like at each size it has ever had.

A checkpoint records the number of floats it was trained on and nothing else, which is enough
to *detect* an observation change and not enough to *survive* one: reconciling weights needs
to know which block grew and by how much, and 1,884 does not say.

So two things live here. New checkpoints carry their layout — :func:`signature` — and are
self-describing forever. Checkpoints written before that existed are looked up in
:data:`HISTORICAL`, which is a short table because this project has had few observations, and
which must **never be edited retroactively**: an entry is a statement about a file that
already exists on disk.

Adding a block, or widening one, needs **no entry here at all**: the checkpoints written
afterwards describe themselves, which is the whole point of :func:`signature`. The table stops
growing where it is. If the size is not in it and the checkpoint does not carry its own layout,
grafting refuses rather than guesses — a wrong guess produces a network that loads, runs, and
plays nonsense.
"""

from catan import encoder

#: Blocks in the order :func:`catan.encoder._build_layout` lays them out. A layout is
#: ``{name: (rows, features)}``, with ``rows`` 1 for the un-repeated blocks.
ORDER = ("tiles", "vertices", "roads", "players", "affordability", "history",
         "rolls", "global")


def signature(module=encoder):
    """The current observation layout, for recording in a checkpoint."""
    out = {}
    for name in ORDER:
        span = module.LAYOUT[name]
        width = span.stop - span.start
        rows, features = module.SHAPES.get(name, (1, width))
        out[name] = (rows, features)
    return out


#: Every observation this repository has shipped a checkpoint against.
#:
#: 1868  the original layout, before the affordability block
#: 1884  affordability added (record 0022)
#: 2503  per-vertex production and harbour reach, per-player production rate, board
#:       scarcity (record 0024)
HISTORICAL = {
    1868: {
        "tiles": (19, 19), "vertices": (54, 16), "roads": (72, 6),
        "players": (4, 29), "affordability": (0, 0), "history": (4, 12),
        "rolls": (1, 12), "global": (1, 35),
    },
    1884: {
        "tiles": (19, 19), "vertices": (54, 16), "roads": (72, 6),
        "players": (4, 29), "affordability": (4, 4), "history": (4, 12),
        "rolls": (1, 12), "global": (1, 35),
    },
}


def total(layout):
    return sum(rows * features for rows, features in layout.values())


def resolve(config):
    """The layout a checkpoint was trained against, or ``None`` if it cannot be known."""
    stored = config.get("layout")
    if stored:
        return {name: tuple(value) for name, value in stored.items()}
    size = config.get("obs_size")
    known = HISTORICAL.get(size)
    if known is not None:
        return dict(known)
    if size == encoder.SIZE:
        return signature()
    return None


def column_map(old, new=None):
    """For each column of a **new** observation, the old column it came from, or ``-1``.

    Only ever describes blocks that were *appended to* or added, which is the shape every
    change to this observation has taken and the shape the "append, never insert" rule in
    ``CLAUDE.md`` requires. A block whose features were reordered would need a different
    function and a much more careful one; this raises rather than pretending.
    """
    new = signature() if new is None else new
    mapping = []
    old_offset = 0
    offsets = {}
    for name in ORDER:
        rows, features = old.get(name, (0, 0))
        offsets[name] = old_offset
        old_offset += rows * features

    for name in ORDER:
        old_rows, old_features = old.get(name, (0, 0))
        rows, features = new[name]
        if old_features > features or old_rows > rows:
            raise ValueError(
                f"the {name!r} block shrank ({old_rows}x{old_features} -> {rows}x{features}); "
                f"this only reconciles blocks that grew"
            )
        for row in range(rows):
            for feature in range(features):
                if row < old_rows and feature < old_features:
                    mapping.append(offsets[name] + row * old_features + feature)
                else:
                    mapping.append(-1)
    return mapping


def row_map(name, old, new=None):
    """For each feature of a **new** row of ``name``, the old feature index, or ``-1``."""
    new = signature() if new is None else new
    _, old_features = old.get(name, (0, 0))
    _, features = new[name]
    return [f if f < old_features else -1 for f in range(features)]
