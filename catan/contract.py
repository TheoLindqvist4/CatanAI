"""What a model has to agree with before it is allowed to play — the CATANIA-1 contract.

A trained model is a pile of numbers that means something only against a *particular*
observation, a particular action space and a particular set of rules. Change any of the
three and the same weights compute a different function of the game. This module is the one
place that says which three, so that a number written down anywhere — a win rate, a rating,
a match result — can name the world it was measured in.

    from catan import contract
    contract.signature()
    'CATANIA-1/engine-0.1.0/obs-1:2503/act-1:325/catan-1v1-v1'

**Derived, never restated.** Every field here is read out of the module that owns it —
:data:`catan.encoder.SIZE`, :data:`catan.action_space.NUM_ACTIONS`,
:mod:`catan.rulesets` — because a contract that is *able* to disagree with the engine is
worse than no contract: it would be believed. The only hand-written numbers are the version
counters, and :func:`rules_digest` exists to stop those being forgotten.

Why this module exists
----------------------
It has already cost this project twice, and both times silently.

``models/champion.json`` records 71.6% against the heuristic. The same weights re-measure at
49.3% [41.4, 57.3], because commit ``e4b0441`` restricted pre-roll development-card plays
*after* the promotion. The figure was never wrong; it simply stopped describing the current
game, and nothing in the file could say so.

Separately, of the 19 recorded games in ``games/``, two no longer replay — ``unknown action
ROLL(0)`` and ``must roll the dice before PLAY_YEAR_OF_PLENTY(ore, ore)``. Seed-plus-moves is
this project's cheapest storage format and its audit primitive, and it turns out to be
silently scoped to a version that was never recorded beside it.

Both are the same missing fact. See ``docs/audit-2026-08-05-public-arena.md`` §B9.

The tripwire
------------
:data:`RULES_VERSION` is hand-written, so it can be forgotten — which is exactly what
happened. :func:`rules_digest` plays a handful of seeded games and hashes what the rules
*offered* at every decision, and ``tests/test_contract.py`` pins the result. So a change to
legality or to ``apply`` fails a test whose only remedy is to bump the version and re-pin the
digest in the same commit. The version becomes a deliberate statement rather than a
convention nobody remembers.

That is the same enforcement the geometry already gets: ``tests/test_topology.py``
transcribes the ids from ``Images/`` independently, so the diagrams and the code cannot
drift.
"""

import hashlib

from catan import __version__, action_space, encoder
from catan.env import CatanEnv
from catan.rulesets import ALL as ALL_RULESETS
from catan.rulesets import BASE_GAME, DEFAULT, RANKED_1V1

#: The name of this contract. A model file that does not claim it is not offered a game.
MODEL_FORMAT = "CATANIA-1"

#: The engine: the rules, the geometry and the state model. Bumped by hand, defended by
#: :func:`rules_digest`.
ENGINE_VERSION = __version__

#: Bumped when *any* ruleset's mechanics change, so a recorded number can never silently
#: belong to a different game. It is deliberately not per-ruleset: the shared machinery —
#: legality, the robber, the awards — is most of what a variant is, and a change there
#: affects every format.
RULES_VERSION = 1

#: Bumped when the meaning of a column changes. **Appending does not bump it**: an appended
#: block leaves every existing column where it was, and ``training.alphazero.network.graft``
#: widens a checkpoint into it with zero columns, so the grafted network computes exactly the
#: function it computed before. Reordering or repurposing a column does bump it, because no
#: graft can rescue that.
OBSERVATION_VERSION = 1

#: Bumped when an existing index changes meaning. Appending does not — ``docs`` and
#: ``training/structured_net.py::_validate`` both hold the action space to *append, never
#: insert*, and an appended index lands correctly with no code change.
ACTION_SPACE_VERSION = 1

#: Network architectures a submitted checkpoint may claim, keyed by the ``kind`` in its
#: config. The absence of ``kind`` means ``flat``, which predates the field — see
#: :func:`training.net.build`.
NETWORKS = {
    "flat": "flat-v1",
    "structured": "structured-v1",
}

#: How a model may be asked for a move. ``policy`` is one forward pass and its argmax;
#: ``alphazero`` is the same network under PUCT over a determinized information set. They are
#: different players at different costs, which is why a rating names one — see
#: ``docs/audit-2026-08-05-public-arena.md`` §5 on ``(weights, simulations)`` identity.
INFERENCE_MODES = ("policy", "alphazero")

#: The only accepted serialization. Emphatically *not* "a PyTorch file": a checkpoint is
#: loaded with ``weights_only=True`` and nothing else, because ``torch.load`` on an
#: untrusted pickle is arbitrary code execution on the host. See :mod:`training.loading`.
SERIALIZATION = "torch-state-dict/weights-only"

#: The largest weight file that will be looked at. The champions are under 1.6 MB; this is
#: room for a much larger network and a refusal well before anything can exhaust memory.
MAX_MODEL_BYTES = 64 * 1024 * 1024


def ruleset_id(ruleset=None):
    """The published identifier for a ruleset, e.g. ``catan-1v1-v1``.

    Two rulesets that differ in any field are different games and get different ids, so a
    result recorded under one can never be pooled with a result recorded under the other.
    """
    ruleset = DEFAULT if ruleset is None else ruleset
    return f"{_RULESET_SLUGS[ruleset.name]}-v{RULES_VERSION}"


_RULESET_SLUGS = {
    RANKED_1V1.name: "catan-1v1",
    BASE_GAME.name: "catan-base",
}


def signature(ruleset=None):
    """One string naming the world a number was measured in.

    Goes in every game row, every rating row and every model record. Two figures may be
    compared if and only if their signatures are equal — which is a rule a database can
    enforce, unlike "we think the rules were the same in August".
    """
    return "/".join((
        MODEL_FORMAT,
        f"engine-{ENGINE_VERSION}",
        f"obs-{OBSERVATION_VERSION}:{encoder.SIZE}",
        f"act-{ACTION_SPACE_VERSION}:{action_space.NUM_ACTIONS}",
        ruleset_id(ruleset),
    ))


def describe(ruleset=None):
    """The whole contract as JSON-ready data, for a health endpoint or a model record.

    Plain builtins only — no imports beyond the standard library are needed to read it, which
    is what lets a service that has never heard of this package check compatibility.
    """
    ruleset = DEFAULT if ruleset is None else ruleset
    return {
        "model_format": MODEL_FORMAT,
        "signature": signature(ruleset),
        "engine_version": ENGINE_VERSION,
        "rules_version": RULES_VERSION,
        "ruleset": {
            "id": ruleset_id(ruleset),
            "name": ruleset.name,
            "victory_points_to_win": ruleset.victory_points_to_win,
            "hand_limit": ruleset.hand_limit,
            "friendly_robber": ruleset.friendly_robber,
            "balanced_dice": ruleset.balanced_dice,
        },
        "observation": {
            "version": OBSERVATION_VERSION,
            "size": encoder.SIZE,
            # The block shapes, so a consumer can tell *how* two sizes differ rather than
            # only that they do. `training.alphazero.layouts` reconciles a checkpoint with
            # exactly this information.
            "blocks": {name: [span.start, span.stop]
                       for name, span in encoder.LAYOUT.items()},
        },
        "action_space": {
            "version": ACTION_SPACE_VERSION,
            "count": action_space.NUM_ACTIONS,
            "blocks": {kind.name: [span.start, span.stop]
                       for kind, span in action_space.SLICES.items()},
        },
        "networks": dict(NETWORKS),
        "inference_modes": list(INFERENCE_MODES),
        "serialization": SERIALIZATION,
        "max_model_bytes": MAX_MODEL_BYTES,
    }


def accepts(observation_size, action_count):
    """Whether a model built for these shapes may be run against this engine.

    Equality, not compatibility. A checkpoint from an older observation loads perfectly well
    and then plays nonsense, which is the failure this answers — a caller that wants the
    older one anyway must go through ``training.alphazero.network.graft`` deliberately and
    end up at the current shapes.
    """
    return (observation_size == encoder.SIZE
            and action_count == action_space.NUM_ACTIONS)


# --------------------------------------------------------------------------- #
# The tripwire                                                                #
# --------------------------------------------------------------------------- #

#: Games played by :func:`rules_digest`. Two per ruleset, played out under a fixed stream:
#: enough that every phase, the robber, discards, development cards, harbour trading and both
#: awards all occur, and few enough to cost about two seconds.
DIGEST_SEEDS = (17, 4_242)

#: Bounded so a pathological game cannot make the digest slow. Games under a random policy
#: run long — 828 decisions is typical — and truncation is fine here: what is being hashed is
#: what the rules offered, not who won.
DIGEST_MAX_TURNS = 400


def rules_digest():
    """A fingerprint of what the rules *offer*, over a fixed set of seeded games.

    Plays each seed in each ruleset with a deterministic stream and hashes, at every single
    decision, the turn, the phase, whose decision it is and **the exact set of legal
    actions**. Then the terminal position.

    So it moves if legality moves, and it moves if ``apply`` moves — a different effect puts
    the game in a different position, which offers a different legal set at the next
    decision. It does *not* move for a refactor, a performance change, or anything else that
    leaves the game the players are playing alone. That is the distinction
    :data:`RULES_VERSION` is supposed to track and cannot track by itself.

    Pinned by ``tests/test_contract.py``. When that test fails, the fix is never to update
    the expected string on its own: bump :data:`RULES_VERSION` in the same commit, or
    establish that the change was not meant to reach the rules.
    """
    import random

    digest = hashlib.sha256()
    for ruleset in ALL_RULESETS:
        digest.update(f"|ruleset:{ruleset.name}|".encode())
        for seed in DIGEST_SEEDS:
            env = CatanEnv(num_players=2, ruleset=ruleset, max_turns=DIGEST_MAX_TURNS)
            _, info = env.reset(seed=seed)
            # Its own stream, seeded from the game's seed. Not `state.rng`: drawing from
            # that would change the dice, so the digest would depend on how many decisions
            # were made rather than on what was offered at each of them.
            stream = random.Random(seed ^ 0x5EED)
            while not info["done"]:
                legal = info["legal"]
                digest.update(f"{info['turn']}:{info['phase'].name}:"
                              f"{info['player']}:{','.join(map(str, legal))};".encode())
                _, _, _, _, info = env.step(stream.choice(legal))
            digest.update(f"end:{info['winner']}:"
                          f"{sorted(info['public_scores'].items())};".encode())
    return digest.hexdigest()[:16]
