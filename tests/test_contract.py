"""The CATANIA-1 runtime contract, and the tripwire that keeps it honest.

Two jobs here, and they are different in kind.

The first is that every field the contract publishes agrees with the module that owns it. A
contract is only worth anything if it *cannot* disagree with the engine, so each assertion
below reaches into the real module rather than restating a number.

The second is :func:`test_the_rules_have_not_changed_without_the_version_changing`, which is
not a test of this module at all. It is a test of the rules, parked here because this is
where the consequence lands. Read its docstring before touching the expected digest.
"""

import json

import pytest

from catan import __version__, action_space, contract, encoder
from catan.rulesets import ALL, BASE_GAME, DEFAULT, RANKED_1V1


# =========================================================================== #
# DERIVED, NEVER RESTATED                                                     #
# =========================================================================== #

def test_the_observation_size_is_read_from_the_encoder():
    """Not written down here. `encoder.SIZE` went 1808 -> 1868 -> 1884 -> 2503 in six days,
    and a copy of it in a second file would have been wrong four times."""
    assert contract.describe()["observation"]["size"] == encoder.SIZE


def test_the_action_count_is_read_from_the_action_space():
    assert contract.describe()["action_space"]["count"] == action_space.NUM_ACTIONS


def test_the_engine_version_is_read_from_the_package():
    assert contract.ENGINE_VERSION == __version__


def test_the_signature_carries_every_field_that_can_invalidate_a_number():
    """A signature exists so that two figures may be compared if and only if it matches.
    That is only true if everything able to change the meaning of a weight appears in it."""
    signature = contract.signature()
    assert contract.MODEL_FORMAT in signature
    assert __version__ in signature
    assert str(encoder.SIZE) in signature
    assert str(action_space.NUM_ACTIONS) in signature
    assert contract.ruleset_id(DEFAULT) in signature


def test_two_rulesets_never_share_a_signature():
    """Ranked 1v1 and the base game are different games. A win rate in one says nothing
    about the other, and the string that would let a database pool them must differ."""
    assert contract.signature(RANKED_1V1) != contract.signature(BASE_GAME)


def test_every_ruleset_has_an_id():
    """A ruleset added without an id would raise at the point a game is recorded, which is
    the worst possible moment to find out."""
    for ruleset in ALL:
        assert contract.ruleset_id(ruleset).endswith(f"-v{contract.RULES_VERSION}")


def test_the_default_ruleset_is_what_you_get_for_nothing():
    assert contract.signature() == contract.signature(DEFAULT)
    assert contract.ruleset_id() == contract.ruleset_id(DEFAULT)


# =========================================================================== #
# READABLE BY SOMETHING THAT HAS NEVER HEARD OF THIS PACKAGE                  #
# =========================================================================== #

def test_the_description_is_plain_json():
    """The whole point of publishing it: a service pins a version and checks compatibility
    without importing the engine. An enum or a NamedTuple in there would serialise to
    something only Python can read back."""
    payload = contract.describe()
    assert json.loads(json.dumps(payload)) == payload


def test_the_description_names_the_blocks_and_not_only_the_totals():
    """2,503 floats could be many layouts, which is exactly why
    `training/alphazero/layouts.py` exists. A consumer needs to see *how* two sizes differ."""
    blocks = contract.describe()["observation"]["blocks"]
    assert blocks == {name: [span.start, span.stop]
                      for name, span in encoder.LAYOUT.items()}
    assert blocks["global"][1] == encoder.SIZE


def test_the_ruleset_block_states_the_mechanics_rather_than_only_the_name():
    """"ranked 1v1" is a label. 15 points and a hand limit of 9 are the game."""
    described = contract.describe(RANKED_1V1)["ruleset"]
    assert described["victory_points_to_win"] == RANKED_1V1.victory_points_to_win
    assert described["hand_limit"] == RANKED_1V1.hand_limit
    assert described["friendly_robber"] is RANKED_1V1.friendly_robber
    assert described["balanced_dice"] is RANKED_1V1.balanced_dice


# =========================================================================== #
# COMPATIBILITY IS EQUALITY                                                   #
# =========================================================================== #

def test_the_current_shapes_are_accepted():
    assert contract.accepts(encoder.SIZE, action_space.NUM_ACTIONS)


@pytest.mark.parametrize("observation, actions", [
    (1884, 325),      # the champion before the per-resource production block landed
    (1868, 325),      # models/champion.pt, promoted at this size
    (2503, 324),
    (2504, 325),
])
def test_a_model_built_for_other_shapes_is_refused(observation, actions):
    """A stale checkpoint loads perfectly well and then fails on the first move. Refusing it
    here is the difference between an error at startup and an error mid-game."""
    assert not contract.accepts(observation, actions)


def test_a_grafted_model_is_accepted_because_grafting_ends_at_the_current_shape():
    """`accepts` is equality, not compatibility, and that is not a gap in it.
    `network.graft` exists to widen an older checkpoint *into* the current observation; what
    comes out the far side is a current-shaped model and passes here. What must not happen is
    an ungrafted one being waved through on the grounds that a graft was available."""
    assert contract.accepts(encoder.SIZE, action_space.NUM_ACTIONS)
    assert not contract.accepts(1884, action_space.NUM_ACTIONS)


# =========================================================================== #
# THE TRIPWIRE                                                                #
# =========================================================================== #

#: What :func:`catan.contract.rules_digest` returns for the rules as they stand.
#:
#: ⚠️ **Do not update this on its own.** See the test below.
RULES_DIGEST = "57abae04b5c9a170"


def test_the_rules_have_not_changed_without_the_version_changing():
    """The rules are what they were when ``RULES_VERSION`` was last set.

    ⚠️ **If this fails, the expected string is not the thing to change first.**

    The digest moves when legality moves and when ``apply`` moves — a different effect puts
    the game in a different position, which offers a different legal set at the next
    decision. It does not move for a refactor, a speed-up, or anything else that leaves the
    game alone. So a failure means one of exactly two things:

    * **The rules changed on purpose.** Bump ``contract.RULES_VERSION`` and re-pin the
      digest, in the same commit. Every number recorded under the old version now belongs to
      a different game and the signature says so, which is the entire point.
    * **The rules changed by accident.** Which is the case this exists for. Commit
      ``e4b0441`` restricted pre-roll development-card plays and turned a recorded 71.6% into
      a measured 49.3% with nothing raising, and two recorded games in ``games/`` stopped
      replaying at some unknown point for the same reason. See
      ``docs/audit-2026-08-05-public-arena.md`` §B9.

    Updating the string alone converts the second case into the first, silently, which is
    worse than deleting the test.
    """
    assert contract.rules_digest() == RULES_DIGEST


def test_the_digest_is_a_pure_function_of_the_rules():
    """Twice in one process must agree, or the digest cannot pin anything. It draws from its
    own stream rather than `state.rng` for exactly this reason — drawing from the game's own
    RNG would make the dice depend on how many decisions had been taken."""
    assert contract.rules_digest() == contract.rules_digest()


def test_the_digest_covers_both_rulesets():
    """Friendly Robber and Balanced Dice are ranked-1v1 only, so a digest over one ruleset
    would be blind to a change in the other — and `catan.rules` is shared, so most changes
    reach both."""
    from catan.contract import DIGEST_SEEDS, rules_digest
    import catan.contract as module

    everything = rules_digest()
    original = module.ALL_RULESETS
    try:
        module.ALL_RULESETS = (RANKED_1V1,)
        assert rules_digest() != everything
    finally:
        module.ALL_RULESETS = original
    assert rules_digest() == everything
    assert len(DIGEST_SEEDS) >= 2
