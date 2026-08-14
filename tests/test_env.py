"""The Gymnasium-style environment."""

import random

import pytest

from catan import action_space, encoder, rules
from catan.actions import ActionType
from catan.dev_cards import DevCard
from catan.env import DEFAULT_MAX_TURNS, LOSS_REWARD, WIN_REWARD, CatanEnv
from catan.rulesets import ALL, BASE_GAME, RANKED_1V1
from catan.state import Phase
from helpers import give


def play(env, seed=0, limit=20_000, on_step=None):
    """Drive a whole game with random legal indices. Returns the final info."""
    obs, info = env.reset(seed=seed)
    rng = random.Random(seed ^ 0xA11CE)
    for _ in range(limit):
        if info["done"]:
            break
        obs, reward, terminated, truncated, info = env.step(rng.choice(info["legal"]))
        if on_step is not None:
            on_step(env, obs, reward, terminated, truncated, info)
    return info


# =========================================================================== #
# RESET                                                                       #
# =========================================================================== #

def test_reset_hands_back_a_playable_first_decision():
    env = CatanEnv()
    obs, info = env.reset(seed=0)

    assert len(obs) == env.observation_size == encoder.SIZE
    assert len(info["mask"]) == env.num_actions == action_space.NUM_ACTIONS
    assert any(info["mask"]), "the first decision must offer something"
    assert info["legal"] == [i for i, flag in enumerate(info["mask"]) if flag]
    assert info["done"] is False
    assert info["phase"] is Phase.SETUP_SETTLEMENT
    assert info["player"] in env.state.players


def test_a_seed_reproduces_the_whole_game():
    a, b = CatanEnv(), CatanEnv()
    first = play(a, seed=11)
    second = play(b, seed=11)
    assert a.state == b.state
    assert first["winner"] == second["winner"]
    assert first["scores"] == second["scores"]


def test_different_seeds_give_different_games():
    winners = {tuple(sorted(play(CatanEnv(), seed=s)["scores"].items()))
               for s in range(6)}
    assert len(winners) > 1


def test_reset_can_shuffle_turn_order():
    orders = set()
    for seed in range(20):
        env = CatanEnv(num_players=4)
        env.reset(seed=seed, randomize_order=True)
        orders.add(tuple(env.state.player_order))
    assert len(orders) > 1


def test_turn_order_is_fixed_unless_asked():
    for seed in range(5):
        env = CatanEnv(num_players=4)
        env.reset(seed=seed)
        assert env.state.player_order == [1, 2, 3, 4]


@pytest.mark.parametrize("ruleset", ALL, ids=lambda r: r.name)
def test_the_ruleset_reaches_the_state(ruleset):
    env = CatanEnv(ruleset=ruleset)
    env.reset(seed=1)
    assert env.state.ruleset is ruleset


# =========================================================================== #
# THE DICE ARE THE ENVIRONMENT'S JOB                                          #
# =========================================================================== #

def test_an_agent_is_never_asked_to_roll():
    """`roll_dice` is stochasticity, not a move, so it never reaches the agent."""
    def check(env, obs, reward, terminated, truncated, info):
        if not info["done"]:
            assert any(info["mask"]), \
                f"nothing to do in {info['phase'].name} — the env should have rolled"

    play(CatanEnv(), seed=4, on_step=check)


def test_the_environment_stops_for_a_pre_roll_development_card():
    """Playing a Knight before the roll is a real choice, so it must be offered."""
    env = CatanEnv()
    env.reset(seed=1)

    # play out setup, then hand the roller a Knight before the dice
    rng = random.Random(0)
    while env.state.in_setup:
        _, _, _, _, info = env.step(rng.choice(env._observe()[1]["legal"]))

    env.state.phase = Phase.ROLL
    env.state.rolled_this_turn = False
    env.state.dev_cards[env.state.current_player][DevCard.KNIGHT] = 1

    _, info = env._observe()
    assert info["phase"] is Phase.ROLL
    offered = {action_space.decode(i).type for i in info["legal"]}

    # Both halves of the choice. Playing a Knight before the roll is *optional*, so offering
    # only the card would force a player holding a Knight to burn it every single turn.
    assert offered == {ActionType.PLAY_KNIGHT, ActionType.ROLL}


def test_rolling_is_skipped_repeatedly_when_nobody_has_a_choice():
    """Several turns can pass with nothing to decide; the env must not stall on one."""
    env = CatanEnv()
    env.reset(seed=2)
    rng = random.Random(2)
    _, info = env._observe()
    while env.state.in_setup:
        _, _, _, _, info = env.step(rng.choice(info["legal"]))

    before = env.state.turn_number
    for _ in range(40):
        if info["done"]:
            break
        assert any(info["mask"])
        end_turn_index = action_space.encode(rules.end_turn())
        if info["mask"][end_turn_index]:
            _, _, _, _, info = env.step(end_turn_index)
        else:
            _, _, _, _, info = env.step(rng.choice(info["legal"]))
    assert env.state.turn_number > before


# =========================================================================== #
# WHOEVER MUST ACT IS THE OBSERVER                                            #
# =========================================================================== #

def test_the_observer_is_whoever_must_act_even_if_that_is_an_opponent():
    env = CatanEnv()
    env.reset(seed=1)
    rng = random.Random(0)
    _, info = env._observe()
    while env.state.in_setup:
        _, _, _, _, info = env.step(rng.choice(info["legal"]))

    state = env.state
    roller = state.turn_player
    other = next(p for p in state.players if p != roller)
    give(state, roller)
    give(state, other, wood=state.ruleset.hand_limit + 4)
    state.phase = Phase.ROLL
    state.rolled_this_turn = True
    rules.begin_robber(state)

    _, info = env._observe()
    assert info["phase"] is Phase.DISCARD
    assert info["player"] == other, "the discarding opponent must be the observer"


def test_the_observation_always_matches_the_reported_player():
    def check(env, obs, reward, terminated, truncated, info):
        if not info["done"]:
            assert obs == encoder.encode(env.state, info["player"])

    play(CatanEnv(num_players=3), seed=5, on_step=check)


def test_observe_works_for_any_player():
    """A self-play loop needs the losing side's view of the final position too."""
    env = CatanEnv()
    play(env, seed=3)
    for player in env.state.players:
        assert len(env.observe(player)) == encoder.SIZE


# =========================================================================== #
# ILLEGAL ACTIONS                                                             #
# =========================================================================== #

def test_an_illegal_index_raises_rather_than_being_ignored():
    """Silently substituting a legal move would teach an agent its choice is irrelevant."""
    env = CatanEnv()
    _, info = env.reset(seed=0)
    illegal = next(i for i in range(env.num_actions) if not info["mask"][i])
    with pytest.raises(rules.IllegalAction):
        env.step(illegal)


@pytest.mark.parametrize("bad", [-1, 325, 10_000])
def test_an_out_of_range_index_raises(bad):
    env = CatanEnv()
    env.reset(seed=0)
    with pytest.raises(IndexError):
        env.step(bad)


@pytest.mark.parametrize("bad", ["0", 1.0, None, True])
def test_a_non_integer_action_raises(bad):
    env = CatanEnv()
    env.reset(seed=0)
    with pytest.raises(TypeError):
        env.step(bad)


def test_stepping_before_reset_raises():
    with pytest.raises(RuntimeError):
        CatanEnv().step(0)


def test_stepping_after_the_game_is_over_raises():
    env = CatanEnv()
    play(env, seed=3)
    assert env.state.phase is Phase.GAME_OVER
    with pytest.raises(RuntimeError):
        env.step(0)


# =========================================================================== #
# REWARDS                                                                     #
# =========================================================================== #

def test_reward_is_zero_until_the_game_ends():
    rewards = []

    def check(env, obs, reward, terminated, truncated, info):
        rewards.append((reward, terminated))

    play(CatanEnv(), seed=3, on_step=check)
    assert all(reward == 0.0 for reward, terminated in rewards[:-1])
    assert rewards[-1][1] is True


def test_the_winner_is_rewarded_and_the_loser_is_penalised():
    env = CatanEnv()
    obs, info = env.reset(seed=3)
    rng = random.Random(3)
    final_reward, actor = None, None
    while not info["done"]:
        actor = info["player"]
        obs, final_reward, terminated, truncated, info = env.step(
            rng.choice(info["legal"]))

    assert terminated and env.state.winner is not None
    expected = WIN_REWARD if env.state.winner == actor else LOSS_REWARD
    assert final_reward == expected
    # the game is zero-sum: whoever acted last either won or lost
    assert final_reward in (WIN_REWARD, LOSS_REWARD)


def test_a_finished_game_reports_a_winner_who_meets_the_target():
    env = CatanEnv(ruleset=RANKED_1V1)
    info = play(env, seed=3)
    assert info["winner"] is not None
    assert info["scores"][info["winner"]] >= RANKED_1V1.victory_points_to_win


# =========================================================================== #
# TRUNCATION                                                                  #
# =========================================================================== #

def test_truncation_is_reported_separately_from_termination():
    """A learner must not read a time-out as a loss."""
    env = CatanEnv(max_turns=6)
    obs, info = env.reset(seed=0)
    rng = random.Random(0)

    truncated = terminated = False
    for _ in range(4000):
        if info["done"]:
            break
        obs, reward, terminated, truncated, info = env.step(rng.choice(info["legal"]))

    assert truncated and not terminated
    assert env.state.winner is None
    assert info["winner"] is None
    assert reward == 0.0, "a truncated game must not pay out"


def test_the_default_turn_cap_is_generous_enough_to_finish():
    assert DEFAULT_MAX_TURNS >= 5_000
    env = CatanEnv()
    info = play(env, seed=3)
    assert info["winner"] is not None


# =========================================================================== #
# INFO                                                                        #
# =========================================================================== #

def test_info_reports_your_own_true_score_because_you_can_see_your_own_hand():
    env = CatanEnv()
    env.reset(seed=1)
    state = env.state
    me = state.current_player
    state.dev_cards[me][DevCard.VICTORY_POINT] = 2
    _, info = env._observe()

    assert info["player"] == me
    assert info["scores"][me] == rules.victory_points(state, me)
    assert info["public_scores"][me] == rules.public_victory_points(state, me)
    assert info["scores"][me] - info["public_scores"][me] == 2


def test_info_does_not_report_an_opponents_hidden_victory_point_cards():
    """⚠️ This test used to assert the opposite.

    ``info["scores"]`` was ``rules.scores(state)`` unconditionally and sat next to
    ``public_scores``, so one subtraction gave an opponent's exact hidden Victory Point count
    — the fact that decides whether they are one build from 15. It is not a sandbox escape:
    it is readable by a well-behaved agent using only documented, public keys, which is why
    neither the observation's leak tests nor ``PublicView`` caught it. See
    ``docs/audit-2026-08-05-public-arena.md`` §B3.
    """
    env = CatanEnv()
    env.reset(seed=1)
    state = env.state
    me = state.current_player
    opponent = next(p for p in state.players if p != me)
    state.dev_cards[opponent][DevCard.VICTORY_POINT] = 2
    _, info = env._observe()

    assert info["scores"][opponent] == info["public_scores"][opponent]
    assert info["scores"][opponent] != rules.victory_points(state, opponent)


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_nothing_in_info_moves_when_the_hidden_state_is_scrambled(seed):
    """The leak test the ``info`` dict never had.

    ``tests/test_encoder.py`` scrambles for the observation and ``tests/test_view.py`` for
    :class:`~catan.view.PublicView`, and between them they were taken to cover what an agent
    can see. They do not: ``info`` is a third channel, handed to the agent at every decision,
    and ``info["scores"]`` was leaking through it in plain sight for as long as it existed.

    So: at *every* decision of three whole games, clone the position, rewrite everything the
    mover may not see at constant public counts, and demand that every key an agent reads
    comes back identical. ``view`` is excluded because it is an object rather than a value —
    it has its own leak test, and its whole surface is checked there.
    """
    from helpers import scramble_hidden_state

    READ_BY_AGENTS = ("player", "mask", "legal", "phase", "turn", "last_roll",
                      "scores", "public_scores", "winner", "done")

    env = CatanEnv()
    observation, info = env.reset(seed=seed)
    rng = random.Random(seed ^ 0xF00D)
    decisions = 0

    while not info["done"] and decisions < 2_000:
        scrambled = env.clone()
        scramble_hidden_state(scrambled.state, info["player"])
        shadow_observation, shadow = scrambled._observe()

        for key in READ_BY_AGENTS:
            assert shadow[key] == info[key], (
                f"info[{key!r}] moved when only hidden state changed, at decision "
                f"{decisions} of seed {seed}")
        assert shadow_observation == observation

        observation, _, _, _, info = env.step(rng.choice(info["legal"]))
        decisions += 1

    assert decisions > 50, "the game ended too early to have tested anything"


def test_every_score_is_revealed_once_the_game_is_over():
    """The winner turns their cards over, so nothing is hidden any more — and the recorder
    needs the true final margin. `interfaces.web.stats` keys its `reveal` on the same fact."""
    env = CatanEnv()
    env.reset(seed=1)
    state = env.state
    opponent = next(p for p in state.players if p != state.current_player)
    state.dev_cards[opponent][DevCard.VICTORY_POINT] = 2

    _, info = env._observe(done=True)
    assert info["scores"][opponent] == rules.victory_points(state, opponent)
    assert info["scores"][opponent] - info["public_scores"][opponent] == 2


def test_info_carries_the_phase_turn_and_last_roll():
    env = CatanEnv()
    _, info = env.reset(seed=1)
    assert info["turn"] == 0
    assert info["last_roll"] is None

    info = play(env, seed=1)
    assert info["turn"] > 0
    assert 2 <= info["last_roll"] <= 12


# =========================================================================== #
# CLONING                                                                     #
# =========================================================================== #

def test_a_clone_is_independent_of_its_original():
    env = CatanEnv()
    env.reset(seed=1)
    rng = random.Random(0)
    for _ in range(30):
        _, info = env._observe()
        if info["done"]:
            break
        env.step(rng.choice(info["legal"]))

    clone = env.clone()
    assert clone.state == env.state
    assert clone.state is not env.state
    assert clone.state.board is env.state.board

    _, info = clone._observe()
    clone.step(rng.choice(info["legal"]))
    assert clone.state != env.state


def test_a_clone_carries_the_configuration():
    env = CatanEnv(num_players=3, ruleset=BASE_GAME, max_turns=99)
    env.reset(seed=1)
    clone = env.clone()
    assert clone.num_players == 3
    assert clone.ruleset is BASE_GAME
    assert clone.max_turns == 99


# =========================================================================== #
# WHOLE GAMES                                                                 #
# =========================================================================== #

@pytest.mark.slow
@pytest.mark.parametrize("ruleset,num_players", [
    (BASE_GAME, 2),
    (BASE_GAME, 4),
    (RANKED_1V1, 2),      # the format's intended player count
], ids=["base-2p", "base-4p", "ranked-2p"])
def test_random_play_finishes_in_the_intended_configurations(ruleset, num_players):
    finished = 0
    for seed in range(4):
        env = CatanEnv(num_players=num_players, ruleset=ruleset)
        info = play(env, seed=seed)
        if info["winner"] is not None:
            finished += 1
    assert finished >= 3, f"{finished}/4 finished for {ruleset.name}, {num_players}p"


@pytest.mark.slow
def test_ranked_1v1_rules_at_four_players_truncate_cleanly_rather_than_breaking():
    """15 points is calibrated for two players and does not reliably terminate at four.

    Measured: random 4-player games under RANKED_1V1 plateau at 13-14 points and hit the
    turn cap — the board is too crowded for four players to each place the nine buildings
    that 15 points needs. That is the ruleset being used outside its format, not a bug.

    What must hold is that it *truncates cleanly*: no crash, no deadlock, a valid final
    state, no winner, and no reward. A learner reading truncation as a loss would be
    learning from noise.
    """
    env = CatanEnv(num_players=4, ruleset=RANKED_1V1, max_turns=250)
    obs, info = env.reset(seed=0)
    rng = random.Random(0)

    truncated = terminated = False
    reward = None
    for _ in range(40_000):
        if info["done"]:
            break
        assert any(info["mask"]), "deadlocked with nothing to do"
        obs, reward, terminated, truncated, info = env.step(rng.choice(info["legal"]))

    assert truncated and not terminated
    assert env.state.winner is None and info["winner"] is None
    assert reward == 0.0
    assert len(obs) == encoder.SIZE


@pytest.mark.slow
def test_the_mask_is_never_wrong_across_a_whole_game():
    """Every offered index must be applicable, at every step of a real game."""
    def check(env, obs, reward, terminated, truncated, info):
        if info["done"]:
            return
        for index in info["legal"]:
            rules.apply(env.state.clone(), action_space.decode(index))

    play(CatanEnv(num_players=3), seed=6, limit=1200, on_step=check)
