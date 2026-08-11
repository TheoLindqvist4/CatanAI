"""The AlphaZero pipeline: the leak boundary, the search, the buffer, and the checklist.

Section 21 of the design guide is a list of ten things to establish before a long run, and
most of this file is those ten written as assertions. They are cheap and they are the
difference between a run that failed and a run that was never going to work:

* deterministic with a fixed seed          :func:`test_generation_is_deterministic`
* legal action mask correct                :func:`test_search_never_returns_an_illegal_move`
* state tensor fixed shape                 :func:`test_replay_round_trips_a_real_observation`
* action ids stable                        ``tests/test_action_space.py`` already
* checkpoint save/load works               :func:`test_checkpoint_round_trips`
* self-play parallelism stable             :func:`test_parallel_pool_generates`
* no rendering or logging during training  :func:`test_generation_is_silent`

The first section is the one that matters most, and it is not on the guide's list because the
guide assumes perfect information. **Search must not see what the player may not see.**
:func:`test_determinize_ignores_hidden_state` holds this package to the same standard the
encoder and the heuristic are held to, using the same scrambler.
"""

import collections
import io
import json
import random
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catan import action_space, encoder, rules
from catan.agents import HeuristicAgent
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.resources import BANK_PER_RESOURCE, NUM_RESOURCES, Resource, total
from catan.rulesets import RANKED_1V1
from catan.state import GameState, Phase
from tests.helpers import complete_setup, scramble_hidden_state
from training.alphazero import replay_buffer
from training.alphazero.config import Config, parse
from training.alphazero.determinize import determinize, unseen_dev_cards, unseen_resources
from training.alphazero.mcts import CHANCE, DECISION, TERMINAL, Search
from training.alphazero.network import graft, new_network
from training.alphazero.replay_buffer import ReplayBuffer, pack_mask, sparse_policy
from training.alphazero.self_play import Generator, to_arrays


# --------------------------------------------------------------------------- #
# A position part-way through a real game, for everything below to work on.   #
# --------------------------------------------------------------------------- #

def played(seed=7, moves=120, ruleset=RANKED_1V1):
    """A state some way into a game, reached by legal play."""
    state = GameState(num_players=2, seed=seed, ruleset=ruleset)
    rng = random.Random(seed)
    complete_setup(state, rng)
    for _ in range(moves):
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.ROLL:
            legal = rules.legal_actions(state)
            if legal and rng.random() < 0.3:
                rules.apply(state, rng.choice(legal))
            else:
                rules.roll_dice(state)
            continue
        legal = rules.legal_actions(state)
        if not legal:
            break
        rules.apply(state, rng.choice(legal))
    return state


def decision_state():
    """A state with a real choice at the root — more than one legal action.

    ``played()`` with its default arguments lands on a ROLL with *zero* legal actions, so
    every test that opened a search on it skipped rather than ran, silently and forever.
    Searching seeds for a decision is two lines and makes the difference between a test and
    a decoration.
    """
    for seed in range(1, 40):
        state = played(seed=seed, moves=120)
        if len(action_space.legal_indices(state)) > 1:
            return state
    pytest.skip("no searchable position found")


def extra_columns(rows, game_id=0):
    """The auxiliary columns :meth:`ReplayBuffer.add` takes beside the original five.

    A helper rather than four literals at every call site: record 0026 added root value,
    final ownership, final margin and a game id, and a test that spelled them out would have
    to be edited again by the next person who adds one.
    """
    return (
        np.zeros(rows, dtype=np.float16),                                    # root_value
        np.zeros((rows, replay_buffer.OWNER_COLUMNS), dtype=np.int8),        # owners
        np.zeros(rows, dtype=np.float16),                                    # margin
        np.full(rows, game_id, dtype=np.int64),                              # game_id
    )


def stub_evaluator(seed=0):
    """A network-shaped function that needs no network. Deterministic given the mask."""
    rng = np.random.default_rng(seed)

    def evaluate(obs, masks):
        scores = rng.random(masks.shape) * masks
        totals = scores.sum(axis=1, keepdims=True)
        probabilities = np.divide(scores, np.where(totals > 0, totals, 1.0))
        return probabilities, np.zeros(len(obs), dtype=np.float32)

    return evaluate


# =========================================================================== #
# The leak boundary                                                          #
# =========================================================================== #

def test_determinize_ignores_hidden_state():
    """Scrambling what the observer may not see must not change the world search gets.

    The same standard, and the same scrambler, that ``tests/test_encoder.py`` holds the
    observation to and ``tests/test_agents.py`` holds the heuristic to. If this fails, the
    search is reading the opponent's hand or the top of the deck — which would not crash,
    would measure as a *stronger* agent, and would produce policy targets the network has no
    way to reproduce from the observation it is given.
    """
    state = played()
    other = state.clone(rng=random.Random(0))
    scramble_hidden_state(other, me=1)

    first = determinize(state, 1, rng=random.Random(99))
    second = determinize(other, 1, rng=random.Random(99))

    assert first.hands == second.hands
    assert first.dev_cards == second.dev_cards
    assert first.dev_cards_new == second.dev_cards_new
    assert first.dev_deck == second.dev_deck
    assert first.dice_deck == second.dice_deck


def test_determinize_keeps_every_public_count():
    state = played()
    world = determinize(state, 1, rng=random.Random(3))

    assert world.hands[1] == state.hands[1], "the observer's own hand must survive"
    assert world.dev_cards[1] == state.dev_cards[1]
    assert world.bank == state.bank
    for player in state.players:
        assert total(world.hands[player]) == total(state.hands[player])
        assert sum(world.dev_cards[player]) == sum(state.dev_cards[player])
        assert sum(world.dev_cards_new[player]) == sum(state.dev_cards_new[player])
    assert len(world.dev_deck) == len(state.dev_deck)
    assert len(world.dice_deck) == len(state.dice_deck)
    assert world.vertex_owner == state.vertex_owner
    assert world.edge_owner == state.edge_owner
    assert world.robber_tile == state.robber_tile
    assert world.turn_number == state.turn_number
    assert world.phase is state.phase


def test_determinize_does_not_touch_the_original():
    state = played()
    before = state.clone(rng=random.Random(0))
    determinize(state, 1, rng=random.Random(5))
    assert state == before


def test_unseen_resources_reads_only_public_counts():
    state = played()
    expected = [BANK_PER_RESOURCE - state.bank[r] - state.hands[1][r]
                for r in range(NUM_RESOURCES)]
    assert unseen_resources(state, 1) == [max(0, e) for e in expected]

    # The opponent's own hand must not appear in the answer.
    state.hands[2] = [0] * NUM_RESOURCES
    state.hands[2][Resource.ORE] = 4
    assert unseen_resources(state, 1) == [max(0, e) for e in expected]


def test_unseen_dev_cards_excludes_my_own_and_played_knights():
    state = played()
    pool = unseen_dev_cards(state, 1)
    assert all(count >= 0 for count in pool)
    assert sum(pool) <= 25
    # A knight I hold is not unseen to me.
    state.dev_cards[1][DevCard.KNIGHT] += 1
    assert unseen_dev_cards(state, 1)[DevCard.KNIGHT] == pool[DevCard.KNIGHT] - 1


def test_determinized_world_is_playable():
    """A resampled world must still be a legal game the engine can be driven through."""
    state = played()
    world = determinize(state, 1, rng=random.Random(11))
    assert action_space.legal_indices(world) == action_space.legal_indices(state), (
        "legality at the root depends only on public facts and the observer's own hand, so "
        "determinization must not change it"
    )
    for _ in range(40):
        if world.phase is Phase.GAME_OVER:
            break
        if world.phase is Phase.ROLL and not rules.legal_actions(world):
            rules.roll_dice(world)
            continue
        legal = rules.legal_actions(world)
        if not legal:
            break
        rules.apply(world, legal[0])


def test_agent_decision_is_unchanged_by_hidden_state():
    """The whole agent, not just the determinizer, must be blind to the hidden state."""
    from training.alphazero.agent import MCTSAgent

    env = CatanEnv(num_players=2, ruleset=RANKED_1V1)
    observation, info = env.reset(seed=4)
    for _ in range(60):
        if info["done"] or len(info["legal"]) > 3:
            break
        observation, _, _, _, info = env.step(info["legal"][0])

    net = new_network()
    torch.manual_seed(0)

    honest = MCTSAgent(net, simulations=12, temperature=0.0, seed=1)
    first = honest(observation, info)

    scramble_hidden_state(env.state, me=info["player"])
    cheat = MCTSAgent(net, simulations=12, temperature=0.0, seed=1)
    second = cheat(observation, info)

    assert first == second


# =========================================================================== #
# The search                                                                 #
# =========================================================================== #

def test_search_classifies_a_roll_as_a_chance_node():
    state = played(moves=30)
    while state.phase is not Phase.ROLL or rules.legal_actions(state):
        if state.phase is Phase.GAME_OVER:
            pytest.skip("game ended before a plain roll")
        legal = rules.legal_actions(state)
        if not legal:
            rules.roll_dice(state)
            continue
        rules.apply(state, legal[0])
    search = Search(state, budget=4, rng=np.random.default_rng(0))
    assert search.root.kind is CHANCE
    assert not search.searchable
    assert search.forced is None


def test_search_produces_visits_over_legal_actions_only():
    state = played()
    if not action_space.legal_indices(state):
        pytest.skip("no decision at this position")
    search = Search(state, budget=32, rng=np.random.default_rng(0), noise=0.0)
    if not search.searchable:
        pytest.skip("forced move")

    evaluate = stub_evaluator()
    while (pending := search.request()) is not None:
        obs, mask = pending
        probabilities, values = evaluate(obs[None, :], mask[None, :])
        search.deliver(probabilities[0], float(values[0]))

    actions, counts = search.visit_counts()
    legal = set(action_space.legal_indices(state))
    assert set(actions.tolist()) <= legal
    assert counts.sum() > 0
    assert search.best_action(0.0) in legal


def test_search_never_returns_an_illegal_move():
    """Over a whole game, every move the search picks must pass the engine's own mask."""
    env = CatanEnv(num_players=2, ruleset=RANKED_1V1, max_turns=60)
    _, info = env.reset(seed=12)
    evaluate = stub_evaluator(1)
    steps = 0

    while not info["done"] and steps < 200:
        state = env.state
        search = Search(determinize(state, info["player"], rng=random.Random(steps)),
                        budget=8, rng=np.random.default_rng(steps), noise=0.0,
                        max_turns=env.max_turns)
        if search.searchable:
            while (pending := search.request()) is not None:
                obs, mask = pending
                probabilities, values = evaluate(obs[None, :], mask[None, :])
                search.deliver(probabilities[0], float(values[0]))
            action = search.best_action(0.0)
        else:
            action = search.forced
        assert info["mask"][action], f"{action_space.describe(action)} is not legal"
        _, _, _, _, info = env.step(action)
        steps += 1


def test_terminal_value_is_zero_sum_and_signed_by_seat():
    """A won position must be +1 for the winner and -1 for the loser, whoever moved last."""
    state = played()
    state.phase = Phase.GAME_OVER
    state.winner = 2
    search = Search(state, budget=1, rng=np.random.default_rng(0))
    assert search.root.kind is TERMINAL
    assert search._terminal_value(search.root) == -1.0        # seat 1's frame

    state.winner = 1
    other = Search(state, budget=1, rng=np.random.default_rng(0))
    assert other._terminal_value(other.root) == 1.0

    state.winner = None
    state.phase = Phase.BUILD
    state.turn_number = 10_000
    draw = Search(state, budget=1, rng=np.random.default_rng(0), max_turns=10)
    assert draw.root.kind is TERMINAL
    assert draw._terminal_value(draw.root) == 0.0


def test_orient_is_its_own_inverse():
    for value in (-1.0, -0.3, 0.0, 0.7, 1.0):
        for player in (1, 2):
            there = Search._orient(value, player)
            assert Search._orient(there, player) == pytest.approx(value)


def test_search_refuses_more_than_two_players():
    state = GameState(num_players=3, seed=0)
    with pytest.raises(ValueError, match="two players"):
        Search(state, budget=4)


def test_search_stays_within_its_budget():
    search = Search(decision_state(), budget=17, rng=np.random.default_rng(0), noise=0.0)
    assert search.searchable
    evaluate = stub_evaluator()
    calls = 0
    while (pending := search.request()) is not None:
        obs, mask = pending
        probabilities, values = evaluate(obs[None, :], mask[None, :])
        search.deliver(probabilities[0], float(values[0]))
        calls += 1
    assert search.simulations == 17
    assert calls <= 17


# =========================================================================== #
# The Gumbel root                                                            #
# =========================================================================== #

def gumbel_search(budget=64, seed=0, **kwargs):
    """A searched position and its search. Shared by the tests below."""
    state = decision_state()
    search = Search(state, budget=budget, rng=np.random.default_rng(seed),
                    noise=0.0, gumbel=True, max_turns=400, **kwargs)
    assert search.searchable
    evaluate = stub_evaluator(seed)
    while (pending := search.request()) is not None:
        obs, mask = pending
        probabilities, values = evaluate(obs[None, :], mask[None, :])
        search.deliver(probabilities[0], float(values[0]))
    return search


def test_gumbel_target_is_a_distribution_over_legal_actions_only():
    search = gumbel_search()
    actions, weights = search.policy_target()
    assert len(actions) == len(search.root.actions)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    legal = set(action_space.legal_indices(search.root.state))
    assert set(actions.tolist()) <= legal


def test_gumbel_stays_within_its_budget():
    """Sequential Halving must not overrun, including when the budget does not divide."""
    for budget in (7, 16, 31, 64):
        search = gumbel_search(budget=budget)
        assert search.simulations == budget


def test_gumbel_picks_a_candidate_it_actually_searched():
    search = gumbel_search()
    chosen = search.best_action()
    survivors = {int(search.root.actions[slot]) for slot in search._candidates}
    assert chosen in survivors, "the played move must come from the surviving set"
    assert search.root.child_n[list(search._candidates)].min() > 0


def test_plain_search_target_is_still_the_visit_counts():
    """``policy_target`` must not change what a non-Gumbel search learns from."""
    search = Search(decision_state(), budget=32, rng=np.random.default_rng(1), noise=0.0)
    assert search.searchable
    evaluate = stub_evaluator(1)
    while (pending := search.request()) is not None:
        obs, mask = pending
        probabilities, values = evaluate(obs[None, :], mask[None, :])
        search.deliver(probabilities[0], float(values[0]))

    actions, weights = search.policy_target()
    counts_actions, counts = search.visit_counts()
    assert actions.tolist() == counts_actions.tolist()
    assert weights == pytest.approx(counts / counts.sum())


def test_gumbel_does_not_add_dirichlet_noise():
    """Gumbel replaces the root noise; applying both would be exploration twice."""
    search = Search(decision_state(), budget=8, rng=np.random.default_rng(2), noise=0.9,
                    gumbel=True, max_turns=400)
    assert search.searchable
    evaluate = stub_evaluator(2)
    pending = search.request()
    obs, mask = pending
    probabilities, values = evaluate(obs[None, :], mask[None, :])
    search.deliver(probabilities[0], float(values[0]))

    expected = probabilities[0][search.root.actions]
    expected = expected / expected.sum()
    assert search.root.prior == pytest.approx(expected), (
        "a noise weight of 0.9 would be unmissable here if it had been applied"
    )


# =========================================================================== #
# Generation                                                                 #
# =========================================================================== #

def test_generation_is_deterministic():
    """The guide's first checklist item. Same seed, same samples."""
    def once():
        generator = Generator(stub_evaluator(0), {"simulations": 8, "max_turns": 80},
                              seed=3, width=3)
        samples, results = generator.run(positions=40)
        return ([(s.index.tolist(), s.probability.tolist(), s.player, outcome)
                 for s, outcome, *_ in samples],
                [r["winner"] for r in results])

    assert once() == once()


def test_generation_records_both_seats():
    generator = Generator(stub_evaluator(0), {"simulations": 6, "max_turns": 60},
                          seed=5, width=4)
    samples, results = generator.run(games=1)
    assert results, "no game finished"
    seats = {sample.player for sample, *_ in samples}
    assert seats == {1, 2}, "a learner that saw one seat trains on winners only"

    finished = [r for r in results if r["winner"] is not None]
    if finished:
        outcomes = {outcome for _, outcome, *_ in samples}
        assert outcomes <= {-1, 0, 1}
        assert 1 in outcomes and -1 in outcomes


def test_generation_skips_forced_moves():
    generator = Generator(stub_evaluator(0), {"simulations": 6, "max_turns": 80},
                          seed=8, width=3)
    _, results = generator.run(games=1)
    assert results
    assert all(r["searched"] < r["decisions"] for r in results), (
        "every game has forced decisions, and recording them wastes buffer on one-hot targets"
    )
    assert all(r["samples"] == r["searched"] for r in results)


def test_generation_is_silent():
    """No printing during training. The guide's section 8, and it costs real time at scale."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        generator = Generator(stub_evaluator(0), {"simulations": 6, "max_turns": 60},
                              seed=2, width=2)
        generator.run(positions=30)
    assert buffer.getvalue() == ""


def test_generation_survives_a_full_game_and_restarts():
    generator = Generator(stub_evaluator(0), {"simulations": 4, "max_turns": 60},
                          seed=1, width=2)
    _, results = generator.run(games=3)
    assert len(results) >= 3
    for result in results:
        assert result["turns"] > 0
        assert result["winner"] in (None, 1, 2)


def test_run_needs_a_stopping_condition():
    generator = Generator(stub_evaluator(0), {"simulations": 4}, seed=0, width=1)
    with pytest.raises(ValueError, match="stopping condition"):
        generator.run()


# =========================================================================== #
# The replay buffer                                                          #
# =========================================================================== #

def test_replay_round_trips_a_real_observation():
    """float16 must not lose anything the encoder actually puts in an observation."""
    state = played()
    observation = np.asarray(encoder.encode(state, 1), dtype=np.float32)
    assert len(observation) == encoder.SIZE
    stored = observation.astype(np.float16).astype(np.float32)
    assert np.allclose(observation, stored, rtol=1e-3, atol=1e-3)


def test_sparse_policy_round_trips_through_the_buffer():
    actions = np.array([0, 5, 200, 324], dtype=np.int64)
    counts = np.array([10.0, 5.0, 4.0, 1.0])
    index, probability = sparse_policy(actions, counts)

    buffer = ReplayBuffer(capacity=8)
    mask = np.zeros(action_space.NUM_ACTIONS, dtype=bool)
    mask[actions] = True
    buffer.add(np.zeros((1, encoder.SIZE), dtype=np.float16), index[None, :],
               probability[None, :], pack_mask(mask)[None, :],
               np.array([1], dtype=np.int8), *extra_columns(1))

    _, policy, stored_mask, value, *_ = buffer.sample(1, np.random.default_rng(0))
    assert policy[0].sum() == pytest.approx(1.0, abs=2e-3)
    assert policy[0, 0] == pytest.approx(0.5, abs=2e-3), "END_TURN must not be lost to padding"
    assert policy[0, 324] == pytest.approx(0.05, abs=2e-3)
    assert stored_mask[0].tolist() == mask.tolist()
    assert value[0] == 1.0
    assert policy[0][~mask].sum() == 0.0


def test_buffer_wraps_and_keeps_the_newest():
    buffer = ReplayBuffer(capacity=4)
    for step in range(6):
        buffer.add(
            np.full((1, encoder.SIZE), step, dtype=np.float16),
            np.zeros((1, replay_buffer.POLICY_TOP_K), dtype=np.int16),
            np.zeros((1, replay_buffer.POLICY_TOP_K), dtype=np.float16),
            np.zeros((1, (action_space.NUM_ACTIONS + 7) // 8), dtype=np.uint8),
            np.array([1], dtype=np.int8), *extra_columns(1, game_id=step),
        )
    assert len(buffer) == 4
    assert buffer.full
    order = buffer._ordered()
    ages = [float(buffer.obs[row, 0]) for row in order]
    assert ages == [2, 3, 4, 5], "oldest first, newest last"


def test_sampling_covers_every_age_band():
    buffer = ReplayBuffer(capacity=400)
    for step in range(400):
        buffer.add(
            np.full((1, encoder.SIZE), step, dtype=np.float16),
            np.zeros((1, replay_buffer.POLICY_TOP_K), dtype=np.int16),
            np.zeros((1, replay_buffer.POLICY_TOP_K), dtype=np.float16),
            np.zeros((1, (action_space.NUM_ACTIONS + 7) // 8), dtype=np.uint8),
            np.array([0], dtype=np.int8), *extra_columns(1, game_id=step),
        )
    obs, *_ = buffer.sample(256, np.random.default_rng(0))
    ages = obs[:, 0]
    for band in range(4):
        low, high = band * 100, (band + 1) * 100
        assert ((ages >= low) & (ages < high)).sum() == 64, (
            "the guide's 25/25/25/25 by age"
        )


def test_add_more_than_capacity_keeps_the_newest():
    buffer = ReplayBuffer(capacity=3)
    rows = 10
    added = buffer.add(
        np.arange(rows, dtype=np.float16)[:, None].repeat(encoder.SIZE, axis=1),
        np.zeros((rows, replay_buffer.POLICY_TOP_K), dtype=np.int16),
        np.zeros((rows, replay_buffer.POLICY_TOP_K), dtype=np.float16),
        np.zeros((rows, (action_space.NUM_ACTIONS + 7) // 8), dtype=np.uint8),
        np.zeros(rows, dtype=np.int8), *extra_columns(rows),
    )
    assert added == 3
    assert sorted(float(v) for v in buffer.obs[:, 0]) == [7, 8, 9]


# =========================================================================== #
# The network, and carrying a checkpoint across an observation change        #
# =========================================================================== #

def test_new_network_matches_the_current_spaces():
    net = new_network()
    assert net.obs_size == encoder.SIZE
    assert net.num_actions == action_space.NUM_ACTIONS
    assert net.value_activation == "tanh"


def test_tanh_value_head_is_bounded_and_off_by_default():
    from training.structured_net import StructuredPolicyValueNet

    default = StructuredPolicyValueNet()
    assert default.value_activation == "linear", (
        "every PPO checkpoint was trained without it; changing the default would silently "
        "change what they mean"
    )
    assert "value_activation" in default.config()

    bounded = new_network()
    with torch.no_grad():
        _, value = bounded(torch.randn(16, encoder.SIZE) * 50)
    assert torch.all(value.abs() <= 1.0)


def test_graft_is_a_no_op_at_the_current_size():
    net = new_network()
    weights, inserted = graft(net.state_dict(), net.config())
    assert inserted == 0
    assert set(weights) == set(net.state_dict())


def test_graft_widens_every_observation_layer_with_zeros():
    """A grafted network must compute exactly what the original did.

    Built by *narrowing* a current network to the previous layout and grafting it back, so
    the test exercises the same path a real checkpoint takes without needing one on disk.
    """
    from training.alphazero import layouts
    from training.alphazero.network import _input_map, _segments

    old_layout = layouts.HISTORICAL[1884]
    net = new_network()
    original = net.state_dict()

    narrow = dict(original)
    for name, segments in _segments(1).items():
        mapping = _input_map(segments, old_layout, layouts.signature())
        keep = [i for i, source in enumerate(mapping) if source >= 0]
        # Undo the graft: drop the columns that would be new, and put the rest back in the
        # order the old layer had them.
        columns = sorted(keep, key=lambda i: mapping[i])
        narrow[name] = original[name][:, columns]

    grafted, inserted = graft(narrow, {**net.config(), "obs_size": 1884, "layout": None})
    assert inserted > 0

    for name, segments in _segments(1).items():
        mapping = _input_map(segments, old_layout, layouts.signature())
        assert grafted[name].shape == original[name].shape, name
        kept = [i for i, source in enumerate(mapping) if source >= 0]
        added = [i for i, source in enumerate(mapping) if source < 0]
        # Surviving columns come back exactly where they were...
        assert torch.allclose(grafted[name][:, kept], original[name][:, kept], atol=1e-6), (
            f"{name} did not put its old columns back where they were"
        )
        # ...and the new ones are zero. Not "close to the original": the original's values
        # there are a fresh random init this test happens to have, and the whole safety
        # argument is that a grafted network gives new features *no* weight until it learns
        # one.
        assert torch.all(grafted[name][:, added] == 0), f"{name} gave a new feature weight"


def test_grafted_columns_cannot_contribute_anything():
    """The whole safety argument, as an assertion: the new features start at zero weight."""
    from training.alphazero import layouts
    from training.alphazero.network import _input_map, _segments

    net = new_network()
    original = net.state_dict()
    narrow = dict(original)
    for name, segments in _segments(1).items():
        mapping = _input_map(segments, layouts.HISTORICAL[1884], layouts.signature())
        columns = sorted((i for i, m in enumerate(mapping) if m >= 0), key=lambda i: mapping[i])
        narrow[name] = original[name][:, columns]

    grafted, _ = graft(narrow, {**net.config(), "obs_size": 1884, "layout": None})
    rebuilt = new_network()
    rebuilt.load_state_dict(grafted)

    # Drive every appended feature hard. The logits must not move at all.
    observation = torch.randn(4, encoder.SIZE)
    changed = observation.clone()
    rows = encoder.SHAPES["vertices"][1]
    base = encoder.LAYOUT["vertices"].start
    for vertex in range(encoder.SHAPES["vertices"][0]):
        at = base + vertex * rows + encoder.VERTEX_OFFSETS["production"]
        changed[:, at:at + 11] += 100.0
    with torch.no_grad():
        assert torch.allclose(rebuilt(observation)[0], rebuilt(changed)[0], atol=1e-4)


def test_graft_refuses_a_layout_it_cannot_know():
    net = new_network()
    with pytest.raises(ValueError, match="carries no layout"):
        graft(net.state_dict(), {**net.config(), "obs_size": 1234, "layout": None})


def test_graft_refuses_a_block_that_shrank():
    from training.alphazero import layouts

    shrunk = dict(layouts.signature())
    shrunk["vertices"] = (54, 99)
    with pytest.raises(ValueError, match="shrank"):
        layouts.column_map(shrunk)


def test_historical_layouts_add_up_to_their_key():
    """An entry is a statement about a file on disk; a wrong one grafts silently."""
    from training.alphazero import layouts

    for size, layout in layouts.HISTORICAL.items():
        assert layouts.total(layout) == size, size
    assert layouts.total(layouts.signature()) == encoder.SIZE


def test_segments_match_the_network_they_describe():
    """The graft's idea of how each layer's input is assembled must match the real one."""
    from training.alphazero import layouts
    from training.alphazero.network import _input_map, _segments

    net = new_network()
    current = layouts.signature()
    for name, segments in _segments(net.hops).items():
        expected = dict(net.state_dict())[name].shape[1]
        assert len(_input_map(segments, current, current)) == expected, name


# =========================================================================== #
# Checkpoints, config, and the pool                                          #
# =========================================================================== #

def test_checkpoint_round_trips(tmp_path):
    from training.alphazero.agent import MCTSAgent

    net = new_network()
    path = tmp_path / "candidate.pt"
    torch.save({"config": net.config(),
                "weights": net.state_dict(),
                "iteration": 3}, path)

    agent = MCTSAgent.load(path, simulations=4)
    assert agent.net.obs_size == encoder.SIZE
    assert agent.net.value_activation == "tanh"
    assert agent.metadata["iteration"] == 3

    observation = torch.randn(2, encoder.SIZE)
    with torch.no_grad():
        assert torch.allclose(net(observation)[1], agent.net(observation)[1])


def test_config_rejects_an_unknown_key():
    with pytest.raises(KeyError, match="unknown setting"):
        Config({"mtcs_simulations": 200})


def test_config_parses_the_shipped_file():
    import pathlib

    values = parse(pathlib.Path("configs/train.yaml").read_text(encoding="utf-8"))
    config = Config(values)
    assert config["mcts_simulations"] > 0
    assert config["batch_size"] > 0
    assert isinstance(config["warm_start"], str)


def test_config_refuses_nesting():
    with pytest.raises(ValueError, match="flat"):
        parse("outer:\n  inner: 1\n")


def test_config_reads_scalars():
    values = parse("a: 1\nb: 2.5\nc: true\nd: text  # comment\ne: 20_000\n")
    assert values == {"a": 1, "b": 2.5, "c": True, "d": "text", "e": 20000}


def test_champion_paths_are_separate():
    from training import champion as ppo
    from training.alphazero import champion as az

    assert az.CHAMPION != ppo.CHAMPION, (
        "the two lineages must not overwrite each other — the PPO champion is kept"
    )
    assert az.RECORD != ppo.RECORD
    assert az.PPO_CHAMPION == ppo.CHAMPION


def test_champion_load_returns_none_for_a_missing_file(tmp_path):
    from training.alphazero import champion as az

    assert az.load(tmp_path / "nothing.pt") is None


def test_describe_reports_the_rung_that_actually_decided(monkeypatch):
    """``beat_champion`` is the gate, so it has to be the line.

    Since ``baseline_games`` began defaulting to 0 the other two figures are ``null`` for
    every chain promotion — gen5 and gen6 both — so a ``describe`` that read only those
    printed a bare date for the reigning champion while the number that earned it sat in the
    record unread.
    """
    from training.alphazero import champion as az

    monkeypatch.setattr(az, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(az, "record", lambda: {
        "promoted_at": "2026-08-06 21:34",
        "beat_champion": 0.5525,        # gen6-gilded-beacon, as promoted
        "beat_heuristic": None,
        "beat_ppo_champion": None,
    })
    line = az.describe()
    assert "2026-08-06 21:34" in line
    assert "55" in line and "champion it replaced" in line
    assert "forced" not in line

    # A forced promotion did not clear the gate, and its number must not read as though it
    # had — the whole point of recording `forced` is that the two stay distinguishable.
    monkeypatch.setattr(az, "record", lambda: {
        "promoted_at": "2026-08-04 11:02",
        "beat_champion": 0.518,
        "forced": True,
    })
    assert "(forced)" in az.describe()


def test_champion_name_is_derived_from_the_weights():
    """The same weights must always produce the same name, whatever order they arrive in.

    The point of deriving the name from a content hash rather than allocating a counter is
    that a checkpoint keeps its identity when it is copied, re-promoted, or listed by a
    tool that happened to build its state dict differently.
    """
    from training.alphazero import champion as az

    weights = {"b": torch.arange(6).float().reshape(2, 3),
               "a": torch.zeros(4)}
    reordered = {key: weights[key] for key in reversed(list(weights))}
    assert az.fingerprint(weights) == az.fingerprint(reordered)
    assert az.champion_name(3, az.fingerprint(weights)).startswith("gen3-")

    # and a different network must not collide onto the same name
    changed = dict(weights)
    changed["a"] = torch.ones(4)
    assert az.fingerprint(changed) != az.fingerprint(weights)


def test_champion_name_survives_a_console_that_cannot_encode_it():
    """The name reaches a cp1252 Windows console intact.

    ``train.py`` already records why this matters: a banner that raises
    UnicodeEncodeError is worse than a plain one. The codename words are ASCII, and this
    pins that rather than trusting it.
    """
    from training.alphazero import champion as az

    for adjective in az._ADJECTIVES:
        adjective.encode("cp1252")
    for noun in az._NOUNS:
        noun.encode("cp1252")
    az.champion_name(9, "0" * 64).encode("cp1252")


def test_first_promotion_is_refused_rather_than_waved_through(tmp_path, monkeypatch):
    """The hole in the older gate must not be reproduced here.

    ``CLAUDE.md`` records ``training.champion.promote`` installing without a match whenever
    no champion loads — and it fires exactly when ``encoder.SIZE`` changed, which is exactly
    when nobody is watching. The heuristic rung used to plug that hole. It is gone, so with
    no reigning champion there is no gate at all, and the only safe answer is to refuse:
    installing the first champion of a lineage is now an explicit ``--force --reason``.
    """
    from training.alphazero import champion as az

    monkeypatch.setattr(az, "MODELS", tmp_path)
    monkeypatch.setattr(az, "CHAMPION", tmp_path / "champion_az.pt")
    monkeypatch.setattr(az, "RECORD", tmp_path / "champion_az.json")
    monkeypatch.setattr(az, "PPO_CHAMPION", tmp_path / "champion.pt")

    net = new_network()
    candidate = tmp_path / "candidate.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, candidate)

    played = []

    def losing(a, b, games=300, seed=0, **kwargs):
        played.append((a["kind"], b["kind"]))
        return {"wins": 10, "losses": 90, "truncated": 0, "games": 100,
                "win_rate": 0.10, "ci": (0.05, 0.18), "ci_width": 0.13}

    monkeypatch.setattr("training.alphazero.arena.compete", losing)
    promoted, reason = az.promote(candidate, games=100, log=lambda *_: None)

    assert not promoted, "an untrained network must not become the champion"
    assert "no reigning champion" in reason
    assert not (tmp_path / "champion_az.pt").exists()

    # And the escape hatch works, and says so in the record.
    promoted, reason = az.promote(candidate, games=100, force=True,
                                  reason="measured 61% by hand", log=lambda *_: None)
    assert promoted and "forced" in reason
    assert json.loads((tmp_path / "champion_az.json").read_text())["forced"] is True


def test_the_heuristic_cannot_veto_a_candidate_that_beat_the_champion(tmp_path, monkeypatch):
    """Beating the champion is the whole gate, deliberately.

    This test used to assert the opposite — that collapsing against the fixed heuristic
    refused a candidate that had beaten the champion. The tripwire was removed on purpose so
    that "promoted" means exactly one measurable thing. The argument for it has not stopped
    being true (self-play is non-transitive, so a candidate can climb by learning the
    champion's habits while getting worse at the game), which is why it is written down here:
    if this behaviour is ever reverted, it should be because someone decided to, not because
    the check quietly came back.
    """
    from training.alphazero import champion as az

    net = new_network()
    candidate = tmp_path / "candidate.pt"
    reigning = tmp_path / "champion_az.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, candidate)
    torch.save({"config": net.config(), "weights": net.state_dict()}, reigning)

    monkeypatch.setattr(az, "MODELS", tmp_path)
    monkeypatch.setattr(az, "CHAMPION", reigning)
    monkeypatch.setattr(az, "RECORD", tmp_path / "champion_az.json")
    monkeypatch.setattr(az, "PPO_CHAMPION", tmp_path / "champion.pt")
    (tmp_path / "champion_az.json").write_text(json.dumps({"beat_heuristic": 0.70}),
                                               encoding="utf-8")

    def results(a, b, games=300, seed=0, **kwargs):
        if b["kind"] == "heuristic":                       # collapsed against the baseline
            return {"wins": 40, "losses": 60, "truncated": 0, "games": 100,
                    "win_rate": 0.40, "ci": (0.31, 0.50), "ci_width": 0.19}
        return {"wins": 70, "losses": 30, "truncated": 0, "games": 100,   # but beat the champ
                "win_rate": 0.70, "ci": (0.60, 0.78), "ci_width": 0.18}

    monkeypatch.setattr("training.alphazero.arena.compete", results)
    promoted, reason = az.promote(candidate, games=100, baseline_games=100,
                                  log=lambda *_: None)

    assert promoted, "the heuristic is recorded, not consulted"
    assert "70.0% against the champion" in reason
    written = json.loads((tmp_path / "champion_az.json").read_text())
    assert written["beat_heuristic"] == 0.40, "asked for, so it must still be recorded"

    # And by default it is not even played.
    played = []
    monkeypatch.setattr("training.alphazero.arena.compete",
                        lambda a, b, **kw: played.append(b["kind"]) or results(a, b, **kw))
    az.promote(candidate, games=100, log=lambda *_: None)
    assert "heuristic" not in played, "the default gate must not spend games on the yardstick"


@pytest.mark.slow
def test_parallel_pool_generates():
    """The pool must start on Windows spawn and come back with usable arrays."""
    from training.alphazero.config import Config
    from training.alphazero.workers import ParallelSelfPlay

    net = new_network()
    config = Config({"self_play_workers": 2, "envs_per_worker": 2,
                     "mcts_simulations": 4, "max_turns": 60})
    pool = ParallelSelfPlay(net, config)
    try:
        arrays, results = pool.generate(60)
    finally:
        pool.close()

    obs, index, probability, mask, value, root_value, owners, margin, game_id = arrays
    assert len(obs) >= 60
    assert obs.shape[1] == encoder.SIZE
    assert index.shape[1] == replay_buffer.POLICY_TOP_K
    assert set(np.unique(value).tolist()) <= {-1, 0, 1}
    assert obs.dtype == np.float16
    # The auxiliary columns must survive the pickle round trip through the pool, which is
    # where a stitching mistake would show up: the arrays would concatenate cleanly into the
    # wrong columns and the run would train on a permuted target without ever raising.
    assert owners.shape == (len(obs), replay_buffer.OWNER_COLUMNS)
    assert set(np.unique(owners).tolist()) <= {0, 1, 2}
    assert root_value.shape == margin.shape == game_id.shape == (len(obs),)
    assert np.all(np.abs(root_value.astype(np.float32)) <= 1.0)
    assert len(np.unique(game_id)) > 1, "every row claiming one game defeats max_per_game"


@pytest.mark.slow
def test_workers_do_not_all_generate_the_same_games():
    """Every worker runs the same code from the same config; without distinct identities the
    pool would burn fourteen cores producing one core's worth of data."""
    from training.alphazero.config import Config
    from training.alphazero.workers import ParallelSelfPlay

    net = new_network()
    config = Config({"self_play_workers": 2, "envs_per_worker": 2,
                     "mcts_simulations": 4, "max_turns": 60})
    pool = ParallelSelfPlay(net, config)
    try:
        (obs, *_), _ = pool.generate(80)
    finally:
        pool.close()

    half = len(obs) // 2
    assert not np.array_equal(obs[:half], obs[half:2 * half])


def test_trainer_learns_a_step(tmp_path):
    """One gradient step must run end to end and move the weights."""
    from training.alphazero.config import Config
    from training.alphazero.trainer import Trainer

    net = new_network()
    config = Config({"run_directory": str(tmp_path), "batch_size": 8,
                     "replay_buffer_size": 64, "learning_rate": 1e-3})
    trainer = Trainer(net, config, log=lambda *_: None)

    rows = 32
    mask = np.zeros((rows, action_space.NUM_ACTIONS), dtype=bool)
    mask[:, :5] = True
    index = np.zeros((rows, replay_buffer.POLICY_TOP_K), dtype=np.int16)
    index[:] = replay_buffer.PAD_INDEX
    index[:, 0] = 3
    probability = np.zeros((rows, replay_buffer.POLICY_TOP_K), dtype=np.float16)
    probability[:, 0] = 1.0
    trainer.buffer.add(
        np.random.default_rng(0).random((rows, encoder.SIZE)).astype(np.float16),
        index, probability,
        np.stack([pack_mask(row) for row in mask]),
        np.ones(rows, dtype=np.int8), *extra_columns(rows),
    )

    before = net.policy_head.weight.detach().clone()
    stats = trainer.learn(3)
    assert stats["batches"] == 3
    assert stats["policy_loss"] > 0
    assert not torch.allclose(before, net.policy_head.weight)

    path = trainer.checkpoint("test")
    assert path.is_file()
    stored = torch.load(path, map_location="cpu", weights_only=False)
    assert stored["config"] == net.config()
    assert stored["lineage"] == "alphazero"

    lines = (tmp_path / "metrics.jsonl")
    trainer._record({"iteration": 1, "total_games": 0, "buffer": rows,
                     "generate_seconds": 0.0, "train_seconds": 0.0,
                     "elapsed_minutes": 0.0})
    assert json.loads(lines.read_text(encoding="utf-8").splitlines()[0])["iteration"] == 1


def test_report_reads_a_run_and_tolerates_a_half_written_line(tmp_path):
    """The trainer appends while the reader reads, so the last line may be incomplete."""
    from training.alphazero import report

    complete = json.dumps({
        "iteration": 1, "total_games": 10, "elapsed_minutes": 1.0, "buffer": 100,
        "generate_seconds": 25.0, "train_seconds": 5.0, "positions_per_second": 300.0,
        "policy_loss": 1.2, "value_loss": 0.3, "entropy": 1.1,
        "evaluation": {"win_rate": 0.6, "ci": [0.5, 0.7], "truncated": 0},
    })
    partial = '{"iteration": 2, "total_ga'
    (tmp_path / "metrics.jsonl").write_text(complete + "\n" + partial, encoding="utf-8")
    entries = report.load(tmp_path)
    assert len(entries) == 1, "a partial line must be skipped, not crash the reader"

    summary = report.summarise(entries)
    assert summary["games"] == 10
    assert summary["best_evaluation"] == 0.6
    assert report.evaluations(entries) == [(1, entries[0]["evaluation"])]

    lines = []
    report.report(tmp_path, full=True, log=lines.append)
    assert any("60.0%" in line for line in lines)


def test_report_says_so_when_there_is_no_run(tmp_path):
    from training.alphazero import report

    lines = []
    assert report.report(tmp_path, log=lines.append) == {}
    assert "has a run started" in " ".join(lines)


@pytest.mark.slow
def test_arena_result_does_not_depend_on_worker_count():
    """An evaluator whose answer moves with the machine it ran on is not an evaluator.

    Not compared against ``play_match``: that lets each agent's RNG carry from one game to
    the next, so game 17 depends on the sixteen before it, which cannot be reproduced when
    the games are dealt out across processes. ``arena`` re-seeds per game instead, which is
    what makes this equality hold. The two therefore sample the same quantity by different
    draws — see the module docstring.
    """
    from training.alphazero.arena import compete

    one = compete({"kind": "heuristic", "seed": 0}, {"kind": "greedy", "seed": 0},
                  games=16, seed=555, workers=1, max_turns=400)
    many = compete({"kind": "heuristic", "seed": 0}, {"kind": "greedy", "seed": 0},
                   games=16, seed=555, workers=4, max_turns=400)

    assert (one["wins"], one["losses"], one["truncated"]) ==            (many["wins"], many["losses"], many["truncated"])
    assert one["wins"] > one["losses"], "the heuristic must still beat greedy"


@pytest.mark.slow
def test_arena_rank_orders_candidates_best_first(tmp_path):
    """The step D17 insists on: candidates are compared by *playing*, not by a stored score."""
    from training.alphazero.arena import rank

    strong = tmp_path / "strong.pt"
    weak = tmp_path / "weak.pt"
    net = new_network()
    torch.save({"config": net.config(), "weights": net.state_dict()}, strong)
    torch.save({"config": net.config(), "weights": net.state_dict()}, weak)

    ordered = rank([strong, weak], opponent={"kind": "random", "seed": 0},
                   games=8, simulations=2, workers=2, log=lambda *_: None)

    rates = [r["win_rate"] for r in ordered]
    assert rates == sorted(rates, reverse=True), "results must come back best-first"
    assert {r["path"] for r in ordered} == {str(strong), str(weak)}, "every candidate played"
    assert all(r["games"] == 8 for r in ordered)
    # The two checkpoints hold identical weights, so a paired comparison — same seed, same
    # opponent, same seat rotation — must give them identical records. If it does not, the
    # candidates are not being compared on the same games and the ranking means nothing.
    assert rates[0] == rates[1], "identical networks must score identically on paired games"


def test_arena_builds_a_checkpoint_from_an_older_observation(tmp_path):
    """A candidate list may include a champion promoted before an encoder change.

    Building it directly raises — the network constructor rejects a mismatched `obs_size` —
    and that happens inside the pool *initializer*, which takes the whole pool down with a
    BrokenProcessPool rather than a readable error. It has to graft instead.
    """
    from training.alphazero import layouts
    from training.alphazero.arena import build_agent
    from training.alphazero.network import _input_map, _segments

    net = new_network()
    original = net.state_dict()
    old_layout = layouts.HISTORICAL[1884]
    narrow = dict(original)
    for name, segments in _segments(1).items():
        mapping = _input_map(segments, old_layout, layouts.signature())
        columns = sorted((i for i, m in enumerate(mapping) if m >= 0),
                         key=lambda i: mapping[i])
        narrow[name] = original[name][:, columns]

    stale = tmp_path / "stale.pt"
    torch.save({"config": {**net.config(), "obs_size": 1884, "layout": None},
                "weights": narrow}, stale)

    agent = build_agent({"kind": "mcts", "path": str(stale), "simulations": 2})
    assert agent is not None
    assert agent.net.obs_size == encoder.SIZE


def test_arena_refuses_a_checkpoint_it_cannot_use(tmp_path):
    from training.alphazero.arena import build_agent

    missing = tmp_path / "nothing.pt"
    with pytest.raises(ValueError, match="not a usable model"):
        build_agent({"kind": "mcts", "path": str(missing)})


def test_arena_builds_every_agent_kind_it_claims_to(tmp_path):
    from training.alphazero.arena import build_agent

    net = new_network()
    path = tmp_path / "candidate.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, path)

    assert build_agent({"kind": "heuristic", "noise": 0}) is not None
    assert build_agent({"kind": "greedy"}) is not None
    assert build_agent({"kind": "random"}) is not None
    assert build_agent({"kind": "mcts", "path": path, "simulations": 4}) is not None
    assert build_agent({"kind": "policy", "path": path}) is not None
    with pytest.raises(ValueError, match="unknown agent kind"):
        build_agent({"kind": "telepathy"})


def test_evaluator_ladder_skips_a_missing_rung(monkeypatch):
    from training.alphazero import evaluator

    monkeypatch.setattr(evaluator, "_opponent", lambda name: None)
    assert evaluator.ladder(object(), include=("champion",)) == {"champion": None}


def test_better_needs_the_interval_to_clear_the_threshold():
    from training.alphazero.evaluator import better

    assert not better({"win_rate": 0.52, "ci": (0.47, 0.57)})
    assert better({"win_rate": 0.60, "ci": (0.55, 0.65)})


@pytest.mark.slow
def test_warm_started_agent_beats_random():
    """A sanity floor for the whole play path: load, determinize, search, move.

    Deliberately measured on the *warm-started* network rather than a fresh one. An untrained
    network searching 16 simulations is a random policy steered by a random value function,
    and the first version of this test asserted it would beat random play: it drew 0-0,
    because neither side can reach 15 points inside the turn cap. That is a fact about
    untrained play, not about the pipeline, so it made a bad floor. The grafted champion is a
    real policy, so this now fails only if something in load/determinize/search/move is
    actually broken.
    """
    import pathlib

    from catan.agents import RandomAgent, play_match
    from training.alphazero.agent import MCTSAgent
    from training.alphazero.network import load_for_alphazero

    source = pathlib.Path("models/champion.pt")
    if not source.is_file():
        pytest.skip("no checkpoint to warm-start from")

    torch.set_num_threads(2)
    net, _ = load_for_alphazero(source)
    agent = MCTSAgent(net, simulations=16, temperature=0.0, seed=0)
    tally = play_match({1: agent, 2: RandomAgent(0)}, games=12, seed=500,
                       ruleset=RANKED_1V1, max_turns=400)
    assert tally[1] > tally[2], tally


# =========================================================================== #
# Watching a run: the study and the dashboard                                 #
# =========================================================================== #

def test_quantile_edges_split_into_equal_groups():
    """Fixed bands put every game in one bucket once; quantiles cannot."""
    from training.alphazero.study import quantile_edges

    values = [0.1 * i for i in range(20)]
    edges = quantile_edges(values, bands=4)
    counts = [sum(1 for v in values if low <= v < high)
              for low, high in zip(edges, edges[1:])]
    assert len(edges) == 5
    assert max(counts) - min(counts) <= 1, counts
    assert quantile_edges([]) == []
    assert len(quantile_edges([3.0] * 10)) >= 2, "a constant series still needs one band"


def test_vertex_production_splits_pips_by_resource():
    """The quantity the observation is missing: which resource the pips come from."""
    from training.alphazero.study import odds, vertex_production

    state = played(moves=0)
    board = state.board
    for vertex in range(1, 55):
        per = vertex_production(board, vertex)
        assert len(per) == NUM_RESOURCES
        expected = sum(odds(board.number_at(t)) for t in
                       __import__("catan.topology", fromlist=["VERTEX_TILES"]).VERTEX_TILES[vertex]
                       if board.resource_at(t) is not None)
        assert sum(per) == pytest.approx(expected)
    assert odds(None) == 0.0
    assert odds(8) == pytest.approx(5 / 36)
    assert odds(2) == pytest.approx(1 / 36)


def test_study_records_an_opening_and_an_outcome():
    from catan.agents import GreedyAgent, HeuristicAgent
    from training.alphazero.study import play_and_record, summarise

    # The heuristic against greedy, not greedy against random: the summary is only defined
    # over *decided* games, and two weak agents do not reliably reach 15 points.
    records = [play_and_record(HeuristicAgent(0), GreedyAgent(0), seed=seed, max_turns=400)
               for seed in range(4)]
    assert any(r["decided"] for r in records), "no game finished, so there is nothing to summarise"
    for record in records:
        assert record["seat"] in (1, 2)
        assert record["pips"] > 0
        assert record["gap_to_best"] >= -1e-9, "cannot beat the best available spot"
        assert 1 <= record["diversity"] <= 5
        assert 0.0 <= record["ore_wheat_sheep"] <= 1.0
        assert record["ore_wheat_sheep"] + record["wood_brick"] == pytest.approx(1.0)
        assert len(record["spots"]) == 2, "two opening settlements"

    summary = summarise(records)
    assert summary["games"] == 4
    assert set(summary["by_resource"]) == {r.name.lower() for r in Resource}


def test_summarise_survives_a_study_with_no_decided_games():
    from training.alphazero.study import summarise

    assert summarise([])["decided"] == 0
    undecided = [{"decided": False, "won": False, "pips": 0.5, "diversity": 3,
                  "ore_wheat_sheep": 0.5, "gap_to_best": 0.0, "spots": [1, 2],
                  "opening_harbours": 0, "harbours_owned": [], "turns": 400,
                  "resources": {r.name.lower(): 0.1 for r in Resource}}]
    assert summarise(undecided)["decided"] == 0


def test_dashboard_builds_a_self_contained_page(tmp_path):
    from training.alphazero import dashboard

    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text("\n".join(
        json.dumps({"iteration": i, "total_games": 10 * i, "elapsed_minutes": i / 2,
                    "buffer": 1000 * i, "generate_seconds": 25.0, "train_seconds": 3.0,
                    "positions_per_second": 150.0 + i, "policy_loss": 1.4 - i / 100,
                    "value_loss": 0.3, "entropy": 1.3, "turns": 110,
                    **({"evaluation": {"win_rate": 0.55, "ci": [0.47, 0.63],
                                       "truncated": 0}} if i % 3 == 0 else {})})
        for i in range(1, 10)), encoding="utf-8")

    page = dashboard.build(run, tmp_path / "missing.json")
    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    # The page is built with f-strings wrapped around CSS, so every CSS brace is doubled in
    # the source. A doubling that leaked through shows up as `{{` in the output.
    assert "{{" not in page and "}}" not in page, "an escaped brace survived into the page"
    assert "color-scheme" in page and "{" in page, "the CSS itself must still be there"
    assert "No study yet" in page, "a missing study is explained, not a crash"
    assert "positions/sec" in page
    # Self-contained: nothing to fetch, so it can be opened from disk or e-mailed.
    for forbidden in ("<script", "src=", "href=", "http://", "https://"):
        assert forbidden not in page, forbidden


def test_dashboard_draws_the_opening_study_when_there_is_one(tmp_path):
    from training.alphazero import dashboard, study

    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text("", encoding="utf-8")

    from catan.agents import GreedyAgent, HeuristicAgent

    records = [study.play_and_record(HeuristicAgent(0), GreedyAgent(0), seed=s, max_turns=400)
               for s in range(6)]
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"summary": study.summarise(records), "games": records}),
                    encoding="utf-8")

    page = dashboard.build(run, path)
    assert "by ore-wheat-sheep share" in page
    assert "by opening production" in page
    assert "harbour" in page


def test_dashboard_survives_a_corrupt_study(tmp_path):
    from training.alphazero import dashboard

    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.jsonl").write_text("", encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert "unreadable" in dashboard.build(run, bad)


def test_a_forced_promotion_must_say_why(tmp_path, monkeypatch):
    """Overriding the gate silently is how a gate stops meaning anything."""
    from training.alphazero import champion as az

    net = new_network()
    candidate = tmp_path / "candidate.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, candidate)
    monkeypatch.setattr(az, "MODELS", tmp_path)
    monkeypatch.setattr(az, "CHAMPION", tmp_path / "champion_az.pt")
    monkeypatch.setattr(az, "RECORD", tmp_path / "champion_az.json")
    monkeypatch.setattr(az, "PPO_CHAMPION", tmp_path / "champion.pt")

    promoted, why = az.promote(candidate, force=True, log=lambda *_: None)
    assert not promoted
    assert "say why" in why
    assert not (tmp_path / "champion_az.pt").exists()


def test_a_forced_promotion_records_the_reason_and_the_head_to_head(tmp_path, monkeypatch):
    """A forced promotion must never be mistakable for one that passed the gate — and it
    must keep the number that made it a judgement call."""
    from training.alphazero import champion as az

    net = new_network()
    candidate = tmp_path / "candidate.pt"
    reigning = tmp_path / "champion_az.pt"
    torch.save({"config": net.config(), "weights": net.state_dict()}, candidate)
    torch.save({"config": net.config(), "weights": net.state_dict()}, reigning)
    monkeypatch.setattr(az, "MODELS", tmp_path)
    monkeypatch.setattr(az, "CHAMPION", reigning)
    monkeypatch.setattr(az, "RECORD", tmp_path / "champion_az.json")
    monkeypatch.setattr(az, "PPO_CHAMPION", tmp_path / "nothing.pt")

    played = []

    def result(a, b, games=400, seed=0, **kwargs):
        played.append(b["kind"])
        return {"wins": 52, "losses": 48, "truncated": 0, "games": 100,
                "win_rate": 0.52, "ci": (0.42, 0.62), "ci_width": 0.20}

    monkeypatch.setattr("training.alphazero.arena.compete", result)
    promoted, why = az.promote(candidate, games=100, force=True,
                               reason="better on the fixed yardstick", log=lambda *_: None)

    assert promoted and "better on the fixed yardstick" in why
    written = json.loads((tmp_path / "champion_az.json").read_text(encoding="utf-8"))
    assert written["forced"] is True
    assert written["forced_reason"] == "better on the fixed yardstick"
    assert written["beat_champion"] == 0.52, (
        "the head-to-head is the whole context for an override; a record without it "
        "cannot be argued with later"
    )
    assert "mcts" in played


# =========================================================================== #
# THE CHANCE-NODE FAST PATH                                                   #
# =========================================================================== #

def test_a_balanced_dice_chance_node_has_only_one_outcome():
    """The premise ``Search._sample_roll``'s fast path rests on, measured rather than argued.

    Under the ranked ruleset the 36-card deck is *consumed*, ``clone`` copies it verbatim and
    ``draw_balanced`` pops the last card — so every clone of one chance node draws the same
    card. The search may therefore key the child off ``dice_deck[-1]`` instead of cloning and
    rolling on every visit.

    If this ever fails, the fast path is wrong and must go, not be patched.
    """
    from catan.rulesets import BASE_GAME
    from tests.helpers import drive

    balanced_positions = 0
    for seed in range(6):
        state = GameState(num_players=2, seed=seed, ruleset=RANKED_1V1)

        def check(state):
            nonlocal balanced_positions
            if state.phase is not Phase.ROLL or rules.legal_actions(state):
                return
            balanced_positions += 1
            predicted = sum(state.dice_deck[-1])
            drawn = {rules.roll_dice(state.clone(rng=state.rng)) for _ in range(16)}
            assert drawn == {predicted}, (
                f"a chance node produced {sorted(drawn)}; the fast path would have "
                f"committed to {predicted}"
            )

        drive(state, random.Random(seed ^ 0xD1CE), max_actions=500, on_step=check)

    assert balanced_positions > 100, "the walk did not reach enough chance nodes to say"


def test_plain_dice_still_resample_on_every_visit():
    """The other half of the guard. ``dice_deck is None`` under the base game, so the fast
    path must fall through and keep the genuine per-visit sampling a chance node needs."""
    from catan.rulesets import BASE_GAME
    from tests.helpers import drive

    spreads = []
    for seed in range(4):
        state = GameState(num_players=2, seed=seed, ruleset=BASE_GAME)

        def check(state):
            if state.phase is not Phase.ROLL or rules.legal_actions(state):
                return
            assert state.dice_deck is None
            spreads.append(len({rules.roll_dice(state.clone(rng=state.rng))
                                for _ in range(24)}))

        drive(state, random.Random(seed ^ 0xBEEF), max_actions=400, on_step=check)

    assert spreads, "no chance node was reached"
    assert max(spreads) > 1, "plain dice stopped being random"


def test_the_search_still_reaches_the_same_positions_through_a_chance_node():
    """The fast path may only skip work, never change where the search ends up.

    A chance node under the ranked ruleset must hold exactly one child, and it must be the
    child a full clone-and-roll would have built.
    """
    seen = 0
    for start in range(6):
        env = CatanEnv(num_players=2, ruleset=RANKED_1V1, max_turns=400)
        observation, info = env.reset(seed=11 + start)
        agents = {1: HeuristicAgent(1), 2: HeuristicAgent(2)}
        for _ in range(20 + 12 * start):
            if info["done"]:
                break
            observation, _, _, _, info = env.step(
                agents[info["player"]](observation, info))
        if info["done"]:
            continue

        world = determinize(env.state, info["player"], rng=random.Random(5 + start))
        search = Search(world, budget=192, rng=np.random.default_rng(start), noise=0.0)
        if not search.searchable:
            continue
        uniform = np.full(action_space.NUM_ACTIONS, 1.0 / action_space.NUM_ACTIONS)
        while search.request() is not None:
            search.deliver(uniform, 0.0)

        stack = [search.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node.kind is not CHANCE:
                continue
            seen += 1
            assert len(node.children) == 1, (
                f"a balanced-dice chance node grew {len(node.children)} children; the "
                f"deck cannot produce two totals from one state"
            )
            (total,) = node.children
            assert total == sum(node.state.dice_deck[-1])

    assert seen > 0, "no search built a chance node, so this proved nothing"


# ---------------------------------------------------------------------------------------- #
# The overnight chain                                                                        #
# ---------------------------------------------------------------------------------------- #

def test_chain_relays_a_child_s_non_ascii_output_without_dying():
    """A ten-hour chain must not be killable by a character.

    ``champion.promote`` ends with ``"kept the current champion — " + reason``. Its stdout is
    a pipe, so Python encodes it with the locale codec; on Windows that is cp1252, where the
    em-dash is the single byte ``0x97``, which is not valid UTF-8. Decoding it with
    ``errors="replace"`` gave ``U+FFFD``, cp1252 cannot encode ``U+FFFD`` on the way back
    out, and the chain died at 02:39 with six hours of its window unused.

    So this asserts the round trip, not the workaround: what the child printed is what the
    chain gets back.
    """
    from training.alphazero import chain

    message = "kept the current champion — not shown better"
    ok, output = chain.run(
        [sys.executable, "-c", f"print({message!r})"], lambda _: None)

    assert ok
    assert message in output, (
        f"the child's output did not survive the pipe: {output!r}"
    )
    assert "\ufffd" not in output, (
        "a replacement character means the two ends disagreed about the encoding; "
        "printing this is what killed the chain"
    )


def test_chain_logging_survives_a_console_that_cannot_encode_the_message():
    """Even if something unencodable does reach the log, printing it must not raise.

    The belt to :func:`test_chain_relays_a_child_s_non_ascii_output_without_dying`'s braces.
    A crash here costs every remaining stage, so the log is allowed to be ugly and is not
    allowed to be fatal.
    """
    from training.alphazero import chain

    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    with redirect_stdout(cp1252):
        chain.use_utf8_console()
        print("kept the current champion \ufffd not shown better")
        sys.stdout.flush()

    assert b"kept the current champion" in raw.getvalue()


# ---------------------------------------------------------------------------------------- #
# The opening: a 54-wide root                                                                #
# ---------------------------------------------------------------------------------------- #

def _opening_search(budget, floor, seed=0, peak=0):
    """A search at the first settlement placement, driven by a deliberately peaked prior.

    Peaked because that is the case the floor exists for: PUCT following a confident prior
    concentrates, which is right at six legal moves and wrong at fifty-four.

    ``peak`` is which legal action carries the mass, counted in the order the mask lists them
    — which is the order the tree assigns slots, so ``peak=k`` puts the prior's best at slot
    ``k``. It defaults to the first, and a test that needs to tell "the prior's choice" apart
    from "the lowest-numbered slot" has to move it, because those are the same thing at 0.
    """
    env = CatanEnv(num_players=2, ruleset=RANKED_1V1, max_turns=400)
    _, info = env.reset(seed=seed)
    assert env.state.phase is Phase.SETUP_SETTLEMENT
    world = determinize(env.state, info["player"], rng=random.Random(seed))
    search = Search(world, budget=budget, rng=np.random.default_rng(seed), noise=0.0,
                    root_min_visits=floor)
    while (pending := search.request()) is not None:
        flags = np.frombuffer(bytes(pending[1]), dtype=np.uint8).astype(np.float64)
        legal = np.flatnonzero(flags)
        probabilities = flags * 1e-3
        probabilities[legal[min(peak, len(legal) - 1)]] = 1.0
        search.deliver(probabilities / probabilities.sum(), 0.0)
    return search


def test_without_the_floor_a_wide_root_is_barely_examined():
    """The behaviour the floor exists to fix, pinned so the comparison below means something.

    Measured on the champion rather than this synthetic prior, at the ``noise=0`` this test
    also uses: a median of 3 of the 54 spots at 96 simulations and 5 at 400. Record 0028 has
    the table at all three noise settings; quote it from there rather than from here. An
    opening chosen from three candidates cannot be reasoning about which *pair* of
    settlements it ends up with.
    """
    search = _opening_search(budget=200, floor=0)
    examined = int((search.root.child_n > 0).sum())
    assert len(search.root.actions) > 40, "this seed is not the wide root the test needs"
    assert examined < len(search.root.actions) / 3, (
        f"PUCT examined {examined} of {len(search.root.actions)} spots; if this has become "
        f"broad on its own, the floor may no longer be earning its complexity"
    )


def test_the_floor_makes_every_opening_spot_get_looked_at():
    search = _opening_search(budget=200, floor=2)
    counts = search.root.child_n
    assert (counts >= 2).all(), (
        f"{int((counts < 2).sum())} of {len(counts)} spots came in under the floor"
    )


def test_the_forced_sweep_does_not_reach_the_policy_target():
    """The floor is measurement, not preference.

    If the guaranteed visits were left in the counts, a 54-wide root with a floor of 2 would
    hand the trainer a target that is 108 parts uniform to a handful of parts opinion — the
    search would have been made *worse* by looking at more of the board. So the target must
    stay at least as sharp as the same search without the floor.
    """
    with_floor = _opening_search(budget=200, floor=2)
    without = _opening_search(budget=200, floor=0)

    def entropy(search):
        _, probabilities = search.policy_target()
        probabilities = probabilities[probabilities > 0]
        return float(-(probabilities * np.log(probabilities)).sum())

    uniform = float(np.log(len(with_floor.root.actions)))
    assert entropy(with_floor) < uniform / 2, "the target came back close to uniform"
    assert entropy(with_floor) <= entropy(without) + 1e-9


def test_the_forced_sweep_does_not_reach_the_move_that_is_played():
    """Self-play plays the opening at temperature 1.0, sampling from the counts.

    With the floor left in, 53 spots at 2 visits against one spot's discretionary handful
    means the sampler picks an arbitrary spot almost every time — the opening would become
    close to random exactly where the extra compute was spent.
    """
    search = _opening_search(budget=200, floor=2)
    picks = collections.Counter(search.best_action(temperature=1.0) for _ in range(200))
    assert len(picks) < len(search.root.actions) / 2, (
        f"sampling reached {len(picks)} distinct spots of {len(search.root.actions)}; the "
        f"forced sweep is leaking into best_action"
    )


def test_a_floor_too_big_for_the_budget_falls_back_to_the_prior():
    """The degenerate case, and the one that used to be answered by the lowest vertex id.

    A floor only works while the whole sweep fits inside the budget. At 54 spots and a floor
    of 8 it does not — 432 forced visits against 199 available — so every spot the sweep
    reached sits at exactly the floor and *nothing* is discretionary. ``_discretionary_counts``
    used to hand back the raw counts here, which are near-uniform by construction, and
    ``argmax`` over that tie is the first slot. Record 0028 measured the real thing at a floor
    of 8 and 400 simulations: discretionary zero on 40 of 40 boards, and one board where
    temperature 0 and temperature 1 returned two different arbitrary spots.

    Returning zeros instead lets both callers reach the branch they already had for an empty
    array, which is the network's prior. Still a degraded search — but ranked rather than
    arbitrary.
    """
    search = _opening_search(budget=200, floor=8, peak=7)
    assert len(search.root.actions) > 40, "this seed is not the wide root the test needs"

    counts = search._discretionary_counts()
    assert counts.sum() == 0, (
        "this test is only meaningful when the sweep cannot complete; it did, so the budget "
        "and floor no longer produce the degenerate case"
    )

    best = int(np.argmax(search.root.prior))
    assert best != 0, (
        "the prior's best spot is also the lowest-numbered one for this seed, so the test "
        "cannot tell the fallback from the bug it replaced"
    )
    assert search.best_action(temperature=0.0) == int(search.root.actions[best])

    _, target = search.policy_target()
    assert np.allclose(target, search.root.prior), "the target should be the prior, unaltered"


def test_root_min_visits_defaults_to_off():
    """Every existing caller must be unchanged, so the default has to be inert."""
    plain = _opening_search(budget=200, floor=0)
    assert plain.root_min_visits == 0
    default = Search(determinize(CatanEnv(num_players=2, ruleset=RANKED_1V1).reset(seed=0)[1]
                                ["view"]._state, 1, rng=random.Random(0)),
                     budget=8, rng=np.random.default_rng(0))
    assert default.root_min_visits == 0
