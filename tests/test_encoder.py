"""The observation encoder.

The important tests here are the **leak detectors**: mutate something a player is not
entitled to see and assert their observation does not move. Everything else is shape.
"""

import random

import pytest

import catan.topology as T
from catan import dice, encoder as E, rules
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DevCard
from catan import resources
from catan.dev_cards import ROAD_BUILDING_ROADS
from catan.resources import (
    BANK_RATE,
    NUM_RESOURCES,
    PURCHASE_NAMES,
    PURCHASES,
    SPECIFIC_HARBOUR_RATE,
    Resource,
)
from catan.rulesets import ALL, BASE_GAME, RANKED_1V1
from catan.state import MAX_PLAYERS, GameState, NO_OWNER, Phase, Piece
from helpers import (
    complete_setup,
    scramble_hidden_state,
    enough_for_everything,
    fresh,
    give,
    in_build_phase,
    play_random_game,
    put_building,
    put_road,
)


@pytest.fixture
def played_game():
    """A finished game, so the public record has something in it."""
    from catan.agents import HeuristicAgent
    from catan.env import CatanEnv

    env = CatanEnv(num_players=2, max_turns=400)
    observation, info = env.reset(seed=4)
    agents = {1: HeuristicAgent(4), 2: HeuristicAgent(99)}
    while not info["done"]:
        observation, _, _, _, info = env.step(agents[info["player"]](observation, info))
    return env.state


def mid_game(seed=1, ruleset=None, num_players=2):
    """A state with buildings, roads, cards and an award in play."""
    state = fresh(seed=seed, ruleset=ruleset, num_players=num_players)
    complete_setup(state)
    in_build_phase(state, 1)
    enough_for_everything(state, 1)
    state.dev_cards[1][DevCard.KNIGHT] = 2
    state.dev_cards[2][DevCard.MONOPOLY] = 1
    state.knights_played[1] = 3
    rules.update_awards(state)
    return state


# =========================================================================== #
# SHAPE                                                                       #
# =========================================================================== #

def test_the_layout_covers_the_vector_exactly_once():
    """Derived from LAYOUT rather than from a written-down list of blocks, so adding a
    block cannot leave a hole that only shows up as a silently mis-read observation."""
    covered = 0
    for name, span in E.LAYOUT.items():
        assert span.start == covered, f"a gap or overlap before {name}"
        covered = span.stop
    assert covered == E.SIZE


def test_repeated_blocks_have_consistent_shapes():
    for name, (rows, width) in E.SHAPES.items():
        span = E.LAYOUT[name]
        assert span.stop - span.start == rows * width
    assert E.SHAPES["tiles"][0] == T.NUM_TILES
    assert E.SHAPES["vertices"][0] == T.NUM_VERTICES
    assert E.SHAPES["roads"][0] == T.NUM_ROADS
    assert E.SHAPES["players"][0] == MAX_PLAYERS
    assert E.SHAPES["affordability"][0] == len(PURCHASES)


def test_the_length_never_varies():
    """A network's input shape cannot depend on the situation."""
    lengths = set()
    for ruleset in ALL:
        for num_players in (2, 3, 4):
            state = fresh(seed=1, ruleset=ruleset, num_players=num_players)
            lengths.add(len(E.encode(state, 1)))
            complete_setup(state)
            lengths.add(len(E.encode(state, 1)))
            state.phase = Phase.GAME_OVER
            state.winner = 1
            lengths.add(len(E.encode(state, 1)))
    assert lengths == {E.SIZE}


def test_every_value_is_scaled_into_the_unit_range():
    for seed in range(8):
        state = play_random_game(seed=seed, max_actions=1200)
        for player in state.players:
            observation = E.encode(state, player)
            assert all(isinstance(v, float) for v in observation)
            out_of_range = [v for v in observation if not 0.0 <= v <= 1.0]
            assert not out_of_range, f"unscaled values: {out_of_range[:5]}"


def test_an_unknown_player_is_rejected():
    state = fresh(seed=1, num_players=2)
    for bad in (0, 3, -1):
        with pytest.raises(ValueError):
            E.encode(state, bad)


def test_the_default_view_is_the_current_player():
    state = mid_game()
    assert E.encode(state) == E.encode(state, state.current_player)


def test_block_pulls_out_reshaped_rows():
    state = mid_game()
    observation = E.encode(state, 1)
    tiles = E.block(observation, "tiles")
    assert len(tiles) == T.NUM_TILES
    assert all(len(row) == E.TILE_FEATURES for row in tiles)
    assert E.block(observation, "global") == observation[E.LAYOUT["global"]]


# =========================================================================== #
# HIDDEN INFORMATION — the leak detectors                                     #
# =========================================================================== #

def test_an_opponents_hand_composition_is_invisible():
    """Only the size is public, because cards are countable but not readable."""
    state = mid_game()
    give(state, 2, wood=5)
    before = E.encode(state, 1)

    give(state, 2, ore=5)          # same size, entirely different cards
    assert E.encode(state, 1) == before

    give(state, 2, wood=6)         # a different size does show
    assert E.encode(state, 1) != before


def test_my_own_hand_composition_is_visible():
    state = mid_game()
    give(state, 1, wood=5)
    before = E.encode(state, 1)
    give(state, 1, ore=5)
    assert E.encode(state, 1) != before


def test_an_opponents_development_card_composition_is_invisible():
    state = mid_game()
    state.dev_cards[2] = [3, 0, 0, 0, 0]      # three Knights
    before = E.encode(state, 1)

    state.dev_cards[2] = [0, 3, 0, 0, 0]      # three Victory Points instead
    assert E.encode(state, 1) == before, "the kind of card leaked"

    state.dev_cards[2] = [0, 4, 0, 0, 0]      # a different count does show
    assert E.encode(state, 1) != before


def test_an_opponents_hidden_victory_points_are_invisible():
    """A Victory Point card is worth a point but stays hidden until it wins."""
    state = mid_game()
    state.dev_cards[2] = [0, 0, 0, 0, 1]
    before = E.encode(state, 1)

    state.dev_cards[2] = [0, 1, 0, 0, 0]      # same count, now worth a point
    assert rules.victory_points(state, 2) != rules.public_victory_points(state, 2)
    assert E.encode(state, 1) == before, "an opponent's hidden points leaked"


def test_my_own_hidden_victory_points_are_visible_to_me():
    state = mid_game()
    before = E.encode(state, 1)
    state.dev_cards[1][DevCard.VICTORY_POINT] += 1
    assert E.encode(state, 1) != before


def test_the_development_deck_order_is_invisible():
    """Only how many are left is public. Seeing the order is seeing the future."""
    state = mid_game()
    before = E.encode(state, 1)

    state.dev_deck.reverse()
    assert E.encode(state, 1) == before

    state.rng.shuffle(state.dev_deck)
    assert E.encode(state, 1) == before

    state.dev_deck.pop()                       # a shorter deck does show
    assert E.encode(state, 1) != before


def test_which_development_cards_remain_is_invisible():
    state = mid_game()
    state.dev_deck = [DevCard.KNIGHT] * 10
    before = E.encode(state, 1)
    state.dev_deck = [DevCard.VICTORY_POINT] * 10
    assert E.encode(state, 1) == before


def test_the_balanced_dice_deck_is_invisible():
    """It is not encoded at all — it would be a preview of the next 24 rolls."""
    state = mid_game(ruleset=RANKED_1V1)
    assert state.dice_deck is not None
    before = E.encode(state, 1)

    state.dice_deck.reverse()
    assert E.encode(state, 1) == before
    state.dice_deck = dice.new_deck(random.Random(99))
    assert E.encode(state, 1) == before


def test_an_opponent_sees_the_mirror_of_what_i_hide():
    """The masking is per-observer, not global: what I hide from them, they hide from me."""
    state = mid_game()
    state.dev_cards[1] = [2, 1, 0, 0, 0]
    state.dev_cards[2] = [0, 0, 1, 1, 1]
    give(state, 1, wood=4)
    give(state, 2, ore=4)

    mine, theirs = E.encode(state, 1), E.encode(state, 2)
    assert mine != theirs

    # my composition changes my view but not theirs
    state.dev_cards[1] = [1, 2, 0, 0, 0]
    assert E.encode(state, 1) != mine
    assert E.encode(state, 2) == theirs


# =========================================================================== #
# PERSPECTIVE ROTATION                                                        #
# =========================================================================== #

def test_i_am_always_in_slot_zero():
    state = mid_game(num_players=4)
    for player in state.players:
        slots = E.player_slots(state, player)
        assert slots[player] == 0
        assert sorted(slots.values()) == [0, 1, 2, 3]


def test_opponents_follow_me_in_turn_order():
    state = fresh(num_players=4, player_order=[3, 1, 4, 2], seed=1)
    assert E.player_slots(state, 3) == {3: 0, 1: 1, 4: 2, 2: 3}
    assert E.player_slots(state, 4) == {4: 0, 2: 1, 3: 2, 1: 3}


def test_the_is_me_flag_marks_slot_zero_and_nothing_else():
    state = mid_game(num_players=4)
    rows = E.block(E.encode(state, 2), "players")
    # feature 0 is "in the game", feature 1 is "is me"
    assert [row[1] for row in rows] == [1.0, 0.0, 0.0, 0.0]
    assert [row[0] for row in rows] == [1.0, 1.0, 1.0, 1.0]


def test_absent_players_leave_their_slot_empty():
    state = mid_game(num_players=2)
    rows = E.block(E.encode(state, 1), "players")
    assert [row[0] for row in rows] == [1.0, 1.0, 0.0, 0.0]
    assert all(value == 0.0 for row in rows[2:] for value in row)


def test_the_same_position_encodes_identically_whichever_number_holds_it():
    """The point of rotation: one network plays every seat."""
    board = fresh(seed=7).board
    observations = []

    for me, them in ((1, 2), (2, 1)):
        state = GameState(num_players=2, seed=7, board=board, player_order=[me, them])
        put_building(state, me, 20, Piece.CITY)
        put_building(state, them, 31, Piece.SETTLEMENT)
        put_road(state, me, 30)
        put_road(state, them, 40)
        give(state, me, wood=3, ore=2)
        give(state, them, sheep=4)
        state.dev_cards[me][DevCard.KNIGHT] = 2
        state.dev_cards[them][DevCard.MONOPOLY] = 1
        state.knights_played[me] = 3
        rules.update_awards(state)
        in_build_phase(state, me)
        observations.append(E.encode(state, me))

    assert observations[0] == observations[1]


# =========================================================================== #
# THE BOARD BLOCK                                                             #
# =========================================================================== #

def test_each_tile_reports_one_resource_and_one_number():
    state = mid_game()
    for index, row in enumerate(E.block(E.encode(state, 1), "tiles")):
        tile = index + 1
        resource_hot = row[:NUM_RESOURCES + 1]
        assert sum(resource_hot) == 1.0
        expected = NUM_RESOURCES if state.board.resource_at(tile) is None \
            else int(state.board.resource_at(tile))
        assert resource_hot[expected] == 1.0

        number_hot = row[NUM_RESOURCES + 1:NUM_RESOURCES + 1 + len(E.ROLLS)]
        assert sum(number_hot) == 1.0
        assert number_hot[E.ROLLS.index(state.board.number_at(tile))] == 1.0


def test_the_desert_shows_no_production_odds_despite_carrying_a_seven():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "tiles")
    odds_at = NUM_RESOURCES + 1 + len(E.ROLLS)
    desert = rows[state.board.desert_tile - 1]
    assert desert[odds_at] == 0.0

    other = next(t for t in range(1, T.NUM_TILES + 1)
                 if state.board.resource_at(t) is not None)
    assert rows[other - 1][odds_at] > 0.0


def test_exactly_one_tile_carries_the_robber():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "tiles")
    robber_at = E.TILE_FEATURES - 1
    flags = [row[robber_at] for row in rows]
    assert sum(flags) == 1.0
    assert flags[state.robber_tile - 1] == 1.0


def test_a_vertex_reports_its_owner_relative_to_me():
    state = mid_game()
    put_building(state, 2, 47, Piece.SETTLEMENT)
    rows = E.block(E.encode(state, 1), "vertices")

    for vertex in range(1, T.NUM_VERTICES + 1):
        owner_hot = rows[vertex - 1][:MAX_PLAYERS + 1]
        assert sum(owner_hot) == 1.0
        owner = state.vertex_owner[vertex]
        expected = 0 if owner == NO_OWNER else 1 + E.player_slots(state, 1)[owner]
        assert owner_hot[expected] == 1.0


def test_a_city_is_flagged_and_a_settlement_is_not():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    city_at = MAX_PLAYERS + 1
    for vertex in range(1, T.NUM_VERTICES + 1):
        expected = 1.0 if state.vertex_piece[vertex] is Piece.CITY else 0.0
        assert rows[vertex - 1][city_at] == expected


def test_a_vertex_reports_its_harbour():
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    harbour_at = MAX_PLAYERS + 2

    for vertex in range(1, T.NUM_VERTICES + 1):
        hot = rows[vertex - 1][harbour_at:harbour_at + E.HARBOUR_KINDS]
        harbours = state.board.harbours_at(vertex)
        if not harbours:
            assert hot[0] == 1.0 and sum(hot) == 1.0
        else:
            assert hot[0] == 0.0
            for harbour in harbours:
                slot = 1 if harbour is GENERIC_HARBOUR else 2 + int(harbour)
                assert hot[slot] == 1.0


def test_pip_potential_is_retired_but_its_slot_is_still_there():
    """The resource-blind pip total is switched off, and the column it occupied is not.

    Removing the field would shift every offset after it inside a vertex row and change
    ``SIZE``. The vertices block would have *shrunk*, which ``layouts.column_map`` refuses
    rather than guesses, so no existing checkpoint could be grafted onto the result and the
    reigning champion would stop loading. (``layouts.HISTORICAL`` is not the obstacle: it is
    keyed 1868 and 1884, and a checkpoint at the current ``SIZE`` carries its own layout.) So
    the slot stays and the value is 0. Both halves matter: the first is the compatibility
    guarantee, the second is the change.
    """
    state = mid_game()
    rows = E.block(E.encode(state, 1), "vertices")
    pips_at = E.VERTEX_OFFSETS["pip_potential"]

    assert not E.PIP_POTENTIAL, "this test describes the retired feature"
    assert pips_at == MAX_PLAYERS + 2 + E.HARBOUR_KINDS, "the slot moved"
    for vertex in range(1, T.NUM_VERTICES + 1):
        assert rows[vertex - 1][pips_at] == 0.0


def test_pip_potential_still_sums_the_adjacent_odds_when_switched_back_on():
    """The arithmetic is retired, not deleted — flipping the flag restores it exactly.

    Kept so that *trying* ``PIP_POTENTIAL`` again is an afternoon rather than an archaeology
    exercise, and so a change to ``expected_production`` cannot silently break the definition
    while the feature happens to be off. Shipping it back on is not one line: the test above
    asserts the flag is off, so a permanent flip has to change that test too.
    """
    state = mid_game()
    pips_at = E.VERTEX_OFFSETS["pip_potential"]

    original = E.PIP_POTENTIAL
    E.PIP_POTENTIAL = True
    try:
        state.board.__dict__.pop("_observation_template", None)   # the template is cached
        rows = E.block(E.encode(state, 1), "vertices")
    finally:
        E.PIP_POTENTIAL = original
        state.board.__dict__.pop("_observation_template", None)

    for vertex in range(1, T.NUM_VERTICES + 1):
        expected = sum(
            (6 - abs(7 - state.board.number_at(tile))) / 36
            for tile in T.VERTEX_TILES[vertex]
            if state.board.resource_at(tile) is not None
        )
        assert rows[vertex - 1][pips_at] == pytest.approx(expected)


def test_buildability_flags_agree_with_the_rules():
    state = mid_game()
    vertex_rows = E.block(E.encode(state, 1), "vertices")
    # By name, not by counting back from the end: features were appended to this row and a
    # `VERTEX_FEATURES - 2` was silently wrong the moment they were.
    distance_at = E.VERTEX_OFFSETS["buildable"]
    connected_at = E.VERTEX_OFFSETS["my_road"]
    for vertex in range(1, T.NUM_VERTICES + 1):
        row = vertex_rows[vertex - 1]
        assert row[distance_at] == float(rules.respects_distance_rule(state, vertex))
        assert row[connected_at] == float(rules.touches_own_road(state, 1, vertex))

    road_rows = E.block(E.encode(state, 1), "roads")
    reach_at = E.ROAD_FEATURES - 1
    for road in range(1, T.NUM_ROADS + 1):
        expected = (state.edge_owner[road] == NO_OWNER
                    and rules.is_road_connected(state, 1, road))
        assert road_rows[road - 1][reach_at] == float(expected)


# =========================================================================== #
# GLOBAL BLOCK                                                                #
# =========================================================================== #

def test_the_phase_is_one_hot():
    state = fresh(seed=1)
    for phase in Phase:
        state.phase = phase
        if phase is Phase.GAME_OVER:
            state.winner = 1
        if phase is Phase.DISCARD:
            state.pending_discards = [1]
            state.discards_owed[1] = 1
        values = E.encode(state, 1)[E.LAYOUT["global"]]
        hot = values[:len(Phase)]
        assert sum(hot) == 1.0
        assert hot[int(phase)] == 1.0


def test_no_roll_yet_has_its_own_slot():
    state = fresh(seed=1)
    values = E.encode(state, 1)[E.LAYOUT["global"]]
    roll_hot = values[len(Phase):len(Phase) + len(E.ROLLS) + 1]
    assert roll_hot[-1] == 1.0, "the 'not rolled yet' slot"

    state.last_roll = 8
    values = E.encode(state, 1)[E.LAYOUT["global"]]
    roll_hot = values[len(Phase):len(Phase) + len(E.ROLLS) + 1]
    assert roll_hot[E.ROLLS.index(8)] == 1.0
    assert roll_hot[-1] == 0.0


def test_the_ruleset_is_part_of_the_observation():
    """One network should be able to play either format, so it has to see which."""
    base = E.encode(mid_game(ruleset=BASE_GAME), 1)
    ranked = E.encode(mid_game(ruleset=RANKED_1V1), 1)
    assert base[E.LAYOUT["global"]] != ranked[E.LAYOUT["global"]]


def test_my_affordability_survives_scrambling_the_opponents_hand(played_game):
    """It is derived from my own hand, so it must be invariant under everything hidden."""
    state = played_game
    for me in state.players:
        before = E.encode(state, me)[E.LAYOUT["affordability"]]
        scrambled = scramble_hidden_state(state.clone(), me)
        after = E.encode(scrambled, me)[E.LAYOUT["affordability"]]
        assert before == after, "affordability moved when hidden cards changed"


def test_the_affordability_block_describes_me_not_whoever_is_deciding():
    """Building it from ``state.current_player`` would pass every other test in this file —
    ``mid_game()`` makes the two the same player. It would also hand over the discarding
    opponent's exact hand composition on every 7, which is why this test is not optional."""
    state = mid_game()
    give(state, 1, wood=1, brick=1)                            # player 1 can afford a road
    give(state, 2, wheat=state.ruleset.hand_limit + 4)          # only player 2 is over
    state.rolled_this_turn = True
    rules.begin_robber(state)
    assert state.phase is Phase.DISCARD
    assert state.current_player == 2, "the decision belongs to the opponent"

    assert afford(state, me=1)[ROAD] == [1.0, 0.0, 0.0, 1.0]
    assert afford(state, me=2)[ROAD][AFFORDABLE] == 0.0, "player 2 holds only wheat"

    before = E.encode(state, 1)[E.LAYOUT["affordability"]]
    state.hands[2][Resource.BRICK] += 5
    assert E.encode(state, 1)[E.LAYOUT["affordability"]] == before, \
        "my affordability moved when the opponent's hand changed"

    # The mirror leg, so "invariant" cannot be passing for the trivial reason that the block
    # never moves. One sheep and one wheat completes the settlement I was two cards from.
    assert afford(state, me=1)[SETTLEMENT][AFFORDABLE] == 0.0
    state.hands[1][Resource.SHEEP] += 1
    state.hands[1][Resource.WHEAT] += 1
    assert afford(state, me=1)[SETTLEMENT] == [1.0, 0.0, 0.0, 1.0]
    assert E.encode(state, 1)[E.LAYOUT["affordability"]] != before


def test_whose_decision_it_is_is_visible():
    """During a discard the decision can belong to an opponent, so an agent must be able
    to tell that it is not being asked to move."""
    state = mid_game()
    give(state, 1)                                            # player 1 holds nothing
    give(state, 2, wood=state.ruleset.hand_limit + 4)          # only player 2 is over
    state.rolled_this_turn = True
    rules.begin_robber(state)
    assert state.phase is Phase.DISCARD
    assert state.current_player == 2, "only player 2 should owe a discard"

    flag_at = (len(Phase) + len(E.ROLLS) + 1 + 1 + NUM_RESOURCES + 1 + 1 + 1 + 1)
    assert E.encode(state, 2)[E.LAYOUT["global"]][flag_at] == 1.0
    assert E.encode(state, 1)[E.LAYOUT["global"]][flag_at] == 0.0


# =========================================================================== #
# AFFORDABILITY — my hand against the cost table                              #
# =========================================================================== #
#
# The four columns of a row, by name, so no test indexes a literal.
AFFORDABLE, CARDS_SHORT, TRADE_PRICE, COVERABLE = range(E.AFFORDABILITY_FEATURES)

ROAD, SETTLEMENT, CITY, DEV_CARD = range(len(PURCHASES))


def no_harbour(seed=1):
    """A build decision for player 1 with no harbour, so every rate is 4:1.

    ``mid_game()`` cannot be used for anything rate-sensitive: its random setup happens to
    put player 1 on vertex 33, which carries a generic harbour, so its rates are 3:1. The
    assertion is the point — a change to board generation should fail loudly here rather
    than quietly change what the tests below measure.
    """
    state = fresh(seed=seed, num_players=2)
    in_build_phase(state, 1)
    assert rules.trade_rates(state, 1) == [BANK_RATE] * NUM_RESOURCES
    return state


def afford(state, me=1):
    return E.block(E.encode(state, me), "affordability")


def harbour_vertex(state, wanted):
    """A vertex carrying ``wanted``, found rather than written down."""
    return next(v for v in range(1, T.NUM_VERTICES + 1)
                if wanted in state.board.harbours_at(v))


def test_the_affordability_block_agrees_with_can_afford():
    """The cross-check that keeps the block honest, like the buildability flags above."""
    state = mid_game()
    for row, cost in enumerate(PURCHASES):
        expected = float(resources.can_afford(state.hands[1], cost))
        assert afford(state)[row][AFFORDABLE] == expected, PURCHASE_NAMES[row]

    give(state, 1, wheat=2)          # affords nothing at all now
    for row, cost in enumerate(PURCHASES):
        expected = float(resources.can_afford(state.hands[1], cost))
        assert afford(state)[row][AFFORDABLE] == expected, PURCHASE_NAMES[row]


def test_one_card_short_of_a_settlement_shows_a_quarter():
    """The judgement the block exists for. Today's observation cannot tell "one card away"
    from "three cards away" — both are simply an illegal action."""
    state = no_harbour()
    give(state, 1, wood=1, brick=1, sheep=1)          # only the wheat is missing
    row = afford(state)[SETTLEMENT]
    assert row[AFFORDABLE] == 0.0
    assert row[CARDS_SHORT] == pytest.approx(1 / sum(resources.SETTLEMENT_COST))


def test_the_deficit_counts_cards_not_kinds():
    state = no_harbour()
    give(state, 1, wheat=2)                            # the city still wants three ore
    row = afford(state)[CITY]
    assert row[CARDS_SHORT] == pytest.approx(3 / sum(resources.CITY_COST))


def test_a_two_for_one_harbour_halves_the_price_of_a_deficit():
    """The whole point of pricing through the rates, and the direction is easy to get
    backwards: ``trade_rates`` is indexed by the resource *given*, so a wheat harbour makes
    wheat cheap to dump and does nothing to help acquire wheat."""
    state = no_harbour()
    give(state, 1, wood=6)                             # one brick short, five spare wood
    assert afford(state)[ROAD][TRADE_PRICE] == pytest.approx(BANK_RATE / E.TRADE_PRICE_SCALE)
    assert afford(state)[ROAD][COVERABLE] == 1.0

    put_building(state, 1, harbour_vertex(state, Resource.WOOD))
    assert rules.trade_rates(state, 1)[Resource.WOOD] == SPECIFIC_HARBOUR_RATE
    assert afford(state)[ROAD][TRADE_PRICE] == pytest.approx(
        SPECIFIC_HARBOUR_RATE / E.TRADE_PRICE_SCALE
    )


def test_surplus_spread_across_resources_cannot_fund_a_trade():
    """A trade is paid with four cards of *one* resource, so the surplus is floored per
    resource. Pooling it first is the natural bug and it is wrong."""
    state = no_harbour()
    give(state, 1, wood=3, sheep=3)                    # six spare cards, no trade fundable
    row = afford(state)[ROAD]
    assert row[COVERABLE] == 0.0
    assert row[TRADE_PRICE] == 1.0


def test_the_cards_a_purchase_needs_are_not_counted_as_surplus():
    state = no_harbour()
    give(state, 1, brick=4)             # the road wants one brick, so the surplus is three
    assert afford(state)[ROAD][COVERABLE] == 0.0

    give(state, 1, brick=5)                            # now four spare, one trade fundable
    assert afford(state)[ROAD][COVERABLE] == 1.0
    assert afford(state)[ROAD][TRADE_PRICE] == pytest.approx(BANK_RATE / E.TRADE_PRICE_SCALE)

    # And the give == take leg: eight spare wheat funds two trades, three ore are needed.
    give(state, 1, wheat=10)
    assert afford(state)[CITY][COVERABLE] == 0.0


def test_the_cheapest_trades_are_counted_first():
    state = no_harbour()
    put_building(state, 1, harbour_vertex(state, Resource.WOOD))
    put_building(state, 1, harbour_vertex(state, GENERIC_HARBOUR))
    assert rules.trade_rates(state, 1) == [2, 3, 3, 3, 3]

    # The city needs two more ore. Wood funds one trade at 2, sheep two at 3, so the
    # cheapest bill is 2 + 3 = 5. Any other ordering pays 6.
    give(state, 1, wood=3, sheep=6, wheat=2, ore=1)
    assert afford(state)[CITY][TRADE_PRICE] == pytest.approx(5 / E.TRADE_PRICE_SCALE)


@pytest.mark.slow
def test_the_cheapest_plan_is_the_cheapest_one():
    """Cheapest-rate-first is claimed to be optimal, not merely plausible — every bank
    trade yields exactly one card whatever it cost, so the cheapest units win. Checked
    against an exhaustive search rather than argued, because a future edit that swaps the
    sort key would still pass every other test here."""
    import itertools

    def brute(hand, cost, rates, bank):
        need = sum(max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES))
        if need == 0:
            return 0, True
        if any(bank[r] < max(0, cost[r] - hand[r]) for r in range(NUM_RESOURCES)):
            return None, False
        # clamped at zero: a resource the purchase is short of has nothing to give
        spare = [max(0, hand[g] - cost[g]) for g in range(NUM_RESOURCES)]
        best = None
        for counts in itertools.product(range(need + 1), repeat=NUM_RESOURCES):
            if sum(counts) != need:
                continue
            if any(counts[g] * rates[g] > spare[g] for g in range(NUM_RESOURCES)):
                continue
            bill = sum(counts[g] * rates[g] for g in range(NUM_RESOURCES))
            best = bill if best is None else min(best, bill)
        return best, best is not None

    rng = random.Random(0)
    state = no_harbour()
    for _ in range(400):
        hand = [rng.randint(0, 8) for _ in range(NUM_RESOURCES)]
        bank = [rng.randint(0, 4) for _ in range(NUM_RESOURCES)]
        state.hands[1] = list(hand)
        state.bank = list(bank)
        rates = rules.trade_rates(state, 1)
        rows = afford(state)

        for row, cost in enumerate(PURCHASES):
            price, coverable = brute(hand, cost, rates, bank)
            assert rows[row][COVERABLE] == float(coverable), (hand, bank, cost)
            if coverable and price:
                assert rows[row][TRADE_PRICE] == pytest.approx(
                    min(price / E.TRADE_PRICE_SCALE, 1.0)
                ), (hand, bank, cost)


def test_an_empty_bank_leaves_a_deficit_uncoverable():
    state = no_harbour()
    give(state, 1, wood=12)                            # one brick short, plenty to trade
    assert afford(state)[ROAD][COVERABLE] == 1.0

    short_before = afford(state)[ROAD][CARDS_SHORT]
    state.bank[Resource.BRICK] = 0
    row = afford(state)[ROAD]
    assert row[COVERABLE] == 0.0
    assert row[TRADE_PRICE] == 1.0
    assert row[CARDS_SHORT] == short_before, "the bank constrains the plan, not the distance"

    # A bank holding *some* of what is needed is still not enough: each trade takes one
    # card, and paying refills the pile you gave from, never the one you took.
    state = no_harbour()
    give(state, 1, wheat=2, sheep=16)                  # three ore short, four trades funded
    assert afford(state)[CITY][COVERABLE] == 1.0
    state.bank[Resource.ORE] = 1
    assert afford(state)[CITY][COVERABLE] == 0.0


def test_a_free_road_needs_no_cards():
    state = no_harbour()
    give(state, 1)                                     # nothing at all in hand
    assert afford(state)[ROAD] == [0.0, 1.0, 1.0, 0.0]

    state.free_roads = ROAD_BUILDING_ROADS
    assert afford(state)[ROAD] == [1.0, 0.0, 0.0, 1.0]
    assert afford(state)[SETTLEMENT] == [0.0, 1.0, 1.0, 0.0], "only the road is free"

    state.free_roads = 0
    assert afford(state)[ROAD] == [0.0, 1.0, 1.0, 0.0]

    # The counter is global and lapses at END_TURN, so credit outstanding while an opponent
    # acts is not mine.
    state.free_roads = 1
    state.turn_number = state.player_order.index(2)
    assert state.current_player == 2
    assert afford(state)[ROAD] == [0.0, 1.0, 1.0, 0.0]


def test_affordability_is_not_legality():
    """The block's inputs are exactly hand, costs, rates, free roads and the bank. Piece
    supply, deck size and placement are already encoded elsewhere, and folding any of them
    in here would make one float mean several things."""
    state = no_harbour()
    give(state, 1, sheep=1, wheat=1, ore=1)

    before = afford(state)[DEV_CARD]
    state.dev_deck.clear()
    assert not rules.can_buy_dev_card(state, 1)
    assert afford(state)[DEV_CARD] == before, "an empty deck is in the global block"

    give(state, 1, wheat=2, ore=3)
    before = afford(state)[CITY]
    state.cities_left[1] = 0
    assert afford(state)[CITY] == before, "the piece supply is in the player block"


def test_the_block_is_live_during_setup():
    """No phase branch. Zeroing it would make 0.0 mean both "cannot afford" and "not
    applicable", forcing the net to learn a product with a phase one-hot it already has."""
    state = fresh(seed=1, num_players=2)
    assert state.phase is Phase.SETUP_SETTLEMENT
    assert afford(state)[SETTLEMENT] == [0.0, 1.0, 1.0, 0.0]


def test_the_price_saturates_rather_than_leaving_the_unit_range():
    state = no_harbour()
    give(state, 1)
    rows = afford(state)
    assert all(row[TRADE_PRICE] == 1.0 for row in rows)

    values = E.encode(state, 1)[E.LAYOUT["affordability"]]
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in values)


# =========================================================================== #
# DETERMINISM AND COST                                                        #
# =========================================================================== #

def test_encoding_is_pure():
    state = mid_game()
    before = state.clone()
    first = E.encode(state, 1)
    assert state == before, "encoding mutated the state"
    assert E.encode(state, 1) == first


@pytest.mark.slow
def test_encoding_works_at_every_point_of_a_real_game():
    def check(state):
        for player in state.players:
            observation = E.encode(state, player)
            assert len(observation) == E.SIZE
            assert all(0.0 <= v <= 1.0 for v in observation)

    for seed in range(4):
        play_random_game(seed=seed, num_players=3, max_actions=900, on_step=check)


def test_encoding_is_fast_enough_to_sit_in_the_loop():
    import timeit
    state = mid_game()
    elapsed = timeit.timeit(lambda: E.encode(state, 1), number=2000) / 2000
    assert elapsed < 5e-3, f"{elapsed * 1e6:.0f} us per observation is too slow"


# =========================================================================== #
# THE TWO ENTRY POINTS                                                        #
# =========================================================================== #

def _same_floats(state, me):
    """``(encode, encode_into)`` for one position, both as tuples of float32."""
    import struct

    buffer = E.observation_buffer()
    E.encode_into(state, me, buffer)
    as_list = E.encode(state, me)
    # Round the list through float32 too: the comparison is "do the two doors agree",
    # not "does float32 have 53 bits of mantissa".
    narrowed = struct.unpack(f"{E.SIZE}f", struct.pack(f"{E.SIZE}f", *as_list))
    return narrowed, tuple(buffer)


def test_encode_into_matches_the_list_encoding_bit_for_bit():
    """The two doors must be one encoder wearing two coats.

    This single assertion subsumes three others: the same values means the leak detectors
    above cover ``encode_into`` too, the same order means the network's input layout is
    unchanged, and the same numbers means a checkpoint trained through one door still means
    something through the other.
    """
    for ruleset in ALL:
        for num_players in (2, 3, 4):
            state = mid_game(seed=3, ruleset=ruleset, num_players=num_players)
            for me in state.players:
                as_list, as_buffer = _same_floats(state, me)
                assert as_list == as_buffer, (
                    f"{ruleset.name}, {num_players}p, seat {me}: the buffer encoding "
                    f"differs at "
                    f"{[i for i, (a, b) in enumerate(zip(as_list, as_buffer)) if a != b][:5]}"
                )


def test_encode_still_returns_python_floats():
    """What stops ``encode`` quietly becoming an array: every other caller indexes it as a
    list, and ``block`` slices it."""
    observation = E.encode(mid_game(), 1)
    assert isinstance(observation, list)
    assert all(isinstance(value, float) for value in observation)


@pytest.mark.slow
def test_the_two_entry_points_agree_over_whole_games():
    def check(state):
        for me in state.players:
            as_list, as_buffer = _same_floats(state, me)
            assert as_list == as_buffer

    for seed in range(3):
        play_random_game(seed=seed, num_players=3, max_actions=700, on_step=check)


def test_a_buffer_is_fully_overwritten_between_positions():
    """The template reset is what makes a reused buffer safe. Without it a stale 1.0 from
    the previous position survives wherever the new one writes nothing."""
    state = mid_game()
    buffer = E.observation_buffer()

    E.encode_into(state, 1, buffer)
    first = tuple(buffer)

    other = mid_game(seed=9)
    E.encode_into(other, 1, buffer)
    E.encode_into(state, 1, buffer)
    assert tuple(buffer) == first, "encoding a second position left residue in the buffer"


# =========================================================================== #
# THE FUSED SURVEY                                                            #
# =========================================================================== #

def test_fused_player_scores_agree_with_the_rules():
    """``encoder._survey`` is a second implementation of scoring, which this repository
    otherwise forbids. This test is the entire justification for it: :mod:`catan.rules`
    stays the authority and the fast route has to keep producing the authority's answer.

    Checked at *every* position of whole games, not at a sampled few — the interesting
    cases are the ones nobody thinks to construct: an award changing hands, a settlement
    becoming a city, a harbour vertex bought late.
    """
    def check(state):
        points, production, rates = E._survey(state)
        for player in state.players:
            assert points[player] == rules.public_victory_points(state, player)
            assert rates[player] == rules.trade_rates(state, player)
            assert production[player] == pytest.approx(
                rules.production_rates(state, player))
            hidden = state.dev_cards[player][DevCard.VICTORY_POINT]
            assert points[player] + hidden == rules.victory_points(state, player)

    for ruleset in ALL:
        for num_players in (2, 3, 4):
            play_random_game(seed=num_players, num_players=num_players,
                             ruleset=ruleset, max_actions=600, on_step=check)


def test_the_survey_survives_a_board_with_no_award_holders():
    """``largest_army_holder`` and ``longest_road_holder`` are ``None``, not 0, until
    somebody earns them — and ``None`` is not a key in the per-player tables."""
    state = fresh(seed=1)
    assert state.largest_army_holder is None and state.longest_road_holder is None
    points, _, _ = E._survey(state)
    assert set(points) == set(state.players)
    assert all(value == 0 for value in points.values())


# =========================================================================== #
# THE PUBLIC RECORD                                                           #
# =========================================================================== #

def test_the_public_record_is_public(played_game):
    """The history block is the newest place a leak could hide, so it gets its own check
    on top of the whole-vector detectors above: rewrite every hidden thing at constant
    public counts and require the block not to move by a single float."""
    state = played_game
    for me in state.players:
        before = E.encode(state, me)[E.LAYOUT["history"]]
        rolls_before = E.encode(state, me)[E.LAYOUT["rolls"]]

        scrambled = scramble_hidden_state(state.clone(), me)
        after = E.encode(scrambled, me)[E.LAYOUT["history"]]
        rolls_after = E.encode(scrambled, me)[E.LAYOUT["rolls"]]

        assert before == after, "the public record moved when hidden cards changed"
        assert rolls_before == rolls_after


def test_the_public_record_actually_records_something(played_game):
    """A block of zeros would pass every leak test ever written."""
    state = played_game
    history = E.encode(state, 1)[E.LAYOUT["history"]]
    rolls = E.encode(state, 1)[E.LAYOUT["rolls"]]
    assert any(v > 0 for v in history), "nothing was recorded"
    assert any(v > 0 for v in rolls)
    assert sum(rolls[:len(E.ROLLS)]) == pytest.approx(1.0), "the histogram is not a distribution"


def test_the_public_record_is_rotated_like_everything_else(played_game):
    """Slot 0 is always me. Without this, one network could not play both seats."""
    state = played_game
    width = E.HISTORY_FEATURES
    for me in state.players:
        block = E.encode(state, me)[E.LAYOUT["history"]]
        mine = block[:width]
        expected = [
            min(state.produced[me][r] / E.PRODUCTION_SCALE, 1.0) for r in range(5)
        ]
        assert mine[:5] == pytest.approx(expected)


def test_production_and_spending_are_counted(played_game):
    state = played_game
    for player in state.players:
        assert sum(state.produced[player]) > 0, "nobody ever produced anything"
        assert sum(state.spent[player]) > 0, "nobody ever paid for anything"
    assert sum(state.roll_counts) > 0


def test_the_record_survives_a_clone(played_game):
    state = played_game
    clone = state.clone()
    assert clone.roll_counts == state.roll_counts
    assert clone.produced == state.produced
    assert clone.spent == state.spent
    assert clone.dev_bought == state.dev_bought
    assert clone.last_build_turn == state.last_build_turn
    # and is a copy, not a reference
    clone.produced[1][0] += 99
    assert clone.produced[1][0] != state.produced[1][0]
