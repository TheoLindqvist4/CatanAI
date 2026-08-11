"""The end-of-game statistics panel.

Two of these carry the weight.

:func:`test_what_the_robber_blocked_agrees_with_the_rules` is the licence for
:func:`interfaces.web.stats.owed` to exist at all. It is a second implementation of the
production walk, which the one-source-of-truth rule would otherwise forbid; what makes it
legal is being compared against :func:`catan.rules.distribute` at every position of whole
games, in the same spirit as ``test_fused_player_scores_agree_with_the_rules`` licences the
encoder's ``_survey``.

:func:`test_the_statistics_never_leak_hidden_information` is the leak test. The panel is a
second response shape reaching the browser, and it reports on the *history* of a game rather
than its position — so "is this public?" has to be asked about a running total rather than
about a card, which is a question the existing leak test over ``view`` never has to ask.
"""

import json
import pathlib
import random
import re

import pytest

from catan import rules
from catan.dev_cards import DevCard
from catan.events import EventKind
from catan.resources import NUM_RESOURCES

api = pytest.importorskip("interfaces.web.api")
from interfaces.web import stats  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def options(view):
    board = [i for targets in view["actions"]["board"].values() for i in targets.values()]
    return board + [entry["index"] for entry in view["actions"]["panel"]]


def play_out(game, limit=4000, rng=None):
    """Click through a whole game the way the client would."""
    rng = rng or random.Random(0)
    view = game.view()
    for _ in range(limit):
        if view["done"]:
            break
        view = game.play(rng.choice(options(view)))
    return view


def finished(seed=4, opponent="hard"):
    game = api.Game(seed=seed, opponent=opponent)
    view = play_out(game, rng=random.Random(seed))
    if not view["done"]:
        pytest.skip("game did not finish inside the step budget")
    return game


# =========================================================================== #
# THE ROBBER'S DAMAGE — the figure the engine throws away                     #
# =========================================================================== #

@pytest.mark.parametrize("seed", [1, 4, 9])
def test_what_the_robber_blocked_agrees_with_the_rules(seed):
    """``owed`` must be exactly the payout ``distribute`` computes before the bank bites.

    The one number in the panel that cannot be counted and has to be derived: the engine
    skips a blocked tile with a bare ``continue`` and the amount is gone. This is what stops
    the derivation drifting away from the rule it mirrors — if the payout ever changes, the
    comparison here is what says so.

    Checked at *every* position of a game and for *every* roll, not just the roll that
    actually came up, so a mistake that only shows on an 8 cannot hide behind a game that
    never rolled one.
    """
    game = api.Game(seed=seed, opponent="hard")
    rng = random.Random(seed)
    view = game.view()

    positions = 0
    for _ in range(400):
        if view["done"]:
            break
        for roll in range(2, 13):
            mine = stats.owed(game.state, roll, game.state.robber_tile)

            # A full bank, because `distribute` applies the shortage rule and `owed` is
            # deliberately the figure *before* it: a card the bank could not pay was not
            # blocked by the robber, and folding the two together would make the panel
            # blame it for the bank running dry.
            theirs = game.state.clone()
            theirs.bank = [19] * NUM_RESOURCES
            paid = rules.distribute(theirs, roll)

            for player in game.state.players:
                assert mine[player] == paid[player], (
                    f"roll {roll}, player {player}: {mine[player]} != {paid[player]}")
        positions += 1
        view = game.play(rng.choice(options(view)))

    assert positions > 20, "the walk ended too early to have tested anything"


def test_nothing_is_blocked_when_the_robber_sits_on_the_desert():
    """Where it starts, and the one tile whose blocking cannot cost anybody a card."""
    game = api.Game(seed=3, opponent="hard")
    state = game.state
    assert state.robber_tile == state.board.desert_tile

    for roll in range(2, 13):
        with_robber = stats.owed(state, roll, state.robber_tile)
        without = stats.owed(state, roll, None)
        assert with_robber == without


def test_the_robbers_damage_is_counted_over_a_whole_game():
    """It has to actually fire — a tally that is always zero would pass every other test."""
    blocked = 0
    for seed in range(1, 9):
        game = finished(seed=seed)
        blocked += sum(p["blockedTotal"] for p in game.statistics()["players"])
    assert blocked > 0, "the robber never blocked anything in eight games"


# =========================================================================== #
# THE DICE                                                                    #
# =========================================================================== #

def test_the_dice_add_up_to_the_engines_own_count():
    """``state.roll_counts`` is the column sum of the per-player table.

    The engine keeps a bare histogram with no attribution, so who rolled what is counted
    here from the events instead. Two counts of one thing is exactly the drift this project
    keeps removing; this is what holds them together.
    """
    game = finished(seed=5)
    tally = game.stats
    for total in range(2, 13):
        counted = sum(tally.rolls[p][total] for p in game.state.players)
        assert counted == game.state.roll_counts[total], f"total {total}"


def test_the_reported_distribution_is_the_per_player_table_summed():
    game = finished(seed=6)
    report = game.statistics()
    for total in range(2, 13):
        per_player = sum(
            report["dice"]["byPlayer"][str(p)][str(total)] for p in game.state.players
        )
        assert report["dice"]["totals"][str(total)] == per_player


def test_every_total_gets_a_row_even_when_it_never_came_up():
    """A gap in the middle of a histogram reads as missing data, not as 'never rolled'."""
    game = api.Game(seed=2, opponent="hard")
    report = game.statistics()
    assert [int(k) for k in report["dice"]["totals"]] == list(range(2, 13))
    assert set(report["dice"]["totals"].values()) == {0}, "nothing has been rolled yet"


def test_a_players_sevens_are_their_own_rolls_not_everyones():
    game = finished(seed=5)
    report = game.statistics()
    for entry in report["players"]:
        assert entry["sevensRolled"] == game.stats.rolls[entry["id"]][7]
    total = sum(entry["sevensRolled"] for entry in report["players"])
    assert total == game.state.roll_counts[7]


# =========================================================================== #
# SEVENS, DISCARDS AND STEALS                                                 #
# =========================================================================== #

def test_cards_discarded_matches_the_log_the_player_read():
    """One event per card, so the tally is countable against the log itself.

    The log rather than ``info["events"]``: an unpaced ``play`` runs the opponent's whole
    reply and ``game.info`` only survives the last step of it, so sampling it after each
    call sees a fraction of what happened. The log is fed from the same place the tally is
    and drops nothing, which is what makes it the oracle here — the panel and the narration
    are supposed to be two readings of one event stream.
    """
    game = finished(seed=8)
    counted = {
        player: sum(1 for line in game.log if line.startswith(f"{name} discarded "))
        for player, name in game.names.items()
    }

    assert sum(counted.values()) > 0, "nobody ever discarded; pick another seed"
    for entry in game.statistics()["players"]:
        assert entry["cardsDiscarded"] == counted[entry["id"]]
        assert sum(entry["discarded"].values()) == counted[entry["id"]]


def test_a_seven_out_counts_the_seven_not_the_cards():
    """Losing four cards to one 7 is being caught out once.

    The two numbers answer different questions — how often you were over the limit, and how
    much it cost — so a panel that conflated them would answer neither.
    """
    game = finished(seed=8)
    for entry in game.statistics()["players"]:
        if entry["cardsDiscarded"]:
            assert entry["sevenOuts"] >= 1
            assert entry["sevenOuts"] <= entry["cardsDiscarded"]
        else:
            assert entry["sevenOuts"] == 0


def test_discards_spread_across_steps_still_count_as_one_seven():
    """One discard is one action, so a four-card discard arrives as four separate steps.

    Counting a seven-out per *event* rather than per 7 would multiply the figure by the size
    of the hand, which is the one mistake this tally can make.
    """
    tally = stats.GameStats([1, 2])
    game = api.Game(seed=1, opponent="hard")
    state = game.state

    from catan.events import Event

    tally.record(state, [Event(EventKind.ROLLED, 1, amount=7)])
    for resource in range(4):
        tally.record(state, [Event(EventKind.DISCARDED, 2, resource=resource)])

    assert tally.seven_outs[2] == 1
    assert sum(tally.discarded[2]) == 4

    # A second 7 is a second time they were caught out.
    tally.record(state, [Event(EventKind.ROLLED, 2, amount=7)])
    tally.record(state, [Event(EventKind.DISCARDED, 2, resource=0)])
    assert tally.seven_outs[2] == 2


def test_a_steal_is_recorded_on_both_sides():
    """The thief gained it and the victim lost it — one event, two tallies."""
    game = finished(seed=6)
    report = {entry["id"]: entry for entry in game.statistics()["players"]}
    for resource in report[1]["stolenBy"]:
        assert report[1]["stolenBy"][resource] == report[2]["stolenFrom"][resource]
        assert report[2]["stolenBy"][resource] == report[1]["stolenFrom"][resource]


def test_knights_come_from_the_engines_own_counter():
    """Largest Army is decided by it, so the panel must not keep a second one."""
    game = finished(seed=5)
    for entry in game.statistics()["players"]:
        assert entry["knights"] == game.state.knights_played[entry["id"]]


# =========================================================================== #
# WHERE THE POINTS CAME FROM                                                  #
# =========================================================================== #

@pytest.mark.parametrize("seed", [2, 5, 11])
def test_the_point_breakdown_adds_up_to_the_score(seed):
    """The parts must sum to the number that decides the game.

    Checked at every position, because a breakdown that is only right at the end is a
    breakdown that is wrong while it is being watched.
    """
    game = api.Game(seed=seed, opponent="hard")
    rng = random.Random(seed)
    view = game.view()

    for _ in range(400):
        for player in game.state.players:
            sources = stats.victory_point_sources(game.state, player)
            assert sum(row["points"] for row in sources) == rules.victory_points(
                game.state, player)
        if view["done"]:
            break
        view = game.play(rng.choice(options(view)))


def test_the_breakdown_names_a_count_as_well_as_the_points():
    """Two points for Longest Road says nothing about how long the road is."""
    game = finished(seed=5)
    for entry in game.statistics()["players"]:
        by_source = {row["source"]: row for row in entry["pointSources"]}
        player = entry["id"]
        assert by_source["cities"]["points"] == by_source["cities"]["count"] * 2
        assert by_source["longest road"]["count"] == rules.longest_road_length(
            game.state, player)
        assert by_source["largest army"]["count"] == game.state.knights_played[player]


def test_an_award_is_worth_nothing_until_it_is_held():
    game = finished(seed=5)
    for entry in game.statistics()["players"]:
        for name in ("longest road", "largest army"):
            row = next(r for r in entry["pointSources"] if r["source"] == name)
            assert row["points"] == (2 if row["held"] else 0)


# =========================================================================== #
# HIDDEN INFORMATION — the leak test                                          #
# =========================================================================== #

def test_the_statistics_never_leak_hidden_information():
    """Walk a whole game and inspect every statistics payload the human could fetch.

    The panel is fetched by a click, so it can be fetched at any moment of the game — which
    makes it exactly as capable of leaking as the view is.
    """
    game = api.Game(seed=7, opponent="hard")
    rng = random.Random(7)
    view = game.view()

    for _ in range(1500):
        report = game.statistics()
        opponent = next(p for p in report["players"] if not p["you"])

        if not view["done"]:
            sources = {row["source"] for row in opponent["pointSources"]}
            assert "victory point cards" not in sources, (
                "the opponent's hidden victory points leaked")
            assert opponent["victoryPointsComplete"] is False
            assert opponent["victoryPoints"] == rules.public_victory_points(
                game.state, opponent["id"])

        # Neither deck may appear anywhere in the payload, at any point.
        blob = json.dumps(report)
        assert "devDeckOrder" not in blob and "diceDeck" not in blob

        if view["done"]:
            break
        view = game.play(rng.choice(options(view)))


def test_the_opponents_hidden_points_are_shown_once_the_game_ends():
    """Hidden during play, revealed at the end so the player can see what beat them —
    exactly the rule ``view`` applies to a hand."""
    game = finished(seed=11)
    report = game.statistics()
    opponent = next(p for p in report["players"] if not p["you"])
    sources = {row["source"] for row in opponent["pointSources"]}
    assert "victory point cards" in sources
    assert opponent["victoryPointsComplete"] is True
    assert opponent["victoryPoints"] == rules.victory_points(game.state, opponent["id"])


def test_you_always_see_your_own_point_breakdown():
    game = api.Game(seed=5, opponent="hard")
    game.state.dev_cards[api.HUMAN][DevCard.VICTORY_POINT] = 2
    you = next(p for p in game.statistics()["players"] if p["you"])
    row = next(r for r in you["pointSources"] if r["source"] == "victory point cards")
    assert row["points"] == 2
    assert you["victoryPointsComplete"] is True


def test_a_watched_game_reveals_nobodys_hidden_cards_early():
    """There is no "you" in a game nobody is playing, so nobody's hand is the viewer's."""
    game = api.Game(opponent="hard", watch="greedy", seed=11)
    for _ in range(30):
        game.advance()
    report = game.statistics()
    assert not any(entry["you"] for entry in report["players"])
    for entry in report["players"]:
        sources = {row["source"] for row in entry["pointSources"]}
        assert "victory point cards" not in sources
        assert entry["victoryPointsComplete"] is False


# =========================================================================== #
# THE PAYLOAD                                                                 #
# =========================================================================== #

def test_the_report_carries_what_the_panel_draws():
    game = finished(seed=4)
    report = game.statistics()
    for key in ("turns", "rolls", "dice", "players"):
        assert key in report
    for key in ("totals", "expected", "ways", "balanced", "byPlayer"):
        assert key in report["dice"], key
    for entry in report["players"]:
        for key in ("id", "name", "you", "rolled", "sevensRolled", "knights",
                    "robberMoves", "sevenOuts", "cardsDiscarded", "discarded",
                    "blocked", "blockedTotal", "stolenBy", "stolenFrom", "produced",
                    "producedTotal", "monopolised", "gained", "gainedTotal", "lost",
                    "lostTotal", "spent", "devBought", "victoryPoints",
                    "victoryPointsComplete", "pointSources"):
            assert key in entry, key


def test_the_report_is_json_serialisable():
    json.dumps(finished(seed=4).statistics())


def test_the_players_are_named_the_way_the_log_names_them():
    """One mapping, so the panel and the log cannot disagree about who is who."""
    game = api.Game(seed=4, opponent="hard")
    assert {e["id"]: e["name"] for e in game.statistics()["players"]} == game.names

    watched = api.Game(opponent="hard", watch="greedy", seed=4)
    names = {e["id"]: e["name"] for e in watched.statistics()["players"]}
    assert names == watched.names == {1: "greedy", 2: "hard"}


def test_production_is_the_engines_own_running_total():
    game = finished(seed=4)
    for entry in game.statistics()["players"]:
        expected = game.state.produced[entry["id"]]
        assert list(entry["produced"].values()) == list(expected)


def test_statistics_work_before_anything_has_happened():
    """The button exists from the first move, so the panel has to survive an empty game."""
    report = api.Game(seed=1, opponent="hard").statistics()
    assert report["rolls"] == 0
    assert all(entry["gainedTotal"] == 0 for entry in report["players"])
    assert all(entry["blockedTotal"] == 0 for entry in report["players"])


# =========================================================================== #
# TRAINING PAYS FOR NONE OF IT                                                #
# =========================================================================== #

def test_the_engine_does_not_know_the_statistics_exist():
    """The whole design in one assertion.

    A counter on ``GameState`` would be copied by ``clone()`` on every MCTS node expansion,
    so a figure read once at the end of one browser game would be paid for by every search
    in training. Everything the panel reports is either already-public bookkeeping the
    engine keeps for the observation, or is derived in the interface from ``info["events"]``.
    """
    # Import statements, not mentions: a docstring is free to say where an agent ends up
    # plugged in, and two of them do. What must not exist is a dependency.
    importing = re.compile(r"^\s*(?:from|import)\s+interfaces", re.MULTILINE)

    offenders = []
    for folder in ("catan", "training", "benchmark"):
        for path in (ROOT / folder).rglob("*.py"):
            if importing.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"the engine or training imports the interface: {offenders}"


def test_the_statistics_are_not_computed_unless_they_are_asked_for():
    """``view`` is built four times a turn; the panel is built when somebody clicks.

    Keeping the report off the per-move payload is what makes "the interface got a stats
    panel" cost the browser game nothing per move, let alone training.
    """
    view = api.Game(seed=1, opponent="hard").view()
    assert "stats" not in view and "statistics" not in view
