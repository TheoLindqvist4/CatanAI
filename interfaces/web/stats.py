"""What happened over a whole game, for the panel behind the Stats button.

**Nothing here touches the engine, and that is the point.** A counter added to
:class:`~catan.state.GameState` is copied by ``clone()`` on *every* MCTS node expansion —
`state.clone` is 4 us and the search calls it hundreds of thousands of times an hour — so a
tally that exists to be read once, at the end of one browser game, would be paid for by
every search in training. Everything below is derived in the interface instead, from
``info["events"]`` and public state the engine already keeps for its own reasons.

Three quarters of it is already recorded and simply needs reading:

``state.roll_counts``      how often each total came up
``state.produced``         cumulative production per player per resource
``state.spent``            cumulative spending per player per resource
``state.knights_played``   knights played per player
``state.dev_bought``       development cards bought per player

Those exist because the observation's ``history`` and ``rolls`` blocks need them, and they
are public by construction — the docstring on that block in ``catan/state.py`` states the
rule: nothing derived from a hidden quantity belongs there.

Two things are recorded nowhere and are accumulated here, a step at a time:

* **Who rolled what.** ``roll_counts`` is a bare per-total histogram with no attribution, so
  "how many 7s have I rolled" cannot be read off it. The ``ROLLED`` event carries the player.
* **The robber's damage.** :func:`catan.rules.distribute` skips a blocked tile with a bare
  ``continue`` and the amount is gone — nothing returns it and no event reports it. It is
  the one figure here that has to be *computed* rather than counted, which is what
  :func:`owed` is for.

⚠️ **This is a second implementation of the production walk**, which the one-source-of-truth
rule would otherwise forbid. What makes it legal is
``tests/test_web_stats.py::test_what_the_robber_blocked_agrees_with_the_rules``, which drives
whole games and compares :func:`owed` against what :func:`catan.rules.distribute` actually
paid — at every position, and for every roll from 2 to 12 rather than only the one that came
up, so a mistake that shows on an 8 cannot hide behind a game that never rolled one. The
comparison is made against a *full* bank on purpose: :func:`owed` is the figure before the
shortage rule, because a card the bank could not pay was not blocked by the robber. If the
payout rules change, that test is what will say so.

**Hidden information.** :func:`report` takes ``reveal`` for the same reason
:func:`interfaces.web.api.view` takes ``info["done"]``: a player's Victory Point cards are
hidden until the game ends, so the victory-point breakdown is the one block here that is not
public knowledge. Everything else already is — production, discards, steals and monopolies
are all narrated in the log both players read, and what the robber blocked is derivable by
anyone who can see the board.

The *condition* is ``view``'s; the *seat* is not. In a watched game
:func:`interfaces.web.api.statistics` passes ``you=None``, so nobody's Victory Point row is
shown early, where ``view`` still treats seat 1 as "you" and hands its hidden cards back.
This panel is the stricter of the two, deliberately — see that function's docstring for why
both are right.

**Some of what is built here is carried and not drawn.** ``gained``/``gainedTotal`` and
``lost``/``lostTotal`` on a player, and ``ways`` in the dice block, reach the browser and
``static/app.js`` renders none of them: it draws ``produced``, ``spent``, ``blocked``,
``discarded`` and the steal and monopoly totals, and reads ``expected`` rather than the
``ways`` it was derived from. ``test_the_report_carries_what_the_panel_draws`` asserts the
keys are present — its name is a shade optimistic for these five — so they will not quietly
vanish. They are a loose end and not a lie: the numbers are right, nothing shows them yet.
"""

from catan import rules
from catan.dev_cards import AWARD_VICTORY_POINTS, DevCard
from catan.events import EventKind
from catan.resources import NUM_RESOURCES, Resource
from catan.state import NO_OWNER, PIECE_YIELD, Piece
from catan.topology import NUM_VERTICES

#: Every dice total, so a distribution has a row for a number that never came up. A gap in
#: the middle of a histogram reads as "no data" rather than as "never rolled".
TOTALS = tuple(range(2, 13))

#: How many of the 36 ordered die pairs make each total. The reference a distribution is
#: read against — 24 rolls with four 7s means nothing until you know 7 is the common one.
WAYS = {total: 6 - abs(total - 7) for total in TOTALS}


def _named(amounts):
    """A per-resource list as ``{"wood": 3, ...}``, keyed the way the view keys a hand."""
    return {Resource(r).name.lower(): amounts[r] for r in range(NUM_RESOURCES)}


def owed(state, roll, blocked_tile):
    """What each player would collect on ``roll``, ignoring tiles under ``blocked_tile``.

    The same walk :func:`catan.rules.distribute` opens with, and deliberately so: passing
    ``state.robber_tile`` reproduces what the engine pays before the bank's shortage rule is
    applied, and passing ``None`` says what it would have paid with no robber on the board.
    The difference between the two is the robber's damage, which is the only reason this
    exists — see :meth:`GameStats.record`.

    Args:
        state: the position at the moment of the roll.
        roll: the total. A 7 produces nothing and yields all zeroes.
        blocked_tile: a tile id to skip, or ``None`` to count every tile.

    Returns:
        dict: ``{player: [amount per resource]}``.
    """
    collected = {player: [0] * NUM_RESOURCES for player in state.players}
    for vertex, productions in state.board.producers_for(roll).items():
        owner = state.vertex_owner[vertex]
        if owner == NO_OWNER:
            continue
        amount = PIECE_YIELD[state.vertex_piece[vertex]]
        for production in productions:
            if production.tile == blocked_tile:
                continue
            collected[owner][production.resource] += amount
    return collected


class GameStats:
    """A running tally, fed one step's events at a time.

    Lives on :class:`interfaces.web.api.Game` and is updated wherever the log is, so it sees
    exactly the events the player is told about and cannot fall behind them.
    """

    def __init__(self, players):
        self.players = tuple(players)
        zero = lambda: {player: 0 for player in self.players}          # noqa: E731
        hand = lambda: {player: [0] * NUM_RESOURCES for player in self.players}  # noqa: E731

        #: Rolls each player made, by total. ``state.roll_counts`` is the column sum, and
        #: ``test_the_dice_add_up_to_the_engines_own_count`` holds the two together.
        self.rolls = {player: [0] * 13 for player in self.players}
        #: Cards each player gave up to the hand limit, per resource.
        self.discarded = hand()
        #: How many separate 7s cost them cards — not how many cards. A player discarding
        #: four cards to one 7 has been caught out once, and the two numbers answer
        #: different questions.
        self.seven_outs = zero()
        #: Cards taken by this player with the robber, and taken *from* them.
        self.stole = hand()
        self.robbed = hand()
        #: Cards gained by playing Monopoly, and lost to somebody else's.
        self.monopolised = hand()
        self.monopolised_from = hand()
        #: Production the robber denied them — the figure the engine throws away.
        self.blocked = hand()
        #: How many times each player moved the robber, from a 7 or a Knight.
        self.robber_moves = zero()

        # Which players have already given up a card to the 7 currently being resolved.
        # Cleared when a new 7 is rolled rather than when the discards finish: one discard
        # is one action, so a four-card discard arrives as four separate steps, and during
        # them the turn can pass between players — the game is not alternating.
        self._discarding = set()

    def record(self, state, events):
        """Fold one step's events in.

        Args:
            state: the position **after** the step. Right for the one thing that needs a
                position rather than a count: within a step the engine applies the action
                first and only then rolls while nobody has a choice
                (:meth:`catan.env.CatanEnv._advance_to_decision`), so anything that moved
                the robber or put a building down did so *before* the roll being recorded,
                and the state as it stands is the state the roll paid out from.
            events: ``info["events"]``.
        """
        for event in events:
            kind = event.kind
            if kind is EventKind.ROLLED:
                self.rolls[event.player][event.amount] += 1
                if event.amount == 7:
                    self._discarding.clear()
                else:
                    self._add_blocked(state, event.amount)
            elif kind is EventKind.DISCARDED:
                self.discarded[event.player][event.resource] += 1
                if event.player not in self._discarding:
                    self._discarding.add(event.player)
                    self.seven_outs[event.player] += 1
            elif kind is EventKind.STOLE:
                self.stole[event.player][event.resource] += 1
                self.robbed[event.other][event.resource] += 1
            elif kind is EventKind.MONOPOLISED:
                self.monopolised[event.player][event.resource] += event.amount
                for other in self.players:
                    if other != event.player:
                        self.monopolised_from[other][event.resource] += event.amount
                # Two players, so the whole haul came from the one opponent. This is written
                # to survive a third seat rather than to serve one: the split between two
                # victims is not on the event, so with more players the loss side would have
                # to be attributed some other way. `Search` already refuses num_players != 2.
            elif kind is EventKind.ROBBER_MOVED:
                self.robber_moves[event.player] += 1

    def _add_blocked(self, state, roll):
        """Tally what the robber stopped on this roll, per player."""
        # This cannot fire in this engine, and is kept as a cheap assertion rather than as a
        # case that happens. The robber is **always** on a tile: `GameState.__init__` starts
        # it on `board.desert_tile`, every board carries exactly one desert
        # (`Board.TILE_COUNTS`) and `desert_tile` is an `index()` lookup that would raise if
        # it did not, and `rules.move_robber` only ever assigns a tile id. Nothing anywhere
        # assigns `None`.
        if state.robber_tile is None:
            return
        without = owed(state, roll, None)
        with_robber = owed(state, roll, state.robber_tile)
        for player in self.players:
            for resource in range(NUM_RESOURCES):
                self.blocked[player][resource] += (
                    without[player][resource] - with_robber[player][resource]
                )


# --------------------------------------------------------------------------- #
# Victory points, broken down                                                 #
# --------------------------------------------------------------------------- #

def victory_point_sources(state, player):
    """Where a player's points came from, as ``[{source, count, points}, ...]``.

    :func:`catan.rules.victory_points` sums the four components inline and returns one
    number; this is the same arithmetic with the terms kept apart, which is what "why am I
    on 11" needs. ``points`` totalling :func:`catan.rules.victory_points` is asserted by
    ``test_the_point_breakdown_adds_up_to_the_score``, so this cannot quietly disagree with
    the rules that decide the game.

    ⚠️ The Victory Point card row is hidden information while the game runs. :func:`report`
    is what decides whether to include it.
    """
    settlements = cities = 0
    for vertex in range(1, NUM_VERTICES + 1):
        if state.vertex_owner[vertex] != player:
            continue
        if state.vertex_piece[vertex] is Piece.CITY:
            cities += 1
        elif state.vertex_piece[vertex] is Piece.SETTLEMENT:
            settlements += 1

    return [
        {"source": "settlements", "count": settlements, "points": settlements},
        {"source": "cities", "count": cities, "points": cities * 2},
        {
            "source": "longest road",
            "count": rules.longest_road_length(state, player),
            "points": AWARD_VICTORY_POINTS if state.longest_road_holder == player else 0,
            "held": state.longest_road_holder == player,
        },
        {
            "source": "largest army",
            "count": state.knights_played[player],
            "points": AWARD_VICTORY_POINTS if state.largest_army_holder == player else 0,
            "held": state.largest_army_holder == player,
        },
        {
            "source": "victory point cards",
            "count": state.dev_cards[player][DevCard.VICTORY_POINT],
            "points": state.dev_cards[player][DevCard.VICTORY_POINT],
            "hidden": True,
        },
    ]


# --------------------------------------------------------------------------- #
# The payload                                                                 #
# --------------------------------------------------------------------------- #

def report(state, tally, names, reveal, you=None):
    """The whole panel, as JSON-ready data.

    Args:
        state: the current position.
        tally: the :class:`GameStats` that has been following the game.
        names: ``{player: label}``, so the panel says "You" and "Opponent" rather than P1.
        reveal: whether hidden information may be shown. The game being over, in practice —
            the same condition :func:`interfaces.web.api.view` reveals a hand on. Only the
            victory-point breakdown is affected: everything else here is public.
        you: the seat the panel is being shown to, or ``None`` when nobody is playing, which
            is the case in a watched game. This is where the panel is stricter than
            :func:`interfaces.web.api.view`, which keeps seat 1 as "you" even when both seats
            are agents — the condition is shared, the seat is not.
    """
    totals = [sum(tally.rolls[p][total] for p in tally.players) for total in range(13)]
    rolled = sum(totals)

    return {
        "turns": state.turn_number,
        "rolls": rolled,
        "dice": {
            "totals": {str(total): totals[total] for total in TOTALS},
            # What a fair 36-roll deck would have given over the same number of rolls, so a
            # bar can be read against something. Under the Balanced Dice ruleset this is not
            # an expectation but a promise: the deck of 36 is consumed rather than resampled.
            "expected": {
                str(total): round(rolled * WAYS[total] / 36, 1) for total in TOTALS
            },
            # Carried but not drawn: the panel plots ``expected``, which is this multiplied
            # through by the number of rolls. Kept because a reader asking "why is 7 the
            # tall one" is asking for this, and it costs eleven integers.
            "ways": {str(total): WAYS[total] for total in TOTALS},
            "balanced": state.ruleset.balanced_dice,
            "byPlayer": {
                str(player): {str(total): tally.rolls[player][total] for total in TOTALS}
                for player in tally.players
            },
        },
        "players": [
            _player_stats(state, tally, player, names, reveal, you)
            for player in state.players
        ],
    }


def _player_stats(state, tally, player, names, reveal, you):
    produced = list(state.produced[player])
    stolen = tally.stole[player]
    monopolised = tally.monopolised[player]
    gained = [
        produced[r] + stolen[r] + monopolised[r] for r in range(NUM_RESOURCES)
    ]
    lost = [
        tally.discarded[player][r] + tally.robbed[player][r]
        + tally.monopolised_from[player][r]
        for r in range(NUM_RESOURCES)
    ]

    sources = victory_point_sources(state, player)
    if not reveal and player != you:
        # A Victory Point card is held, never played, and invisible until it wins the game.
        # Dropping the row rather than zeroing it: a "0" here would be a claim, and the
        # honest statement is that this panel does not know.
        sources = [row for row in sources if not row.get("hidden")]

    return {
        "id": player,
        "name": names.get(player, f"P{player}"),
        "you": player == you,
        # --- the dice, from this player's side -------------------------------
        "rolled": sum(tally.rolls[player]),
        "sevensRolled": tally.rolls[player][7],
        # --- the robber ------------------------------------------------------
        "knights": state.knights_played[player],
        "robberMoves": tally.robber_moves[player],
        "sevenOuts": tally.seven_outs[player],
        "cardsDiscarded": sum(tally.discarded[player]),
        "discarded": _named(tally.discarded[player]),
        "blocked": _named(tally.blocked[player]),
        "blockedTotal": sum(tally.blocked[player]),
        "stolenBy": _named(stolen),
        "stolenFrom": _named(tally.robbed[player]),
        # --- cards over the whole game ---------------------------------------
        "produced": _named(produced),
        "producedTotal": sum(produced),
        "monopolised": _named(monopolised),
        # Every card in and every card out, the four of them carried but not currently drawn.
        # ``app.js`` shows most of the components — produced, discarded, stolen either way,
        # taken by monopoly — and never their sums; the loss side of a monopoly is not a key
        # at all and reaches the browser only inside ``lost``. The arithmetic is here rather
        # than in the browser because that is where the rest of it is, and
        # ``test_the_report_carries_what_the_panel_draws`` keeps the keys from disappearing
        # before anything renders them.
        "gained": _named(gained),
        "gainedTotal": sum(gained),
        "lost": _named(lost),
        "lostTotal": sum(lost),
        "spent": _named(state.spent[player]),
        "devBought": state.dev_bought[player],
        # --- where the points came from --------------------------------------
        "victoryPoints": sum(row["points"] for row in sources),
        "victoryPointsComplete": reveal or player == you,
        "pointSources": sources,
    }
