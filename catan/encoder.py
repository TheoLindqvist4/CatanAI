"""Turn a :class:`~catan.state.GameState` into a fixed-length observation vector.

Three properties matter, and each is a test rather than an intention:

**Fixed length.** :data:`SIZE` floats, always — same in every phase, every ruleset, and at
every player count. A network's input shape cannot depend on the situation.

**Perspective rotation.** ``encode(state, me)`` always puts *me* in player slot 0, with
opponents following in turn order. So one network plays every seat, and a position is
encoded identically no matter which player number happens to hold it.

**Hidden information stays hidden.** An observation contains only what that player may
actually see. Specifically it never contains:

* another player's **hand composition** — only its size, which is public because cards are
  countable
* another player's **development-card composition** — only how many they hold, and how many
  Knights they have played, both public
* the **development deck order**, or how many of each kind remain — only the count
* the **Balanced Dice deck** at all

``tests/test_encoder.py`` checks this by mutating hidden state and asserting the observation
does not move. That is a leak detector, not a documentation exercise.

Two entry points, one body
--------------------------
:func:`encode` returns a list of Python floats and is what every interface, test and PPO
call site uses. :func:`encode_into` writes the *same numbers* into a caller-owned
``array('f')`` from :func:`observation_buffer`, which a search can then hand to numpy as a
view rather than converting element by element.

That second door exists because of one measurement. A leaf evaluation encodes and then
converts, and the conversion was costing almost half as much as the encoding itself:
``np.fromiter`` over 2,503 Python floats is 42 us against 95 us for the encode, and it runs
once per simulation. Writing float32 in the first place makes the conversion
``np.frombuffer`` — 0.5 us — and the template reset a memcpy rather than a list copy. The
arithmetic is unchanged and happens in float64 either way; only the *store* is narrower, and
the consumer was casting to float32 regardless. ``tests/test_encoder.py`` asserts the two
doors agree bit for bit, which is what makes this a representation change rather than a
second encoder.

Layout
------
:data:`LAYOUT` maps a name to the ``slice`` it occupies, so a consumer can pull out the
per-tile, per-vertex or per-road block and reshape it — a graph or convolutional model wants
``(19, TILE_FEATURES)`` rather than a flat run. The blocks are:

    tiles         19 x 19   resource, number, production odds, robber
    vertices      54 x 27   owner, piece, harbour, buildability, per-resource production,
                            harbour nearness — and one retired slot, see :data:`PIP_POTENTIAL`
    roads         72 x  6   owner, buildability
    players        4 x 34   hands and holdings, masked for opponents, production rate
    affordability  4 x  4   my hand against each purchase, priced through my trade rates
    history        4 x 12   the public record: production, spending, purchases, idleness
    rolls              12   how often each total has come up, and how far in we are
    global             40   phase, last roll, bank, ruleset, turn bookkeeping, board scarcity

Every value is scaled into roughly ``[0, 1]``, using exact maxima where one exists (a
resource count cannot exceed the bank's 19) and a documented soft cap otherwise.
"""

import array

from catan import rules
from catan.board import GENERIC_HARBOUR
from catan.dev_cards import DECK_SIZE as DEV_DECK_SIZE
from catan.dev_cards import (
    AWARD_VICTORY_POINTS,
    DECK_COUNTS,
    ROAD_BUILDING_ROADS,
    DevCard,
)
from catan.resources import (
    BANK_PER_RESOURCE,
    BANK_RATE,
    GENERIC_HARBOUR_RATE,
    NUM_RESOURCES,
    PURCHASE_NAMES,
    PURCHASES,
    Resource,
    SPECIFIC_HARBOUR_RATE,
    total,
)
from catan.state import (
    MAX_CITIES,
    MAX_PLAYERS,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NO_OWNER,
    PIECE_VICTORY_POINTS,
    PIECE_YIELD,
    Phase,
    Piece,
)
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_TILES,
)

# --------------------------------------------------------------------------- #
# Scaling                                                                     #
# --------------------------------------------------------------------------- #

#: Every card of one resource, so a per-resource count cannot exceed it.
MAX_OF_ONE_RESOURCE = BANK_PER_RESOURCE
#: Every resource card in the game, so a hand total cannot exceed it.
MAX_CARDS = BANK_PER_RESOURCE * NUM_RESOURCES
#: The most Knights in the deck, so a knight count cannot exceed it.
MAX_KNIGHTS = DECK_COUNTS[DevCard.KNIGHT]

#: Turn counts are unbounded in principle; games run a few hundred turns, so this is the
#: soft cap beyond which the feature saturates.
TURN_SCALE = 400
#: Turns without building. Ten is already a long drought in a 1v1 game.
IDLE_SCALE = 20.0

#: Rolls run 2..12.
ROLLS = tuple(range(2, 13))

#: Harbour slots per vertex: none, generic, then one per resource.
HARBOUR_KINDS = 2 + NUM_RESOURCES


def _pips(number):
    """Ways to roll ``number`` with two dice, 0..6. 7 is the most likely."""
    return 6 - abs(7 - number)


def _odds(number):
    """Probability of ``number`` on two dice."""
    return _pips(number) / 36


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #

TILE_FEATURES = (
    (NUM_RESOURCES + 1)   # resource one-hot, plus desert
    + len(ROLLS)          # number one-hot
    + 1                   # production odds (0 for the desert)
    + 1                   # robber here
)

#: Harbour kinds a vertex can *reach*: generic, plus one per resource. One fewer than
#: :data:`HARBOUR_KINDS`, whose first slot means "there is no harbour here" — a thing a
#: distance cannot express.
HARBOUR_REACH_KINDS = HARBOUR_KINDS - 1

#: Roads beyond which a harbour is treated as unreachable. Six is most of the board; the
#: value is a scale, not a rule, so a distance of exactly six and one of nine are equally
#: "far" and the network is not asked to care.
HARBOUR_REACH_LIMIT = 6

VERTEX_FEATURES = (
    (MAX_PLAYERS + 1)     # owner: empty, then one slot per player
    + 1                   # is a city (a settlement is owner set and this clear)
    + HARBOUR_KINDS       # harbour one-hot
    + 1                   # retired: written 0.0, the slot kept. See PIP_POTENTIAL
    + 1                   # satisfies the distance rule
    + 1                   # reachable from my road network
    # ---- appended; everything above keeps the offset it had ----------------- #
    + NUM_RESOURCES       # expected cards a roll of *each* resource, from this corner
    + HARBOUR_REACH_KINDS  # how near the nearest harbour of each kind is, 1 = adjacent
)

ROAD_FEATURES = (
    (MAX_PLAYERS + 1)     # owner
    + 1                   # I could build here, cost aside
)

PLAYER_FEATURES = (
    1                     # in the game
    + 1                   # is me
    + NUM_RESOURCES       # hand composition   (me only; zeros for opponents)
    + 1                   # hand size          (public)
    + len(DevCard)        # dev-card composition (me only)
    + 1                   # dev cards held     (public)
    + 1                   # knights played     (public)
    + 1                   # public victory points
    + 1                   # true victory points (me only)
    + 1                   # holds Largest Army
    + 1                   # holds Longest Road
    + 1                   # longest road length
    + 3                   # settlements, cities, roads left
    + NUM_RESOURCES       # bank trade rates
    + 1                   # discards still owed
    # ---- appended; everything above keeps the offset it had ----------------- #
    + NUM_RESOURCES       # expected cards a roll of each resource, from their buildings
)

#: Where each field sits inside one vertex row, by name.
#:
#: Written down once, here, because three places used to count to it: the encoder, the static
#: template and the tests — and the tests counted *backwards from the end*, which was correct
#: until features were appended and then silently wrong. A name cannot drift the way an
#: arithmetic expression can.
VERTEX_OFFSETS = {
    "owner": 0,
    "city": MAX_PLAYERS + 1,
    "harbour": MAX_PLAYERS + 2,
    "pip_potential": MAX_PLAYERS + 2 + HARBOUR_KINDS,
    "buildable": MAX_PLAYERS + 3 + HARBOUR_KINDS,
    "my_road": MAX_PLAYERS + 4 + HARBOUR_KINDS,
    "production": MAX_PLAYERS + 5 + HARBOUR_KINDS,
    "harbour_reach": MAX_PLAYERS + 5 + HARBOUR_KINDS + NUM_RESOURCES,
}

#: Whether the per-vertex ``pip_potential`` slot carries its value or a constant 0.
#:
#: **The slot stays either way.** Deleting the field would move every offset after it inside
#: a vertex row and change :data:`SIZE`. The vertices block would have *shrunk*, and
#: ``layouts.column_map`` reconciles only blocks that grew — it raises rather than invent a
#: column correspondence that does not exist, so no existing checkpoint could be grafted onto
#: the result. ``layouts.HISTORICAL`` is not what stands in the way, whatever this comment
#: used to say: it is keyed 1868 and 1884 only, and a 2503 checkpoint carries its own layout,
#: so the table is never consulted for one. Writing 0.0 removes the *information* at no
#: compatibility cost and leaves ``SIZE`` alone, so ``models/champion_az.pt`` keeps loading —
#: which matters, because with no loadable champion the AlphaZero gate has nothing to measure
#: against and ``training.alphazero.champion`` refuses to promote at all.
#:
#: **Reversing it is not one line.** ``test_pip_potential_is_retired_but_its_slot_is_still_there``
#: asserts the flag is off, so flipping it here fails a test. The arithmetic is kept and
#: tested, so *trying* it costs nothing; shipping it flipped is a two-file change, which is
#: the point.
#:
#: **Why it is off.** It is the summed odds of the adjacent tiles with the resources thrown
#: away, so three sheep and an even three-way spread are the same number. It was the only
#: placement signal the observation had, and the agent maximised it essentially perfectly —
#: 0.005 pips off the best available spot. Record 0024 added the per-resource production that
#: supersedes it. That more pips *lose* is not established and is not the argument: the bands
#: hold 39 to 77 games each, they are not monotone, and their intervals overlap — record 0024
#: says of its own version of that table "at 240 games that did not hold". Resource diversity
#: is the result that held. See ``docs/decisions/0029-retiring-pip-potential.md`` for the
#: tables and the argument; this retires the guide that came first rather than leaving it to
#: be unlearned.
PIP_POTENTIAL = False

#: Where each appended field sits inside one player row.
PLAYER_OFFSETS = {"production": PLAYER_FEATURES - NUM_RESOURCES}

#: Divides the per-resource production rate into roughly [0, 1]. A player producing one card
#: of a resource every three rolls is doing very well, so 0.33 is the top of the useful range.
PRODUCTION_RATE_SCALE = 1 / 3

#: Per purchase in :data:`catan.resources.PURCHASES`, how far my hand is from paying for it.
#:
#: The cost *table* is deliberately not encoded: the four costs are identical in every state
#: a network will ever see, so a constant input carries no information and folds into a bias
#: in one gradient step. What decides a turn is the state-dependent difference between the
#: hand and the price — and, since a deficit is closed at the bank, that difference priced
#: through my own harbours. One brick short holding six wood is two cards away at a 2:1 wood
#: harbour, three at a 3:1, four at 4:1, and out of reach on three wood.
#:
#: Affordability, not legality: an empty development deck, a spent city piece and a vertex
#: with nowhere legal to build do not move these rows. Those facts are already in the global
#: block, the player block and the per-vertex flags respectively, and folding them in here
#: would make one float mean three things. ``_encode_roads`` reports reachability "cost
#: aside" for the same reason; this block is the mirror, and the two compose.
AFFORDABILITY_FEATURES = (
    1                     # affordable now (a Road Building credit counts as paid)
    + 1                   # cards short, as a fraction of what this purchase costs
    + 1                   # cheapest cards given to the bank to close the gap; 1.0 out of reach
    + 1                   # the gap can be closed now, by trades the bank can honour
)

#: Cards handed to the bank to close a gap. An exact maximum exists — five trades at
#: ``BANK_RATE`` is 20 — but it is unreachable in play: over 63,372 measured
#: ``(player, purchase)`` samples, only 0.02% of coverable gaps cost more than two 4:1
#: trades and none cost 20, so dividing by the exact maximum would leave every live value in
#: the bottom fifth of the range. A soft cap, therefore, clipped — and out of reach saturates
#: at the same 1.0, so the column is monotone "cards this would cost me" throughout. The
#: collision at the top is separated by the coverable flag in the same row.
TRADE_PRICE_SCALE = float(2 * BANK_RATE)

#: Exact maximum of the cards-short column: every card of the cost is missing.
_PURCHASE_TOTALS = tuple(float(sum(cost)) for cost in PURCHASES)

#: The one purchase a development card can make free, so the credit is folded in.
ROAD_ROW = PURCHASE_NAMES.index("road")

#: Per player, in the public record: cumulative production and cumulative spending, both
#: per resource, plus development cards bought and how long since they last built.
HISTORY_FEATURES = NUM_RESOURCES + NUM_RESOURCES + 1 + 1

#: Rolls 2..12, plus how far into the game the histogram is.
ROLL_HISTORY_FEATURES = len(ROLLS) + 1

#: Everything ever produced by one player is bounded well below this; used only to keep the
#: value in a sane range rather than as a hard cap, so it is clipped.
PRODUCTION_SCALE = 120.0

GLOBAL_FEATURES = (
    len(Phase)            # phase one-hot
    + len(ROLLS) + 1      # last roll one-hot, plus "not rolled yet"
    + 1                   # turn number
    + NUM_RESOURCES       # the bank
    + 1                   # development cards left in the deck
    + 1                   # free roads owed
    + 1                   # a development card was played this turn
    + 1                   # the dice have been rolled this turn
    + 1                   # it is my decision right now
    + 1                   # players in the game
    + 4                   # ruleset: win target, hand limit, friendly robber, balanced dice
    # ---- appended, and board-static: written by the template, not by _encode_global,
    # which writes forwards from the start of the block and stops before here ---------- #
    + NUM_RESOURCES       # what the whole board makes of each resource — scarcity
)


def _build_layout():
    spans, offset = {}, 0
    for name, width in (
        ("tiles", NUM_TILES * TILE_FEATURES),
        ("vertices", NUM_VERTICES * VERTEX_FEATURES),
        ("roads", NUM_ROADS * ROAD_FEATURES),
        ("players", MAX_PLAYERS * PLAYER_FEATURES),
        ("affordability", len(PURCHASES) * AFFORDABILITY_FEATURES),
        ("history", MAX_PLAYERS * HISTORY_FEATURES),
        ("rolls", ROLL_HISTORY_FEATURES),
        ("global", GLOBAL_FEATURES),
    ):
        spans[name] = slice(offset, offset + width)
        offset += width
    return spans, offset


LAYOUT, SIZE = _build_layout()

#: Rows and columns of each repeated block, for reshaping.
SHAPES = {
    "tiles": (NUM_TILES, TILE_FEATURES),
    "vertices": (NUM_VERTICES, VERTEX_FEATURES),
    "roads": (NUM_ROADS, ROAD_FEATURES),
    "players": (MAX_PLAYERS, PLAYER_FEATURES),
    "affordability": (len(PURCHASES), AFFORDABILITY_FEATURES),
    "history": (MAX_PLAYERS, HISTORY_FEATURES),
}


# --------------------------------------------------------------------------- #
# Perspective                                                                 #
# --------------------------------------------------------------------------- #

def player_slots(state, me):
    """``{player: slot}`` with ``me`` at 0 and the others in turn order after.

    Rotating rather than using raw player numbers is what lets one network play every
    seat: the same position always encodes the same way.
    """
    order = state.player_order
    start = order.index(me)
    rotated = order[start:] + order[:start]
    return {player: slot for slot, player in enumerate(rotated)}


# --------------------------------------------------------------------------- #
# The board-static half of an observation                                     #
# --------------------------------------------------------------------------- #

def _static_template(board):
    """The parts of an observation that depend on the *layout* and never on play.

    Which resource sits on a tile, its number token, its odds, which harbours a vertex can
    reach and how far off they are, and what each corner expects to produce of each resource
    are all fixed the moment the board is generated.
    Recomputing them on every encode is most of what encoding costs — profiling a training
    rollout put ``_encode_vertices`` at 45% of the total, nearly all of it in one generator
    expression summing three tiles' odds for a board that had not changed in 14,000 calls.

    So it is computed once per :class:`~catan.board.Board` and cached on it. The board is
    immutable and shared across clones, so one template serves an entire training run.
    """
    template = board.__dict__.get("_observation_template")
    if template is not None:
        return template

    out = [0.0] * SIZE

    base = LAYOUT["tiles"].start
    for tile in range(1, NUM_TILES + 1):
        at = base + (tile - 1) * TILE_FEATURES
        resource = board.resource_at(tile)
        number = board.number_at(tile)

        out[at + (NUM_RESOURCES if resource is None else int(resource))] = 1.0
        at += NUM_RESOURCES + 1
        out[at + ROLLS.index(number)] = 1.0
        at += len(ROLLS)
        # a desert never pays out, whatever its token says
        out[at] = 0.0 if resource is None else _odds(number)
        # at + 1 is the robber flag, which moves — left at 0 for the caller

    distances = board.harbour_distances()
    base = LAYOUT["vertices"].start
    for vertex in range(1, NUM_VERTICES + 1):
        # owner one-hot and the city flag are play, not layout; skip to the harbours
        at = base + (vertex - 1) * VERTEX_FEATURES + (MAX_PLAYERS + 1) + 1

        harbours = board.harbours_at(vertex)
        if not harbours:
            out[at] = 1.0
        else:
            for harbour in harbours:
                out[at + (1 if harbour is GENERIC_HARBOUR else 2 + int(harbour))] = 1.0
        at += HARBOUR_KINDS

        # The classic settlement heuristic, retired — this slot is written 0.0. See
        # PIP_POTENTIAL. The production tuple is read either way: the per-resource split
        # below is what replaced it.
        # The 0.0 start to sum() no longer does anything, and is kept only so the retired
        # branch stays byte-identical to what it computed before. It guarded an earlier
        # version that summed a *generator* over the adjacent tiles, where a corner touching
        # only the desert summed an empty one and bare sum() returned int 0 into a float
        # vector. expected_production now returns a 5-tuple of floats at every vertex, so
        # that sum is 0.0 with or without the start.
        production = board.expected_production(vertex)
        out[at] = sum(production, 0.0) if PIP_POTENTIAL else 0.0
        at += 1

        # Past the two buildability flags, which are play and are written per encode.
        at += 2

        # The same production, *split by resource*. The sum above cannot tell three sheep
        # from an even spread across three resources, and choosing between those is most of
        # what opening placement is. See docs/decisions/0024-what-a-placement-can-see.md.
        for resource in range(NUM_RESOURCES):
            out[at + resource] = production[resource]
        at += NUM_RESOURCES

        # How near the nearest harbour of each kind is, as 1 for "here" falling to 0 at
        # HARBOUR_REACH_LIMIT roads away. A vertex already knew whether it *was* a harbour;
        # it did not know one was two roads off, which is the form the information is
        # actually used in.
        for kind, steps in enumerate(distances[vertex]):
            if steps is not None:
                out[at + kind] = max(0.0, 1.0 - steps / HARBOUR_REACH_LIMIT)

    # What this board makes of each resource in total. Balanced generation fixes the
    # resource counts but not which numbers they land on, so this varies board to board and
    # is what "ore is scarce here" means.
    at = LAYOUT["global"].stop - NUM_RESOURCES
    for resource, rate in enumerate(board.resource_scarcity()):
        out[at + resource] = rate

    board.__dict__["_observation_template"] = out
    return out


def _static_template_array(board):
    """The same template as an ``array('f')``, so resetting a buffer is a memcpy.

    Cached beside the list one rather than derived from it per call: the conversion is the
    very cost :func:`encode_into` exists to avoid, and a board outlives a whole game.
    """
    cached = board.__dict__.get("_observation_template_f")
    if cached is None:
        cached = array.array("f", _static_template(board))
        board.__dict__["_observation_template_f"] = cached
    return cached


# --------------------------------------------------------------------------- #
# Encoding                                                                    #
# --------------------------------------------------------------------------- #

def _observer(state, me):
    if me is None:
        me = state.current_player
    if me not in state.players:
        raise ValueError(f"player must be in 1..{state.num_players}, got {me}")
    return me


def encode(state, me=None):
    """The observation ``me`` is entitled to, as a list of :data:`SIZE` floats.

    Args:
        state: the game.
        me: whose view to build. Defaults to ``state.current_player`` — but pass it
            explicitly in search, because during a discard the current player may be an
            opponent.
    """
    me = _observer(state, me)
    return _fill(state, me, _static_template(state.board).copy())


def observation_buffer():
    """A reusable float32 buffer for :func:`encode_into`.

    ``array`` rather than a list because the caller wants bytes: a search converts every
    leaf observation to a numpy row, and from an ``array('f')`` that is ``np.frombuffer`` —
    a cast of a pointer — where from a list it is a 2,503-element unboxing loop.
    """
    return array.array("f", bytes(SIZE * 4))


def encode_into(state, me, out):
    """:func:`encode`'s numbers, written into ``out`` from :func:`observation_buffer`.

    Returns ``out``, so a caller can wrap the call in ``np.frombuffer``.

    ⚠️ ``out`` is **overwritten**, and a caller that hands the same buffer to two searches
    gets one observation twice. Give each search its own; they are 10 KB.
    """
    me = _observer(state, me)
    out[:] = _static_template_array(state.board)
    return _fill(state, me, out)


def _fill(state, me, out):
    """Everything that is play rather than layout, written over a reset template.

    Shared by both entry points so the list and the buffer cannot drift apart —
    ``test_encode_into_matches_the_list_encoding_bit_for_bit`` is what proves they do not.
    """
    slots = player_slots(state, me)
    # One walk over the board, rather than the nine that four rules functions cost between
    # them. See _survey.
    points, production, rates_by_player = _survey(state)
    # Hoisted because two blocks want it: the player block reports it as a feature, and the
    # affordability block prices deficits through it. `rates` is always *me*'s.
    rates = rates_by_player[me]

    _encode_tiles(state, out)
    _encode_board(state, out, me, slots)
    _encode_players(state, out, me, slots, points, production, rates_by_player)
    _encode_affordability(state, out, me, rates)
    _encode_history(state, out, me, slots)
    _encode_global(state, out, me)
    return out


def _survey(state):
    """Victory points, production and trade rates for every player, in one walk.

    Returns ``({player: points}, {player: [rate per resource]}, {player: [give per
    resource]})``, where ``points`` excludes hidden Victory Point cards — it is
    :func:`catan.rules.public_victory_points`.

    **This is a second implementation of scoring, and that is normally forbidden here.** The
    justification is arithmetic: :func:`catan.rules.victory_points`,
    :func:`~catan.rules.public_victory_points`, :func:`~catan.rules.production_rates` and
    :func:`~catan.rules.trade_rates` each walk all 54 vertices, and one encode called them
    nine times between them for two players — nine passes to answer three questions about
    the same buildings. The block measured 43-45% of an encode, and an encode runs once per
    simulation.

    What makes it legal rather than a rule waiting to diverge is
    ``test_fused_player_scores_agree_with_the_rules``, which drives whole games at two, three
    and four players and asserts every value here equals the rules' own at *every* position.
    :mod:`catan.rules` remains the authority; this is a faster route to the same answer, and
    the test is what says so.
    """
    from catan.resources import NUM_RESOURCES

    owners = state.vertex_owner
    pieces = state.vertex_piece
    board = state.board
    harbours_at = board.harbours_at
    expected = board.expected_production

    points = {player: 0 for player in state.players}
    production = {player: [0.0] * NUM_RESOURCES for player in state.players}
    rates = {player: [BANK_RATE] * NUM_RESOURCES for player in state.players}

    for vertex in range(1, NUM_VERTICES + 1):
        owner = owners[vertex]
        if owner == NO_OWNER:
            continue
        piece = pieces[vertex]

        # Before the zero-yield short-circuit below: a piece worth points but yielding
        # nothing is not a case today, and silently losing its points the day it is would be
        # a scoring bug nothing raises on.
        points[owner] += PIECE_VICTORY_POINTS[piece]

        for harbour in harbours_at(vertex):
            if harbour is GENERIC_HARBOUR:
                mine = rates[owner]
                for resource in range(NUM_RESOURCES):
                    if mine[resource] > GENERIC_HARBOUR_RATE:
                        mine[resource] = GENERIC_HARBOUR_RATE
            elif rates[owner][harbour] > SPECIFIC_HARBOUR_RATE:
                rates[owner][harbour] = SPECIFIC_HARBOUR_RATE

        yields = PIECE_YIELD[piece]
        if not yields:
            continue
        mine = production[owner]
        for resource, rate in enumerate(expected(vertex)):
            mine[resource] += rate * yields

    # `None` is the "nobody holds it" sentinel, not player 0 — indexing with it raises.
    for holder in (state.largest_army_holder, state.longest_road_holder):
        if holder in points:
            points[holder] += AWARD_VICTORY_POINTS
    return points, production, rates


def _encode_tiles(state, out):
    """Only the robber moves; everything else about a tile is in the static template."""
    at = LAYOUT["tiles"].start + (state.robber_tile - 1) * TILE_FEATURES
    out[at + NUM_RESOURCES + 1 + len(ROLLS) + 1] = 1.0


def _encode_board(state, out, me, slots):
    """The vertices block and the roads block, which want the same walk.

    Harbours, harbour nearness, the per-resource production and the retired pip slot all come
    from the template; what is written here is ownership and the three buildability flags.

    The flags are derived from what is *owned* rather than by asking
    :func:`catan.rules.respects_distance_rule` and :func:`catan.rules.touches_own_road` per
    vertex. Those were 108 calls per encode and 38% of what encoding cost after the static
    template; there are at most ten settlements and fifteen roads to walk instead of
    fifty-four vertices to interrogate.

    Vertices and roads are one function because the two blocks were computing the same thing
    twice: which vertices my roads touch decides both a vertex's "reachable from my road
    network" flag and a road's "I could build here". The roads block walked all 72 roads to
    rebuild a set the vertex loop had already built, and then took a third pass over the 54
    vertices to turn it into reachability. Now the vertex loop, which is running anyway,
    accumulates it.

    The rules remain the authority — ``test_buildability_flags_agree_with_the_rules``
    cross-checks against them, which is what keeps this shortcut honest.
    """
    owners = state.vertex_owner
    pieces = state.vertex_piece
    edges = state.edge_owner

    blocked = set()
    for vertex in range(1, NUM_VERTICES + 1):
        if owners[vertex] != NO_OWNER:
            blocked.add(vertex)
            blocked.update(VERTEX_NEIGHBOURS[vertex])

    my_junctions = set()
    for road in range(1, NUM_ROADS + 1):
        if edges[road] == me:
            my_junctions.update(ROAD_VERTICES[road])

    # offset of the two buildability flags within a vertex block, past the harbour
    # one-hot and the retired pip slot, which is still there and still one wide
    flags = (MAX_PLAYERS + 1) + 1 + HARBOUR_KINDS + 1
    base = LAYOUT["vertices"].start
    # Vertices I can build outward from: mine, or empty and met by one of my roads — the
    # same rule as :func:`catan.rules.is_road_connected`, for the whole board at once.
    reachable = set()

    for vertex in range(1, NUM_VERTICES + 1):
        at = base + (vertex - 1) * VERTEX_FEATURES
        owner = owners[vertex]

        out[at + (0 if owner == NO_OWNER else 1 + slots[owner])] = 1.0
        if pieces[vertex] is Piece.CITY:
            out[at + MAX_PLAYERS + 1] = 1.0

        at += flags
        if vertex not in blocked:
            out[at] = 1.0
        if vertex in my_junctions:
            out[at + 1] = 1.0

        if owner == me or (owner == NO_OWNER and vertex in my_junctions):
            reachable.add(vertex)

    base = LAYOUT["roads"].start
    for road in range(1, NUM_ROADS + 1):
        at = base + (road - 1) * ROAD_FEATURES
        owner = edges[road]
        out[at + (0 if owner == NO_OWNER else 1 + slots[owner])] = 1.0
        # Cost aside, so the feature says "reachable" rather than "affordable right now" —
        # affordability is already visible from my hand.
        #
        # Only the 1.0 is written. This block of the static template is all zeros and both
        # entry points reset from it, so an explicit `else 0.0` would be 64 stores a call to
        # write a zero over a zero. That coupling is invisible from here, which is why it is
        # stated: if the roads block ever acquires a non-zero template value, this needs an
        # else again.
        if owner == NO_OWNER and not reachable.isdisjoint(ROAD_VERTICES[road]):
            out[at + MAX_PLAYERS + 1] = 1.0


def _encode_players(state, out, me, slots, points, production, rates_by_player):
    """One row per player, from the survey rather than from nine walks over the board.

    ``points`` is public victory points; my hidden Victory Point cards are added back for my
    own row alone, which is the whole reason the two totals are different numbers.
    """
    base = LAYOUT["players"].start
    for player, slot in slots.items():
        at = base + slot * PLAYER_FEATURES
        mine = player == me
        hand = state.hands[player]
        held = state.dev_cards[player]

        out[at] = 1.0
        at += 1
        out[at] = 1.0 if mine else 0.0
        at += 1

        # composition is mine alone; the size is public because cards are countable
        if mine:
            for resource in range(NUM_RESOURCES):
                out[at + resource] = hand[resource] / MAX_OF_ONE_RESOURCE
        at += NUM_RESOURCES
        out[at] = total(hand) / MAX_CARDS
        at += 1

        if mine:
            for card in range(len(DevCard)):
                out[at + card] = held[card] / DEV_DECK_SIZE
        at += len(DevCard)
        out[at] = sum(held) / DEV_DECK_SIZE
        at += 1

        out[at] = state.knights_played[player] / MAX_KNIGHTS
        at += 1

        target = state.ruleset.victory_points_to_win
        public = points[player]
        out[at] = min(public / target, 1.0)
        at += 1
        # only I know my hidden Victory Point cards, so only I see my true total
        if mine:
            out[at] = min((public + held[DevCard.VICTORY_POINT]) / target, 1.0)
        at += 1

        out[at] = 1.0 if state.largest_army_holder == player else 0.0
        at += 1
        out[at] = 1.0 if state.longest_road_holder == player else 0.0
        at += 1
        out[at] = rules.longest_road_length(state, player) / MAX_ROADS
        at += 1

        out[at] = state.settlements_left[player] / MAX_SETTLEMENTS
        out[at + 1] = state.cities_left[player] / MAX_CITIES
        out[at + 2] = state.roads_left[player] / MAX_ROADS
        at += 3

        for resource, rate in enumerate(rates_by_player[player]):
            out[at + resource] = rate / BANK_RATE
        at += NUM_RESOURCES

        out[at] = state.discards_owed[player] / MAX_OF_ONE_RESOURCE
        at += 1

        # What this player's board *makes*, per resource, per roll. Public: buildings and
        # number tokens are both on the table, and it is what a person means by "they have
        # no brick". Encoded for opponents too, for the same reason.
        for resource, rate in enumerate(production[player]):
            out[at + resource] = min(rate / PRODUCTION_RATE_SCALE, 1.0)


def _encode_affordability(state, out, me, rates):
    """How far my hand is from each purchase, priced through my own bank rates.

    See :data:`AFFORDABILITY_FEATURES` for why the cost table itself is not encoded.

    Two of the four columns are conjunctions the first layer cannot form. A per-resource
    shortfall is a hinge on one input coordinate at an integer threshold, and the first
    ``Linear`` produces those for free from the hand composition — but "every one of four
    resources is covered" is not a threshold on any sum (four wood passes every sum test and
    affords nothing), and neither is ``sum(surplus[g] // rates[g]) >= need``, which divides by
    a rate that moves with harbour ownership. The other two columns are the graded distances
    underneath, which is what the critic needs: masking touches only the policy logits, so the
    value head never sees legality at all and would otherwise infer progress from five raw
    card counts.

    **Mine alone.** Every value is a function of ``me``'s hand. Filling it from
    ``state.current_player`` would hand over an opponent's exact hand composition on every
    discard, since the discard decision belongs to whoever is over the limit.

    ⚠️ Encoding from inside :func:`catan.rules.can_play_road_building`'s probe window would
    read inflated Road Building credit — it temporarily increments ``state.free_roads`` and
    restores it in a ``finally``. Nothing in the current call graph does.
    """
    base = LAYOUT["affordability"].start
    hand = state.hands[me]
    bank = state.bank
    # Credit is spent in preference to cards and cannot be declined, so while it is
    # outstanding a road genuinely costs nothing. The counter is global and lapses at
    # END_TURN, so it is only mine while I am the one acting.
    free_road = state.free_roads > 0 and state.current_player == me

    for row, cost in enumerate(PURCHASES):
        at = base + row * AFFORDABILITY_FEATURES

        need, stocked = 0, True
        if not (free_road and row == ROAD_ROW):
            for resource in range(NUM_RESOURCES):
                short = cost[resource] - hand[resource]
                if short > 0:
                    need += short
                    # A trade returns exactly one card, and paying refills the bank's pile
                    # for what I *gave*, never for what I take. So covering a deficit of
                    # `short` needs `short` of them left, not merely one.
                    if bank[resource] < short:
                        stocked = False

        if need == 0:
            out[at] = 1.0          # affordable now
            out[at + 3] = 1.0      # and so trivially coverable
            continue               # cards short and price stay at the template's 0.0

        # Each trade is paid with `rates[give]` cards of a *single* resource held at once, so
        # fundable trades are floored per resource. Pooling the surplus first is the natural
        # bug and is wrong: three wood and three sheep at 4:1 fund nothing. Surplus is
        # measured *above the cost*, which protects the cards this purchase needs and, since
        # surplus in `give` implies no shortfall in `give`, makes the engine's give != take
        # rule impossible to violate rather than separately enforced. That same filter keeps a
        # negative surplus out of the floor division, where -1 // 4 is -1 rather than 0.
        offers = sorted(
            (rates[give], (hand[give] - cost[give]) // rates[give])
            for give in range(NUM_RESOURCES)
            if hand[give] - cost[give] >= rates[give]
        )
        # Cheapest rate first is provably the minimum bill: every trade yields exactly one
        # card whatever it cost, so covering a fixed number of cards means buying the
        # cheapest units. `test_the_cheapest_plan_is_the_cheapest_one` checks it exhaustively
        # rather than taking the argument on trust.
        price, left = 0, need
        for rate, fundable in offers:
            taken = fundable if fundable < left else left
            price += taken * rate
            left -= taken
            if not left:
                break

        coverable = left == 0 and stocked
        out[at + 1] = need / _PURCHASE_TOTALS[row]
        out[at + 2] = min(price / TRADE_PRICE_SCALE, 1.0) if coverable else 1.0
        out[at + 3] = 1.0 if coverable else 0.0


def _encode_history(state, out, me, slots):
    """The public record: what everyone at the table has watched happen.

    Without this the observation is a pure snapshot, and a snapshot cannot express "they
    have hoarded ore for three turns, cities are coming". Every number here is one any
    player could have written down while watching — see the note on ``GameState``'s public
    record for why there is no opponent hand estimate among them.

    Rotated into slot order like the player block, so one network still plays every seat.
    """
    base = LAYOUT["history"].start
    for player, slot in slots.items():
        at = base + slot * HISTORY_FEATURES

        produced = state.produced[player]
        spent = state.spent[player]
        for resource in range(NUM_RESOURCES):
            out[at + resource] = min(produced[resource] / PRODUCTION_SCALE, 1.0)
            out[at + NUM_RESOURCES + resource] = min(spent[resource] / PRODUCTION_SCALE, 1.0)
        at += 2 * NUM_RESOURCES

        out[at] = min(state.dev_bought[player] / DEV_DECK_SIZE, 1.0)
        at += 1
        # how long they have gone without putting anything down
        idle = state.turn_number - state.last_build_turn[player]
        out[at] = min(idle / IDLE_SCALE, 1.0)

    at = LAYOUT["rolls"].start
    total = sum(state.roll_counts)
    for position, roll in enumerate(ROLLS):
        out[at + position] = state.roll_counts[roll] / total if total else 0.0
    out[at + len(ROLLS)] = min(total / TURN_SCALE, 1.0)


def _encode_global(state, out, me):
    at = LAYOUT["global"].start

    out[at + int(state.phase)] = 1.0
    at += len(Phase)

    if state.last_roll is None:
        out[at + len(ROLLS)] = 1.0
    else:
        out[at + ROLLS.index(state.last_roll)] = 1.0
    at += len(ROLLS) + 1

    out[at] = min(state.turn_number / TURN_SCALE, 1.0)
    at += 1

    for resource in range(NUM_RESOURCES):
        out[at + resource] = state.bank[resource] / MAX_OF_ONE_RESOURCE
    at += NUM_RESOURCES

    # how many cards are left is public; which cards they are is not
    out[at] = len(state.dev_deck) / DEV_DECK_SIZE
    at += 1
    out[at] = state.free_roads / ROAD_BUILDING_ROADS
    at += 1
    out[at] = 1.0 if state.dev_card_played_this_turn else 0.0
    at += 1
    out[at] = 1.0 if state.rolled_this_turn else 0.0
    at += 1
    # during a discard the decision may belong to an opponent
    out[at] = 1.0 if state.current_player == me else 0.0
    at += 1
    out[at] = state.num_players / MAX_PLAYERS
    at += 1

    ruleset = state.ruleset
    out[at] = ruleset.victory_points_to_win / 20
    out[at + 1] = ruleset.hand_limit / 20
    out[at + 2] = 1.0 if ruleset.friendly_robber else 0.0
    out[at + 3] = 1.0 if ruleset.balanced_dice else 0.0


# --------------------------------------------------------------------------- #
# Introspection                                                               #
# --------------------------------------------------------------------------- #

def block(observation, name):
    """One named block of an observation, as a list of rows.

    ``block(obs, "tiles")[3]`` is tile 4's features.
    """
    values = observation[LAYOUT[name]]
    if name not in SHAPES:
        return values
    rows, width = SHAPES[name]
    return [values[i * width:(i + 1) * width] for i in range(rows)]


def _validate():
    """Layout consistency, at import."""
    order = ("tiles", "vertices", "roads", "players", "affordability",
             "history", "rolls", "global")
    # Without this, a block can be added to `_build_layout` and forgotten here, and the
    # start/covered check below catches it only indirectly — or, if it is appended last,
    # not at all.
    assert tuple(LAYOUT) == order, "a block was added to the layout but not to this list"
    assert LAYOUT["global"].stop == SIZE
    covered = 0
    for name in order:
        assert LAYOUT[name].start == covered, f"{name} does not follow the previous block"
        covered = LAYOUT[name].stop
    assert covered == SIZE
    for name, (rows, width) in SHAPES.items():
        assert LAYOUT[name].stop - LAYOUT[name].start == rows * width


_validate()
