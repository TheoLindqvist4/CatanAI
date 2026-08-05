"""Play or watch a game in the terminal.

    python -m interfaces.cli                        you vs the strongest AI
    python -m interfaces.cli --agents human easy     an easier opponent
    python -m interfaces.cli --agents hard greedy    watch two agents
    python -m interfaces.cli --games 20 --quiet      benchmark, results only
    python -m interfaces.cli --render out/           also write a PNG each turn

**This is the only module allowed to call ``print`` or ``input``.** Everything under
``catan/`` is I/O-free so it can be driven by a training loop; the moment a rule needs to
ask a question, it stops being usable at scale. See
``docs/decisions/0003-io-free-core-and-injected-randomness.md``.

The board is drawn from :mod:`catan.topology`'s lattice, the same coordinates
:mod:`interfaces.render` uses for PNGs — a character grid instead of pixels. Vertices land
on their own rows and columns, roads occupy the gaps between them, and tile labels sit in
the middle of their hexagon, so nothing has to be positioned by hand.
"""

import argparse
import pathlib
import sys

from catan import action_space, rules
from catan.actions import ActionType
from catan.agents import DIFFICULTY, GreedyAgent, HeuristicAgent, RandomAgent
from catan.dev_cards import DevCard
from catan.env import CatanEnv
from catan.resources import Resource
from catan.rulesets import BASE_GAME, RANKED_1V1
from catan.state import NO_OWNER, Phase, Piece
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    TILE_XY,
    VERTEX_XY,
)

# --------------------------------------------------------------------------- #
# Board drawing                                                               #
# --------------------------------------------------------------------------- #

#: Character columns per lattice x unit, and rows per lattice y unit. Chosen so a slanted
#: road has a row of its own between the vertex rows, and space for two characters.
COLS_PER_X = 3
ROWS_PER_Y = 2

RESOURCE_CODES = {
    Resource.WOOD: "Wd",
    Resource.BRICK: "Br",
    Resource.SHEEP: "Sh",
    Resource.WHEAT: "Wh",
    Resource.ORE: "Or",
    None: "Ds",
}

LEGEND = (
    "digits = settlements, ABCD = cities, |/\\ = roads, (R) = robber, "
    "+ = empty spot, . = you could build"
)

#: Settlements print as the player's digit, cities as a letter, so they are distinguishable
#: without relying on colour.
CITY_LETTERS = "ABCD"

ANSI = {
    1: "\033[91m",   # red
    2: "\033[94m",   # blue
    3: "\033[93m",   # orange-ish
    4: "\033[92m",   # green
}
RESET = "\033[0m"
DIM = "\033[2m"


class _Grid:
    """A character canvas addressed in lattice coordinates."""

    def __init__(self):
        xs = [x for x, _ in VERTEX_XY[1:]]
        ys = [y for _, y in VERTEX_XY[1:]]
        self.min_x, self.min_y = min(xs), min(ys)
        self.width = (max(xs) - self.min_x) * COLS_PER_X + 1
        self.height = (max(ys) - self.min_y) * ROWS_PER_Y + 1
        self.cells = [[" "] * self.width for _ in range(self.height)]
        self.colours = [[None] * self.width for _ in range(self.height)]

    def at(self, lattice):
        x, y = lattice
        return ((y - self.min_y) * ROWS_PER_Y, (x - self.min_x) * COLS_PER_X)

    def put(self, row, col, text, colour=None):
        for offset, char in enumerate(text):
            if 0 <= row < self.height and 0 <= col + offset < self.width:
                self.cells[row][col + offset] = char
                self.colours[row][col + offset] = colour

    def centred(self, row, col, text, colour=None):
        self.put(row, col - len(text) // 2, text, colour)

    def render(self, colour=True):
        lines = []
        for row in range(self.height):
            out, active = [], None
            for col in range(self.width):
                want = self.colours[row][col] if colour else None
                if want != active:
                    out.append(RESET if want is None else want)
                    active = want
                out.append(self.cells[row][col])
            if active is not None:
                out.append(RESET)
            lines.append("".join(out).rstrip())
        return "\n".join(lines)


def text_board(state, colour=True, spots_for=None):
    """The board as text, laid out from the same lattice the PNG renderer uses."""
    grid = _Grid()

    _draw_text_roads(grid, state)
    _draw_text_tiles(grid, state)
    _draw_text_vertices(grid, state, spots_for)
    return grid.render(colour=colour)


def _draw_text_roads(grid, state):
    for road in range(1, NUM_ROADS + 1):
        owner = state.edge_owner[road]
        if owner == NO_OWNER:
            continue
        first, second = (VERTEX_XY[v] for v in ROAD_VERTICES[road])
        if first[1] > second[1]:
            first, second = second, first
        row, col = grid.at(first)
        colour = ANSI.get(owner)

        if first[0] == second[0]:                       # vertical: spans two rows
            for step in range(1, ROWS_PER_Y * 2):
                grid.put(row + step, col, "|", colour)
        elif second[0] > first[0]:                      # down and to the right
            grid.put(row + 1, col + 1, "\\", colour)
            grid.put(row + 1, col + 2, "\\", colour)
        else:                                           # down and to the left
            grid.put(row + 1, col - 1, "/", colour)
            grid.put(row + 1, col - 2, "/", colour)


def _draw_text_tiles(grid, state):
    for tile in range(1, NUM_TILES + 1):
        row, col = grid.at(TILE_XY[tile])
        resource = state.board.resource_at(tile)
        label = RESOURCE_CODES[resource]
        number = "" if resource is None else f"{state.board.number_at(tile):>2}"
        grid.centred(row, col, f"{label}{number}")
        if state.robber_tile == tile:
            grid.centred(row + 1, col, "(R)")


def _draw_text_vertices(grid, state, spots_for):
    for vertex in range(1, NUM_VERTICES + 1):
        row, col = grid.at(VERTEX_XY[vertex])
        owner = state.vertex_owner[vertex]
        if owner != NO_OWNER:
            piece = state.vertex_piece[vertex]
            mark = (CITY_LETTERS[owner - 1] if piece is Piece.CITY else str(owner))
            grid.put(row, col, mark, ANSI.get(owner))
        elif spots_for is not None and rules.respects_distance_rule(state, vertex) and (
            state.in_setup or rules.touches_own_road(state, spots_for, vertex)
        ):
            grid.put(row, col, ".", None)
        else:
            grid.put(row, col, "+", None)


# --------------------------------------------------------------------------- #
# Status                                                                      #
# --------------------------------------------------------------------------- #

def hand_summary(state, player, hidden=False):
    hand = state.hands[player]
    if hidden:
        return f"{sum(hand)} cards"
    parts = [f"{Resource(r).name.lower()[:2]} {n}" for r, n in enumerate(hand) if n]
    return ", ".join(parts) if parts else "nothing"


def player_summary(state, player, reveal=False):
    """One line per player. Hidden holdings stay hidden unless ``reveal``."""
    tag = f"P{player}"
    if state.largest_army_holder == player:
        tag += " [army]"
    if state.longest_road_holder == player:
        tag += " [road]"

    points = (rules.victory_points(state, player) if reveal
              else rules.public_victory_points(state, player))
    label = "vp" if reveal else "public vp"
    dev = sum(state.dev_cards[player])
    dev_text = (
        f"{dev} dev" if not reveal
        else ", ".join(f"{DevCard(c).name.lower()} {n}"
                       for c, n in enumerate(state.dev_cards[player]) if n) or "no dev"
    )
    return (
        f"{tag:<16} {label} {points:<3} | {hand_summary(state, player, hidden=not reveal)}"
        f" | {dev_text} | knights {state.knights_played[player]}"
        f" | road {rules.longest_road_length(state, player)}"
    )


def describe(index):
    """An action index as something a person can read."""
    action = action_space.decode(index)
    kind = action.type
    if kind is ActionType.END_TURN:
        return "end turn"
    if kind is ActionType.BUILD_ROAD:
        return f"road at {action.position} {tuple(ROAD_VERTICES[action.position])}"
    if kind is ActionType.BUILD_SETTLEMENT:
        return f"settlement at {action.position}"
    if kind is ActionType.BUILD_CITY:
        return f"city at {action.position}"
    if kind is ActionType.TRADE_WITH_BANK:
        return (f"trade {Resource(action.position).name.lower()}"
                f" -> {Resource(action.extra).name.lower()}")
    if kind is ActionType.MOVE_ROBBER:
        victim = f", rob P{action.extra}" if action.extra else ""
        return f"robber to tile {action.position}{victim}"
    if kind is ActionType.DISCARD:
        return f"discard {Resource(action.position).name.lower()}"
    if kind is ActionType.BUY_DEV_CARD:
        return "buy a development card"
    if kind is ActionType.PLAY_YEAR_OF_PLENTY:
        return (f"year of plenty: {Resource(action.position).name.lower()}"
                f" + {Resource(action.extra).name.lower()}")
    if kind is ActionType.PLAY_MONOPOLY:
        return f"monopoly on {Resource(action.position).name.lower()}"
    return kind.name.replace("PLAY_", "play ").replace("_", " ").lower()


# --------------------------------------------------------------------------- #
# Agents                                                                      #
# --------------------------------------------------------------------------- #

class HumanAgent:
    """Prompts for a choice. The only place ``input`` is called."""

    def __init__(self, colour=True):
        self.colour = colour

    def __call__(self, observation, info):
        legal = info["legal"]
        print()
        for position, index in enumerate(legal):
            print(f"  [{position:>3}] {describe(index)}")

        while True:
            try:
                answer = input(f"\nchoose 0-{len(legal) - 1} (or 'q' to quit): ").strip()
            except EOFError:
                raise SystemExit("\ninput closed")
            if answer.lower() in {"q", "quit", "exit"}:
                raise SystemExit("bye")
            if answer.isdigit() and int(answer) < len(legal):
                return legal[int(answer)]
            print(f"  not a choice — enter a number from 0 to {len(legal) - 1}")


AGENTS = {
    "human": lambda seed, colour: HumanAgent(colour=colour),
    "hard": lambda seed, colour: HeuristicAgent(seed, noise=DIFFICULTY["hard"]),
    "medium": lambda seed, colour: HeuristicAgent(seed, noise=DIFFICULTY["medium"]),
    "easy": lambda seed, colour: HeuristicAgent(seed, noise=DIFFICULTY["easy"]),
    "greedy": lambda seed, colour: GreedyAgent(seed),
    "random": lambda seed, colour: RandomAgent(seed),
}

# Each champion joins the roster when there is one. Optional: the CLI must work on a
# checkout with no PyTorch, and must not be disturbed by a training run in progress — both
# are read from models/, which only a promotion writes.
try:
    from training import champion as _champion
    from training.alphazero import champion as _az_champion

    if _champion.load() is not None:
        AGENTS["learned"] = lambda seed, colour: _champion.load(temperature=0.35, seed=seed)
    elif _az_champion.load_previous_technique() is not None:
        # Promoted at an older encoder.SIZE, so champion.load refuses it. The observation
        # only grew, and the graft gives the new columns zero weight, so this is the same
        # function that was measured — see interfaces/web/api.py and decision record 0023.
        AGENTS["learned"] = lambda seed, colour: _az_champion.load_previous_technique(
            temperature=0.35, seed=seed)

    if _az_champion.load() is not None:
        AGENTS["alphazero"] = lambda seed, colour: _az_champion.load(
            simulations=_az_champion.CHAMPION_SIMULATIONS, temperature=0.35, seed=seed)
except ImportError:
    pass


# --------------------------------------------------------------------------- #
# Driving                                                                     #
# --------------------------------------------------------------------------- #

def run_game(agents, seed, ruleset, quiet=False, colour=True, render_to=None,
             human_seats=(), max_turns=None):
    """Play one game. Returns the final ``info``."""
    env = CatanEnv(num_players=len(agents), ruleset=ruleset,
                   **({} if max_turns is None else {"max_turns": max_turns}))
    observation, info = env.reset(seed=seed)
    frame = 0

    while not info["done"]:
        player = info["player"]
        is_human = player in human_seats

        if not quiet and (is_human or not human_seats):
            _show(env, info, colour=colour, reveal_for=player if is_human else None)

        action = agents[player](observation, info)

        if not quiet and not is_human:
            print(f"  P{player}: {describe(action)}")

        observation, reward, terminated, truncated, info = env.step(action)

        if render_to is not None:
            from interfaces.render import save
            frame += 1
            save(env.state, pathlib.Path(render_to) / f"turn_{frame:04d}.png")

    if not quiet:
        _show(env, info, colour=colour, reveal_for="all")
        if info["winner"] is None:
            print("\n  the game was truncated — no winner")
        else:
            print(f"\n  P{info['winner']} wins with {info['scores'][info['winner']]} points")
    return info


def _show(env, info, colour=True, reveal_for=None):
    state = env.state
    print()
    print("=" * 68)
    print(text_board(state, colour=colour,
                     spots_for=reveal_for if isinstance(reveal_for, int) else None))
    print()
    for player in state.players:
        reveal = reveal_for == "all" or reveal_for == player
        print("  " + player_summary(state, player, reveal=reveal))
    roll = "-" if state.last_roll is None else state.last_roll
    print(f"\n  turn {info['turn']} | {state.phase.name} | last roll {roll} "
          f"| bank {sum(state.bank)} | dev deck {len(state.dev_deck)}")
    print(f"  {DIM if colour else ''}{LEGEND}{RESET if colour else ''}")
    if not info["done"]:
        print(f"  waiting on P{info['player']}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m interfaces.cli",
        description="Play or watch a game of Catan.",
    )
    # The strongest available opponent, in order of preference. AlphaZero first when it
    # exists: its promotion gate refuses to install one that has not beaten the fixed
    # heuristic, so its presence is itself the evidence.
    strongest = next((name for name in ("alphazero", "learned") if name in AGENTS), "hard")
    parser.add_argument("--agents", nargs="+",
                        default=["human", strongest],
                        metavar="AGENT",
                        help=f"one per seat: {', '.join(AGENTS)} "
                             f"(default: human {strongest})")
    parser.add_argument("--seed", type=int, default=None, help="reproduces a whole game")
    parser.add_argument("--games", type=int, default=1, help="play this many")
    parser.add_argument("--rules", choices=["ranked1v1", "base"], default="ranked1v1")
    parser.add_argument("--quiet", action="store_true", help="results only")
    parser.add_argument("--no-color", dest="colour", action="store_false")
    parser.add_argument("--render", metavar="DIR",
                        help="also write a PNG per action into DIR")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="stop a game early; it is then reported as truncated")
    args = parser.parse_args(argv)

    unknown = [name for name in args.agents if name not in AGENTS]
    if unknown:
        parser.error(f"unknown agent(s): {', '.join(unknown)}. Choose from {', '.join(AGENTS)}")
    if not 2 <= len(args.agents) <= 4:
        parser.error("between 2 and 4 agents are needed")

    ruleset = RANKED_1V1 if args.rules == "ranked1v1" else BASE_GAME
    if args.render:
        pathlib.Path(args.render).mkdir(parents=True, exist_ok=True)

    human_seats = {seat for seat, name in enumerate(args.agents, start=1)
                   if name == "human"}
    if human_seats and args.games > 1:
        parser.error("a human plays one game at a time")

    # Say *which* champion is playing. "alphazero" names a file that is overwritten every
    # promotion, so a result recorded against it is ambiguous a week later; the name is not.
    if "alphazero" in args.agents and not args.quiet:
        try:
            from training.alphazero import champion as _named
            print(f"alphazero: champion {_named.name()} "
                  f"at {_named.CHAMPION_SIMULATIONS} simulations/move")
        except ImportError:
            pass

    tally = {seat: 0 for seat in range(1, len(args.agents) + 1)}
    tally["truncated"] = 0

    for game in range(args.games):
        seed = None if args.seed is None else args.seed + game
        agents = {
            seat: AGENTS[name](None if seed is None else seed + seat, args.colour)
            for seat, name in enumerate(args.agents, start=1)
        }
        if not args.quiet and args.games > 1:
            print(f"\n### game {game + 1} of {args.games} (seed {seed})")
        info = run_game(agents, seed, ruleset, quiet=args.quiet, colour=args.colour,
                        render_to=args.render, human_seats=human_seats,
                        max_turns=args.max_turns)
        if info["winner"] is None:
            tally["truncated"] += 1
        else:
            tally[info["winner"]] += 1

    if args.games > 1:
        print("\n" + "=" * 68)
        for seat, name in enumerate(args.agents, start=1):
            print(f"  P{seat} ({name}): {tally[seat]} wins")
        if tally["truncated"]:
            print(f"  truncated: {tally['truncated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
