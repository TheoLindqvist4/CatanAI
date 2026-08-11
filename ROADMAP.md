# CatanIA — Roadmap

**Goal.** A complete, rules-correct Catan engine whose entire state is machine-readable, so a
reinforcement-learning agent can be trained on it. A playable human interface is a *consumer* of
that engine, never a part of it.

Ordering principle: **engine first, rules second, AI surface third, UI last.** Every phase leaves
the repo in a working, tested state.

This file is the *plan*. For what has already been established — how the geometry works, the
initial audit, and the reasoning behind each decision taken — see **[`docs/`](docs/README.md)**.

---

## Status

| Phase | Scope | State |
|---|---|---|
| **0** | Unblock: correctness, performance, determinism, tests | ✅ **done** |
| **1** | Real state model + economy | ✅ **done** |
| **2** | Complete the rules, incl. the ranked 1v1 ruleset | ✅ **done** |
| **3** | AI surface (action space, observations, env) | ✅ **done** |
| **4** | Interfaces (renderer, CLI) | ✅ **done** |
| **5** | Engine support for an interface: events, click targets | ✅ **done** |
| **6** | The web game — play against it in a browser | ✅ **done** |
| **7** | A better opponent: the heuristic agent | ✅ **done** |
| **8** | PPO self-play, the champion and its gate | ✅ **done** |
| **9** | AlphaZero: determinized search, self-play, a second champion | ✅ **done** — six champions so far |
| **10** | The opening: a search wide enough to see it, a gate with one rung | ✅ **done** |

935 tests, and they are no longer quick. A full run takes **about six minutes** on this
machine (374.8 s), and `python -m pytest -m "not slow"` deselects 35 and takes about six
minutes too (353.6 s) — deselecting the slow marker buys back barely 20 seconds, because the
cost is not the fuzzing. It is the web tests playing whole games over HTTP, and those are not
marked slow. Two tests are timing-based and flake under load — re-run them alone before
believing a failure, and check what else is running first.

The default ruleset is **Colonist ranked 1v1** — 15 points, hand limit 9, Friendly Robber,
Balanced Dice — with base-game Catan available as a control
([decision 0013](docs/decisions/0013-ranked-1v1-ruleset.md)).

---

## Target architecture

```
catan/
  topology.py      # ✅ Geometry GENERATED from ROW_LENGTHS, frozen at import. O(1) lookups.
  rulesets.py      # ✅ RuleSet: base game vs ranked 1v1 (the default)
  resources.py     # ✅ the five resources; costs as fixed-width vectors
  dev_cards.py     # ✅ the 25-card deck, award thresholds
  dice.py          # ✅ plain 2d6, or Colonist's 36-card Balanced Dice deck
  board.py         # ✅ one layout: numbers, resources, production index. IMMUTABLE.
  state.py         # ✅ GameState: vertex_owner[54], vertex_piece[54], edge_owner[72],
                   #    hands, supplies, phase, turn.  clone() / __eq__
  actions.py       # ✅ Action = (type, position, extra)
  rules.py         # ✅ legal_actions / apply — the single legality authority
  action_space.py  # ✅ 325 flat indices + legal_mask(state). Append, never insert
  encoder.py       # ✅ encoder.SIZE-float observation, perspective-rotated, hidden-info masked
  env.py           # ✅ Gymnasium-style reset(seed) / step(index)
  agents.py        # ✅ random and greedy baselines + play_match
interfaces/
  render.py        # ✅ draws a GameState as a PNG, straight from the topology lattice
  cli.py           # ✅ play or watch in the terminal — the ONLY place print()/input() appear
  static/images/   # ✅ board art vendored from FullStackCatan, + the card and robber sets
  web/
    api.py         # ✅ the game as plain dicts — no HTTP, so it is testable
    server.py      # ✅ a stdlib HTTP shim over api.py
    static/        # ✅ the browser client: SVG board, click to play
tests/
```

See [`docs/engine.md`](docs/engine.md) for how the layers fit together and how to drive a
game.

**Non-negotiables**

1. `rules.py` is pure and I/O-free. `legal_actions` is the *single* legality authority — no
   duplicated `check_*` logic in a `Game` class.
2. Per-game `random.Random(seed)`. Never the global `random` module.
3. **Perspective rotation** in the encoder: "me" is always player index 0, so one network plays
   every seat.
4. **Hidden-information masking**: opponents expose hand *size* and played dev cards, never
   contents.
5. `GameState.clone()` must be cheap — MCTS and self-play both depend on it.

---

## Phase 0 — Unblock ✅

Prerequisite work that everything else was blocked on.

- [x] **Generate the geometry instead of hand-writing it.** The whole board now derives from one
      line — `ROW_LENGTHS = (3, 4, 5, 4, 3)`. Hex centres go on an integer lattice, corners are
      deduplicated by exact equality, and ids are assigned by position: vertices by `(y, x)`, roads
      by `(min y, x₁+x₂)`. ~440 lines of hand-maintained tables → 0, and the generated ids are
      *identical* to the ones drawn in `Images/`, pinned by tests. This structurally eliminates the
      bug found at old `Board.py:324` / `Board.py:373` (roads 2 and 51 were each missing a
      neighbour, corrupting longest-road and legal-move enumeration).
      → [decision 0001](docs/decisions/0001-generate-geometry-from-the-row-structure.md),
      [docs/board-geometry.md](docs/board-geometry.md)
- [x] **Freeze the topology.** Lookups were rebuilding a 54- or 72-entry dict literal on every
      call (~5.5 µs). Now module-level tuples indexed in O(1) with zero allocation.
- [x] **Flat tile arrays instead of ragged rows.** `Board` indexes tiles by id; the 3-4-5-4-3 row
      view is rebuilt on demand for display only. Number placement no longer uses a half-filled
      row's length as an implicit progress marker.
      → [decision 0002](docs/decisions/0002-flat-tile-arrays-not-ragged-rows.md)
- [x] **Real production structure.** `positions_grid` (`{vertex: [{number: resource}, …]}`, which
      discarded tile identity) is replaced by
      `vertex_production: vertex → tuple[Production(tile, number, resource)]` plus a payout index
      keyed by roll. A dice payout is now one dict lookup rather than a 54-vertex scan, and a 7
      pays nobody *structurally* because the desert is excluded from the index.
- [x] **Coastline derived.** `COASTAL_ROADS` (30), `PERIMETER_VERTICES` (30) and `CORNER_VERTICES`
      (18) are named apart so Phase 2 harbours cannot attach to the wrong set.
- [x] **Remove I/O from the core.** No `print` in `Board.__init__`; `display_board()` returns a
      string; no module-level game instantiation; `if __name__ == "__main__"` guard.
- [x] **Determinism.** Injectable `random.Random` through `Board`, `Dice`, `Game`. Bounded
      board-generation retries. All legal-move enumerations return sorted lists.
- [x] **Instance state, not class state.** `player_order`, `turn_number`, `dice_value` and the deck
      counts were class attributes shared across every instance — fatal for parallel self-play.
- [x] **Ids raise instead of returning a sentinel.** The adjacency lookups used to return the
      *string* `"The number must be between 1 and 54."` on bad input, so a caller iterating the
      result silently looped over its characters.
- [x] **Test suite.** 129 tests covering the numbering drawn in `Images/`, geometry invariants,
      the lattice, the coastline, board-generation invariants, determinism, I/O-freedom, two
      performance floors, and regressions pinning roads 2/51 and the longest-road branching case.
- [x] **`docs/`** — the [audit](docs/audit-2026-07-30.md), a
      [geometry reference](docs/board-geometry.md), and numbered
      [decision records](docs/decisions/).

Three items were pulled forward from Phase 1 because they sat inside code being rewritten anyway,
and leaving a known-wrong version in place was worse than the small scope increase:

- [x] `get_available_road_from_settlement` now filters against occupancy, and the setup-phase road
      loop no longer `break`s on a rejected choice. Together these were costing a player their
      starting road: the offered list included roads that were already built, and picking one
      consumed the prompt without placing anything.
- [x] The resource supply in `Deck` was 21 per type; standard Catan is 19.
- [x] The production data structure (listed under Phase 1 as replacing `positions_grid`), since the
      board's layout code was being rewritten around it.

**Measured effect** (same board layouts, same longest-road answers):

| | before | after | |
|---|---|---|---|
| adjacency lookup | 5,490 ns | 38 ns | 145× |
| board construction | 2,753/s | 6,637/s | 2.4× |
| `find_longest_path` (8 roads) | 319 µs | 81 µs | 3.9× |
| hand-maintained geometry data | ~440 lines | 0 | — |

`find_longest_path` is still an exponential search; memoising it is Phase 3 work, once
`GameState` exists to key a cache on.

## Phase 1 — Real state model + economy ✅

The blocking issue was that the board could not say *who owns what*: `settlement_positions` was one
flat "available" list, so **empty**, **blocked by the distance rule** and **occupied by player N**
all collapsed into a single bit.

- [x] **`catan/` package** built fresh, per
      [decision 0007](docs/decisions/0007-package-layout-rewrite-vs-incremental.md).
      `topology.py` moved in unchanged and is the only module shared with the legacy engine —
      deliberately not copied, since two copies of the geometry would drift.
- [x] **`GameState`** with `vertex_owner[54]`, `vertex_piece[54]`, `edge_owner[72]`, hands and
      supplies as parallel arrays. Availability is *derived* from ownership; nothing is deleted.
- [x] **The board is immutable**, so `clone()` shares it —
      [decision 0009](docs/decisions/0009-immutable-board-mutable-state.md).
- [x] **One legality authority.** `legal_actions` and `apply` share the same `can_*` predicates,
      so a move can never be offered by one and rejected by the other.
      `apply` mutates; `clone()` is the explicit copy —
      [decision 0008](docs/decisions/0008-mutating-apply-plus-clone.md).
- [x] **Resources charged on build.** One brick + one wood now builds exactly one road.
- [x] **Connectivity enforced**, including the rule that a road may not be extended through an
      opponent's building.
- [x] **Cities**: 2 wheat + 3 ore, must upgrade an own settlement, double production, 2 VP, and the
      settlement returns to the supply.
- [x] **Victory points + the win at 10**, derived from the board so they cannot drift.
- [x] **Setup** as an atomic action per placement, snake order, and the second settlement's payout.
- [x] **Longest road** under [decision 0006](docs/decisions/0006-longest-road-intersection-reuse.md):
      strict simple path, and an opponent's building breaks a chain. The *award* is Phase 2.
- [x] **2–4 players.**
- [x] **Naming normalised**: `'Weat'` is gone, resources are a `Resource` IntEnum, hands are
      vectors rather than dicts keyed by misspelled strings.
- [x] **147 new tests**, including whole-game fuzzing that re-checks every state invariant after
      every mutation.

**Found while building it:** `Action.__repr__` crashed on a bogus action type — inside the error
message raised *because* the type was bogus. And `GameState.__eq__` required board *identity*, so
replaying a seed compared unequal; `Board` now has value equality and a hash.

## Phase 2 — Complete the rules ✅

Trading was moved to the front of this phase because Phase 1 measured only **4 of 40** random
games reaching 10 points: a settlement needs four *different* resources, most players' buildings
reach only three, and there was no way to convert a surplus. That made the reward signal
almost always zero.

- [x] **The bank**: 19 cards per resource, with cards conserved — every card is either in the
      bank or in a hand, so the total is always 95. Paying for a build returns the cards.
      Production is bank-limited with the official shortage rule: if the bank cannot cover
      everything owed of a resource, nobody gets any unless exactly one player is owed it.
- [x] **4:1 bank trading**, and **harbours** at 3:1 and 2:1. Nine harbours on coastal roads,
      evenly spaced, both endpoints granting the port; positions fixed, types shuffled by seed —
      [decision 0010](docs/decisions/0010-harbour-placement.md), which records that the
      positions are *not* the official ones.
      → **39 of 40 games now finish**, median 286 turns.
- [x] **The robber**, on `GameState` since it moves during play. Starts on the desert, must
      always move, blocks its tile's production for everyone, and steals one card drawn
      uniformly over the victim's *cards* — not their resource types. Cannot rob you, or anyone
      holding nothing.
- [x] **Discard on 7**: everyone over 7 cards gives up half, rounded down, in seat order from
      the roller. **One card per action**, so the choice stays a small discrete action instead of
      a multiset. The count is fixed when the 7 is rolled — recomputing it from the shrinking
      hand moved the target and stopped discards at 7 cards instead of at half, which is a bug
      this caught.
      → still **40 of 40** random games finish, median 393 turns (up from 286; the robber
      slows the game, as it should).
- [x] **Harbours randomised** per board rather than fixed, after confirming the rules sanction
      it and that no official coastal-edge list is published —
      [decision 0010](docs/decisions/0010-harbour-placement.md). 280 distinct position sets,
      gaps always 3 or 4 roads so they never cluster.
- [x] **Development cards**: the 25-card deck, buying, and both timing rules — one per turn,
      and not the turn you bought it. All five cards. Only a Knight may be played *before*
      rolling, which is how it blocks a tile before that tile produces; the other three have
      the same effect either side of the dice, so they wait for the roll
      (`catan.actions.PRE_ROLL_PLAYS`). Victory Point cards are never played: they
      count while held and stay hidden, so `public_victory_points` exists alongside
      `victory_points` for what an opponent can see.
      → [decision 0012](docs/decisions/0012-development-card-modelling.md)
- [x] **Largest Army** (3+ knights) and **Longest Road** (5+ segments), 2 points each, sharing
      one implementation: a minimum to qualify, sole leadership to take it, holder keeps it on a
      tie. Rechecked after *every* build — a settlement can break an opponent's road and take
      Longest Road off them.
      → **40 of 40** random games finish, median 349 turns. Largest Army is held in 39 and
      Longest Road in 37, so both are contested. Winners' points: buildings 203, VP cards 92,
      army 58, road 50.
- [x] **The ranked 1v1 ruleset**, after reading the published Colonist settings: 15 points to
      win, hand limit 9, **Friendly Robber** (a player at or below 2 *public* points cannot be
      robbed, and their tiles cannot even be blocked), and **Balanced Dice** (a 36-card deck of
      every two-dice combination, replaced when 12 remain). Made configuration rather than
      special cases, so base-game Catan stays runnable as a control.
      → [decision 0013](docs/decisions/0013-ranked-1v1-ruleset.md)
      → both rulesets finish 30/30 random games; ranked 1v1 takes a median 435 turns to 15
      points against 326 to 10.
- [ ] **Official spiral layout** as an alternative to balanced generation, behind a config flag.
      The house rule makes double-production vertices impossible, so an agent trained only on it
      never learns to value a "double 6" spot —
      [decision 0005](docs/decisions/0005-balanced-board-generation.md).

**Out of scope for this version:** player-to-player trading
([decision 0011](docs/decisions/0011-no-player-to-player-trading.md)). The target is 1v1, where
trading hands resources to the only opponent who can beat you, and an unrestricted offer is a
multiset-for-multiset exchange that does not flatten into a discrete action space. Revisit when
adding 3–4 player training or human play.

## Phase 3 — AI surface ✅

- [x] **`action_space.py`**: **324** flat indices, in contiguous blocks by action type, plus
      `legal_mask(state) -> bytearray`. The size is independent of the player count, so a network
      trained on 1v1 has the same output shape as one on four players. The mask is *translated*
      from `rules.legal_actions`, never re-derived — a second legality authority is the exact bug
      the engine was rebuilt to remove. Costs +2% over `legal_actions`.
      The load-bearing test asserts every action the rules can ever offer is expressible: if one
      were not, the mask would drop it silently and an agent could never choose it.
- [x] **`encoder.py`**: a fixed-length observation (**1808** floats when this phase closed,
      2,503 today) with named blocks (`LAYOUT`, `SHAPES`) so a
      graph or convolutional model can reshape rather than being forced through an MLP.
      Perspective-rotated — *me* is always player slot 0, so one network plays every seat.
      Hidden information masked per observer, enforced by **leak detectors** that mutate the
      hidden thing and assert the observation does not move.
- [x] **`env.py`**: `reset(seed)` / `step(index)`. Rolls the dice for you (it is stochasticity,
      not a move) *except* when a development card can be played first, which is a real choice.
      `info["player"]` is **who must act** — during a discard that is usually an opponent.
      Terminal zero-sum reward, and truncation reported separately from termination.
- [x] **`agents.py`**: random and greedy baselines, plus `play_match`, which **swaps seats every
      other game** because Catan's first-player advantage is real. Greedy beats random ~70%.
- [x] **Performance.** Memoising longest road was the big win — it is an exponential search and
      both `update_awards` (after every build) and the encoder wanted it per player. The memo is
      keyed on the ownership arrays themselves rather than invalidated by hand, so a future
      mutation site cannot forget. `legal_actions` also gained affordability gates.

      | | before | after |
      |---|---|---|
      | `update_awards` | 83 µs | **1 µs** |
      | `encoder.encode` | 458 µs | **250 µs** |
      | `legal_mask`, poor hand | 245 µs | **16 µs** |
      | per env step, typical | ~700 µs | **~270 µs** (~3,700/s) |

      → [docs/ai-surface.md](docs/ai-surface.md),
      [decision 0014](docs/decisions/0014-ai-surface.md)
- [x] **Belief sampling — a prerequisite for MCTS.** `clone(rng=state.rng)` copies `dice_deck`,
      `dev_deck` and opponents' `dev_cards` verbatim, so a rollout replays the same future rather
      than sampling one. Correct (these are hidden, not random) but it means search must reshuffle
      the unseen parts. Left out here deliberately, because the right approach depends on the
      algorithm; built in Phase 9 as `training/alphazero/determinize.py`.
- [ ] Scale `tests/test_selfplay.py` up to the 10k-game harness. The invariant checks are written;
      it is currently 60 games at 1,500 actions.

## Phase 4 — Interfaces ✅

- [x] **`interfaces/render.py`** — draws a `GameState` as a PNG. The engine already knows where
      everything is: `topology` places tiles, vertices *and* roads on an integer lattice, so
      rendering is one linear map from lattice units to pixels. No separate layout logic, and
      nothing to keep in sync with the rules.
      Board art vendored from [FullStackCatan](https://github.com/TheoLindqvist4/FullStackCatan)
      — 32 cropped PNGs for tiles, number tokens, settlements, cities, roads and spot markers.
- [x] **`interfaces/cli.py`** — play or watch a game in the terminal, and the only module that
      calls `print` or `input`. The board is drawn from the same `topology` lattice the PNG
      renderer uses, as a character grid: vertices on their own rows and columns, roads in the
      gaps between them, tile labels in the middle of their hexagon. Nothing positioned by hand.
      Hidden information is respected — an opponent's hand shows as a count, and victory-point
      cards stay hidden until the game ends.

      ```
      python -m interfaces.cli                          you vs the greedy agent
      python -m interfaces.cli --agents greedy random   watch two agents
      python -m interfaces.cli --games 20 --quiet       benchmark, results only
      python -m interfaces.cli --render out/            write a PNG per action
      ```
- [x] **Deleted the legacy engine** — `Board.py`, `Player.py`, `Deck.py`, `Dice.py` and
      `Game_2_players.py`, along with their tests. The CLI replaces them as the playable entry
      point, and git history keeps them if they are ever wanted.

**Out of scope: a FullStackCatan adapter.** The two projects stay independent. The board art was
*copied* here, which is not a coupling; nothing imports across repos, and neither has to move in
step with the other. If a web view is wanted later it should be built here, rendering engine
state from the same lattice `render.py` uses — not by having the other project generate its own
board, which is the second-authority problem this engine was rebuilt to remove.

---

# Playing against the AI

Phases 5–7 turn the engine into a game a person actually wants to play: a **board in the browser**
you click, against an opponent worth beating.

**Interface first, opponent second.** The web game ships against the existing `GreedyAgent` so
the interface and the rules can be confirmed working end to end; the opponent is then improved
without the interface changing, because an agent is only a
`(observation, info) -> index` callable.

## What is missing, measured

The engine is complete and the CLI works, but three things stand between that and a good game.

**The engine cannot say what happened.** `apply` mutates and returns; nothing records the story.
`_steal_one_card` moves a card in silence, and `distribute` computes who received what but no
caller reads it. So an interface currently *cannot* tell you "the AI rolled 8, you got 2 wood" or
"the AI stole your wheat" — the two things a player most needs to see. This is an engine gap, not
a presentation one, which is why it comes first.

**Choice lists are too long for a person.** Measured over a full game: setup offers all **54**
vertices as one flat list, and mid-game decisions peak between 40 and 54. Most turns are under 10,
but the worst cases are exactly the interesting ones. A numbered list is fine for an agent and
wrong for a human — the fix is to pick an action *type* first, then click the board.

**The opponent picks *what*, not *where*.** `GreedyAgent` orders action types sensibly and then
chooses a position **at random**. It beats `RandomAgent` about 70% of the time, which says more
about random than about greedy. There is no positional judgement in the project at all yet.

---

## Phase 5 — Engine support for an interface ✅

Headless and fully testable. Everything here is a prerequisite for the game, and none of it cost
the training loop anything measurable.

- [x] **An event stream.** `catan.events` with a small fixed-arity `Event` — rolled, produced,
      stole, robber moved, discarded, built, traded, bought, played, monopolised, award, turn
      ended, won. The engine already computed all of it and threw it away.
      - **The rules only append; clearing is the caller's job.** An earlier version cleared in
        both `apply` and `roll_dice`, and silently lost every action that happened to precede an
        automatic roll — visible as a missing setup road in the log. `CatanEnv.step` clears once
        and returns the lot in `info["events"]`.
      - ✅ **Measurement gate passed.** End-to-end throughput is **3,718 steps/sec, unchanged**;
        `update_awards` went 1.0 → 1.5 µs and nothing else moved. No opt-in flag needed.
- [x] **Click targets.** `action_space.clickable(state)` maps each board action type to the
      elements that are legal targets, and `grouped(state)` arranges everything by type. This
      lives next to the action space, not in the UI: it is the same information the mask carries,
      shaped for a person, and here it can be tested.
- [x] **`events.describe`** shared by every interface, so none of them can disagree about what
      just happened.
- [ ] **Undo** — a bounded stack of `state.clone()` snapshots before each human action. Cloning is
      ~2 µs, so the cost is nothing; the work is deciding what undo means across a dice roll.
      Deferred: not needed to confirm the game works.
- [ ] **Save and load** — `GameState` to JSON and back. Deferred for the same reason, though it
      would also let a bug report carry the position it happened in.

## Phase 6 — The web game ✅

A local web app: Python serves the engine, the browser draws the board and posts clicks.

    python -m interfaces.web        then open http://127.0.0.1:8000

- [x] **The standard library, and nothing else.** A single-player local game does not need a
      framework, and adding one would mean a dependency, an install and a version to pin for
      something this thin. `interfaces/web/server.py` is ~150 lines of `http.server`.
- [x] **All the thinking in `interfaces/web/api.py`**, which knows nothing about HTTP — plain
      dicts in and out, so every decision is testable in Python. Swapping in FastAPI later
      rewrites `server.py` and touches nothing else.
- [x] **The server decides everything; the client renders and reports clicks.** The browser holds
      no legality, no scoring and no board generation. The last time this project had board logic
      in JavaScript it was a second implementation that could disagree with the engine.
- [x] **Click to play.** The board is SVG, positioned from the same lattice `render.py` maps to
      pixels, with the vendored art dropped in. Pick a build type, the legal targets pulse, click
      one. Fat invisible hit-areas so a road is easy to hit.
- [x] **Everything a player needs to see**: the dice each turn, both hands (yours in full, the
      opponent's as a count), development cards, knights, longest road, pieces left, the bank, the
      dev deck, the robber, harbours, whose turn it is, and what is being asked of you.
- [x] **A running log** — every roll, payout, theft, trade, purchase and award, in plain English
      from `events.describe`, so the opponent's turn is readable rather than a board that changed
      while you watched.
- [x] **Hidden information filtered server-side**, in one function. `tests/test_web.py` walks a
      whole game asserting no response ever carries the opponent's hand, their development cards
      or either deck — the same leak detectors the encoder has, applied to JSON.
- [x] **45 tests**, including a full game played over real HTTP, illegal actions returning 400,
      unknown games 404, and path traversal refused. **134 today**, across `test_web.py` and
      `test_web_stats.py`.

Added since, and worth listing because they are what a person actually sees:

- [x] **The game says when it is over.** The board goes quiet by itself when someone wins —
      nothing left to place, no actions offered — so the result used to be one grey line in the
      hint bar that was genuinely easy to miss. An overlay announces it, with both scores and
      where to go next. Nothing is asked of the server: `done`, `winner` and both scores are
      already in the view that drew the last move.
- [x] **A statistics panel**, behind a Stats button and available mid-game rather than only at
      the end — what the dice have been doing is worth knowing while there is still something
      to do about it. Dice distribution against what a fair deck of 36 would have given, per
      player and per total; what each 7 cost; what the robber denied; cards produced, stolen,
      monopolised and spent; and where every victory point came from.
      - **Derived in the interface, not counted in the engine.** A tally on `GameState` is
        copied by `clone()` on every MCTS node expansion — `clone` is ~4 µs and the search
        calls it hundreds of thousands of times an hour — so a figure read once at the end of
        one browser game would be paid for by every search in training. Three quarters of it
        was already public state the observation needs anyway; only two things are accumulated
        (who rolled what, and what the robber blocked).
      - ⚠️ **`interfaces/web/stats.py::owed` is a second implementation of the production
        walk**, which the one-source-of-truth rule would otherwise forbid — `rules.distribute`
        skips a blocked tile with a bare `continue` and the amount is gone, so the robber's
        damage has to be computed rather than counted.
        `test_what_the_robber_blocked_agrees_with_the_rules` drives whole games and compares
        the two at every roll, which is what makes it legal.
      - **Its own endpoint**, fetched on click rather than four times a turn, so nothing here
        is on the path of a move. Victory Point cards stay hidden until the game ends, on
        exactly the condition `view` already uses.

## Phase 7 — A better opponent ✅

The interface is done and the rules are confirmed working, so the opponent was improved
**without touching the interface** — an agent is only a `(observation, info) -> index` callable,
and `interfaces/web/api.py` picks it from a dict.

`GreedyAgent` ordered action types sensibly and then chose a position **at random**, which is
why a person beat it 16–3 on the first try. `HeuristicAgent` supplies the missing half.

- [x] **Position evaluation.** `catan/heuristics.py` — pure functions over a `PublicView`. Value
      is **marginal**: a tile is discounted by how much of that resource you already produce, so
      a spot covering three resources you lack beats a richer one covering a fourth wheat. The
      encoder's pip-potential feature was *not* reused — it is absolute production, and marginal
      is the whole point.
- [x] **Opening placement.** Every legal vertex evaluated; the setup road points at the best spot
      it opens rather than a random neighbour. `test_the_opening_is_not_random` pins it.
- [x] **Build policy.** A handler chain — city, settlement, dev-card play, road, buy, trade —
      each choosing *where* by evaluation. Roads only when they reach somewhere: a road is worth
      half the best spot it brings in reach, so one that leads nowhere scores zero.
- [x] **Robber policy.** Blocks the tile costing opponents most, weighted by settlement vs city,
      and robs whoever holds most. (Friendly Robber is enforced by the rules, so the agent only
      ever sees legal targets.)
- [x] **Card policy.** Monopoly on the resource the *bank* is missing — public arithmetic, no
      peeking at hands; Year of Plenty to complete a build this turn; discard the biggest pile of
      the cheapest thing.
- [x] **Difficulty levels.** One knob: noise added to each evaluation. `easy` misjudges spots,
      `hard` does not. Exposed in the web dropdown and `--agents` on the CLI.
- [x] **It cannot cheat.** Agents see a `PublicView` with an explicit allow-list, so hidden cards
      raise `AttributeError` rather than being available to read. A leak test replays a whole game
      and demands the same move at every decision once the opponent's hidden cards are rewritten.
- [x] **Measure it.** 60 games per pairing, seats swapped:

      hard   vs random     98.3%        hard   vs medium    73.7%
      hard   vs greedy     96.7%        hard   vs easy      80.0%
      medium vs greedy     96.6%        medium vs easy      69.5%
      easy   vs greedy     91.7%        greedy vs random    75.0%

      Monotone, which is what a difficulty setting has to be to mean anything. The knob does
      saturate: noise only degrades *position* choice, never the action-type priority, so `easy`
      still beats greedy 91.7% of the time — `greedy` and `random` remain the rungs below it.

**Found on the way:** `robber_damage` indexed the vertex→tiles table with a *tile* id, so robber
placement was scored against unrelated positions. Both tables are keyed by integers, so nothing
complained. See [decision 0016](docs/decisions/0016-heuristic-opponent-and-difficulty.md).

## Phase 8 — Self-play ✅ (pipeline) / 🔬 (still improving the result)

`HeuristicAgent` was the baseline Phase 7 existed to create. The question is no longer "can it
beat random" but "can it beat 96.7%-against-greedy".

- [x] **PPO, not AlphaZero.** MCTS needs a state it can roll forward; `clone()` copies the dev
      deck, the dice deck and opponents' cards verbatim, so a rollout replays the same future
      rather than sampling one. Belief sampling is a prerequisite and is not built.
      See [decision 0017](docs/decisions/0017-ppo-self-play.md).
- [x] **Torch (CPU) as the first dependency.** The engine stays dependency-free; `training/`
      is the only package that imports it, and both interfaces work without it.
- [x] **`training/`** — `net`, `rollout`, `ppo`, `pool`, `clone`, `evaluate`, `agent`, `train`.
- [x] **Three engine invariants pinned.** The winner is always the actor (so `step()`'s reward
      is always `+1` and the loser's never arrives), the terminal observation is the winner's,
      the terminal mask is empty. All three fail silently; all three are now regression tests.
- [x] **Self-play against a frozen pool**, drawn per game — 60% live policy, 15% heuristic as
      an external anchor, the rest past selves.
- [x] **Behaviour cloning first.** 300 heuristic games, 86,551 decisions, 47 seconds to
      generate — reaches 30.8% against the heuristic in about four minutes, matching 70 minutes
      of from-scratch self-play. See [decision 0018](docs/decisions/0018-clone-before-self-play.md).
- [x] **The encoder was 57% of training time.** The board-static ~40% of an observation is now
      computed once per `Board`; verified bit-identical over 3,200 encodings.
      See [decision 0019](docs/decisions/0019-cache-the-board-static-observation.md).
- [x] **Measured honestly**, on 1,000 games (±3.1 points):

      vs heuristic hard      507-467   52.1%  [48.9, 55.2]
      vs heuristic medium    299- 97   75.5%  [71.0, 79.5]
      vs heuristic easy      341- 57   85.7%  [81.9, 88.8]
      vs greedy              399-  1   99.8%  [98.6, 100.0]
      vs random              399-  1   99.8%  [98.6, 100.0]

### Where it stands

**The learned policy is now the strongest player in the project.** 1,000 games against
`HeuristicAgent(noise=0)`:

      vs heuristic hard      735-253   74.4%  [71.6, 77.0]
      vs heuristic medium    374- 26   93.5%  [90.6, 95.5]
      vs greedy              400-  0  100.0%  [99.0, 100.0]
      vs random              400-  0  100.0%  [99.0, 100.0]

It is the default opponent in both interfaces whenever `checkpoints/policy.pt` exists —
which is not in git, so a fresh clone (or one without PyTorch) falls back to `hard`.

Getting there took about 33 minutes of CPU: four to clone the heuristic, 29 to fine-tune.
The path mattered more than the budget — the first flat-network attempt spent 68 minutes to
reach parity (52.1%) and could not get past it.

### What was in the way, and what it turned out to be worth

All four items were implemented and measured. Ranked by what they actually delivered rather
than by what was expected:

- [x] **The network was the binding constraint.** An MLP over the flat 1808-vector had to
      rediscover board geometry `topology.py` already knows, and could spend the constant
      tiles block on memorising board identity. `training/structured_net.py` shares weights
      across all 54 vertices / 72 roads / 19 tiles and produces per-position logits from each
      position's own embedding. Held-out cloning agreement **69.6% -> 80.3%**, train/test gap
      **13.9 -> 2.2 points**, with 7.3x fewer parameters. Fine-tuned, it went from the flat
      network's 52.1% against the heuristic to **77.6%**.
      See [decision 0021](docs/decisions/0021-structure-aware-network.md).
- [x] **The observation had no memory.** Five counters — roll histogram, cumulative production
      and spending per player, development cards bought, turns since last build — rotated per
      seat. `SIZE` 1808 -> 1868. Deliberately *not* a per-player hand estimate: a robber steal
      moves a card whose identity only the two players involved ever learn, so any running
      total of an opponent's hand would be either wrong or a leak.
- [x] **The rollout was single-process.** Now **1,879 -> 23,599 transitions/sec** across 16
      workers; iteration time 10-12s -> 1.5-3s.
      See [decision 0020](docs/decisions/0020-parallel-rollouts-and-lookahead.md).
- [x] **One-ply lookahead — no measurable gain.** Implemented and leak-safe: `clone()` copies
      the dev deck, the dice deck and opponents' hands verbatim, so `DETERMINISTIC_TYPES` is a
      correctness boundary excluding the five action types whose application would reveal
      hidden state. At 800 games it scores 53.4% against 52.2% for no search — intervals
      overlapping almost entirely. Kept, off by default. Recorded rather than quietly dropped,
      because an unwritten negative result gets re-attempted.

### What was still open, and how it closed

- [x] **Belief sampling.** Built in Phase 9 as `training/alphazero/determinize.py`. It was
      the prerequisite for every search idea, and it is what makes the AlphaZero tree legal.
- [x] **A stronger critic.** The structured network's value head was worse than the flat
      one's (MAE 0.210 against 0.074) — the one place the old architecture still won, and the
      likely reason PPO's one-ply lookahead bought nothing. AlphaZero trains it on game
      outcomes directly, and that alone was not enough: five runs in a row did not move at
      all, `policy_loss` sitting between 1.22 and 1.24 from first iteration to last. Measuring
      the *labels* rather than the loss curve found a value head explaining **14.4%** of
      held-out outcome variance while fitting the buffer it was trained on to 0.28 MSE — it
      had learned to recognise the board, which is constant within a game and sits in the
      observation, and recall the result. Six changes, none of them capacity, took held-out
      variance explained to **35.4%**, and the candidate cleared the gate at **71.0%**
      [66.4, 75.2] over 400 games against the champion it replaced.
      → [decision 0026](docs/decisions/0026-why-the-run-stopped-learning.md)

      The honest half of that result: **the policy label still teaches almost nothing** —
      +0.7 points over the raw prior, p=0.85. The gain came through the critic and the data,
      not through better policy targets, and that is the largest opportunity still open.

---

## Phase 9 — AlphaZero ✅

`docs/catan_alphazero_implementation_guide.md` asked for a self-play pipeline with search.
The full reasoning, and every place this departs from the guide, is
[decision 0023](docs/decisions/0023-alphazero-self-play.md). The short version:

**The result this phase closed on.** The first champion of the lineage, promoted 2026-08-04 at
32 simulations a move:

| | | |
|---|---:|---|
| against the fixed heuristic | **74.7%** | 293-99, [70.2, 78.8], 400 games |
| against the PPO champion | **76.5%** | 306-94, [72.1, 80.4], 400 games |

Three stages, ~2h40m of training on 20 CPU cores, 615,307 positions in the final stage alone.

⚠️ **Five promotions later those numbers belong to a model nobody plays.** `models/champion_az.pt`
is `gen6-gilded-beacon`, promoted 2026-08-06: **55.25%** over 400 games against the champion
before it, [50.35, 60.05]. Its record carries no heuristic figure at all, because the gate
stopped playing the heuristic — Phase 10. The last champion measured there is `gen4-ashen-wheat`
at 92.7%, two promotions ago. Every promotion overwrites one file, so "the champion scores X"
is a sentence that rots; a champion is now named from a digest of its own weights, and the name
is what fixes the reference.

Training is **pure self-play** — one network plays both seats, there is no opponent pool, and
the heuristic never generates a training position. It appears only as a yardstick.

**What was built.** `training/alphazero/` — determinized PUCT search with the dice as chance
nodes, many games in flight per worker feeding one batched evaluator, a ring buffer sampled in
equal parts by age, the continuous generate/learn/measure loop, parallel head-to-head matches,
and a second champion under its own name with its own gate. Plus `benchmark/` and
`configs/train.yaml`, which the guide asks for and the repository did not have.

**The part that mattered.** Search over hidden information needed `determinize.py`, which was
the item sitting open under Phase 8. It resamples every hidden quantity from public facts
only, so scrambling the hidden state at constant public counts leaves the searched world
identical — asserted with the same scrambler the encoder and the heuristic are held to.

**What was found on the way.** `models/champion.pt` had not loaded since the affordability
block changed `encoder.SIZE` from 1868 to 1884, so both interfaces had silently been offering
the heuristic. One tensor was the wrong shape; `network.graft` widens it with zero columns,
which reproduces the original function exactly, and the interfaces offer it again.

**Where the speed went.** The guide makes simulation throughput priority one.
`python -m benchmark.benchmark` is now the number to optimise against. Four measured fixes
(the leaf deriving legality twice, `np.errstate` inside the selection rule, a Python loop
gathering priors, and `np.asarray` over lists of lists) were worth 13% together; switching the
pool from a sample-count share to a wall-clock slice was worth considerably more, because
samples bank in cohorts and `pool.map` waits for the slowest worker.

---

## Phase 10 — The opening, and what "promoted" means ✅

The pipeline worked. What the search was *looking at*, what the observation was still telling
it, and what the word "promoted" meant each turned out to need changing — and the champion was
playing at 32 simulations when the latency budget allowed 64. Each has a record.

- [x] **The opening is fifty-four moves wide, and PUCT was tuned for six.** A settlement
      placement offers 54 legal spots against a normal turn's six, and a player's two
      placements are seven plies apart, so this is where the *pair* is decided. Measured on
      `gen6-gilded-beacon` at the settings the agent actually plays at — no root noise, 40
      distinct boards (24 at 1,600 simulations) — the search examines:

      | simulations | spots examined, of 54 |
      |---:|---|
      | 64 | 3.6 (median 3, range 1–11) |
      | 400 | 5.9 (median 5, 2–54) |
      | 1600 | 8.2 (median 7, 2–54) |

      **Budget buys breadth very slowly**: 25x the simulations takes it from 3.6 to 8.2, so the
      opening was being chosen among about three spots the prior already liked. Depth was never
      the problem. `root_min_visits` gives every legal spot a floor of visits before PUCT may
      concentrate; at 400 simulations with a floor of 4 the search reaches **all 54 spots on 40
      of 40 boards**, and every legal spot at all four placements, for **0.319 s against plain
      PUCT's 0.335 s** (n=40, one torch thread) — slightly cheaper, because the forced sweep
      builds a shallower tree. Settlements get their own budget (`setup_simulations`, 400), and
      setup is exempt from the playout cap and always recorded, including the road.
      → [decision 0028](docs/decisions/0028-the-opening-is-fifty-four-moves-wide.md)

      Two things that reading "54 of 54" would otherwise get wrong. **The sweep is measurement,
      not preference**: `_discretionary_counts` subtracts the floor before the move is chosen
      and before the policy target is recorded, so the target still concentrates on a median 3
      spots — the breadth is in the value estimates. And **a floor of 8 degenerates silently**
      at this budget: 8 x 54 = 432 forced visits against the 399 available — 400 simulations
      leave 399 root visits, because the first expands the root and backs up along an empty
      path — so the sweep never completes, the discretionary counts are all zero and the
      fallback returns a near-uniform target. The shipped 4 needs 216 and is clear of it;
      there is not room for much more.

      ⚠️ **None of this measures whether the agent plays better for it.** It measures what the
      search looks at. A strength claim is what the promotion gate is for.

- [x] **`pip_potential` retired to 0.0.** The observation's oldest placement feature sums the
      odds of a vertex's tiles and throws the resources away, so three sheep and an even
      three-way spread are the same number. It worked, in the sense that the agent had learned
      it exactly — 0.005 pips off the best available spot. The evidence against it: openings
      covering five resources win where three-resource ones do not, [81.9, 98.5] against
      [45.8, 70.4] and non-overlapping, which is precisely the distinction the feature cannot
      make; record 0024 had already added the per-resource production that supersedes it. The
      pips-against-wins cut is *weak* — four bands, not monotone, every interval overlapping
      every other — and the decision does not rest on it. **The slot stays and writes 0.0**
      rather than being deleted: removing it would move every offset after it inside a vertex
      row and stop `models/champion_az.pt` loading, which per `CLAUDE.md` makes the next
      promotion ungated by construction.
      → [decision 0029](docs/decisions/0029-retiring-pip-potential.md)

- [x] **One rung in the AlphaZero gate.** It was three — the heuristic, the PPO champion, the
      reigning AlphaZero champion — which makes "promoted" a conjunction of three noisy
      measurements and lets two promotions mean different things. Now a candidate is installed
      when it beats `models/champion_az.pt` with its Wilson lower bound above 50% over 400
      games, and for no other reason. The other two are *recorded, never consulted* — but
      only the heuristic is off by default. `--ppo-games` falls back to the head-to-head game
      count, so a bare `promote` plays two matches whenever a PPO champion loads, and only
      the first can refuse; `chain.py` passes 0 for that rung as well, which is why an
      overnight run plays one. That is the chain's choice, not the gate's default.
      The argument for the heuristic rung is still true — self-play is non-transitive, so a
      candidate can climb by learning the champion's habits while getting worse at the game —
      and it was dropped anyway, deliberately, so a test pins its absence rather than leaving
      it to drift back.
      **A first promotion of a lineage is now refused** instead of installing unmeasured,
      which is what the PPO gate still does and which fires exactly when `encoder.SIZE` has
      changed and nobody is watching; installing one is an explicit `--force --reason` that
      the record carries forever as `forced`.
      → [decision 0030](docs/decisions/0030-one-rung.md)

- [x] **Champions have names, and runs chain overnight.** Every promotion overwrites one file,
      so a name is derived from a SHA-256 of the weights themselves — the same weights always
      produce the same name, and `gen6-gilded-beacon` says both which and when.
      `training.alphazero.chain` runs train → rank-with-search → gate in a loop against a
      wall-clock deadline, each stage in its own directory and nothing ever deleted, because a
      losing checkpoint is still a rung somebody can start from. Stages run as subprocesses so
      that one bad iteration costs its own hours and not the night.

- [x] **The champion plays at 64 simulations, not 32.** Search was still buying strength at
      that network size: measured against the fixed heuristic over 200 games apiece, 0 sims
      (raw policy) 64.8%, 16 sims 73.4%, 32 sims 74.4%, 64 sims 78.9%, 128 sims 81.9%. Still
      climbing at 128, so this is a **latency** choice and not a strength one — one decision
      costs 52 ms at 32, 101 ms at 64 and 207 ms at 128 on one thread, and 207 ms does not fit
      inside the 200 ms pace a watched game is played back at. `CHAMPION_SIMULATIONS` is
      imported by both interfaces rather than copied, because a win rate belongs to a
      `(weights, simulations)` pair.

## What this deliberately does not include

- **Multi-machine training.** The guide's stage 3 — workers on several machines feeding a
  central replay buffer. `ParallelSelfPlay` is a process pool on one box, and the interface it
  presents (`generate(seconds=...)` returning arrays) is the right seam to replace, but nothing
  distributed is built.
- **Multiplayer over a network.** One human against the AI, locally. Accounts, lobbies and
  matchmaking are a different project.
- **A FullStackCatan adapter**, per the note above: the two projects stay independent.

---

## Decisions

All 30 records in [`docs/decisions/`](docs/decisions/) are settled. Two were resolved going into
Phase 1: longest road uses a **strict simple path** with an opponent's building breaking a chain
([0006](docs/decisions/0006-longest-road-intersection-reuse.md)), and `catan/` was **built fresh**
([0007](docs/decisions/0007-package-layout-rewrite-vs-incremental.md)).

Consequences that are easy to trip over are collected under
[Worth knowing](docs/README.md#worth-knowing) — including that a closed loop of six roads counts as
five, and that double-production vertices cannot exist on this board.

Deviations from official Catan are recorded in
[`docs/README.md`](docs/README.md#deviations-from-official-catan) so they are not mistaken for
bugs — most importantly that balanced generation makes double-production vertices impossible.
