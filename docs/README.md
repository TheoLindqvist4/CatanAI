# Documentation

Reference material for CatanIA. The goal of the project is a rules-correct Catan engine
whose entire state is machine-readable, so an agent can be trained on it; the playable
interface consumes that engine rather than being part of it.

For *what happens next*, see [website-split-plan.md](website-split-plan.md) — packaging this
repository as a runtime that a separate public platform consumes, without either of them
gaining a second copy of the rules. This folder is otherwise for what has been established:
how things work, and why they were decided that way.

## Contents

| Document | What it covers |
|---|---|
| [board-geometry.md](board-geometry.md) | How the 19 tiles, 54 settlement positions and 72 road positions are numbered, the coordinate lattice, and the coastline. **Start here** to understand the board. |
| [engine.md](engine.md) | How the engine fits together: the layers, the state model, and how to drive a game. **Start here** to use the code. |
| [ai-surface.md](ai-surface.md) | How to train against it: the action space, the observation, the environment, and the baselines. |
| [images/board-example.png](images/board-example.png) | A game rendered by `interfaces/render.py`. |
| [audit-2026-07-30.md](audit-2026-07-30.md) | Full audit of the codebase at commit `e0f91a3`: verified bugs, missing rules, AI-readiness blockers, and what Phase 0 measured. |
| [audit-2026-08-05-public-arena.md](audit-2026-08-05-public-arena.md) | What it would take to let strangers submit agents, rank them, and play the best of them: the reuse inventory, the verified containment failures, why the seed defeats every submission format, measured throughput and cost, and the decisions to take before writing code. |
| [website-split-plan.md](website-split-plan.md) | The plan for the two-repository split: what `catania-runtime` contains, the five phases that turn this repository into a consumable one, and what CatanIA-Web is allowed to hold. Reconciles the go-live PDF against the audit above, which contradicts it in nine places. |

Everything that prints lives in `interfaces/`: `render.py` draws a PNG, `cli.py` plays or
watches a game in the terminal. Both lay the board out from the same `topology` lattice, so
neither carries positioning logic of its own.

## Decisions

Numbered, immutable records. A decision is superseded by a new record, not edited away —
the reasoning at the time is the point.

| # | Decision | Status |
|---|---|---|
| [0001](decisions/0001-generate-geometry-from-the-row-structure.md) | Generate the board geometry from the row structure | accepted |
| [0002](decisions/0002-flat-tile-arrays-not-ragged-rows.md) | Store tiles in flat arrays, not ragged rows | accepted |
| [0003](decisions/0003-io-free-core-and-injected-randomness.md) | I/O-free core, injected randomness, instance state | accepted |
| [0004](decisions/0004-desert-as-the-seven-tile.md) | Model the desert as the tile numbered 7 | accepted |
| [0005](decisions/0005-balanced-board-generation.md) | Balanced board generation (a house rule) | accepted, revisit Phase 2 |
| [0006](decisions/0006-longest-road-intersection-reuse.md) | Longest road: strict simple path, and an opponent's building breaks a road | accepted |
| [0007](decisions/0007-package-layout-rewrite-vs-incremental.md) | Build `catan/` fresh | accepted |
| [0008](decisions/0008-mutating-apply-plus-clone.md) | `apply` mutates; copy with `clone()` | accepted |
| [0009](decisions/0009-immutable-board-mutable-state.md) | The board is immutable; mutable state lives in `GameState` | accepted |
| [0010](decisions/0010-harbour-placement.md) | Harbours are randomised per board, but evenly spaced | accepted |
| [0011](decisions/0011-no-player-to-player-trading.md) | No player-to-player trading in this version | accepted — deferred |
| [0012](decisions/0012-development-card-modelling.md) | How development cards are modelled | accepted |
| [0013](decisions/0013-ranked-1v1-ruleset.md) | Ranked 1v1 is the target ruleset | accepted |
| [0014](decisions/0014-ai-surface.md) | The AI surface: action space, observations, environment | accepted |
| [0015](decisions/0015-public-view-instead-of-a-cheating-agent.md) | Agents see a `PublicView`, not the state | accepted |
| [0016](decisions/0016-heuristic-opponent-and-difficulty.md) | A heuristic opponent, with difficulty as noise | accepted |
| [0017](decisions/0017-ppo-self-play.md) | PPO self-play, not AlphaZero | accepted |
| [0018](decisions/0018-clone-before-self-play.md) | Clone the heuristic, then self-play | accepted |
| [0019](decisions/0019-cache-the-board-static-observation.md) | Cache the board-static half of an observation | accepted |
| [0020](decisions/0020-parallel-rollouts-and-lookahead.md) | Parallel rollouts, and a lookahead that does not cheat | accepted |
| [0021](decisions/0021-structure-aware-network.md) | A network that knows the board has a shape | accepted |
| [0022](decisions/0022-affordability-features.md) | The agent can see how far its hand is from a purchase | accepted |
| [0023](decisions/0023-alphazero-self-play.md) | AlphaZero-style self-play, beside the PPO lineage | accepted |
| [0024](decisions/0024-what-a-placement-can-see.md) | What a placement can see: per-resource production, harbour reach | accepted |
| [0025](decisions/0025-how-big-should-the-network-be.md) | How big the network should be, measured rather than borrowed from Go | accepted |
| [0026](decisions/0026-why-the-run-stopped-learning.md) | Why the run stopped learning, and what actually fixed it | accepted |
| [0027](decisions/0027-where-a-searched-decision-goes.md) | Where a searched decision actually goes, and the six changes that came out of it | accepted |
| [0028](decisions/0028-the-opening-is-fifty-four-moves-wide.md) | The opening is fifty-four moves wide, and PUCT was tuned for six | accepted |
| [0029](decisions/0029-retiring-pip-potential.md) | Retiring pip potential: the guide that came first | accepted |
| [0030](decisions/0030-one-rung.md) | One rung: the AlphaZero gate is the reigning champion and nothing else | accepted |

## Worth knowing

Consequences that are easy to trip over:

- **A closed loop of six roads counts as five**
  ([0006](decisions/0006-longest-road-intersection-reuse.md)). A simple path can visit a
  hex ring's six vertices but only five of its roads. Consistent with the strict ruling, but
  not obvious from it.
- **Double-production vertices cannot exist**
  ([0005](decisions/0005-balanced-board-generation.md)). The "double 6" spot of official
  Catan does not occur on this board, so an agent trained here never learns to value one.
- **Search should clone with a shared stream**
  ([0008](decisions/0008-mutating-apply-plus-clone.md)). `state.clone(rng=state.rng)` is
  13× cheaper than the default snapshot, and divergent rollouts are what sampling wants.
- **Harbours attach to `PERIMETER_VERTICES`, not `CORNER_VERTICES`**
  ([board-geometry.md](board-geometry.md#6-coastline)). 30 versus 18 — the 12 notch
  vertices are on the coast too.
- **Trading is what made games finishable — 4 of 40 games reached 10 points before it,
  39 of 40 after** ([engine.md](engine.md#trading-is-what-made-games-finishable)). A
  settlement needs four different resources and most players' buildings reach only three, so
  without a way to convert a surplus they stopped permanently, one of them holding 113 cards.
- **Harbours are randomised, and each serves only one player**
  ([0010](decisions/0010-harbour-placement.md)). The rules sanction randomising them and no
  official coastal-edge list is published, so positions vary per board — 280 layouts, gaps
  always 3 or 4 roads. A harbour is on an *edge*, so both endpoints grant it, but they are
  adjacent, so the distance rule means the first building to claim one excludes the other.
- **A discard count must be fixed when the 7 is rolled**, not recomputed. Half of a
  *shrinking* hand is a moving target: it stopped discards at 7 cards instead of at half.
  `state.discards_owed` stores it, and `rules.begin_robber` is the single place that sets it —
  a test helper that duplicated that logic is what let the bug through.
- **During a discard, `current_player` is not the turn holder.** It is whoever owes cards,
  usually an opponent. Use `state.turn_player` for the turn holder.
- **No player-to-player trading** ([0011](decisions/0011-no-player-to-player-trading.md)),
  deliberately, for the 1v1 target.
- **The default ruleset is ranked 1v1, not base Catan**
  ([0013](decisions/0013-ranked-1v1-ruleset.md)): 15 points to win, hand limit 9, Friendly
  Robber, Balanced Dice. `GameState(ruleset=BASE_GAME)` gives the printed rules. Tests that
  care about the *mechanism* read `state.ruleset.*`; tests that pin a printed number say which
  ruleset they mean.
- **Friendly Robber blocks placement, not just stealing.** A player at or below 2 *public*
  points cannot be robbed **and** their tiles cannot be blocked. Hidden Victory Point cards do
  not count toward the threshold, which is what `public_victory_points` is for.
- **Balanced Dice makes clones replay the same rolls, even sharing the RNG**
  ([0013](decisions/0013-ranked-1v1-ruleset.md)). The 36-card deck is copied on clone, so
  search cannot get divergent rollouts from the RNG alone — it must reshuffle the unseen
  remainder itself. The same applies to `dev_deck` and opponents' `dev_cards`: three pieces of
  hidden state Phase 3 must sample over together.
- **`legal_actions` is not empty during `Phase.ROLL`.** A development card may be played
  before the dice, so a driver must branch on `phase is Phase.ROLL` rather than on
  "`legal_actions` came back empty" ([0012](decisions/0012-development-card-modelling.md)).
- **`state.dev_deck` and opponents' `dev_cards` are hidden information.** The deck is stored
  in its shuffled order so clones replay identically; a search that reads it knows every
  future purchase. Phase 3's encoder must mask both, and use `public_victory_points` rather
  than `victory_points` for what an opponent can see.
- **Longest Road must be rechecked after *every* build, not just after a road.** A settlement
  or city can break an opponent's road and take the award off them. `update_awards` does this;
  it is also the most expensive thing on the hot path.
- **`info["player"]` is who must act, not the turn holder.** During a discard it is usually
  an opponent. Assuming alternation is the classic multi-agent environment bug
  ([0014](decisions/0014-ai-surface.md)).
- **Search must reshuffle hidden state before it samples anything.** `clone(rng=state.rng)`
  copies `dice_deck`, `dev_deck` and opponents' `dev_cards` verbatim, so a rollout replays the
  same future rather than sampling one. Correct — these are hidden, not random — but it means
  belief-sampling is a prerequisite for MCTS, and it is deliberately not implemented
  ([ai-surface.md](ai-surface.md#-search-needs-to-sample-hidden-state-and-does-not-yet)).
- **The action-space size does not depend on the player count**, so weights transfer between
  1v1 and 4-player. Robber actions naming an absent player are simply never legal.
- **Rendering needs no layout logic of its own.** `topology` already places tiles, vertices
  and roads on an integer lattice, so `interfaces/render.py` is one linear map to pixels.
  The earlier FullStackCatan page stepped columns by `tileWidth * 0.87` and overlapped
  neighbouring tiles by 13%; pointy-top hexes tile at exactly one full width.
- **An agent is handed a `PublicView`, and hidden fields raise `AttributeError`.** The
  allow-list is deny-by-default, so a new field on `GameState` is invisible to agents until
  someone adds it deliberately ([0015](decisions/0015-public-view-instead-of-a-cheating-agent.md)).
  `view.my_hand` returns a *copy* — an agent cannot edit the game it is playing.
- **…but the allow-list stops accidents, not adversaries.** `_state` is a `__slots__` entry, so
  attribute lookup *succeeds* and `__getattr__` — the allow-list — is never consulted:
  `info["view"]._state` is the live `GameState`, including the opponent's hand, the dev deck
  order and, under Balanced Dice, the literal next rolls. Two in-repo agents read it
  deliberately and are right to. An agent that never touches the view reaches the same object
  through `sys._getframe`. Fine for first-party code, which is what 0015 designed it for;
  it is not a boundary for code you did not write
  ([audit](audit-2026-08-05-public-arena.md#2-blockers-verified-by-execution)).
- **`info["scores"]` is *true* victory points, including hidden VP cards.** `info["public_scores"]`
  sits next to it and is the safe one; subtracting gives an opponent's exact hidden VP count.
  The existing leak tests point at the observation and the view, not at `info`, so nothing
  catches it ([audit](audit-2026-08-05-public-arena.md#2-blockers-verified-by-execution)).
- **Every performance constant written down in this repo is 2.4–3.7× optimistic** — `arena.py:5`
  (56 ms / 5.2 s), `alphazero/agent.py:24` (25 ms), `api.py:60` (32 ms), `benchmark.py:13`
  (36,000 actions/sec). Re-measure rather than quoting them; a plan built on those
  under-provisions by about 3×
  ([audit](audit-2026-08-05-public-arena.md#6-measurements)).
- **The first-player advantage is zero here, not "real and large".** 49.8% [48.7, 50.9] over
  8,000 heuristic mirror games, because `setup_sequence` is a snake `[1,2,2,1]` and seat 2 gets
  picks 2 and 3. Several docstrings still repeat the 4-player folklore. Seat swapping is cheap
  insurance, not a mandatory variance reducer
  ([audit](audit-2026-08-05-public-arena.md#first-player-advantage-is-zero-in-this-engine)).
- **Position value is marginal, not absolute.** `heuristics.settlement_value` divides each
  tile's rate by what the player already produces of that resource. Do not "simplify" it back
  to a sum of pips: that is the encoder's `pip_potential` feature, which answers a different
  question ([0016](decisions/0016-heuristic-opponent-and-difficulty.md)).
- **`VERTEX_TILES` and `TILE_VERTICES` are both keyed by plain integers**, so swapping them
  type-checks, runs, and silently produces nonsense. It happened once, in `robber_damage`.
  When indexing either, the variable name should say which id it holds.
- **`step()`'s reward is always `+1`, and the loser never receives theirs.** The winner is
  always the player who just acted, so `LOSS_REWARD` is unreachable on the normal path. A
  learner that consumed the returned reward would train on winners only, its critic would
  converge to `V = 1`, and nothing would crash. Read `info["winner"]` and write both seats
  ([0017](decisions/0017-ppo-self-play.md)).
- **At `GAME_OVER`, `current_player` becomes the winner**, so the final observation is the
  *winner's* view and the final mask is all-zero. Bootstrapping the loser's trajectory from
  that observation is an exact sign flip on half the data.
- **~40% of an observation never changes during a game.** It is cached on the `Board` and
  copied. Do not add a board-static feature to the per-encode path — put it in
  `_static_template` ([0019](decisions/0019-cache-the-board-static-observation.md)).
- **12.3% of decisions have exactly one legal action.** No policy gradient exists for them,
  so training skips them — but a forced action can still end the game, and its outcome must
  land on the seat's previous stored transition.
- **Applying an action to a `clone()` can reveal hidden information.** The dev deck, the dice
  deck and opponents' hands are copied verbatim, so a lookahead over `BUY_DEV_CARD`,
  `MOVE_ROBBER`, `PLAY_KNIGHT`, `END_TURN` or `PLAY_MONOPOLY` *sees the real outcome*.
  `training.agent.DETERMINISTIC_TYPES` is the boundary and is a correctness constraint, not a
  performance one ([0020](decisions/0020-parallel-rollouts-and-lookahead.md)).
- **Benchmark a persistent collector with warm-up calls.** Transitions are banked in bursts
  when games finish, so a short measurement measures luck — the first parallel benchmark
  reported 4 workers as faster than 8 for exactly that reason.
- **A repr must never raise.** `Action.__repr__` appears inside the `IllegalAction` message
  raised *because* an action is malformed. It crashed there twice — once on a bad action
  type, once on a bad resource index — replacing a clear error with a confusing one.
  `test_the_repr_of_a_malformed_action_never_raises` pins it.

## Deviations from official Catan

Kept together so they are not mistaken for bugs:

- **[0004](decisions/0004-desert-as-the-seven-tile.md)** — the desert carries a 7 token.
  Equivalent for production; official Catan gives it no token.
- **[0005](decisions/0005-balanced-board-generation.md)** — balanced generation forbids
  equal adjacent numbers and 6/8, 2/12 pairs. Official Catan uses a fixed spiral. This has
  a real strategic consequence: **double-production vertices cannot occur**, so the "double
  6" spot does not exist on this board.

## Conventions

- The ids in [`../Images/`](../Images) are the contract for board numbering.
  `tests/test_topology.py` transcribes them independently and asserts the generated tables
  match, so the drawings and the code cannot diverge silently.
- Tests are the enforcement mechanism for decisions. Each record lists the tests that pin
  it; changing the behaviour means changing a named test, deliberately.
