# Working notes

Things that are true about this repository and cost time to discover. Read this before
changing the engine, the observation, or the agents.

`docs/decisions/` holds the full reasoning; this is the short version plus the traps.

---

## Ground rules this codebase actually holds to

**One source of truth.** If a fact exists in two places it will diverge. Geometry is
generated from `ROW_LENGTHS`, not written down. The browser draws what the server sends and
decides nothing. The web client is told which image file goes with which resource, because
the art set calls wheat `weat` and ore `stone` and a second copy of that mapping in
JavaScript would be a bug waiting to happen. The resource *cards* under
`static/images/ressources/` are a second, separate set — a tile is a forest, a card is one
tree. They are named after the resource, so that mapping is an identity; it is served anyway,
together with its shape in `art.aspects`, because a portrait card sized by a ratio guessed in
JavaScript is a stretched card that reads as bad art rather than as a bug. The robber is
served the same way, and its `scales`/`offsets`/`aspects` entries are one decision rather than
three: together they are what keeps it standing *above* the number token instead of over it,
which is the one thing a player needs from a blocked tile.

**Hidden information is impossible, not merely absent.** `PublicView` is an allow-list — a
new field on `GameState` is invisible to agents until someone adds it deliberately. Anything
that could leak has a test that mutates the hidden state at constant public counts and demands
the observable output not move. `tests/helpers.py::scramble_hidden_state` is the shared
scrambler; use it rather than writing a weaker one.

**Measure before claiming.** Several "obvious" improvements in this project measured as no
better, and one measured as the opposite. Numbers or it did not happen.

**Confidence intervals, not point estimates.** 200 games gives about ±7 points, which cannot
tell 45% from 55%. `training/evaluate.py` has a Wilson interval; use it. Several results have
flipped meaning between 200 and 800 games.

⚠️ **Picking the best of N candidates on one sample and then quoting that sample is
winner's curse, and it has bitten here twice.** A checkpoint chosen as the best of six at 200
games measured 83.9% and then 80.1% on a fresh 400 — the entire apparent gain over the
champion was the selection. `training/alphazero/arena.py::rank` is for *ordering* candidates;
the number that gets written down has to come from a fresh match, which is what the promotion
gate is. Never promote on the ranking's own figure.

---

## Engine invariants the training code depends on

All three are undocumented properties of `CatanEnv`, all three are load-bearing, and all three
fail *silently*. Pinned in `tests/test_training.py`.

1. **The winner is always the player who just acted.** So `step()`'s reward is always `+1`
   and `LOSS_REWARD` is unreachable on the normal path. A learner that consumed the returned
   reward trains on winners only, its critic converges to `V = 1`, every advantage collapses
   to zero, and nothing crashes. Read `info["winner"]` and write both seats.
2. **At `GAME_OVER`, `current_player` becomes the winner**, so the terminal observation is
   the *winner's* view. Bootstrapping the loser's trajectory from it is an exact sign flip on
   half the data.
3. **The terminal mask is all-zero.** A masked softmax over nothing.

Also: the game is **not alternating** — during a discard the decision belongs to whoever is
over the hand limit. `info["player"]` is the authority.

---

## The observation

2,503 floats. `LAYOUT` maps a name to its slice; `VERTEX_OFFSETS` names the fields inside a
vertex row, because counting to them by hand is how a test broke.

```
tiles         19 x 19   resource, number, odds, robber
vertices      54 x 27   owner, city, harbour, pip potential (retired, 0.0),
                        buildability, per-resource production, harbour nearness
roads         72 x  6   owner, buildability
players        4 x 34   hands and holdings (masked for opponents), production rate
affordability  4 x  4   my hand against each purchase, priced through my trade rates
history        4 x 12   production, spending, purchases, idleness  (the public record)
rolls             12   how often each total has come up
global            40   phase, last roll, bank, ruleset, bookkeeping, board scarcity
```

**About 40% of it never changes during a game** and is cached on the `Board`
(`encoder._static_template`). If you add a board-static feature, put it there — the per-encode
path was 57% of training time before this existed. A feature that depends on a *hand* cannot
go there: the `Board` is shared across every clone and every observer, so a cached per-observer
value is both stale and a leak.

**Two entry points, one body.** `encode()` returns a list of Python floats and is what every
interface, test and PPO call site uses — 13 tests assert `isinstance(v, float)`, so it must
stay that way. `encode_into(state, me, buffer)` writes the same numbers into an `array('f')`
from `observation_buffer()`, which numpy takes as a buffer instead of unboxing 2,503 Python
floats. The search uses the second; everything else uses the first, and
`test_encode_into_matches_the_list_encoding_bit_for_bit` is what stops them drifting. See
record 0027.

⚠️ **`encoder._survey` is a second implementation of scoring**, and the one-source-of-truth
rule would otherwise forbid it: it answers what `victory_points`, `public_victory_points`,
`production_rates` and `trade_rates` answer, in one walk over the board instead of nine.
`catan.rules` is still the authority. What makes this legal is
`test_fused_player_scores_agree_with_the_rules`, which drives whole games at 2/3/4 players in
both rulesets and compares at *every* position. If you change scoring, that test is what will
tell you the encoder disagrees.

⚠️ **`interfaces/web/stats.py::owed` is a second implementation of the production walk**,
written in that same idiom and legal for the same reason. `catan.rules.distribute` skips a
tile under the robber with a bare `continue` and the amount is gone — nothing returns it and
no event reports it — so the end-of-game panel's "what the robber cost you" is the one figure
there that has to be *computed*, by walking the producers with and without the blocked tile
and differencing. `catan.rules` is still the authority. What makes it legal is
`tests/test_web_stats.py::test_what_the_robber_blocked_agrees_with_the_rules`, which drives
whole games and compares `owed` against what `distribute` actually pays, for every total 2-12
at every position — against a full bank, because `owed` is deliberately the payout *before*
the shortage rule. If you change `distribute`, that test is what will tell you the panel
disagrees.

**Encoding a constant is worth nothing.** The cost table is identical in every state, so it
folds into a bias in one gradient step. What the `affordability` block encodes is the
*state-dependent* part: cards short, and what closing the gap would cost at the bank given my
own harbours. See `docs/decisions/0022-affordability-features.md`, which is also where the
argument for four columns rather than twenty lives.

**What is NOT in it, and people assume is:**
- ~~**Which numbers a vertex touches.**~~ Fixed in record 0024: a vertex now carries its
  expected cards *per resource*, and how near the closest harbour of each kind is. The old
  resource-blind `pip potential` slot is **still in the vertex row and now carries a constant
  0.0** on all 54 vertices — `encoder.PIP_POTENTIAL` is `False` and the one write site is the
  static template. The slot stays because deleting it would move every offset after it inside
  a vertex row and change `SIZE`: the vertices block would have *shrunk*, and
  `layouts.column_map` reconciles only blocks that grew — it raises rather than invent a
  column correspondence that does not exist — so no existing checkpoint could be grafted onto
  the result and `models/champion_az.pt` would stop loading, which leaves the AlphaZero gate
  with nothing to measure against. `layouts.HISTORICAL` is **not** what stands in the way: it
  is keyed 1868 and 1884 only, and a checkpoint at the current `SIZE` carries its own layout,
  so the table is never consulted for one. Writing 0.0 removes the *information* at no
  compatibility cost and leaves `SIZE` at 2,503. **Restoring it is not one line** —
  `tests/test_encoder.py::test_pip_potential_is_retired_but_its_slot_is_still_there` asserts
  the flag is off, so shipping it flipped is a two-file change. The arithmetic is kept and
  tested by `test_pip_potential_still_sums_the_adjacent_odds_when_switched_back_on`, so
  *trying* it is an afternoon rather than archaeology. It was the only placement signal the
  observation had and the agent maximised it essentially perfectly —
  0.005 pips off the best available spot — which is what a resource-blind signal gets you;
  the per-resource production record 0024 added supersedes it, so this retires the guide that
  came first rather than leaving it to be unlearned. Record 0029.
- **Adjacency**, for the flat MLP. It must infer that vertex 23 neighbours 24 from
  correlations, though `topology.py` knows.
- **Any estimate of an opponent's hand.** Deliberate — a robber steal moves a card only two
  players ever see. The `history` block gives cumulative production and spending per player,
  which bounds it; deriving the bound is left to the network.

Changing `encoder.SIZE` no longer invalidates a checkpoint, as long as the change **appends**:
`training/alphazero/layouts.py` records the block shapes, new checkpoints carry their own, and
`network.graft` widens every observation-width layer with zero columns. Verified rather than
assumed — the champion *of the time* measured 74.7% against the heuristic before the
1884 → 2503 change and 75.6% after being grafted (record 0024); several promotions have
happened since, so neither figure describes the reigning champion. You do **not** need to add
an entry to `layouts.HISTORICAL` for a new size — every checkpoint written since
`layouts.signature` existed carries its own block shapes, and the table is consulted only for
the two that predate it (1868 and 1884). **Never edit an existing entry**: it is a statement
about a file already on disk. The interfaces check `obs_size` *and*
`num_actions` before offering a model, because a stale one loads fine and then fails on the
first move.

---

## The action space

325 flat indices. **Append, never insert** — every existing index keeps its meaning, and the
structured network's final concatenation slices `other[:, 21:]` open-ended so an appended
action lands correctly with no code change.

The positional blocks are **not contiguous**: `TRADE_WITH_BANK` sits between the city and
robber blocks. `training/structured_net.py::_validate` checks this at import; it has caught
two mistakes already, including one of its own.

---

## Traps

**`VERTEX_TILES` and `TILE_VERTICES` are both keyed by plain integers.** Swapping them
type-checks, runs, and silently produces nonsense. It happened in `robber_damage`. Name the
variable after the id it holds.

**Applying an action to a `clone()` can reveal hidden information.** The dev deck, dice deck
and opponents' hands are copied verbatim, so a lookahead over `BUY_DEV_CARD`, `MOVE_ROBBER`,
`PLAY_KNIGHT`, `END_TURN` or `PLAY_MONOPOLY` sees the *real* outcome.
`training.agent.DETERMINISTIC_TYPES` is a correctness boundary, not an optimisation. The
AlphaZero package answers the same problem the other way — `training/alphazero/determinize.py`
resamples everything hidden from what is public, so its search may go as deep as it likes.
**Anything that searches must go through one of those two doors.**

**`gumbel` and `root_min_visits` are mutually exclusive, and nothing says so.**
`Search._descend` takes the Gumbel root when `gumbel` is set and only reaches the floor in the
`elif`, and `best_action`/`policy_target` skip `_discretionary_counts` on that path too. So a
config carrying both `gumbel: true` and `setup_root_min_visits: 4` is accepted, runs, and
searches the opening exactly as it did before the floor existed. The Gumbel root is off by
default — record 0026 measured it worse here — so this only bites whoever turns it back on.

**`models/champion.pt` has not loaded since the affordability block landed.** It was promoted
at `encoder.SIZE == 1868`; the encoder is now 1884, so `training.champion.load()` returns
`None` and both interfaces silently fell back to the heuristic for a while. Nothing raised.
`training.alphazero.network.graft` reconciles it — the only wrong-shaped tensor is
`context_mlp.0.weight`, and inserting the 16 new columns as **zeros** reproduces the original
function exactly — and the interfaces now offer the grafted model. Check `champion.load()` is
not `None` after any observation change.

⚠️ **`models/champion.json` says 71.6% against the heuristic; it measures 49.3% today**
(150 games, [41.4, 57.3]). Not a broken graft — that was checked three ways. Commit `e4b0441`
restricted pre-roll development-card plays to the Knight *after* the champion was promoted, so
the recorded number belongs to a slightly different game. This is the rule below about the
fixed yardstick, biting for real: **a `beat_heuristic` figure is only comparable within one
version of the rules.** Re-measure before trusting any number in a champion record.

**Benchmark persistent collectors with warm-up calls.** Transitions bank in bursts when games
finish, so a short measurement measures luck. An early benchmark reported 4 workers as faster
than 8 for exactly this reason.

⚠️ **A sequential A/B cannot be run on this machine.** Throughput decays monotonically under
sustained load — measured at 594.9 → 422.5 → 425.6 → 351.2 → 353.2 decisions/sec across five
minutes — so whichever arm runs *second* loses, whatever it contains. A round-per-variant
design returned median 1.0013x with range [0.843, 1.170] on a change independently measured at
1.054x. **Alternate the arms, alternate the order within each pair, and report the median of
adjacent-pair ratios**, or do not write the number down. Record 0027.

**A searched decision is 32% observation, 19% network, 16% legal-move generation.** Record 0025
split it 80.9% engine / 19.1% network, which is right but hides that "engine" is two functions:
`encoder.encode` plus its float32 conversion, and `rules.legal_actions` at 173 calls per
decision. Record 0027 has the full table and took ~1.2x out of it. What is left is
`legal_actions`, and the remaining win there needs incremental legality, which is a project.

**Windows spawns rather than forks.** Worker functions must be module-level; a script run from
a heredoc cannot be a multiprocessing parent because the child cannot re-import `__main__`.
Torch sizes its OpenMP pool at import, so thread env vars must be set in the *parent* before
the pool is created.

**A `\b` inside a JavaScript template literal is a backspace**, not a word boundary. The
resulting pattern matches nothing while reading as though it works. There is a test for stray
control characters in `app.js`.

**`python -m` with a heredoc and a pipe buffers output.** Use `python -u`, and read
`checkpoints/metrics.jsonl` for training progress rather than the piped stdout. `| tail` is
worse than buffering — it prints *nothing* until the process exits, so a three-hour run looks
hung. Redirect to a file and read the file.

**Self-play samples bank in cohorts, not continuously.** A worker's `envs_per_worker` games
advance in lockstep — one simulation each per round — so they also *finish* together. Nothing
comes back for the first ~30 s of worker time and then several thousand positions arrive at
once. Two consequences, both measured:

* a `generate(positions=N)` call takes a wildly variable time (5 s for one worker, 46 s for
  another on the same request), and `pool.map` waits for the slowest, so most of the pool
  idles. **Time-box the slice instead** — `generate(seconds=...)`. Count-based sharing measured
  172 positions/sec across 14 workers where the clock-based one gives ~400.
* `envs_per_worker` is a latency knob as much as a batch-size knob.

---

## Agents and training

**The heuristic is the behaviour-cloning teacher**, so its judgement is where the trained
policy starts. Improving it raises the ceiling for both. It is also the fixed yardstick — if
you change it, *win rates recorded against it before and after are not comparable*. Say so
when reporting.

**Resource weights are tuned for 2-player 15-point play and invert 4-player folklore.**
Missing brick is the worst opening deficiency (36% win rate), missing wheat nearly free (49%).
With no player trading, a missing resource costs 4:1 at the bank, so expansion gates the game.
Do not "fix" these back toward wheat-and-ore without 1v1 evidence.

**Fine-tuning from a clone is not training from scratch.** Use `lr 3e-5`–`1e-4` and
`entropy 5e-3`. The defaults will destroy what cloning learned before the outcome signal
replaces it.

**Warm-started runs look like failures for their first ~100 iterations.** One went
27.8 → 21.9 → 31.6 → 31.1 before climbing to 77.6%. Another started *below* its predecessor
and overtook at iteration 79. Judged at iteration 59 both would have been abandoned. Check
whether the intervals overlap before concluding anything.

**Two lineages, two champions.** `models/champion.pt` is PPO; `models/champion_az.pt` is
AlphaZero. They are separate files on purpose and the interfaces offer both. Do not train one
into the other's file.

**The AlphaZero package is where search lives, and its shape follows from the game.** Dice are
chance nodes keyed by roll *total*, resampled per visit. Values propagate in seat 1's frame
rather than flipping by depth, because the game is not alternating — during a discard the mover
may be either player. `Search` refuses `num_players != 2` for that reason. Forced moves
(≈30% of decisions) are collapsed during descent and never recorded: a one-hot policy target
teaches nothing and costs a network evaluation.

**The opening root is 54 moves wide, and PUCT was tuned for six.** A first settlement offers
54 legal spots — exactly 54 on 40 boards out of 40 — against a normal turn's six, and PUCT
concentrates, which is right at six and wrong at fifty-four. Measured on the reigning champion
at the `noise=0` that play and evaluation use, 40 boards: a 400-simulation search examines a
mean of **5.9 of the 54**, 4.2 at 96 simulations, 8.2 at 1,600 (24 boards there). The
distribution has a long right tail — one board in forty examines all 54 — so the medians are
the honest summary: 5, 3 and 7. Budget buys breadth
very slowly, so the opening was being chosen from about three spots the prior already liked.
`Search(root_min_visits=4)` gives every spot four visits before PUCT may concentrate and
reaches **all 54, on 40 boards out of 40**, at each of the four placements of an opening, for
the same cost — 0.319 s against plain PUCT's 0.335 s at 400 simulations on one thread, because
the forced sweep builds a shallower tree. `MCTSAgent` applies both at `SETUP_SETTLEMENT`, so
every agent that searches now plays its opening this way; self-play also exempts setup from
the playout cap and records every placement, road included, since there are only two
settlements per player and the cap was keeping a fraction of them. Record 0028.

⚠️ **The forced sweep is measurement, not preference, and must reach neither the move played
nor the label.** `Search._discretionary_counts` subtracts the floor from the root counts before
`best_action` and `policy_target` read them — **read that docstring before changing either.**
Without the subtraction the 4 x 54 = 216 forced visits swamp a best spot's handful of chosen
ones, and sampling the raw counts at temperature 1.0 picks an arbitrary spot. What the floor
buys is breadth in what the search *looks at*: after the subtraction the recorded target is
still concentrated on a mean of 3.5 spots (median 3, range 1-9; 400 simulations, floor 4, 40
boards), about the same handful plain PUCT would have examined. The margin is thinner than it
looks. 400 simulations over 54 spots is 7.4 visits each, so a floor of 8 never finishes its
sweep — 8 x 54 = 432 against the root's 399 visits — and the discretionary counts come back
identically zero on 40 boards out of 40. That case used to return the raw near-uniform counts,
which put the failure the subtraction exists to prevent straight back; it now returns zeros, so
both callers reach the branch they already had and play the network's prior instead of the
lowest vertex id. Sane rather than arbitrary — but a floor that will not fit its budget still
records no preference at all, and nothing raises.

**The AlphaZero run is warm-started, and that is a choice, not a default.** At the simulation
counts a CPU affords, MCTS is a modest improvement over its prior, so starting from a policy
that already plays is what makes a few hours worth anything. `--cold` does it the guide's way.
Say which one produced a number.

⚠️ **A warm start builds the *checkpoint's* shape, not `new_network`'s.** `load_for_alphazero`
reads `checkpoint["config"]`, which is exactly right for an observation change — `graft` widens
the affected layers and every weight keeps its meaning — and useless for a `width`, `depth` or
`trunk` change, where no column correspondence exists. So changing `new_network`'s defaults
while `warm_start` points at an older checkpoint **does nothing**. That is not hypothetical: it
is what happened when the defaults moved to 374,331 parameters — runs kept training the
reigning 200,379-parameter shape with `aux=False`, so the auxiliary targets self-play was
computing were dropped by the loss, and nothing raised. `build_network` now prints every key
that differs. To actually change shape: `training/alphazero/distil.py`, or `--cold`.

⚠️ **The AlphaZero loop's win-rate column measures the raw policy, and the champion plays with
search. They move in opposite directions — twice measured, in both directions.** In the run on
the 2,503-float observation the policy score went 62.5 → 60.4 → 50.5 → 51.0 and read as "peaked
at iteration 20". Ranked *with search* on identical games, the order was almost exactly
reversed: iteration 95 best at 74.7%, iteration 20 **worst** at 69.0%. A declining policy
column is not evidence a run has peaked. The earlier measurement: On identical games, between two checkpoints of one
run: raw policy 47.4% → 43.8%, the same weights *with 32 simulations* 57.1% → **60.8%**. The
value head is what search leans on hardest and it was still improving while the policy's
argmax drifted. So `checkpoints/alphazero/best.pt` is best-*policy*, not best-*player* —
re-measure candidates with `training/alphazero/arena.py` before promoting, and never read the
in-loop curve as the run's progress.

⚠️ **When the policy will not learn, measure the labels — not the loss curve.** A run at 48
simulations with AlphaZero's noise of 0.25 sat flat for 60 iterations. Everything looked
plausible and everything *was* correct. The answer came from asking how often the recorded
target picks the same move as a clean 400-simulation search:

```
 48 sims, noise 0.25   62%      <- what it was training on
 48 sims, noise 0.10   68%
 96 sims, noise 0.10   78%      <- now the default
160 sims, noise 0.10   80%
raw policy argmax      58%      <- what it already knew
```

Four points of signal. Root noise alone flipped 24% of the top moves — 0.25 is calibrated for
800 simulations, where it barely perturbs the visit counts, and is destructive at 48. Loss
curves and win rates cannot tell "learning slowly" from "learning from nothing"; this
measurement takes three minutes and does.

**The champion is not the newest model.** A champion changes only through `promote`, and the
gate has already refused a completed run. Never copy a checkpoint into `models/` by hand.

**The two lineages gate differently, and on purpose.**

* `training.alphazero.champion promote` — **one rung decides: the reigning champion.** Wilson
  lower bound over 400 games above 50%, and nothing else can refuse. Two matches are *played*
  by default, though, and the distinction matters when you are budgeting time: `ppo_games`
  defaults to `games`, so the PPO champion is played whenever one loads, and that figure is
  recorded and cannot veto. The heuristic is the rung that is off by default —
  `baseline_games` defaults to 0, and `--baseline-games` plays it if asked and *records* the
  number, but it cannot refuse either. `training/alphazero/chain.py` passes `--ppo-games 0`,
  so the overnight chain plays exactly one match; that is the chain's choice, not the gate's
  default. The tripwire argument is still true — self-play is non-transitive, so a candidate
  can climb the ladder by learning the champion's habits while getting worse at the game — and
  the veto was removed anyway, so that "promoted" means exactly one measurable thing.
  `tests/test_alphazero.py::test_the_heuristic_cannot_veto_a_candidate_that_beat_the_champion`
  says so, so that a revert is a decision rather than a drift.
* `training.champion promote` — the PPO lineage, unchanged: Wilson bound **and** no
  regression against the fixed heuristic.

⚠️ **With no loadable champion the AlphaZero gate has nothing to measure against, so it
refuses.** The heuristic used to fill that hole; it no longer does. This fires exactly when
`encoder.SIZE` has changed, because the reigning champion stops loading — and CLAUDE.md used
to record the *opposite* failure here, `promote` installing immediately with no Wilson bound
at all. Installing the first champion of a lineage is now an explicit `--force --reason`,
which the record carries forever as `forced`. Measure by hand first and put the number in the
reason. **The PPO gate still has the old hole** — `reigning is None` there still installs
immediately.

---

## What has been tried and did not work

Recorded so it is not re-attempted.

- **1-ply lookahead with the value head.** Leak-safe, correct, 53.4% vs 52.2% over 800 games.
  Probably because the trunk is shared, so `V(s')` restates what the policy already encodes.
- **The opening bug fix alone.** Improved every structural measure of the opening and won no
  more games, because the road threshold prevented the agent from acting on it.
- **Gather-and-pool over embeddings** for neighbour aggregation: 3.8 ms at batch 512 for one
  relation, where the whole flat trunk cost 6.6 ms. Constant incidence matrices on raw
  features do the same job in 0.33 ms.
- **Behaviour cloning on a noisy teacher's actual moves.** Caps achievable agreement at the
  71.2% the noisy teacher shares with the noiseless one. Play noisy, label clean.

---

## Where to look

| | |
|---|---|
| Why something is the way it is | `docs/decisions/` — 30 records |
| What is done and what is next | `docs/website-split-plan.md` |
| Whether a change helped | `training/evaluate.py`, and use enough games |
| How fast anything is | `python -m benchmark.benchmark`, and warm up first |
| How a training run went | `python -m training.alphazero.report` |
| Where the time goes | `python -m benchmark.profiler selfplay` |
| What the bot did in a real game | `python -m interfaces.web.recorder --margin 5` |

Run the full suite before committing: `python -m pytest tests -q` — 936 passed, 1 skipped in
375 s on a quiet machine, so budget **about six minutes**. `-m "not slow"` does not buy that
back: it deselects 35 tests and still takes 354 s, because the cost is the web tests playing
whole games rather than the fuzzing.

Five tests are timing-based, and two of them assert an absolute figure close enough to what
the code actually measures to flake under load — re-run them alone before believing a failure,
**and check `tasklist | grep python` first, because "alone" is not "unloaded"**.
`test_sharing_the_stream_makes_cloning_much_cheaper` asserts an absolute `< 10 us` and read
15.8 µs against a live training run, where the same unchanged code measures 4.4 µs on a quiet
machine; `test_constructing_one_is_cheap` asserts `< 5 us` on the same kind of margin. The
thresholds are right; the box was busy. The other three — a clone under 100 µs, an encode
under 5 ms, the mask within 1.5x of `legal_actions` — have room to spare, so a failure there
is a regression rather than a busy machine.
