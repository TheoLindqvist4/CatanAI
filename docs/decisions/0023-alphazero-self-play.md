# 23. AlphaZero-style self-play, beside the PPO lineage

Date: 2026-08-04
Status: accepted

## Context

`docs/catan_alphazero_implementation_guide.md` specifies an AlphaZero pipeline: self-play
workers, a replay buffer, a policy/value network, MCTS, an evaluator, and gated promotion,
running as one continuous loop. The repository already has a working PPO self-play trainer
(record 0017), a structure-aware network (0021), a leak-proof observation (0015, 0022) and a
promotion gate (`training/champion.py`).

The guide is written for a generic Catan repository with a GPU. This project is a specific
Catan repository with 20 CPU cores and no GPU, and its engine, observation and action space
are already settled and heavily tested. So the question was never "implement the guide" but
"which parts of the guide are load-bearing here, and what replaces the rest".

Everything below is a decision that could reasonably have gone the other way. They are written
down so they can be revisited without re-deriving them.

---

## Decisions

### D1 — Keep `catan/`; do not adopt the guide's directory names

The guide's target tree is `engine/`, `environment/`, `models/`, `training/`, `benchmark/`,
`configs/`. This project has `catan/` doing the first two — `board.py`, `rules.py`,
`actions.py`, `state.py` are the engine, `env.py`, `encoder.py`, `action_space.py` are the
environment — and it already satisfies the guide's architectural rules: the engine contains no
AI code, training talks to the engine only through `CatanEnv`, and the network is replaceable
without touching the engine.

Renaming would have broken every import, every test, both interfaces and every checkpoint, to
satisfy a naming convention. `benchmark/` and `configs/` did not exist and were added as the
guide names them.

**Revisit if** the project ever grows a second game, at which point the `engine` /
`environment` split earns its keep.

### D2 — `models/` stays a data directory

The guide has `models/network.py` and `models/checkpoints/`. Here `models/` is where the
*playable champion file* lives, and the interfaces read it. Turning it into a Python package
would put a `models/__init__.py` next to `champion.pt` and invite exactly the confusion the
`checkpoints/` vs `models/` split exists to prevent. Network code lives in
`training/alphazero/network.py`; run checkpoints live in `checkpoints/alphazero/`.

### D3 — A subpackage, `training/alphazero/`, not files alongside the PPO trainer

The guide puts `self_play.py`, `replay_buffer.py`, `trainer.py`, `evaluator.py` directly in
`training/`. That directory already holds `train.py`, `ppo.py`, `rollout.py`, `parallel.py`,
`agent.py`, `champion.py` and `evaluate.py` for the PPO lineage. Mixing the two would give
`evaluate.py` and `evaluator.py` in one directory, which is a name collision waiting to be
imported wrongly, and would make "which technique does this file belong to" unanswerable.

### D4 — `encoder.SIZE` is not changed

`CLAUDE.md` records that changing it invalidates every checkpoint *and* that the first
promotion after such a change is ungated by construction. The AlphaZero network therefore
reads exactly the observation the PPO one does. This also means the two lineages can play each
other, which is what makes D10's ladder possible.

**Cost:** the largest known gap in the observation — a vertex does not know *which numbers* it
touches, only an aggregate pip potential — is inherited rather than fixed. Search partly
compensates by looking at what actually happens after a roll.

### D5 — The network is the existing structured net, with `tanh` on the value

The guide recommends a GNN "because Catan is naturally a graph". `StructuredPolicyValueNet` is
already that: 19 tiles, 54 vertices and 72 roads embedded by shared per-entity MLPs, with
neighbourhood aggregation through constant incidence matrices from `catan.topology`. Record
0021 measured the gather-and-pool formulation of the same idea at 3.8 ms against 0.33 ms for
the matrix form, so the "unusual" implementation is the fast one, not a compromise.

The one change is `value_activation="tanh"`, added as an option defaulting to `"linear"` so
every existing PPO checkpoint rebuilds identically. AlphaZero's target is exactly ±1 and its
loss is a squared error; an unbounded head spends its first epochs learning the range.

### D6 — Search runs on a determinized information set. **This is the correctness boundary.**

Textbook AlphaZero assumes perfect information. Catan does not have it, and `GameState.clone`
copies the development deck, the dice deck and opponents' hands *verbatim* — so a tree built
on a clone is a tree that has read the opponent's hand. `training.agent.DETERMINISTIC_TYPES`
solves this for the one-ply lookahead agent by refusing to search anything stochastic, which
caps that agent at one ply forever.

`training/alphazero/determinize.py` solves it the other way: resample everything hidden from
what is public, then search the resulting world freely.

- **Resources** — cards are conserved, so the number of resource *r* held by everyone else is
  `BANK_PER_RESOURCE - bank[r] - my_hand[r]`, both terms public. Dealt out by hand *size*,
  also public.
- **Development cards** — each opponent keeps their card *count*; identities are redrawn from
  `DECK_COUNTS` minus my holding minus everyone's visibly played knights.
- **Dice deck** — length public (the reshuffle rule is printed), contents not, so a fresh
  shuffle of the same length.

Because every input is public or mine, scrambling the hidden state at constant public counts
leaves the determinized world *identical*. `tests/test_alphazero.py::test_determinize_ignores_hidden_state`
asserts that as an equality, using `tests/helpers.py::scramble_hidden_state` — the same
scrambler the encoder and the heuristic are held to.

**Known approximation:** Road Building, Year of Plenty and Monopoly plays are public but the
engine keeps no per-player tally, so they are not subtracted from the unseen pool. The pool is
therefore up to four cards larger than a perfect card counter's. That errs toward the searcher
knowing *less*, which is the safe direction.

**One particle per search, not an average over many.** Averaging over *k* determinizations is
the textbook improvement and costs *k* times the budget. At 48 simulations a move there is not
enough budget to divide. The policy target is averaged over games and positions anyway.

**Revisit if** simulation counts rise above ~200, where root-parallel determinization starts
to pay.

### D7 — Dice are chance nodes, sampled per visit and keyed by total

A node that can only be resolved by rolling is a chance node; descending samples a roll and
steps into the child for that *total* (5+2 and 3+4 are the same game). Revisiting re-samples,
so children are visited in proportion to the dice distribution — correct in expectation, and
far cheaper than expanding all eleven totals at every roll.

**Cost:** a child's stored state was built from whichever sample created it, so under balanced
dice its deck differs slightly from a later sample of the same total. Irrelevant, because the
deck was reshuffled by D6 anyway.

### D8 — Values propagate in seat 1's frame, and two players are required

Catan is not alternating: during a discard the decision belongs to whoever is over the hand
limit, and after a 7 the roller acts twice in a row. Flipping the sign by depth — the usual
AlphaZero shortcut — is therefore wrong here. Every node records which seat acts and values are
carried in a fixed frame. `Search` raises on `num_players != 2`, because a zero-sum scalar has
no meaning with three.

### D9 — Forced moves are collapsed, and never recorded

About 30% of Catan decisions have exactly one legal action. Searching them buys a one-hot
policy target that teaches nothing, at the cost of a network evaluation. Descent applies them
immediately; self-play plays them without recording a sample.

### D10 — Two champions, and the first promotion is gated

`models/champion.pt` (PPO) is untouched. `models/champion_az.pt` is the AlphaZero lineage.
Both are offered by the web and CLI interfaces; AlphaZero is preferred as the default when it
exists.

`training/alphazero/champion.py` closes the hole `CLAUDE.md` records in the older gate: when no
champion loads, `training.champion.promote` installs *immediately* — no Wilson bound, no
regression check — and then writes the baseline every later candidate is measured against.
Here a first candidate must still beat `HeuristicAgent(0)` with its Wilson lower bound above
50%, and the record says `first_of_lineage: true`.

The ladder has three rungs rather than two: the fixed heuristic, the *PPO champion*, and the
reigning AlphaZero champion. Without the middle rung the two lineages never meet and "which
should the interface offer" has no answer.

> **Amended, 2026-08-11. The three paragraphs above describe the gate as it was built and are
> no longer true of the code.** They are kept because two champions in
> `models/champion_az.json` were promoted under them — the `first_of_lineage: true` entry of
> 2026-08-04, and the forced `iter_168` promotion later that day — and a record that quietly
> rewrote itself would make those entries unreadable.
>
> **What superseded them.** The AlphaZero gate is now **one rung: the reigning champion**.
> Wilson lower bound over 400 games, and nothing else, decides. The heuristic rung and its
> `MAX_BASELINE_REGRESSION` check are removed from `training.alphazero.champion.promote`;
> `--baseline-games` defaults to 0 and, when asked for, records the number without ever
> vetoing, as does `--ppo-games`. A first promotion of a lineage is **refused** rather than
> gated on the heuristic — with no reigning champion there is nothing to measure against —
> and installing one is an explicit `--force --reason` that the record carries forever.
>
> So the ladder is one rung, which is *fewer* than the PPO lineage's two, not more. And
> `first_of_lineage: true` is no longer written by any code path: it survives only on the
> 2026-08-04 entry that was promoted while the flag existed.
>
> The reasoning, both sides of the trade-off, and the accepted cost — `beat_heuristic` is now
> `null` for gen5 and gen6, so the lineage has no external anchor unless somebody asks for one
> — are in **[record 0030](0030-one-rung.md)**. The PPO gate in `training/champion.py` is
> unchanged, still has both checks, and still installs immediately when `reigning is None`.

### D11 — The run is warm-started from the PPO champion, via a zero-column graft

This is the largest departure from the guide, which says "learns entirely from self-play".

At 48 simulations a move — all a 20-core CPU affords, see D13 — MCTS is a *modest* improvement
over its prior. Starting from a random prior, that improvement is small in absolute terms and
the run spends its first many hours learning what the repository already knows. The PPO
champion beats the heuristic about 71% of the time. Starting there makes search policy
*improvement over something that already plays*, which is the regime AlphaZero is strong in.

**A graft was needed, and finding out why was itself a finding.** `models/champion.pt` was
promoted when `encoder.SIZE` was 1868. The affordability block (record 0022) added 16 floats at
offset 1773, so the encoder is now 1884 — and **`training.champion.load()` has been returning
`None` ever since**. Both interfaces have been silently falling back to the heuristic. That was
not known before this work.

Exactly one tensor has the wrong shape: `context_mlp.0.weight`, whose input is the
un-positional tail of the observation. The tiles/vertices/roads encoders read fixed-width rows
and affordability is not one of them. `network.graft` inserts the 16 new columns **as zeros**,
so the grafted network computes the same function as the original and the new features start
neutral. `insertion_point` derives *where* from `encoder.LAYOUT` rather than a written-down
offset, and refuses when the size difference does not match exactly one non-positional block.

The PPO value head is **not** carried over: it is an unbounded return predictor on a different
scale, and `tanh` would saturate it into a confident wrong answer. It is re-initialised small.
The policy — the part warm-starting is for — is kept in full.

**What is given up:** the run is no longer a clean test of "AlphaZero from scratch", and the
policy inherits the heuristic's biases through behaviour cloning's descendant. `--cold` runs
from scratch and the pipeline supports it.

**The warm start is weaker than its record says, and the record is not wrong.**
`models/champion.json` says 71.6% against the heuristic. Measured today over 150 games, the
grafted network scores **49.3%, interval [41.4, 57.3]**. Three things were checked before
concluding anything, because a bad graft would look exactly like this:

- `training/structured_net.py` changed in the affordability commit, which would misalign every
  weight if the *architecture* had moved. The diff is documentation and one new assertion.
- `catan/encoder.py`'s layout changed only by appending a block.
- `graft` gives the new columns zero weight, and the test drives 100.0 through the
  affordability block and asserts the logits do not move.

The graft is exact. What changed is the **game**: commit `e4b0441` restricted pre-roll
development-card plays to the Knight alone, after the champion was promoted. `CLAUDE.md`
already states the rule this falls under — change the fixed opponent or the rules, and win
rates recorded before and after are not comparable. 71.6% was true of a slightly different
game.

So the AlphaZero run starts from a coin flip against the heuristic, not from 71.6%, and its
promotion gate has to earn about +8 points from there. That bar is left where it is.

### D12 — Flat-YAML parsing instead of a PyYAML dependency

`configs/train.yaml` exists as the guide asks. The guide's example configuration is twenty
`key: value` lines, and PyYAML is not currently a dependency. `config.parse` handles that
subset and **rejects** anything more with the offending line, rather than half-parsing. `Config`
refuses unknown keys, so `mtcs_simulations` is an error and not a silently ignored setting.

### D13 — Numbers that differ from the guide's recommended configuration

| Setting | Guide | Here | Why |
|---|---:|---:|---|
| `mcts_simulations` | 200 | 48 | Measured: 60 positions/sec/worker at 48 sims. At 200 the whole machine produces ~4 games/sec. |
| `self_play_workers` | 16 | 14 | 20 cores, less the parent's gradient step and headroom so the web interface stays responsive during training. |
| `replay_buffer_size` | 2,000,000 | 220,000 | 2 M × 1,884 float32 is 15 GB. See D14. |
| `games_per_iteration` | 20,000 | ~180 (12,000 positions) | 20,000 games is 15 hours here. The loop must turn over faster than the budget. |
| `training_batches` | 1,000 | 250 | Kept in ratio with the smaller iteration, so the samples-per-gradient-step ratio is comparable. |
| `learning_rate` | 1e-3 | 2e-4 | Warm-started. `CLAUDE.md`: the defaults destroy what cloning learned before the outcome signal replaces it. |
| `evaluation_games` | 1,000 | 200 in-loop, 400 to promote | The Wilson bound decides, not the count. 400 gives ±5 points, which is what a 55% threshold needs. |

### D14 — Replay storage: float16 observations, sparse policy targets

Observations are stored as `float16` (halves the bill; the encoder emits counts and ratios,
nothing outside float16's exact-integer range of 2048, checked by round-tripping a real
observation). Policy targets keep the top 48 entries and renormalise — after 48 simulations
there are at most 48 non-zero visits. Masks are bit-packed to 41 bytes. Sampling is the guide's
25/25/25/25 by age band: a batch drawn uniformly from a ring being appended to at speed is
dominated by the newest games, and every position in one game shares one outcome.

### D15b — 96 simulations and 0.10 noise, because the first run's labels were nearly empty

The first run — 48 simulations, AlphaZero's noise of 0.25 — was flat for 60 iterations: win
rates of 49.3 / 55.3 / 49.7 / 43.8 against the heuristic, and 52.5% head-to-head against the
network it started from. Flat, not broken; every piece of machinery measured correct.

The question worth asking of a policy that will not learn is **how good its labels are**.
Taking a clean 400-simulation search as the reference and asking how often each setting picks
the same best move, over 50 real decisions from a live checkpoint:

| setting | agrees with the reference | edge over the policy |
|---|---:|---:|
| 48 sims, noise 0.25 — *the first run* | 62% | **+4** |
| 48 sims, noise 0.10 | 68% | +10 |
| **96 sims, noise 0.10** | **78%** | **+20** |
| 160 sims, noise 0.10 | 80% | +22 |

The raw policy already agrees 58% of the time. **So the first run was training on labels four
points better than what the network already knew** — and root noise alone flips 24% of the
top moves at 48 simulations (noised-48 agrees with clean-48 only 76% of the time). There was
nothing to learn, and enough label noise to undo what little there was.

AlphaZero's 0.25 is calibrated for 800 simulations, where a quarter of the prior mass being
replaced barely moves a distribution built from 800 visits. At 48 it is destructive.

96/0.10 buys five times the learning signal for twice the cost per label. 160 buys two more
points for another 40% of the throughput, which is not a trade worth making — better labels
matter far more than more of them when the signal is this thin, but only up to the point where
the curve flattens, and it flattens at 96.

The second run keeps the first's **network** — its policy is at parity with the warm start and
its value head is trained, which is worth more than the reset one — and discards its buffer,
because 220,000 badly-labelled positions are what needed to go.

**The general lesson:** when a policy will not learn, measure the labels before touching the
optimiser. Loss curves and win rates cannot distinguish "learning slowly" from "learning from
nothing"; a label-agreement measurement against a deeper search takes three minutes and does.

### D15a — The rising policy entropy is not a problem, and here is the measurement

Recorded because it looks alarming and cost time to resolve. Through the first 45 iterations
the policy loss and the policy's entropy both rise monotonically — 1.30 → 1.42 and 1.09 →
1.41 — while the win rate against the heuristic sits at 49.3 / 55.3 / 49.7, which at 150 games
(±8 points) is three readings of the same number. The obvious reading is a feedback loop:
soft targets at 48 simulations flatten the policy, which flattens the next round's visits.

Measured instead of assumed. The entropy of the stored **targets**, from the live checkpoint:

| | target entropy | top move's mass | actions with any visits |
|---|---:|---:|---:|
| as configured, noise 0.25 | 1.14 | 0.54 | 5.1 |
| noise 0.10 | 1.07 | 0.57 | 4.7 |
| target sharpened by ^(1/0.7) | 0.97 | 0.62 | 5.1 |

The target is **sharp**: 48 simulations put all their visits on about five moves and the best
one takes over half. That is a good learning signal, not noise, and the two obvious
interventions move it by 0.07 and 0.17 nats — not worth a mid-run change.

What is actually happening is a re-calibration. The warm start is a PPO policy, which is far
more confident than an MCTS visit distribution ever is; the entropy rise is that policy
relaxing toward its target's confidence, and it asymptotes there. The two numbers are also not
directly comparable — the logged entropy spans every *legal* action, the target only the ones
with visits.

**The lesson, not the number:** a monotonic loss trend and a flat win rate is not evidence of a
broken run. `CLAUDE.md` already says warm-started runs look like failures for their first ~100
iterations. The cheap way to tell the difference is to measure the *target*, which takes a
minute, rather than to reason about the loss curve.

### D18 — The run peaked and then drifted; the fix was the learning rate

Stage 2 ran 100 iterations. Three independent signals, taken together, say it peaked around
iteration 61:

| signal | iter 20 | 40 | 61 | 80 | 100 |
|---|---:|---:|---:|---:|---:|
| raw policy vs heuristic | 51.0% | 47.4% | 46.5% | 43.8% | **34.7%** |
| value loss | 0.149 | 0.180 | 0.168 | 0.190 | **0.230** |
| **with search**, identical games | — | 57.1%¹ | **60.8%** | — | 56.8%² |

¹ iteration 32  ² iteration 91

No one of these is conclusive — the searcher's 60.8 → 56.8 is inside a ±10 interval at 100
games. Together they are: the policy fell 16 points, which at 150 games is far outside noise,
and the value loss rose by half. The searcher held up longest because search compensates for a
drifting prior, but its prior and its value both come from the same network.

**The learning rate was above the band this repository already knows about.** `CLAUDE.md`
records 3e-5–1e-4 for fine-tuning from a warm start and warns that the defaults destroy what
cloning learned. Stage 2 ran at 2e-4. Sample reuse was also high: 100 batches of 512 against
about 3,100 fresh positions an iteration is 16 passes over each position.

Stage 3 resumes **from the iteration-61 snapshot** — not from the degraded head of the run —
at 7e-5, with 60 batches and a longer generation slice (35 s), which brings reuse to about
seven. Iteration 61 is kept regardless, so the downside of the attempt is bounded: worst case
it is the model that gets promoted.

**It worked.** Stage 3 at iteration 24, on the same 100 games as every figure above:
**63.0%** [53.2, 71.8] against the heuristic, with **zero truncations** — against the
iteration-61 snapshot's 60.8% [50.9, 69.9] with three. The raw-policy column recovered too,
37.8% → 48.3% over twenty iterations, the first time in either run it moved *up*.

**The lesson:** a warm-started run that peaks and then slides is usually taking steps too
large for the distance it has left to travel, and the cheapest evidence is that the value loss
turns around. Snapshot on a timer, not only on a "best" metric — especially when, per D17, the
"best" metric is measuring something other than the deliverable.

### D17 — The in-loop evaluation measures the policy; the gate measures the player

`eval_simulations` is 0, so the number logged each iteration is the **raw policy** playing
argmax — one forward pass a move. That is deliberate and is explained at the setting: a
searching agent evaluates the network once per simulation at batch 1, so 200 games at 32
simulations costs thirteen minutes, and every six iterations that is most of a training run.

But it means the logged number is *not* the strength of the thing that gets promoted. Measured
on one mid-run checkpoint, at 32 simulations:

| | raw policy | with search |
|---|---:|---:|
| vs the heuristic | 46.5% | **57.1%** [47.3, 66.5] |
| vs the PPO champion | — | **74.5%** [65.0, 82.1] |

Search is worth roughly ten points, which is the whole premise of the method working.

**The consequence, and it is a real one:** `best.pt` is chosen by the *policy-only* score, so
"best" means best-policy, not best-player. At the end of a run the candidates should be
re-measured **with search** and the winner submitted to the gate, rather than assuming
`best.pt` is it. Cheap now that `arena.py` exists; it was not cheap when the selection rule
was written.

**And the two do not merely differ, they move in opposite directions.** Measured on identical
games (same seed, same 100 matchups), over the same stretch of one run:

| iteration | raw policy vs heuristic | **with search** vs heuristic |
|---:|---:|---:|
| 32 | 47.4% | 57.1% [47.3, 66.5] |
| 61 | 43.8% | **60.8%** [50.9, 69.9] |

The policy's argmax got worse by 3.6 points while the player got better by 3.7. The value head
is what search leans on hardest, and it was still improving (loss 0.19 and falling) while the
policy's top choice drifted.

So the in-loop curve in `metrics.jsonl` is a *lower bound with a loose relationship to the
thing being built*. Reading it as the run's progress — which is the natural thing to do, since
it is the only number the loop prints — would have got this run abandoned twice.

**Measured a second time, and it is worse than "loosely related".** On the run that followed
the observation change of record 0024, the in-loop column read 62.5 → 60.4 → 50.5 → 51.0 and
said the run peaked at iteration 20. Every checkpoint then played 200 identical games *with*
search:

```
iteration 95 (final)   74.7%   <- the policy column's worst reading
iteration 48           72.3%
iteration 72           70.2%
iteration 24           69.7%
iteration 20           69.0%   <- the policy column's best reading, and best.pt
```

The ranking is close to **reversed**. `best.pt` was the weakest candidate of the five.
Search leans hardest on the value head, and the value head kept improving while the policy's
argmax drifted — so a falling policy column is not evidence that a run has peaked, and acting
on it would have thrown away the best model twice.

**What should change:** the loop should evaluate with a small number of simulations rather
than zero, accepting the cost, or `best.pt` should be selected on something else entirely.
Left as it is for this run because changing the selection rule mid-run is worse than working
around it; snapshots were taken instead so the end-of-run choice had real candidates.

### D16 — Promotion matches run across processes

`catan.agents.play_match` is sequential, which is right for a heuristic at 56 ms a game and
impractical for an agent that searches: 5.2 seconds a game at 32 simulations makes the guide's
400-game match half an hour *per rung*, and its recommended 1,000 games an hour and a half. A
gate that costs more than the run it gates is a gate that stops being run.

`training/alphazero/arena.py` deals the games out by index across processes. The invariant it
is held to is **the answer must not depend on how many workers ran it**, which the tests
assert as an equality between a 1-worker and a 4-worker match.

Getting there required one change worth stating plainly: agents are **stateful**. Each holds
an RNG seeded once at construction, and `play_match` lets that state carry from game to game,
so game 17 depends on the sixteen before it. That cannot be reproduced when games are dealt
across processes, so the arena **re-seeds every agent per game**. The consequence is that an
arena match is not the same *draw* as a `play_match` match — it is an independent sample of
the same quantity. Not hidden, because a faster evaluator that quietly measures something
slightly different is worse than a slow one.

Found while writing it: `def load(path=CHAMPION, ...)` captures the module constant at import,
so redirecting `champion.CHAMPION` leaves `load()` reading the real file. Now resolved at call
time. The same pattern is still in `training/champion.py`, left alone because changing the PPO
gate is not this work's business.

### D15 — Measured optimisations, and one that was left

The guide makes simulation speed priority #1. Baseline: 56 ms per random-play game, 36,000
engine steps/sec, and self-play at 53.6 positions/sec per worker. Four changes were measured
and kept, worth 13% together:

- The leaf computed legality **twice** — `legal_indices` to build the node and `legal_mask` for
  the network — at ~42 µs each. The mask is now built from the node's cached actions.
- `_select` used `np.errstate` and `np.where` on 20-element arrays: 7.5 µs a call, several
  calls per simulation. `child_q` is now maintained on backup: 2.7 µs.
- Priors were gathered with a Python loop over a numpy array (6.8 µs); fancy indexing is 0.2 µs.
- The evaluation batch was `np.asarray` over 24 Python lists (824 µs); rows are now converted
  once with `np.fromiter` and stacked (14.5 µs).

**Left on the table:** `encoder.encode` returns a Python **list** of 1,884 floats, so every
leaf pays ~27 µs to convert it — about 7% of self-play. Fixing it means changing the encoder's
return type, which is shared with the PPO trainer and both interfaces and has 837 lines of
tests written against it. Recorded rather than done.

---

## Result

Three stages, ~2h40m of `train.py` on 20 CPU cores. Promoted **stage 3, iteration 118** as the
first AlphaZero champion, at 32 simulations a move:

| | | |
|---|---:|---|
| against the fixed heuristic | **74.7%** | 293-99, [70.2, 78.8], 400 games |
| against the PPO champion | **76.5%** | 306-94, [72.1, 80.4], 400 games |

**Training is pure self-play.** One network plays both seats; there is no opponent pool and
the heuristic never generates a training position. That is a deliberate difference from the
PPO trainer, which samples 60% self-play, 15% heuristic and 25% frozen past selves. The
heuristic appears only as a yardstick — the in-loop check and the gate. The warm start (D11)
is the one place the heuristic's judgement enters, at second hand.

### Selecting the candidate was itself a trap

Seven checkpoints, each played 200 games against the heuristic on identical games:

```
stage2_iter_61    62.6%  [55.7, 69.1]
stage3_iter_24    72.8%  [66.2, 78.6]   <- highest point estimate
stage3_iter_48    61.3%  [54.3, 67.9]
stage3_iter_72    63.6%  [56.6, 70.0]
stage3_iter_96    71.9%  [65.3, 77.8]
stage3_iter_118   70.9%  [64.2, 76.8]
stage3_best       71.3%  [64.6, 77.2]
```

The top four are statistically indistinguishable, and picking the maximum of seven noisy
estimates is **winner's curse** — the winner is partly whoever drew the friendliest games.
Played head to head on fresh games, the order reverses: **iteration 118 beat iteration 24
112-74 (60.2%)**, with iteration 24's interval topping out at 47%.

So the 72.8% was an inflated draw, and the model with the most training behind it was the
stronger player all along. **When candidates are within each other's intervals, rank them
against each other, not against a third party.**

### The gate refusing a better-looking model, and why that is the right answer

A two-hour run produced `iter_168`, which measured **better than the reigning champion against
both fixed opponents** and was still refused:

| rung | candidate | reigning champion |
|---|---:|---:|
| the fixed heuristic | **80.6%** [76.4, 84.2] | 74.4% [69.8, 78.4] |
| the PPO champion | **77.6%** [73.3, 81.5] | 76.1% [71.6, 80.0] |
| **head to head** | **51.8%** [46.8, 56.8] | — |

Six points better against the yardstick, and a coin flip against the champion. The gate's rule
is the head-to-head lower bound clearing 50%, so it refused.

That is non-transitivity, and it is exactly what the rung exists to catch — but it is worth
being clear about which way the conservatism cuts. The rule protects against a candidate that
beats the champion by learning its habits; here the candidate is *not worse* by any measure and
is probably better by one. Refusing costs a likely-small improvement. Promoting on the
heuristic rung alone would have thrown away the property that makes a self-play ladder mean
anything.

More games would not settle it: at 51.8%, a thousand games gives about [48.7, 54.9] and still
includes 50%. The two models are close to equal in a direct match.

**And on this occasion the override was the right call.** Reading the rungs against each other
rather than against the rule:

- 80.6% vs 74.4% against the fixed heuristic is **significant** — two-proportion z = 2.10,
  p = 0.036. Not an overlapping-interval judgement; a test.
- 77.6% vs 76.1% against the PPO champion: better, not significantly.
- 51.8% head to head: better, not significantly.

Better on every rung, worse on none, and significantly better on the yardstick this project
says is the comparable measure across time. The head-to-head rung is also a *weak
discriminator here specifically*, because the candidate is the champion's direct descendant,
warm-started from it — two closely related networks drift toward 50% against each other even
when one is stronger against the field.

So `iter_168` was promoted with `--force`, and the record says `"forced": true` with the
numbers and the argument in `forced_reason`.

**Two things were deliberately not done.** The gate's rule was *not* rewritten to make this
candidate pass: "significantly better against the fixed yardstick, not worse head to head" may
well be a legitimate second path to promotion, but designing it while looking at the candidate
it would bless is how a gate stops meaning anything. That question is worth deciding on its own
merits, separately. And `--force` now **requires** `--reason`, refuses without one, and plays
the head-to-head rung anyway so the record keeps the number that made it a judgement call —
the first version skipped it and wrote `beat_champion: null` onto the one promotion where that
number mattered most.

---

## Consequences

- Two lineages exist and can be compared. The PPO champion file is untouched.
- **The PPO champion has not been loadable since the affordability block landed.** Recorded
  here rather than quietly fixed by re-promoting it, because a promotion is a decision the
  gate should make, not a side effect of this work.
- Search is leak-free by construction and tested against the repository's own scrambler.
- A parallel run is not bit-reproducible — which worker claims which identity is arbitrary —
  but a single-process `Generator` is, and that is what the determinism test pins.
- `benchmark/benchmark.py` is now the number to optimise against, as the guide asks.
