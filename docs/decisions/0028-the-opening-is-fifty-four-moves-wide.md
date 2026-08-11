# 28. The opening is fifty-four moves wide, and PUCT was tuned for six

Date: 2026-08-11
Status: accepted

## Context

Record 0024 watched the champion place its opening settlements and found it maximising total
pips essentially perfectly — 0.005 pips off the best available spot — and concluded that the
gap was in the observation. It was, in part, and 0024 fixed that part. This is the other
part: even with per-resource production in the observation, the *search* was choosing the
opening from about three spots.

A settlement placement is not one more decision. Measured on 40 fresh two-player
`RANKED_1V1` boards, the first settlement offers **exactly 54 legal actions on 40 of 40
boards**; the later placements offer 50, 46.4 and 42.6 on average as spots are taken and the
distance rule bites. A normal turn offers a handful. `mcts.py`'s constants were chosen
against "a typical Catan position offers 5–30 moves, not 250", and the package's own measured
mean is 9.6 legal moves per decision.

So the opening is the one place in a Catan game where the root is an order of magnitude wider
than the exploration constant was tuned for. It is also the decision a person judges the agent
by, and — because a player's two settlements are seven plies apart — the place where the
*pair* is decided rather than a single spot.

## The two constants, and why they do not settle it

```
C_PUCT        = 1.5      exploration weight
FPU_REDUCTION = 0.25     an unvisited child is scored at (this node's value - 0.25)
```

`Search._select` scores a slot as
`child_q + c_puct * sqrt(node.visits) * prior / (1 + child_n)`. An unvisited slot carries no
`child_q` of its own, so it sits at the node's value less the FPU reduction, and the
exploration term is what has to make that 0.25 back.

An earlier draft of this record derived the starvation from those two constants alone, by
dividing `c_puct` by the width: 1.5/6 = 0.25 at a six-wide root, exactly the reduction,
against 1.5/54 = 0.028 at a settlement — nine times too small. **That derivation drops the
`sqrt(node.visits)` factor and is wrong.** It holds only on the root's first visit; the bonus
grows with the square root of the budget spent so far, and by the 399 root visits a
400-simulation search ends with, an average prior of 1/54 gives
`1.5 x sqrt(399) / 54 = 0.55` — twice the FPU reduction rather than a ninth of it.

Whether a 54-wide root actually gets looked at therefore depends on how the prior is
distributed across the 54, how fast the visited slots' `child_n` climbs against a bonus that
is growing for everyone, and what the values coming back are. Two constants do not settle
that, in either direction. What follows is the measurement, and it is the whole of the
evidence for this change.

## What the search examines, measured

"Spots examined" is the number of distinct root slots with at least one visit. Measured on the
reigning champion **gen6-gilded-beacon** (375,106 parameters, observation 2503) at the first
settlement of a fresh two-player `RANKED_1V1` game, with the search built exactly as
`MCTSAgent.__call__` builds it — determinize, `Search`, batch-1 evaluate — over 40 distinct
boards, 24 at 1,600 simulations. Legal actions: 54 on every board.

Root Dirichlet noise turns out to matter more than the budget does, so all three settings the
repository actually uses are given. `noise=0` is what `MCTSAgent`, the arena and the promotion
gate pass; `0.10` is the configured `dirichlet_weight` for self-play; `0.25` is AlphaZero's
default and is used here by nothing.

| simulations | noise 0 — play, arena, gate | noise 0.10 — self-play | noise 0.25 |
|---:|---|---|---|
| 64 | **3.6** (median 3, range 1–11) | — | 5.0 (median 4, 1–21) |
| 96 | **4.2** (median 3, 1–25) | 4.7 (median 3, 1–30) | 6.3 (median 4.5, 1–31) |
| 400 | **5.9** (median 5, 2–54) | 7.2 (median 5.5, 1–54) | 12.4 (median 11, 4–54) |
| 1600 | **8.2** (median 7, 2–54) | 14.4 (median 12.5, 5–54) | 24.1 (median 22.5, 10–54) |

**Budget buys breadth very slowly.** At the settings the agent plays at, 25x the simulations —
64 to 1,600 — takes the search from 3.6 spots to 8.2. Depth is not the problem; breadth is.

With the floor, at 400 simulations:

| | spots examined |
|---|---|
| `root_min_visits=0` | 5.9 (median 5) |
| **`root_min_visits=4`** | **54 of 54, on 40 of 40 boards** (identical at noise 0.10) |
| `root_min_visits=8` | 50 — the sweep never completes: 8 x 54 = 432 against a 400 budget |

And across a whole opening — all four settlement placements, 16 seeds x 4, noise 0:

| | placement 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| legal spots | 54 | 50 | 46.4 | 42.6 |
| examined, floor 0 | 8.1 | 8.7 | 7.8 | 7.1 |
| examined, **floor 4** | **54/54** | **50/50** | **46/46** | **42/42** |

The floor reaches every legal spot at every placement, not only at the first.

⚠️ **The distribution has a heavy right tail and a mean is a poor summary of it.** At 400
simulations with no floor, 39 of the 40 boards examine 2–10 spots and one (seed 8) examines
all 54; that single board lifts the mean from 5.1 to 5.9. Medians are given beside every mean
for this reason, and it is also why a small-sample measurement of this quantity can land
almost anywhere.

## Decision

Four changes, all confined to the opening.

**1. Opening settlements get their own budget.** `setup_simulations`, default **400**, applied
at `Phase.SETUP_SETTLEMENT` only — the setup road has three options and does not need it.
`MCTSAgent` carries the same default in `DEFAULT_SETUP_SIMULATIONS`. There are four such
decisions in a two-player game, so the cost is bounded by construction.

**2. A root visit floor.** `setup_root_min_visits`, default **4**. `Search._select_root_floor`
hands back a root slot still short of the floor — highest prior first, so a budget exhausted
mid-sweep has spent itself on the likeliest spots rather than on whichever slot sorts first —
and returns `None` once every slot has met it, at which point plain PUCT resumes. 4 x 54 = 216
of the 400 simulations.

**3. Setup is exempt from the playout cap and is always recorded**, including the setup road.
Pointing the opening road at what you intend to expand toward is part of the placement, there
are only three roads' worth of it, and it costs almost nothing to search.

**4. The forced sweep is stripped from both the move played and the target learned.**
`Search._discretionary_counts` subtracts the floor from the root visit counts, and both
`best_action` and `policy_target` read it rather than the raw counts.

## The floor's visits are measurement, not preference, and that is the whole design

This is the part that is easy to get wrong, and the reason point 4 exists.

The floor spends 216 visits looking at spots PUCT would never have tried. Those visits say
nothing about which spot is *better* — they are there so that the value estimates exist at all.
If they reached the policy target, the recorded label would be very nearly uniform over 54
moves, and sampling it at `temperature=1.0` in the opening would pick an arbitrary spot. **The
search would have been made worse by looking at more of the board**, which is a genuinely
perverse outcome and would have been easy to ship without noticing: the loss curve would look
fine and the win rate would take a hundred iterations to say anything.

Subtracting the floor leaves exactly the visits PUCT chose to spend, which is what plain
AlphaZero's target already is.

Measured, at 400 simulations with the floor at 4: the root receives **399** visits, not 400 —
the first simulation expands the root itself and backs up along an empty path. 216 are the
forced sweep and **183 are discretionary, identically on all 40 boards**, since the sweep
always completes. Those 183 land on a mean of **3.5 distinct spots** (median 3, range 1–9),
and the top spot alone takes between 50 and 187 of them.

So "54 of 54 spots" is a statement about what the search *looked at*, and it is worth saying
plainly that the recorded policy target is still concentrated on about three and a half spots
— roughly the handful plain PUCT would have examined. That is the intended behaviour. What
changed is that those three and a half are now chosen after every spot has been valued, rather
than being the first three the prior happened to like.

## Why the opening was under-sampled as well as under-searched

The playout cap (record 0026) records a quarter of searched decisions. Applied to setup, that
meant most placements never reached the buffer at all.

⚠️ **The figure in the code — "setup is 4.6% of searchable decisions and the cap recorded a
quarter of them, so about one placement per game reached the buffer" — is the change author's,
and this record could not confirm it.** It appears in `self_play.py`, `config.py` and
`configs/train_v2.yaml` as a claim, and no independent measurement of it was made here. The
*direction* is not in doubt — the cap is 0.25 by configuration and setup was subject to it —
but the 4.6% is unverified and should be treated as such until somebody counts.

## The cost

Measured at one torch thread, which is what `arena.py` and `workers.py` pin, and the setting
every number above was taken at.

| 400-simulation setup search | mean | median | range | n |
|---|---:|---:|---|---:|
| `root_min_visits=0` | 0.335 s | 0.333 s | [0.30, 0.40] | 40 |
| **`root_min_visits=4`** | **0.319 s** | 0.316 s | [0.30, 0.36] | 40 |

**The floor is free, and very slightly better than free** — the forced sweep builds a
shallower tree, so it does marginally less work than the same budget spent descending. For
reference at the same one thread: 64 simulations 0.050 s, 96 simulations 0.075 s, 1,600
simulations 1.40 s.

⚠️ **Pin your threads.** At 4 threads the same searches cost 0.342 s (floor 0) and 0.419 s
(floor 4), n=8; at 20 threads, 0.79 s and 0.71 s, n=8. Batch-1 forward passes get *slower*
with more threads, so a caller that does not set `torch.set_num_threads` pays roughly double.

## Known limitations

One is open. The other was found by this record's own measurement and fixed before the record
was committed; it is kept here because the failure is worth knowing about and because the fix
is a behaviour change that someone will one day want the reasoning for.

⚠️ **The Gumbel root silently overrides the floor.** In `_descend`, the root branch is
`if self.gumbel: ... elif self.root_min_visits: ...`, so with `gumbel=True` the floor is never
consulted; `best_action` and `policy_target` also take their Gumbel paths and never call
`_discretionary_counts`. Nothing warns. Gumbel is off by default and measured worse in record
0026, so nothing currently runs into this — but the two settings are independently
configurable, and a run that turns Gumbel on will lose the opening breadth without any signal
that it did.

✅ **Fixed: `_discretionary_counts` no longer falls back to the raw counts.** The line used to
be `return discretionary if discretionary.sum() > 0 else counts`. At a floor of 8 with the
shipped 400-simulation setup budget, 8 x 54 = 432 forced visits against 399 available: the sweep
stops at 50 spots, **the discretionary counts are identically zero on 40 of 40 boards**, and the
fallback returned near-uniform raw counts. Verified on one board: `best_action(temperature=0)`
returned action 75 and `best_action(temperature=1)` returned 78 — an arbitrary pick among 49
spots tied at 8 visits, because `argmax` over ties takes the first, which is the lowest-numbered
vertex. That is exactly the failure `_discretionary_counts`'s own docstring says subtracting the
floor prevents, arriving through the fallback instead.

It is now `return np.maximum(counts - self.root_min_visits, 0.0)` with no fallback. Both callers
already had a better branch for an all-zero array and it was being pre-empted: `best_action`
returns `argmax(prior)` and `policy_target` returns the prior itself. So a search whose floor
does not fit its budget now plays the network's best guess rather than the lowest vertex id.
Pinned by `test_a_floor_too_big_for_the_budget_falls_back_to_the_prior`.

This makes the degradation sane rather than absent, and the arithmetic that causes it is
unchanged. The shipped floor of 4 is well clear — 216 forced of 399 — but the margin is only
400/54 = 7.4 visits per spot, so **any floor of 8 or more at this budget still records no
preference at all**; it simply records the prior now instead of an arbitrary spot. A floor that
cannot complete its sweep within the budget should arguably refuse or reduce itself. It does
not, and the docstring on `root_min_visits` says so.

## The figures in the code were wrong, and this record is the correction

Nine claims were copied between `agent.py`, `mcts.py`, `config.py`, `self_play.py` and
`configs/train_v2.yaml` before this measurement existed, and they are tabulated below. Every
claim about the **floor** holds up, and so do the two breadth figures at 64 and 96 simulations
if they are read as medians. The two breadth figures at 400 and 1,600 were taken under root
exploration noise and are 1.9x–2.3x too high for the `noise=0` configuration the agent and the
promotion gate actually use.

| written in the code | correct at noise 0 | verdict |
|---|---|---|
| "3.0 of 54 at 64 simulations" (`agent.py`) | mean 3.6, **median 3.0** | right, as a median |
| "3.0 at 96 simulations" (five places, four files) | median 3.0, mean 4.2 | right as a median, understated as a mean |
| **"13.8 at 400"** (`agent.py`, `config.py`, `train_v2.yaml`) | **5.9** (median 5) | **wrong** — 13.8 is only reproducible near noise 0.25, where it measures 12.4 |
| **"at 1,600 it examines 15.6"** (`mcts.py`) | **8.2** (n=24) | **wrong** at play settings; 14.4 at self-play's 0.10 |
| "54.0 of 54 spots" with the floor | exact, and unconditional | right |
| "4 x 54 = 216 of the 400" (`config.py`) | exact | right |
| "for the same cost" | 0.319 s against 0.335 s | right, and slightly better than right |
| "400 simulations costs ~0.38 s" (`agent.py`) | 0.335 s at one thread | right |
| "a search of 64 simulations costs about 25 ms" (`agent.py`) | 0.050 s | optimistic by 2x |

The two wrong figures described a search their own module never runs: `agent.py` passes
`noise=0.0` and then quoted a breadth measured under noise. The table above is a record of
what was corrected, not of what the code says now — the comments were brought to these numbers
in the same pass, and re-reading the five files afterwards gives:

- `mcts.py`'s `Search.root_min_visits` docstring carries the breadth table itself, at all three
  noise settings, and names 13.8 and 15.6 as "the figures this table replaced", reproducible
  only near noise 0.25.
- "13.8 at 400" is gone from `agent.py`, `config.py` and `configs/train_v2.yaml`, and so is
  every other loose breadth figure: those three now point at the `mcts.py` docstring and at
  this record rather than carrying a number of their own. `configs/train_v2.yaml` says why —
  "quote it from there rather than copying a number into a sixth file".
- `agent.py`'s module docstring now reads "about 50 ms on one torch thread at an opening
  settlement" for 64 simulations, and `DEFAULT_SETUP_SIMULATIONS` reads "400 simulations costs
  ~0.33 s at one thread".

**The measurement is written down once and pointed at from everywhere else**, which is the
arrangement meant to stop this happening a second time. The one figure still quoted in the
code and not established here is the 4.6% above.

## Caveats, stated rather than buried

**Breadth is a property of the (weights, settings) pair, not of the search.** A flatter prior
explores wider. Every figure above belongs to gen6-gilded-beacon at the stated noise; figures
recorded against an earlier champion are not strictly comparable, which is why the table in
`mcts.py` names the network and the noise setting rather than quoting a bare number.

**Nothing here measures whether the floor makes the agent play better.** It measures what the
search looks at. The claim being settled is a breadth claim. A strength claim is what the
promotion gate is for, and per record 0030 that gate is the reigning champion over 400 games.

**The original 13.8 and 15.6 could in principle have been small-sample draws** rather than a
noise-setting difference — the right tail is heavy enough to permit it. The noise sweep
reproduces them too cleanly for that to be the likely explanation, but it is not ruled out.
