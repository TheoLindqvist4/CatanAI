# 29. Retiring pip potential: the guide that came first

Date: 2026-08-11
Status: accepted

## Context

`pip_potential` is one float per vertex: the summed production odds of the tiles that corner
touches. It was in the observation before almost anything else, and for a long time it was the
**only** placement signal the network had. A corner with an 8 on ore, a 6 on wheat and a 5 on
sheep and a corner with three sheep totalling the same are the same number.

Record 0024 measured what a resource-blind guide buys. Over 30 games at 32 simulations, the
champion placed **0.005 pips off the best available spot** against the heuristic's 0.024 — it
maximised the feature essentially perfectly — while owning 0.87 harbours at game end against
1.43 and making 26 two-for-one trades against 95. The agent was not placing badly. It was
placing to the one number it had been given, exactly as well as that number can be followed.

0024's fix was to add what was missing: per-vertex expected cards **of each resource**, harbour
nearness, per-player production rates, board scarcity. That is the information `pip_potential`
is a lossy projection of. This record is the other half — retiring the projection now that the
thing it approximated is in the observation.

## Decision

**The per-vertex `pip_potential` slot carries a constant 0.0, behind
`catan.encoder.PIP_POTENTIAL`.**

**The slot stays.** Deleting the field is the change that is not available: `pip_potential`
sits at offset `MAX_PLAYERS + 2 + HARBOUR_KINDS` inside a 27-float vertex row, and removing it
would shift `buildable`, `my_road`, `production` and `harbour_reach` down by one in all 54
rows and take `encoder.SIZE` with them. Writing 0.0 removes the *information* at no
compatibility cost and leaves `SIZE` alone.

Two tests hold both halves. `test_pip_potential_is_retired_but_its_slot_is_still_there` pins
the slot's position and its constant.
`test_pip_potential_still_sums_the_adjacent_odds_when_switched_back_on` flips the flag and
checks each vertex against `(6 - abs(7 - number)) / 36` summed over the tiles that corner
touches — recomputed inside the test, not read back from `board.expected_production`. The
independence is the point: the test would catch a change to `expected_production` that
silently redefined the feature while it happens to be switched off, which a test that called
the same function could not.

⚠️ **Reversing this is not one line.** The first of those tests asserts `not
E.PIP_POTENTIAL`, so flipping the flag in `catan/encoder.py` fails it: shipping the feature
back on is a two-file change, and `catan/encoder.py`'s own comment says so. *Trying* it is
still cheap — the arithmetic is kept and tested, so the experiment is an afternoon rather than
an archaeology exercise — but a permanent flip is a decision that has to be made twice, which
is the point of pinning it.

## The evidence, weighed honestly

Three arguments were available. Two of them are worth less than they look, and the record
should say which is which.

### The pips half is weak, and quoting two bands of it makes it look stronger than it is

`training.alphazero.study` plays games and records the opening against the outcome.
`catan/encoder.py`'s own comment cites it as "lowest band 79.5%, highest 70.1%" — more pips, no
more wins. Those are the two ends of a four-band table, and the middle is not between them.
The bands, as reported to this record, with Wilson intervals computed by
`training/evaluate.py::confidence_interval`:

| pips band | win rate | n | 95% interval |
|---|---:|---:|---|
| lowest | 79.5% | 39 | [64.5, 89.2] |
| second | 65.2% | 69 | [53.4, 75.4] |
| third | 73.6% | 53 | [60.4, 83.6] |
| highest | 70.1% | 77 | [59.2, 79.2] |

**The table is not monotone and every interval overlaps every other.** The lowest and highest
bands — the two the comment quotes — share [64.5, 79.2], nearly fifteen points of common
range. The second band, which should sit near the top if the slope were real, is the worst of
the four. Total n is 238, which is consistent with the 240-game study record 0024 reports.

0024 already ran into exactly this, on a different cut of the same study, and wrote the
sentence this record should not have to repeat: a clean monotonic result at 40 games, and
**"at 240 games that did not hold"**, followed by "so 'balanced openings win' is suggestive and
not established". The pips cut is the same kind of evidence and deserves the same treatment.

**Suggestive, not decisive.** It is consistent with pips not being what wins games. It does not
establish it, and this decision does not rest on it.

### The diversity half is strong

The same study's resource-diversity cut separates cleanly:

| distinct resources in the opening | 95% interval |
|---|---|
| 5 | **[81.9, 98.5]** |
| 3 | **[45.8, 70.4]** |

The intervals do not overlap — 81.9 is above 70.4 — which is the substantive claim and is
checkable from the intervals alone. The five-resource figure is the 94.4% the encoder comment
quotes: `confidence_interval` returns exactly [81.9, 98.5] for 34 wins in 36, which is what the
interval reconstructs to.

This is the finding that matters, and note what it is *not* evidence for. It says diversity
wins; it says nothing about whether the network needs `pip_potential` to find diversity. It
tells you which direction the missing information points.

### The structural argument carries the decision, and it should

The feature is **resource-blind by construction**. It cannot distinguish three sheep from an
even spread, which is precisely the distinction the diversity result says decides games.
Record 0024 added `production` — five floats per vertex, expected cards per roll of each
resource — from which `pip_potential` is recoverable by summing. The old feature is now a
**derived quantity that the network can compute and that discards the pairing it would need**.

That is the argument. It does not need a win-rate table, and 0024's case for the new features
was made the same way: "the case for these features does not rest on the win-rate study at all
— it rests on the observation provably not containing the information." The mirror of that
sentence is this one. Keeping a lossy summary alongside the thing it summarises is keeping a
guide that points slightly wrong, and — because it was there first and the network was warm
started through several lineages — a guide the run has to spend capacity *un*learning.

## The audit: the champion is already not using it

Before switching it off, the obvious question is what breaks. The answer, from an audit of the
weights across two promotions, is nothing.

`vertex_embed.weight` has **81 input columns**: the vertex's own 27 features, two global
broadcast slots, and the 19 + 27 + 6 neighbourhood aggregates that `hops=1` concatenates. Two
of those columns read `pip_potential` — the vertex's own value, and the vertex-neighbour
average of it.

Comparing gen4-ashen-wheat with gen6-gilded-beacon: **max |ΔW| = 0.0 on both pip columns,
while 73 of the 81 columns moved.** The reigning champion was trained with the column already
zeroed, so those two columns have had no gradient to receive. Switching the feature off changes
nothing under the model that is currently playing; what it changes is what the *next* run is
given.

That is an audit finding, not a win-rate measurement, and it settles a different question:
whether this change is safe to ship under the current champion. It is. Whether it makes the
next champion better is not measured here and would be settled by the promotion gate.

## What the compatibility argument actually is

The reason the slot stays is worth stating precisely, because the version `catan/encoder.py`'s
comment carried before this pass was not right. The comment has been corrected to the below
and says so about itself; this is the argument it was corrected to.

**What is true:** deleting the field moves every later offset inside a vertex row and changes
`encoder.SIZE`. Per `CLAUDE.md`, a `SIZE` change stops the reigning champion loading, and per
record 0030 the AlphaZero gate then has nothing to measure against and refuses outright.

**What is not true:** that it would "invalidate the `layouts.HISTORICAL` entry for 2503".
`training/alphazero/layouts.py::HISTORICAL` is keyed `{1868, 1884}` and has **no 2503 entry**,
by design — checkpoints written since record 0024 carry their own layout via `signature()`,
which is what makes them self-describing forever. A 2503 checkpoint is reconciled from its own
record, not from the table.

The distinction matters because it changes what a future author must do. Deleting the field
would not corrupt a historical table; it would require a *new* `HISTORICAL` entry for the new
size, plus a `graft` path for a block that **shrank** — which `layouts.column_map` explicitly
raises on, since every change to this observation so far has appended. That is a real project,
which is the honest reason not to do it for one float.

## Consequences

- The observation is the same 2,503 floats and every offset is unchanged. No checkpoint is
  affected; nothing needs grafting.
- One float per vertex, 54 in all, is now a constant. A constant input folds into a bias in one
  gradient step, so it costs the network nothing beyond 54 dead columns' worth of dispatch —
  the same argument record 0022 makes for not encoding the cost table.
- **The feature is retired, not deleted.** `PIP_POTENTIAL = True` restores it exactly, and a
  test asserts that it does — while the other test pins the flag off, so restoring it is an
  experiment rather than something that can be shipped by accident.
- `README.md` and `training/structured_net.py` both described pip potential as the placement
  signal the observation gives a vertex, which record 0024 had already superseded. Both were
  brought forward in the same pass as this change: `README.md` now records the retirement and
  the diversity intervals it rests on, and `structured_net.py`'s neighbourhood section says
  what a vertex knows today — 0024's per-resource expected cards, with the resource-blind
  total retired here — and what it still does not.
- **This is not measured as a strength improvement, and no such claim is made.** It removes a
  known-lossy signal that the current champion demonstrably does not use, on a structural
  argument, with the strong half of the study behind it and the weak half labelled.
