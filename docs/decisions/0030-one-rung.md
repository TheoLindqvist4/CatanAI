# 30. One rung: the AlphaZero gate is the reigning champion and nothing else

Date: 2026-08-11
Status: accepted

## Context

Record 0023's D10 gave the AlphaZero lineage a three-rung ladder: the fixed heuristic, the PPO
champion, and the reigning AlphaZero champion. A first candidate of the lineage had to beat
`HeuristicAgent(0)` with its Wilson lower bound above 50%, and the record said
`first_of_lineage: true`. The heuristic rung also carried a regression check inherited from the
PPO gate — a candidate that beat the champion but fell more than `MAX_BASELINE_REGRESSION`
against the yardstick was refused as probably overfitted to the champion.

Three rungs means "promoted" is a conjunction, and a conjunction of three noisy measurements is
hard to reason about later. It also means the word does not name a single quantity: two
promotions can both be "promoted" and have cleared different things.

## Decision

**One rung decides: the reigning champion.** A candidate is installed when it beats
`models/champion_az.pt` with its Wilson lower bound above 50% over 400 games, and for no other
reason.

- `MAX_BASELINE_REGRESSION` is **gone** from `training/alphazero/champion.py`. Nothing in the
  AlphaZero gate can refuse on the yardstick.
- `--baseline-games` **defaults to 0** and does not play the heuristic at all. Pass a positive
  number and the match is played and the win rate written into the record — recorded, never
  consulted. `training/alphazero/chain.py` passes 0, which is its own default.
- `--ppo-games` defaults to the head-to-head game count when unspecified — `champion.py` reads
  `ppo_games = games if ppo_games is None else int(ppo_games)` — so a hand-run promotion plays
  the PPO champion too and records the cross-lineage number. It is likewise recorded and never
  a veto. `chain.py` passes 0 to skip it, which is the chain's choice and not the gate's
  default.
- **A first promotion of a lineage is refused, not waved through.** With the heuristic rung
  gone there is nothing left to gate a first candidate on, so `promote` returns
  `False, "there is no reigning champion to measure against …"`. Installing one is an explicit
  `--force --reason`, and the record carries `"forced": true` with the reason forever.
- The forced path is otherwise unchanged: it still *plays* the head-to-head match and records
  the number, because a record that omits it cannot be argued with later.

**The ladder is now one rung, which is fewer than the PPO lineage's two.** That inversion is
deliberate and is the point of this record.

## Both sides, fairly

This is a trade-off, not a victory, and the argument that was removed has not stopped being
true.

**For the heuristic rung.** Self-play is non-transitive. A candidate can beat the reigning
champion by learning its habits — its openings, its robber patterns, the positions it
mishandles — while getting *worse* at Catan. A fixed external opponent is the only thing in the
pipeline that would notice, because every other measurement in the loop is against a model
descended from the same weights. Record 0023 documents this failure mode in the other
direction, with a candidate that was better against both fixed opponents and a coin flip head
to head, and observes that "two closely related networks drift toward 50% against each other
even when one is stronger against the field". The rung existed for a real reason.

**For removing it.** A gate with three rungs answers three questions and reports one word. When
a promotion is read six months later, "promoted" has to mean something checkable, and with a
conjunction it means "cleared whichever combination of thresholds was in force that week".
Worse, the heuristic is not a fixed yardstick across time: `CLAUDE.md` records
`models/champion.json`'s 71.6% measuring 49.3% today, because commit `e4b0441` changed the game
after the number was written down. **A rung whose meaning drifts cannot refuse a candidate
honestly** — it will refuse on rule changes as readily as on overfitting, and the two look
identical from inside the gate.

So the rung was removed on purpose, so that "promoted" means exactly one measurable thing: this
model beat the model it replaced, over 400 games, with the interval clearing 50%.

## The cost, accepted

**The fixed yardstick is no longer measured by default, so the lineage has no external anchor.**
`models/champion_az.json` says so plainly:

| champion | promoted | vs the champion it replaced | vs the heuristic |
|---|---|---:|---|
| gen4-ashen-wheat | 2026-08-05 | 71.0% | **92.7%** |
| gen5-lucky-ferry | 2026-08-06 | 59.4% | **null** |
| **gen6-gilded-beacon** | 2026-08-06 | **55.25%**, [50.35, 60.05] | **null** |

All at 400 games and 64 simulations. Two consecutive promotions carry `beat_heuristic: null`,
and that null is written deliberately rather than inherited: the record must not carry a stale
figure from the previous champion as though it had been measured for this one.

The consequence is concrete. **No document may quote a heuristic win rate for the current
champion**, because none exists. The last measured one belongs to gen4-ashen-wheat and two
promotions have happened since. `champion.describe()` omits the yardstick line entirely when
it is null, so the interfaces no longer mention it either.

This is the accepted downside, and it is not free: a chain of self-play promotions with no
external measurement is exactly the situation the removed rung was there to catch. What
replaces it is a decision to ask on purpose — `--baseline-games 400` at whatever interval
somebody chooses — rather than a check that fires automatically and can refuse for the wrong
reason.

It also removes a second thing quietly. Record 0023's forced promotion of `iter_168` argued
from the yardstick: "80.6% against the fixed heuristic against the reigning champion's 74.4%
(two-proportion z = 2.10, p = 0.036)". That argument is still available, but it now requires
asking for the baseline match explicitly before making it. The override path did not get
easier; the evidence for using it stopped being collected by default.

## What did *not* change

**The gate did not get more permissive.** The head-to-head rung is unchanged — 400 games,
Wilson lower bound above 50% — and it is the rung that refused `iter_168` in record 0023 at
51.8% [46.8, 56.8]. That candidate would be refused today for the same reason and would still
need `--force --reason`.

**The PPO lineage is untouched.** `training/champion.py` still runs both checks:
`MAX_BASELINE_REGRESSION = 0.05` refuses a candidate that beat the champion but fell more than
five points against the heuristic, and the heuristic match is played on every promotion. Two
lineages, two gates, on purpose.

⚠️ **The PPO gate still has the hole this one closed.** `training/champion.py:129-132`:

```python
if reigning is None:
    _install(candidate_path, {"beat_heuristic": against_baseline["win_rate"],
                              "beat_champion": None, "games": games})
    return True, "no reigning champion; installed"
```

No Wilson bound, no threshold — it installs, and then writes the baseline every later candidate
is measured against. It fires exactly when `encoder.SIZE` has changed and the reigning champion
stops loading, which is exactly when nobody is watching. The AlphaZero gate refuses in that
situation instead; the PPO gate was left alone because changing it is not this change's
business, and it is recorded here so that it is a known hole rather than a surprise.

## What stops this drifting back

Two tests, both written to be read rather than only to pass.

`tests/test_alphazero.py::test_the_heuristic_cannot_veto_a_candidate_that_beat_the_champion`
gives the candidate 40% against the heuristic and 70% against the champion, asserts it is
promoted, and asserts the 0.40 is still written into the record because it was asked for. Its
docstring says the test used to assert the *opposite*, and why: "if this behaviour is ever
reverted, it should be because someone decided to, not because the check quietly came back."
A second assertion checks that the default gate does not even play the heuristic.

`tests/test_alphazero.py::test_first_promotion_is_refused_rather_than_waved_through` submits an
untrained network with no reigning champion, asserts the refusal and that no file was written,
then asserts that `--force --reason` installs it *and* stamps `"forced": true`.

## Consequences

- **"Promoted" names one measurement.** Beat the previous champion over 400 games, lower bound
  above 50%. Nothing else can install a champion except an explicit, recorded override.
- The gate is cheaper, and how much cheaper depends on who is running it. It used to play
  three matches. **By default it now plays two** — the 400 head-to-head games that decide, and
  the PPO match, which `--ppo-games` defaults to the same count and which is played whenever
  `models/champion.pt` loads, recorded and never consulted. Only the heuristic rung is off by
  default. `training/alphazero/chain.py` passes `--ppo-games 0`
  explicitly, so an unattended chain plays **one** 400-game match and nothing else, which is
  what makes a chain of stages fit in its evaluation window. That is the chain's choice, not
  the gate's.
- The lineage's external calibration is now a manual act. If the yardstick matters for a
  particular question, `--baseline-games` is how to get it, and the answer goes into that
  promotion's record.
- ⚠️ **Non-transitive drift is now unmonitored by construction.** Nothing in the pipeline will
  notice a lineage that is climbing its own ladder while getting worse at the game. That is the
  price of the single rung, it was paid knowingly, and the cheapest insurance against it is to
  spend 400 games on the heuristic every few generations and write the number down.
