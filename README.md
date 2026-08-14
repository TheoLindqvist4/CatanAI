# CatanIA

A Catan engine built so that a machine can learn to play it, and a person can play against
what it learned.

The engine is the point. It is dependency-free Python, exhaustively tested, and every rule
lives in exactly one place — so an agent and a human are always playing the same game, and
"the interface disagreed with the rules" is not a bug this project can have.

```sh
python -m interfaces.web        # then open http://127.0.0.1:8000
```

![Board](Images/Catan_board.png)

---

## What is here

| | |
|---|---|
| **A complete Catan implementation** | Ranked 1v1 rules by default: 15 points, hand limit 9, Friendly Robber, Balanced Dice |
| **A playable web interface** | Click the board to build. Painted artwork, resources as cards, full game log, and a statistics panel that says where the game went |
| **A hand-written opponent** | Positional judgement from marginal value — beats a naive greedy agent 96.7% |
| **Two trained opponents** | PPO self-play, and AlphaZero self-play with search. The AlphaZero lineage is six champions deep; the reigning one took the place at **55.25%** over 400 games against the one before it |
| **Search that cannot cheat** | MCTS over a resampled information set — hidden cards are redrawn from public facts before the tree is built |
| **The machinery to improve it** | 2,503-float observation, 325 discrete actions, parallel self-play, a promotion gate per lineage |
| **937 tests** | Including leak detectors that prove no agent — and no search — can see hidden information |

## Quick start

```sh
git clone https://github.com/TheoLindqvist4/CatanIA.git
cd CatanIA
python -m interfaces.web                       # play in the browser

python -m interfaces.cli                       # or in the terminal
python -m interfaces.cli --agents hard easy    # or watch two bots
```

The engine needs **no dependencies at all**. Only training needs PyTorch:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

It also installs as a package, which is how the [website](https://github.com/TheoLindqvist4/CatanAI-Website)
consumes it without gaining a second copy of the rules:

```sh
pip install catania-runtime                 # the engine. no dependencies at all
pip install catania-runtime[inference]      # + torch, to load and play a model
```

That empty dependency list is checked rather than claimed — `tests/test_packaging.py` parses
every file in `catan/` and fails if one imports anything outside the standard library. See
[`docs/model-contract.md`](docs/model-contract.md).

---

## The three ideas this project is built on

### 1. One source of truth, always

The rules live in `catan/rules.py` and nowhere else. `legal_actions` and `apply` share the
same `can_*` predicates, so an action is legal for exactly one reason. The browser draws what
the server sends and reports clicks; it holds no rules, no board generation, no scoring. The
last time board logic existed in JavaScript it was a second implementation that could
disagree with the engine.

The same principle removed 440 lines of hand-written geometry. `catan/topology.py` generates
every vertex, road and adjacency from one line:

```python
ROW_LENGTHS = (3, 4, 5, 4, 3)
```

The generated ids are **identical** to the ones drawn in `Images/`, which the tests check — so
the diagrams and the code cannot drift apart. Two entries in the old hand-written road table
were wrong, which had been silently corrupting the Longest Road calculation.

### 2. Hidden information is hidden by construction, not by care

An agent receives a `PublicView` with an explicit allow-list. Reading an opponent's hand
raises `AttributeError`. A new field on `GameState` is invisible to agents until someone adds
it deliberately — the opposite of a deny-list, where forgetting once leaks forever.

The observation vector, the web responses and the game log are all filtered the same way, and
each has a **leak test**: rewrite the opponent's hidden cards at constant public counts, and
demand that nothing observable changes.

### 3. Measure it, or do not claim it

Every performance claim in this repository has a number behind it, and several turned out to
be the opposite of what seemed obvious. The records in `docs/decisions/` exist so the
reasoning survives — including the things that did not work.

---

## The opponents

| name | what it is |
|---|---|
| `hard` / `medium` / `easy` | The heuristic, with noise added to its evaluations. Difficulty is *misjudgement*, not amputated rules |
| `greedy` | Sensible build order, random placement |
| `random` | Uniform over legal moves |
| `learned` | The PPO champion, when one is installed |
| `alphazero` | The AlphaZero champion — the same network, plus 64 simulations of search per move, and 400 at an opening settlement. `gen6-gilded-beacon` today |

The heuristic's central idea is **marginal value**: a settlement is worth what its tiles add
to what you already produce, not the sum of its pips. A third wheat is worth far less than a
first ore.

Its resource weights are tuned for **two-player, 15-point** play, which inverts four-player
folklore. Competitive 1v1 data on this exact ruleset gives the win rate for a player who
starts with no production of a resource:

```
brick 36%    wood 40%    sheep 42%    ore 43%    wheat 49%      (50% = even)
```

Missing brick is the worst thing that can happen to an opening; missing wheat is nearly free.
With no player-to-player trading, a resource you do not produce costs 4:1 at the bank, so
*expansion* is what gates a 15-point run.

---

## Training

Two lineages, trained differently, kept separately, and played against each other to find out
which is better.

### PPO

```sh
python -m training.clone --games 300 --net structured     # imitate the heuristic, ~4 min
python -m training.train --resume checkpoints/cloned.pt --workers 12 --lr 3e-5
python -m training.champion promote checkpoints/best.pt   # only if measurably better
```

**Cloning first.** Self-play from random spends millions of steps rediscovering things
`catan/heuristics.py` states outright. Cloning reaches useful play in four minutes.

### AlphaZero

```sh
python -m benchmark.benchmark                             # measure before optimising
python -u -m training.alphazero.train --hours 3           # self-play, search, learn
python -m training.alphazero.champion promote checkpoints/alphazero/best.pt
python -m training.alphazero.chain --until "09:00"        # or all three, overnight
```

`chain` is the same three steps in a loop with a deadline: train a stage, rank its snapshots
*with search*, and offer the best one to the gate. Every stage keeps its own directory and
nothing is deleted, because a losing checkpoint is still a rung somebody can start from.

Search over a game with hidden information needs a state you can roll forward *without
reading what the opponent holds*, and `clone()` copies the development deck, the dice deck and
opponents' hands verbatim. `training/alphazero/determinize.py` is that missing piece: it
resamples every hidden quantity from what is public — resources from the bank complement, dev
cards from the unplayed pool, the dice deck by length — so the tree is built on a world the
searching player could actually be in.

That it works is not an argument, it is a test. Scrambling the hidden state at constant public
counts must leave the determinized world *identical*, and
`tests/test_alphazero.py::test_determinize_ignores_hidden_state` asserts it with the same
scrambler the encoder and the heuristic are held to.

The rest is AlphaZero as written, with the dice as explicit chance nodes: sample a roll, key
the child by the total, revisit and resample. Values propagate in a fixed frame rather than
flipping by depth, because Catan is not alternating — during a discard the decision belongs to
whoever is over the hand limit.

**The network knows the board has a shape.** `training/structured_net.py` shares weights
across all 54 vertices, 72 roads and 19 tiles, and produces per-position logits from each
position's own embedding. Against a flat MLP on the same data: held-out agreement 69.6% →
**80.3%**, overfitting gap 13.9 → **2.2 points**, with 7.3x fewer parameters.

### The champions, and why training cannot break your game

```
checkpoints/            scratch. A run owns it and rewrites it. Not in git.
models/champion.pt      the PPO champion.        Changes only by promotion. In git.
models/champion_az.pt   the AlphaZero champion.  Changes only by promotion. In git.
```

The interfaces read `models/` and never `checkpoints/`, so a run in progress cannot disturb a
game in progress — which is the point: you can train in one window and play in another.
Promotion is earned: a candidate plays 400 games against the reigning champion and is refused
unless the Wilson lower bound clears 50%.

**The two lineages gate differently, and on purpose.** The PPO gate in `training/champion.py`
also plays the fixed heuristic and refuses a candidate that dropped more than 5 points against
it — self-play is non-transitive, so a policy can beat the champion by learning its habits
while getting worse at the game, and a fixed external opponent is the only thing that notices.
The AlphaZero gate in `training/alphazero/champion.py` had that rung and **it was removed**:
one rung decides, the reigning champion, and nothing else, so that "promoted" means exactly
one measurable thing. Removed from the *decision*, not from the record, and the two rungs
differ in what they cost. The heuristic is played only when `--baseline-games` asks for it.
The PPO champion is played **by default** — `--ppo-games` falls back to the head-to-head
game count — so a bare `promote` plays two 400-game matches whenever a PPO champion loads,
and only the first of them can refuse. `chain.py` passes `--ppo-games 0` explicitly, which
is why an overnight run plays one match and nothing else; that is the chain's choice, not
the gate's default.
`tests/test_alphazero.py::test_the_heuristic_cannot_veto_a_candidate_that_beat_the_champion`
holds it there, so putting the rung back is a decision rather than a drift
([decision 0030](docs/decisions/0030-one-rung.md)).

This is not theoretical. The gate has already refused a finished training run that scored
48.2% against the champion.

**The first promotion of a lineage is refused, not waved through.** With one rung there is
nothing for a first candidate to be measured against, so the AlphaZero gate stops rather than
installing something unmeasured — which is what the PPO gate still does when no champion
loads, and it fires exactly when `encoder.SIZE` has changed and nobody is watching. Installing
the first champion of a lineage is an explicit `--force --reason`; the head-to-head is still
played when there is anything to play it against, and the record carries `"forced": true` with
the stated reason forever, so a promotion that skipped the gate can never be mistaken for one
that passed it.

### Recorded games

Games you play in the browser are written to `games/`. The record is the **seed and the move
indices**, which is complete rather than a summary: the engine is deterministic, so replaying
reproduces the board, the decks, every roll and every observation. Each decision also keeps
*what else was on offer*, because "it had fourteen options and chose that one" is the question
a lopsided game needs answered.

```sh
python -m interfaces.web.recorder --margin 5 --verify    # the lopsided ones
```

Only games a person actually played are recorded — tests and scripts drive the same code and
leave no trace.

---

## Layout

```
catan/                 the engine — no dependencies
  topology.py            geometry, generated from ROW_LENGTHS
  board.py               one immutable layout
  state.py               everything that changes during a game
  rules.py               legal_actions / apply — the only legality authority
  action_space.py        325 flat indices + a legality mask
  encoder.py             the 2,503-float observation
  view.py                PublicView — what a player may see
  heuristics.py          position evaluation
  agents.py              the baseline agents and a match harness
  env.py                 reset / step

interfaces/            the only parts that display anything
  render.py              board -> PNG
  cli.py                 play or watch in a terminal
  web/                   the browser game, the recorder, the statistics panel, a stdlib
                         HTTP server

training/              the only package that imports PyTorch
  net.py structured_net.py    the policy/value networks
  rollout.py ppo.py pool.py   PPO self-play, the update, the opponent pool
  clone.py                    warm start by imitating the heuristic
  champion.py                 the PPO champion, and its promotion gate
  alphazero/                  the AlphaZero lineage
    determinize.py              resample what the searcher may not see — the leak boundary
    mcts.py                     PUCT, with the dice as chance nodes
    self_play.py workers.py     many games in flight, one batched evaluator, many processes
    replay_buffer.py            a ring, sampled in equal parts by age
    trainer.py train.py         the continuous loop
    report.py                   what a run did, read back from metrics.jsonl
    agent.py champion.py        what you play against, and its gate
    arena.py                    head-to-head matches across processes
    chain.py                    train, rank, promote, repeat, until a deadline
    distil.py                   changing the network's shape without starting over
    layouts.py network.py       carrying a checkpoint across an observation change
    study.py dashboard.py       what openings win, and a page showing a run

benchmark/             games/sec, ms/game, and where the time goes
configs/               train.yaml, train_v2.yaml — a run's settings

docs/decisions/        30 records of why things are the way they are
```

---

## Things that turned out to be the opposite of obvious

Each cost real time to discover, and all are written up in `docs/decisions/`.

- **The opening evaluator was a pip count in disguise.** `settlement_value` never accumulated
  within a vertex, so with an empty hand it collapsed to exactly 4x weighted pips — identical
  on 54 of 54 vertices. A spot with three wheat tiles rated as highly as one with wheat, ore
  and brick.
- **Fixing it won no more games.** The road threshold sat at the 88th percentile of road
  values: on 85.2% of decisions where a road was legal, *every* option was below it. The agent
  could not expand, so a better opening had nothing to express. Fixing both: **70.7%** against
  the old agent, and truncated games fell from ~90 per 800 to 8.
- **The winner is always the player who just acted**, so `step()`'s reward is always `+1` and
  the loser's never arrives. A learner that consumed it would train on winners only, its
  critic would converge to `V = 1`, and nothing would crash.
- **`encoder.encode` was 57% of training time**, recomputing per-vertex harbours and pip
  potential for a board that had not changed in 14,000 calls.
- **A `\b` inside a JavaScript template literal is a backspace**, not a word boundary — a
  pattern that matches nothing while reading as though it should work.
- **1-ply lookahead does not help.** Leak-safe and correct, and 53.4% against 52.2% over 800
  games. Recorded because an unwritten negative result gets re-attempted.
- **Budget does not buy breadth.** An opening settlement offers 54 legal spots and PUCT is
  tuned for a normal turn's six: at the settings the agent plays at, 64 simulations examine a
  median 3 of the 54, and 25x the budget takes that to a median 7 (40 boards, 24 at 1,600
  simulations). Giving every spot a floor of 4 visits reaches all 54 on 40 of 40 boards, and
  costs slightly *less* than plain PUCT — 0.319 s against 0.335 s at 400 simulations — because
  the forced sweep builds a shallower tree.
- **A feature the agent had learned perfectly still had to go.** `pip potential` sums a
  vertex's odds and throws the resources away, so three sheep and an even spread are the same
  number. The champion placed 0.005 pips off the best available spot — and openings covering
  five resources win where three-resource ones do not, [81.9, 98.5] against [45.8, 70.4],
  which is the one distinction the feature cannot make. It now writes 0.0.

---

## Where it stands

The engine and both interfaces are complete and tested.

The strongest player is the AlphaZero champion, `gen6-gilded-beacon`, promoted 2026-08-06:
**55.25%** over 400 games against the champion before it, interval [50.35, 60.05]. That is the
only figure its record carries. The gate is the reigning champion and nothing else, so the
heuristic was not played at all, and this file will not guess what it would have said — the
last champion measured there is two promotions old (`gen4-ashen-wheat`, 92.7% over 400 games
at 64 simulations, in its promotion record). The paired comparison against the champion it
replaced is a different, smaller match: 92.00% [88.4, 94.6] against 80.13% [75.2, 84.3] over
300 games on identical games
([decision 0026](docs/decisions/0026-why-the-run-stopped-learning.md)).

Two lessons worth carrying, both learned here the expensive way. **A win rate against the
heuristic is only comparable within one version of the rules**: `models/champion.json` records
71.6%, and the same weights re-measured at 49.3% over 150 games, [41.4, 57.3], because a
commit restricted pre-roll development-card plays after that promotion. And **the champion is
not the newest model** — it changes only through the gate, which has already refused a
finished run.

What is known to be missing:

1. **Roads have one step of lookahead**, no plan. There is no notion of a route.
2. **Adjacency, for the flat network.** It has to infer that vertex 23 neighbours 24 from
   correlations, though `topology.py` knows. The structured network answers this by sharing
   weights across positions rather than by adding a feature, which is why it is the default.
3. **Any estimate of what the opponent holds.** Deliberate: a robber steal moves a card only
   the two players involved ever see, so a running total would be either wrong or a leak. The
   `history` block's cumulative production and spending bounds it, and deriving the bound is
   left to the network.

Three items have left that list, which is why it is no longer ranked. Build costs: the
observation said nothing about what a road cost, and the `affordability` block now encodes how
far the hand is from each purchase and what closing the gap would cost at the bank
([0022](docs/decisions/0022-affordability-features.md)). Which numbers a vertex touches: a
vertex carries its expected cards *per resource* and how near the closest harbour of each kind
is ([0024](docs/decisions/0024-what-a-placement-can-see.md)), and the resource-blind aggregate
that used to stand in for it now writes 0.0
([0029](docs/decisions/0029-retiring-pip-potential.md)). And belief sampling, which every
search idea needed, is `training/alphazero/determinize.py`.

See [`CLAUDE.md`](CLAUDE.md) for working notes and [`docs/`](docs/) for the decision records.

**This repository is becoming half of a pair.** `catan/` and the inference half of `training/`
are being packaged as `catania-runtime`, which a separate public platform
([CatanAI-Website](https://github.com/TheoLindqvist4/CatanAI-Website)) consumes across a
versioned contract — so the rules, the observation and the hidden-information filter keep
living in exactly one place. Nothing here becomes less usable on its own: a clone with no
dependencies still plays and still trains. See
[`docs/website-split-plan.md`](docs/website-split-plan.md).

## Tests

```sh
python -m pytest tests -q       # 937 tests, about six minutes
```
