# The AI surface

How to train against the engine. For the rules themselves see [engine.md](engine.md); for why
each piece is shaped this way see
[decision 0014](decisions/0014-ai-surface.md) — which was written against 324 actions and a
1,808-float observation, so read it for the reasoning and not for the numbers.

```
catan/action_space.py    325 flat indices  +  legal_mask(state)
catan/encoder.py         2,503-float observation, perspective-rotated, hidden-info masked
catan/env.py             reset(seed) / step(index)
catan/agents.py          random, greedy and heuristic baselines  +  play_match
```

---

## The loop

```python
from catan.env import CatanEnv

env = CatanEnv(num_players=2)              # ranked 1v1 rules by default
obs, info = env.reset(seed=0)

while not info["done"]:
    action = my_agent(obs, info)           # an index; must be legal
    obs, reward, terminated, truncated, info = env.step(action)

print(info["winner"], info["scores"])
```

An agent is any callable `(observation, info) -> index`. That is the whole interface — a network
fits it, and so does `RandomAgent`.

### What `info` carries

| key | |
|---|---|
| `player` | **who must act** — not always the turn holder |
| `mask` | `bytearray` of 325 flags, 1 where legal |
| `legal` | the same thing as a list of indices |
| `view` | a `PublicView` — what this player may see, for agents that reason about positions rather than vectors |
| `phase`, `turn`, `last_roll` | where the game is |
| `scores` | true victory points, including hidden cards |
| `public_scores` | what an opponent can see |
| `events` | what happened during this step, including any dice rolled on the way to it |
| `winner`, `done` | `winner` is `None` if truncated |

**Read `info["player"]`; do not assume turn order.** During a discard the decision belongs to
whoever is over the hand limit, which is usually an opponent. Assuming alternation is the
classic multi-agent environment bug, and the observation is built from *that* player's view.

---

## The action space

325 indices, in contiguous blocks by type — so `mask[SLICES[ActionType.BUILD_ROAD]]` is exactly
the roads:

| block | indices | |
|---|---|---|
| `END_TURN` | 0 | |
| `BUILD_ROAD` | 1–72 | road id |
| `BUILD_SETTLEMENT` | 73–126 | vertex id |
| `BUILD_CITY` | 127–180 | vertex id |
| `TRADE_WITH_BANK` | 181–200 | 20 ordered pairs of distinct resources |
| `MOVE_ROBBER` | 201–295 | 19 tiles × (nobody, or one of four players) |
| `DISCARD` | 296–300 | one per resource |
| `BUY_DEV_CARD` | 301 | |
| `PLAY_KNIGHT` | 302 | |
| `PLAY_ROAD_BUILDING` | 303 | |
| `PLAY_YEAR_OF_PLENTY` | 304–318 | 15 **sorted** pairs, doubles included |
| `PLAY_MONOPOLY` | 319–323 | one per resource |
| `ROLL` | 324 | appended, so every index above kept the value it had |

`action_space.SLICES` and `COUNTS` are derived at import and are the authority; the table is a
convenience, and it is checkable against them in one line.

**`ROLL` is offered only when declining to roll is a real choice.** The dice are environment
stochasticity rather than a move, and `env.step` rolls them for you. The exception is a Knight
played *before* the roll — it decides which tile pays out this turn — so whenever such a card is
playable `legal_actions` returns it *and* `roll()`, because a player holding a Knight who cannot
decline is forced to play it every turn, which is not the game. With no card to play the list is
empty and the environment rolls by itself rather than asking for a click with one answer.

**Append, never insert.** Every index keeps the meaning it had, so a recorded game and a trained
policy head still mean what they meant. `_validate()` asserts at import that the blocks are
contiguous, in order, and cover the space exactly once. Note that the *positional* blocks are not
adjacent to each other — `TRADE_WITH_BANK` sits between the cities and the robber — which
`training/structured_net.py::_validate` checks separately, and has caught two mistakes.

The size is **independent of the player count**, so weights transfer between 1v1 and 4-player
and evaluation code does not branch.

`encode` / `decode` convert between an `Action` and its index. `encode` raises on an
inexpressible action rather than dropping it — a silent drop would make that move permanently
unreachable, and would look like a policy that simply never learns it.

---

## The observation

`encoder.SIZE` floats, always — 2,503 today. `LAYOUT` gives the named spans and `SHAPES` the
row/column counts, so a graph or convolutional model can reshape rather than being forced
through an MLP:

```python
from catan import encoder

encoder.SIZE                               # 2503
encoder.LAYOUT["vertices"]                 # slice(361, 1819)
encoder.SHAPES["vertices"]                 # (54, 27)

obs = encoder.encode(state, me)
encoder.block(obs, "tiles")[3]             # tile 4, as 19 features
```

**The block-by-block table that used to sit here is gone rather than corrected.**
`catan/encoder.py`'s module docstring owns it and `CLAUDE.md` restates it; a further copy in a
document nobody opens when the encoder changes is exactly how this section came to claim 1,884
floats, `54 × 16` vertices, `4 × 29` players and a live pip potential long after all four had
stopped being true. Read `LAYOUT` and `SHAPES`, which cannot go stale, or the docstring.

Three things a consumer still has to be told:

**Every value is scaled into roughly `[0, 1]`** — an exact maximum where one exists (a resource
count cannot exceed the bank's 19) and a documented soft cap, clipped, otherwise.

**The per-vertex `pip_potential` slot is retired and carries a constant `0.0`**
([record 0029](decisions/0029-retiring-pip-potential.md)). The slot stays: deleting the field
would shift every later offset inside a 27-float vertex row and take `encoder.SIZE` with it, and
a block that *shrank* has no column correspondence for `graft` to use. `encoder.PIP_POTENTIAL`
turns the arithmetic back on, and that is a **two-file** change rather than a one-line one —
`test_pip_potential_is_retired_but_its_slot_is_still_there` asserts the flag is off, so shipping
it flipped means editing the test too. What supersedes it is per-vertex expected cards *of each
resource* and how near the nearest harbour of each kind is, both added by record 0024; a summed,
resource-blind pip count cannot tell three sheep from an even three-way spread.

**There are two entry points and one body.** `encode(state, me)` returns a list of Python floats
and is what every interface, test and PPO call site uses — 13 tests assert `isinstance(v, float)`.
`encode_into(state, me, buffer)` writes the same numbers into an `array('f')` from
`observation_buffer()`, which numpy takes as a buffer instead of unboxing 2,503 Python floats;
that is the door a search should use, and each search needs its own buffer, because
`encode_into` overwrites. `test_encode_into_matches_the_list_encoding_bit_for_bit` is what stops
the two drifting apart.

### Perspective rotation

`encode(state, me)` puts **me in player slot 0**, opponents following in turn order. So one
network plays every seat, and a position encodes identically whichever player number holds it.
Absent players leave their slot zeroed.

### Hidden information

An observation contains only what that player may see:

| hidden | public instead |
|---|---|
| an opponent's hand *composition* | its size — cards are countable |
| an opponent's dev-card *composition* | how many they hold, and Knights played |
| the dev deck order and contents | how many remain |
| the Balanced Dice deck | nothing |

This is tested by mutating the hidden thing and asserting the observation does not move — swap
an opponent's three Knights for three Victory Points and nothing changes.

Use `rules.public_victory_points` wherever you mean "what an opponent can see"; `scores` in
`info` is the true total and includes hidden cards.

---

## Baselines

```python
from catan.agents import GreedyAgent, RandomAgent, play_match

play_match({1: GreedyAgent(0), 2: RandomAgent(0)}, games=40, seed=100)
# {1: 27, 2: 13, 'truncated': 0}
```

`RandomAgent` is the floor — anything that cannot beat it is broken. `GreedyAgent` picks the
highest-priority *action type* available, with no idea *where* to build, and still wins about
70% against random. `HeuristicAgent` chooses *where* as well, entirely from `info["view"]`, so it
cannot read an opponent's cards even by mistake; it is the behaviour-cloning teacher and the
fixed yardstick, which is why changing it makes win rates recorded before and after it
incomparable.

`play_match` **swaps seats every other game**, because Catan's first-player advantage is real
and large; a fixed-seat result measures the seat as much as the agent.

For agents that *search*, `play_match` is too slow to be useful — `training/alphazero/arena.py`
records a searched game at 5.2 seconds at 32 simulations against the heuristic's 56 milliseconds,
so the 400-game promotion match is over half an hour per rung, which is how a gate stops being
run. `training.alphazero.arena.compete` plays the same match across processes and returns the
same shape of result. It re-seeds each agent per game so the answer does not depend on the worker
count; `play_match` lets agent RNG carry between games, which is why the two sample the same
quantity by different draws.

---

## The champion, and what promoting one means

Two lineages, two files. `models/champion.pt` is PPO and `models/champion_az.pt` is AlphaZero;
they are separate on purpose and both interfaces offer both.

`training.alphazero.champion.load()` returns the AlphaZero champion as a playable agent, or
`None` — never a raise. A missing file, a missing PyTorch and a checkpoint built for a different
observation or action space all mean one thing to a caller: offer something else. A stale
checkpoint loads perfectly well and then fails on the first move, which is the failure that
`load` exists to prevent.

It is built at `CHAMPION_SIMULATIONS`, **64**, which is both the count the champion is measured
at and the count it is played at — a win rate is a property of the `(weights, simulations)` pair,
so measuring at one and playing at another publishes a figure for a player nobody faces. 64
rather than 128 is a latency choice and not a strength one: one decision costs 52 ms at 32,
101 ms at 64 and 207 ms at 128 on one thread, and 207 ms does not fit inside the 200 ms pace a
watched game is played back at.

The gate is one rung — 400 games (`PROMOTION_GAMES`) against the reigning champion, Wilson lower
bound above 50%, and nothing else decides. Two things about it are easy to get wrong:

- **`promote` plays two matches by default, not one.** `--baseline-games` defaults to 0, so the
  heuristic is not played unless it is asked for; `--ppo-games` defaults to `games`, so the PPO
  champion *is* played whenever one loads. Both are recorded in `models/champion_az.json` and
  neither can veto. `training/alphazero/chain.py` passes `--ppo-games 0` explicitly, so the
  overnight chain plays a single match — that is the chain's choice, not the gate's default.
- **With no loadable champion the gate refuses.** The head-to-head rung is the whole gate, so
  there is nothing left to measure against, and this fires exactly when `encoder.SIZE` has
  changed — which is exactly when nobody is watching. Installing the first champion of a lineage
  is an explicit `--force --reason`, and the record carries `"forced": true` forever.

See [decision 0030](decisions/0030-one-rung.md).

---

## Performance

The table this section used to carry was record 0014's, measured on the 1,808-float observation
of the time, and it is removed rather than updated:
[record 0027](decisions/0027-where-a-searched-decision-goes.md) owns the current breakdown, and
keeping a copy here is what made the old one wrong. Its shape:

- **The observation is 32.1% of a searched decision** — more than the network — and
  `rules.legal_actions` is another 15.8%, at ~173 calls per decision. Two functions are half the
  loop.
- `encoder.encode` followed by `np.fromiter` measured 166.71 µs a leaf; `encode_into` into an
  `observation_buffer()` followed by `np.frombuffer` measures 87.48 µs, bit-identical on every
  one of 30 real positions (min of 9 interleaved rounds).
- Generating a board is **0.0018%** of self-play, because `clone` shares the board by reference.
  Pre-generating a pool of them was built and measured at +0.64% ± 2.26%, and is not worth doing.

`python -m benchmark.benchmark` re-measures, and `python -m benchmark.profiler selfplay` says
where the time goes. Warm anything up first — self-play transitions bank in cohorts rather than
continuously, so a short measurement measures luck.

⚠️ A sequential A/B does not work on this machine: throughput decays monotonically under
sustained load, so whichever arm runs second loses whatever it contains. Alternate the arms,
alternate the order within each pair, and report the median of adjacent-pair ratios. Record 0027
has the numbers.

---

## ⚠️ Search must sample hidden state — and there are exactly two ways to do it

`state.clone(rng=state.rng)` diverges for plain dice, but **three pieces of hidden state are
copied verbatim** and therefore replay identically:

- `dice_deck` — with Balanced Dice, the next ~24 rolls are already determined
- `dev_deck` — the next purchases are already determined
- opponents' `dev_cards`

That is *correct*: these are hidden, not random. But it means a rollout from a cloned state is
not a sample of the future — it is *the* future, and searching it is reading the opponent's
cards. `test_with_balanced_dice_a_clone_replays_the_same_rolls_even_sharing_the_rng` pins the
behaviour so it is not discovered by surprise.

**Anything that searches must go through one of two doors.**

`training.agent.DETERMINISTIC_TYPES`
    Search only the actions whose outcome is fully determined by public information — build,
    trade, discard, Year of Plenty, Road Building. Simple, and it caps the agent at one ply
    forever, because a second ply needs the opponent's reply.

`training.alphazero.determinize.determinize`
    Resample everything hidden from what is public — resources from the bank complement, dev
    cards from the unplayed pool, the dice deck by length — and then search the resulting
    world as deeply as you like. This is belief sampling, and it is what MCTS is built on.

The second is held to the same standard as the observation: scrambling the hidden state at
constant public counts must leave the resampled world *identical*, asserted with
`tests/helpers.py::scramble_hidden_state` in
`tests/test_alphazero.py::test_determinize_ignores_hidden_state`.
