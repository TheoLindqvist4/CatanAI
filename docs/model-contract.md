# CATANIA-1 — the model and runtime contract

What a set of weights has to agree with before this engine will play it, and what a recorded
number has to name before it means anything.

This document is prose. **`catan/contract.py` is the specification**, and every field below is
read out of the module that owns it rather than written down twice — the numbers here are a
snapshot for a reader, and the code is what a service checks against.

```python
from catan import contract
contract.signature()
# 'CATANIA-1/engine-0.1.0/obs-1:2503/act-1:325/catan-1v1-v1'
contract.describe()      # the whole thing as plain JSON
```

---

## The signature

```
CATANIA-1 / engine-0.1.0 / obs-1:2503 / act-1:325 / catan-1v1-v1
    │           │              │            │            │
    │           │              │            │            └── ruleset id and rules version
    │           │              │            └── action-space version : count
    │           │              └── observation version : size
    │           └── engine version (catan.__version__)
    └── model format
```

**Two figures may be compared if and only if their signatures match.** That is a rule a
database can enforce, unlike "we think the rules were the same in August". Store it beside
every game, every rating row and every accepted model.

### Why this exists

It has cost this project twice, both times silently.

`models/champion.json` records **71.6%** against the heuristic. The same weights re-measure at
**49.3%**, interval [41.4, 57.3], because commit `e4b0441` restricted pre-roll development-card
plays *after* that promotion. The figure was never wrong; it stopped describing the current
game, and nothing in the file could say so.

Separately, of the 19 recorded games in `games/`, two no longer replay at all — `unknown action
ROLL(0)` and `must roll the dice before PLAY_YEAR_OF_PLENTY(ore, ore)`. Seed-plus-moves is this
project's audit primitive and its cheapest storage format, and it turned out to be silently
scoped to a version nobody had recorded beside it.

See [`audit-2026-08-05-public-arena.md`](audit-2026-08-05-public-arena.md) §B9.

---

## The fields

| field | value today | changes when |
|---|---|---|
| `model_format` | `CATANIA-1` | the shape of this contract changes |
| `engine_version` | `0.1.0` | the rules, the geometry or the state model change |
| `rules_version` | `1` | any ruleset's mechanics change |
| `ruleset.id` | `catan-1v1-v1` | per ruleset; the base game is `catan-base-v1` |
| `observation.version` | `1` | a column changes meaning. **Appending does not** — see below |
| `observation.size` | `2503` | any block grows |
| `action_space.version` | `1` | an existing index changes meaning. Appending does not |
| `action_space.count` | `325` | an action is added |
| `networks` | `flat-v1`, `structured-v1` | an architecture is added |
| `inference_modes` | `policy`, `alphazero` | — |
| `serialization` | `torch-state-dict/weights-only` | — |
| `max_model_bytes` | 67,108,864 | — |

### Appending does not bump a version

An appended observation block leaves every existing column where it was, and
`training/alphazero/network.py::graft` widens an older checkpoint into it with **zero**
columns — so the grafted network computes exactly the function it computed before. That was
verified rather than assumed: the champion of the time measured 74.7% against the heuristic
before the 1,884 → 2,503 change and 75.6% after being grafted.

Reordering or repurposing a column does bump it, because no graft can rescue that.

The same rule holds for the action space: **append, never insert**. Every existing index keeps
its meaning, and `training/structured_net.py::_validate` checks the block layout at import.

### `accepts` is equality, not compatibility

```python
contract.accepts(observation_size, action_count)
```

A checkpoint from an older observation loads perfectly well and then plays nonsense on the
first move. A caller that wants an older one anyway must go through `graft` deliberately and
end up at the current shapes.

---

## Submitting weights

**Weights only. No executable code, ever.** A submission supplies learned numbers; the platform
supplies the architecture, the observation encoder, the legal-action mask, the engine and the
inference policy.

This is not a stepping stone toward running submitted code. In-process isolation of CPython is
**not achievable** — `sys._getframe(1).f_locals` reaches `env` from inside an agent's
`__call__`, `info["view"]._state` is the live `GameState` because `__slots__` lookup succeeds
before `__getattr__` is consulted, and `import catan.rules; catan.rules.apply = evil` is
equally available. All three were reproduced by execution. Weights-only is the reason none of
that matters.

State the restriction honestly to submitters: this is a **weights competition, not an agent
competition** — everyone submits into one architecture.

### Validation

```sh
python -m training.validate path/to/weights.pt          # locally, before submitting
python -m training.validate path/to/weights.pt --json   # what the platform stores
```

`training/validate.py::validate(path)` is a pure function — no network, no database, nothing
written beside the file — so it runs in a worker with no production credentials and no
outbound access. Every rejection has a stable code:

| code | meaning |
|---|---|
| `unreadable` | not a file, not a checkpoint, or refused by the restricted unpickler |
| `oversize` | over `max_model_bytes`, refused **before** the file is parsed |
| `incompatible_observation` | built for a different `obs_size` |
| `incompatible_actions` | built for a different `num_actions` |
| `unbuildable` | the config does not describe a network this repository can build |
| `missing_tensors` | the architecture has a parameter the file does not |
| `unexpected_tensors` | the file has a parameter the architecture does not |
| `shape_mismatch` | a tensor is the wrong shape |
| `bad_dtype` | not `float32` — refused rather than cast, so the submitter runs what they measured |
| `non_finite` | a NaN or an infinity |
| `probe_failed` | it loaded and then misbehaved on a real position |

The last one is the only check that *runs* the file instead of describing it. Weights can be
the right name, shape and dtype with every value finite and still produce `nan` logits —
float32 overflows between a large matrix and a real activation — and the place that must not
happen is mid-game. The probe asks for one move on a fresh opening at a fixed seed and checks
that the best legal move it picks is legal.

**The checksum is returned either way.** A rejected file still needs a name in an audit log,
and *"the same bad file has now been uploaded forty times"* is a question only a hash can
answer. **The verdict carries the signature**, so a stored one can be re-checked for staleness
rather than trusted forever: a model validated under one observation is not validated under the
next.

### Loading is not running

`training/loading.py` is the **only** module in this repository allowed to call `torch.load`,
and `tests/test_loading.py` fails on the tenth call site. `weights_only=True` throughout — it
stops the pickle executing, and it was verified to load both shipped champions and a full
training checkpoint unchanged.

That is only half of it. `build(dict(checkpoint["config"]))` allocates from numbers in the
file, so a config claiming a two-billion-wide layer is an out-of-memory kill before any tensor
is inspected. `check_config` bounds every dimension, closes over their *product* — sixteen
16,384-wide layers is inside every individual bound and is four billion parameters — and
**raises**, so a refusal cannot be mistaken for "no model present".

---

## The official champion

| | |
|---|---|
| name | `gen6-gilded-beacon` |
| file | `models/champion_az.pt` |
| sha256 | `01893a0c0c1f3f943397d40e756736e66c4252e174bb3d5aacd3d08ce0a1c0a0` |
| network | `structured-v1`, 375,106 parameters |
| promoted | 2026-08-06, **55.25%** over 400 games against the champion before it, [50.35, 60.05] |
| plays at | 64 simulations (`champion.CHAMPION_SIMULATIONS`) |

⚠️ **A rating identity is `(weights, simulations)`, not weights alone.** The same file at 32
and at 64 simulations is two different players at two different costs. A published number that
does not name the budget is a number for a player nobody faces.

⚠️ `models/champion.pt` — the PPO lineage — was promoted at `obs_size` 1868 and **does not
load** at 2503. That is correct behaviour, not a bug; `network.graft` reconciles it and the
interfaces offer the grafted model separately.

⚠️ **Freeze content, not paths.** `models/champion_az.json` records three promotions on one
day. Ranking two submissions "against the champion" on either side of that is two different
opponents. An anchor is `anchors/<sha256>.pt`, and every rating row names the hash.

---

## Installing the runtime

```sh
pip install catania-runtime                 # the engine. no dependencies at all
pip install catania-runtime[inference]      # + torch, to load and play a model
pip install catania-runtime[web]            # + pillow, for the session and the board payload
```

The empty dependency list is the most important line in `pyproject.toml`, and
`tests/test_packaging.py` parses every file in `catan/` and fails if one imports anything
outside the standard library. A consumer that only runs games installs 3.3 MB rather than
torch's 191 MB per process.

During early development the platform pins a git ref rather than an index release:

```
catania-runtime[inference,web] @ git+https://github.com/TheoLindqvist4/CatanAI@runtime-v0.1.0
```

⚠️ **Pin threads before importing torch.** Everything here is a batch of one, so extra threads
add a barrier to wait on and nothing to do inside it: measured **median 1,758 ms against
179 ms**, p90 17,983 ms against 237 ms. Torch sizes its pool at *import*, so this is the first
line of any entrypoint:

```python
from training.threads import pin
pin()                                       # returns False if torch was already imported
```

---

## The rules tripwire

`RULES_VERSION` is hand-written, and a hand-written field is what got forgotten last time.
`contract.rules_digest()` plays four seeded games across both rulesets and hashes the turn, the
phase, the mover and the **full legal set** at every one of 2,178 decisions, then the terminal
position. It moves when legality moves and when `apply` moves; it does not move for a refactor.

`tests/test_contract.py` pins it.

> ⚠️ When that test fails, the expected string is **not** the thing to change first. Bump
> `RULES_VERSION` in the same commit, or find out why legality moved. Updating the string alone
> converts "the rules changed by accident" into "the rules changed on purpose", silently —
> which is worse than deleting the test.
