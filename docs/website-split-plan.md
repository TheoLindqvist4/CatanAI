# The two-repository split, and what it costs

How CatanIA becomes a runtime that a public website can consume, and what CatanIA-Web is
allowed to contain. This is the implementation plan for
[`CatanIA_Dual_Repository_Website_Architecture_and_Go_Live_Plan.pdf`](CatanIA_Dual_Repository_Website_Architecture_and_Go_Live_Plan.pdf),
tailored to what is actually in this repository rather than to what a README describes.

    CatanIA          the engine and the laboratory     github.com/TheoLindqvist4/CatanAI
    CatanIA-Web      the product and the platform      github.com/TheoLindqvist4/CatanAI-Website

The bridge is a versioned package, `catania-runtime`, built from this repository. **The web
repository never contains a second copy of the Catan rules** — not the legality check, not the
observation, not the public-information filter, and not the board-to-pixels layout.

---

## Where this stands — 15 August 2026

**Phase A is complete and tagged `runtime-v0.1.0`** (commit `9d3008d`, branch
`website-split`, pushed to origin).

**Phase B is under way** in `CatanAI-Website` on branch `platform`, pushed. B1, B2 and B4 are
done; **B5, the online game loop, is next**, and B3 (frontend) follows it.

| | | |
|---|---|---|
| B1 | skeleton, compose stack, CI, the boundary | `0237bd5` |
| B2 | PostgreSQL, migrations, accounts | `0780281` |
| B4 | the AI registry, geometry and board art | `1a3036c` |

123 backend tests in 7.5 s. The stack rebuilds from an empty volume: migrations apply, five
built-in opponents seed themselves, `/api/runtime` reports
`CATANIA-1/engine-0.1.0/obs-1:2503/act-1:325/catan-1v1-v1` from the **installed** package.

What the website has established that this repository should know:

- **`app/runtime.py` is the only module allowed to import `catan`, `training` or
  `interfaces`**, enforced by a test that walks every other file's syntax tree — plus a second
  test hunting for rules *retyped* rather than imported.
- The API image is **270 MB with neither torch nor numpy in it**, and plays the heuristic
  perfectly well. `catania-runtime[web]` is enough for a real game; the trained champions are
  what need the worker.
- **`models/*.pt` does not travel with the wheel**, so the registry records the champion and
  reports that this deployment cannot run it. Fetch-and-verify-by-checksum lands with the
  worker.
- ⚠️ **`localhost` costs 2 seconds per connection on Windows** — IPv6 first, containers on
  IPv4 only. Measured 2.083 s against 0.015 s. It was 172 seconds of a 176-second test suite.
- ⚠️ **FastAPI's `@router.get` does not add HEAD**, so cacheable endpoints answered 405 to a
  CDN, a load-balancer probe and `curl -I`.

| | | |
|---|---|---|
| A1 | version identity and the contract | `02f2f92` |
| A2 | one loader, and it does not execute what it is given | `f9df582` |
| A3 | the weight validator | `9bb2b26` |
| A4 | the two leaks closed | `51caac2` |
| A5 | packaging, the thread pin, the contract document | `9d3008d` |

The full suite is **1,029 passed, 1 skipped in 425 s** — budget seven minutes, not six.

Facts established while doing it, which the rest of the plan now depends on:

- `contract.signature()` is `CATANIA-1/engine-0.1.0/obs-1:2503/act-1:325/catan-1v1-v1`.
  The rules digest is `57abae04b5c9a170` over 2,178 decisions.
- **`weights_only=True` loads everything this repository writes** — both champions and a full
  training checkpoint, including `pool.snapshot()`'s tuples. `weights_only=False` now appears
  nowhere outside one warned fallback, and a test walks the tree to keep it that way.
- The official champion is `gen6-gilded-beacon`, sha256
  `01893a0c0c1f3f943397d40e756736e66c4252e174bb3d5aacd3d08ce0a1c0a0`, `structured-v1`,
  **375,106 parameters**, played at 64 simulations. `models/champion.pt` is 1868-wide and does
  not load, which is correct.
- The dependency-free claim is **verified end to end**: the wheel installs into a venv with
  nothing else and plays a 493-decision game with numpy, torch and PIL absent from
  `sys.modules`. Pillow is the `web` extra because `interfaces/web/api.py` reaches `Geometry`
  through `interfaces/render.py`, which imports PIL at module level.
- `api.view(game, seat)` and `api.statistics(game, seat)` exist and are leak-tested on both
  seats. `Game.awaiting(seat)` joins `awaiting_opponent`. The **`log` is still written from
  seat 1's perspective** — public content, wrong pronoun — and a second human seat needs it
  rendered per seat rather than stored once.

### What Phase B has to decide on its first day

- **Docker was not installed on the machine when Phase A was written**, so the compose stack
  has never been started here. Nothing in Phase A depends on it.
- **The champion is not in the wheel.** `packages` lists code, and `models/` is not a package.
  The website should fetch `models/champion_az.pt` from the pinned tag and verify the sha256
  above through `training.validate` — which is the content-addressed anchor rule from §B4
  arriving early, rather than a workaround.
- **`requires-python = ">=3.11"` is a claim about syntax, not a measurement.** Nothing in the
  tree uses `match` or `X | Y` annotations; it has only ever been run on 3.14. Pin the
  backend image deliberately.

---

## What the audit changed

The PDF is written from the README. This repository also contains
[`audit-2026-08-05-public-arena.md`](audit-2026-08-05-public-arena.md), which asked the same
question five days earlier and answered it **by execution**: every containment claim was
re-tested by writing the exploit. Nine of its findings change this plan, and four of them are
work that must happen in *this* repository before the website can accept a single upload.

| Audit finding | What the PDF says | What this plan does |
|---|---|---|
| **B4** — `torch.load(weights_only=False)` at **nine** sites is remote code execution, one of them inside a worker-pool initialiser | "do not deserialize arbitrary executable Python objects" | Phase A2. One loader, `weights_only=True`, config **ranges** validated in a function that raises, and a test that fails on a new `torch.load` call site anywhere in the tree |
| **B9** — no `__version__` anywhere in `catan/`, `training/`, `interfaces/`; no recorded number carries a rules identity | contract must identify ruleset and engine version | Phase A1. `catan/contract.py`, derived from the modules rather than restated beside them |
| **B3** — `info["scores"]` counts hidden Victory Point cards next to `public_scores`; one subtraction reads the opponent's hidden VPs | "private information cannot cross the player boundary" | Phase A4, and it is a leak in the *documented* interface that a well-behaved agent can read — currently pinned as intended by `tests/test_env.py` |
| **B8** — `api.view()` reveals seat 1's hand **unconditionally**, so `GET /api/game/2` reads a stranger's cards | "a player receives only their authorized public state" | Phase A4. One seat-parameterised serializer, per the audit's decision 7: *one function builds the public payload; the browser and the wire are two callers* |
| **§3** — the board is a perfect fingerprint of the seed (20,000 seeds → 20,000 distinct layouts, zero collisions), and seeds were recovered in **63 ms** and **360 ms** | "store the seed" | Phase B5. The seed is 128 bits from `SystemRandom`, stored, and **never served** while a game is live. Replays are released only after a seed set is retired |
| **§5** — Elo/Glicko/TrueSkill are online filters built to *forget*; submitted weights are frozen files whose strength is exactly stationary | "Leaderboard and ELO presentation" | Phase B8. Batch Bradley-Terry MAP, refit from scratch on every update, anchored at `HeuristicAgent(noise=0) = 0`. Displayed on an Elo scale so the word still means something to a visitor |
| **§5** — 695 Elo per unit win rate at p=0.5, so 400 games is **±34 Elo** and the gap between the last two champions is ~12 | "show confidence/sample size information where useful" | The interval is the **primary** visual, not a tooltip. A table sorted on point estimates reorders itself on noise every refit |
| **§7** — torch threads are never pinned in the web process: median **1,758 ms** against **179 ms** at one thread, p90 17,983 ms | not mentioned | Phase A5 ships the helper; every API and worker entrypoint calls it before the first forward pass. 10x median, ~75x tail, one line |
| **§6** — torch is **191 MB per process** and costs **3.4 s** to import; `POST /api/game` currently rebuilds the network per request, measured **1.94 s on the request thread** | "the worker should be the process that loads CatanIA and model weights" | Phase B5. Models load once per worker into a process-local cache, keyed by checksum. The API process does not import torch at all |

Two further audit results size the platform rather than change it. A searching game costs
**56x** a heuristic one (8.9-9.2 s against 0.161 s), so a 400-game ranked match is ~3,600 CPU-
seconds; and a 16-worker match holds ~7 GB, so **one ranked match runs at a time** on a box
this size. Arena scheduling is a queue with a concurrency limit of one by default, not a
thread pool.

⚠️ **Every performance constant written in this repository is 2.4-3.7x optimistic** (audit
§6). Capacity planning uses the audit's measured figures, not the docstrings.

---

## What is already right, and must not be rebuilt

The reuse inventory here is unusually good, because this project has repeatedly built the
general thing rather than the specific one.

- **`interfaces/web/api.py` knows nothing about HTTP.** Its own docstring predicted this
  split: *"Swapping in FastAPI later would rewrite this file and touch nothing else."* That
  holds for routing. It fails for the *process model* — `Games` is a process-local dict — which
  is exactly the part the website replaces.
- **`api.view()` touches only nine attributes of `Game`.** So the wire payload, the browser
  payload and a future replay scrubber are one serializer with three callers.
- **The record is the seed and the move indices**, and `recorder.replay()` proves it
  reconstructs the game exactly. The website persists the same thing, which makes reconnect,
  crash recovery, Arena TV and dispute resolution one mechanism instead of four.
- **An agent is `callable(observation, info) -> int`.** A rated submission becomes playable
  with one dict entry in `OPPONENTS` and no client change.
- **`arena.compete`** is already paired, seat-swapped and worker-count independent.
- **`champion.promote`** is already a Wilson-gated ladder with a `forced` audit trail.
- **`layouts.py` + `network.graft`** already carry a checkpoint across an observation change.

The local app stays local and unchanged. That is audit decision 9.8, taken deliberately: it is
the only part of this repository with users, and it should not carry a regression risk for a
platform that does not exist yet. `python -m interfaces.web` keeps working with zero
dependencies for as long as this repository exists.

---

# Phase A — CatanIA becomes a runtime

Five phases, all in this repository, all with the full suite green
(`python -m pytest tests -q`, ~936 tests, about six minutes).

### A1 — Version identity and the runtime contract

`catan/contract.py`, dependency-free, **derived and never restated**: it imports
`encoder.SIZE`, `action_space.NUM_ACTIONS` and the rulesets rather than writing the numbers
down a second time, because a contract that can disagree with the engine is worse than no
contract.

```
CATANIA-1
engine_version        from catan.__version__
ruleset_version       catan-1v1-v1        (per ruleset, not global)
observation_version   1     size 2503     (from encoder.SIZE)
action_space_version  1     count 325     (from action_space.NUM_ACTIONS)
network               structured-v1
inference_modes       policy | alphazero
serialization         state_dict, weights_only-loadable
```

`contract.signature()` is the string that goes in every game row, every rating row and every
model record. Fixes **B9**.

*Done when:* a test asserts every field agrees with the module it came from, the contract is
JSON round-trippable with no imports beyond the standard library, and `signature()` changes if
and only if one of its inputs does.

### A2 — One loader, and it does not execute what it is given

`training/loading.py::load_checkpoint(path, *, trusted)` is the only module in the tree allowed
to call `torch.load`. Untrusted means `weights_only=True` — which the audit verified **already
loads both shipped champions unchanged** on torch 2.13.0+cpu, so there is no migration and no
metadata sidecar, only one word per site. All nine call sites move onto it.

`weights_only=True` is not the whole trust story: `network.py` does `build(dict(ckpt["config"]))`
before any tensor is inspected, so hostile dimensions are an allocation bomb. The loader
validates config **ranges** and **raises** — `champion.load`'s bare `except Exception: return
None` makes a hostile file indistinguishable from a missing one.

*Done when:* nine sites converted; a test greps the tree and fails on any new `torch.load`;
hostile fixtures (a pickle that would execute, a 2^31-wide layer, a missing key) are refused
with a named exception; both champions still load.

### A3 — The weight validator

`training/validate.py::validate(path) -> Report`. A pure function: no network, no database, no
filesystem beyond the file it is handed. This is what CatanIA-Web's validation worker calls,
and it lives here because the authority on what a valid model *is* belongs beside the
architecture that defines it.

Checks, in order, each with its own rejection reason: loadable under `weights_only=True` ·
config within range · `obs_size == contract.OBSERVATION_SIZE` · `num_actions ==
contract.ACTION_COUNT` · tensor names and shapes identical to a freshly built net · dtype ·
every value finite · size cap. Returns the SHA-256 either way, because a rejected file still
needs a name in the audit log.

*Done when:* each rejection reason has a test with a fixture that triggers it and only it, and
`models/champion_az.pt` validates clean.

### A4 — Close the two leaks the website would otherwise inherit

**B3.** `info["scores"]` counts hidden Victory Point cards. `public_scores` is beside it, and
the subtraction is the fact that decides whether an opponent is one build from winning. It is
readable with public keys only, so no sandbox and no process boundary closes it.
`tests/test_env.py` currently pins the difference as intended; that test is the change.
`tests/helpers.py::scramble_hidden_state` grows to cover `info`, which it has never pointed at.

**B8-serializer.** `api.view(game)` hardcodes `HUMAN = 1`. It gains a seat, defaulting to the
current behaviour so the local app is untouched, and the leak tests run against **both** seats.
Audit decision 7 is explicit that this must be one function: three serialisations of the
hidden-information filter, in the one area this project has been most careful about, is how the
filter comes to disagree with itself.

*Done when:* a leak test drives whole games and asserts that seat 2's payload never moves when
seat 2's hidden state is scrambled at constant public counts, and that neither payload nor
`info` bounds the opponent's hidden VP count.

### A5 — Packaging, and the thread pin

`pyproject.toml` publishing **`catania-runtime`**: `catan` with no dependencies at all, plus
extras — `inference` (torch), `render` (pillow, which `interfaces.render` imports at module
level and `api.py` pulls in for `Geometry`), `dev` (pytest).

`training/threads.py::pin(n=1)` sets the OpenMP variables **before torch is imported**, because
the pool is sized at import. Every website entrypoint calls it first. Worth 10x median and ~75x
tail latency; CLAUDE.md already records why, and the web process never got the treatment the
training code did.

`docs/model-contract.md` is the human-readable CATANIA-1 specification, and the website
repository links to it rather than copying it.

*Done when:* `pip install -e .` in a clean venv gives a working `python -m interfaces.web` with
**no** dependencies installed; `pip install -e .[inference]` adds the champions; the full suite
is green; and the commit is tagged **`runtime-v0.1.0`** — the PDF's Step 1 frozen baseline.

---

# Phase B — CatanIA-Web

The web repository pins `catania-runtime @ git+https://github.com/TheoLindqvist4/CatanAI@runtime-v0.1.0`.
That is the PDF's "alternative during early development", chosen over a published package
because there is no index to publish to yet; the `pyproject.toml` from A5 makes a real release a
one-line change when there is.

### B1 — Skeleton

README, `docs/architecture.md`, `.env.example`, `.gitignore`, `docker-compose.yml`
(frontend · api · worker · scheduler · postgres · redis · caddy), GitHub Actions running lint,
types, unit tests, migration check, dependency scan, docker build and a stack smoke test.

*Done when:* `docker compose up` starts every service and CI is green on a pull request.

### B2 — Backend core, and a health endpoint that proves the boundary

FastAPI, typed settings, structured JSON logs with correlation ids, security headers, CORS
restricted to the real frontend origins.

`GET /api/health` returns `contract.signature()` **read from the installed package**. That is
the PDF's proof that the two repositories work together, and it fails loudly if the pin drifts.
The API process does not import torch — 191 MB and 3.4 s per process buys it nothing.

### B3 — Database, migrations, accounts

SQLAlchemy + Alembic over PostgreSQL, tables per PDF §13. Argon2id. HttpOnly, SameSite,
Secure cookies. Sessions revocable. Redis-backed rate limits on login, registration and reset.
Roles: `anonymous · user · creator · moderator · admin · service`.

Every game row carries `contract.signature()`, so a number is never comparable to one from a
different engine by accident. That is B9 closed at the platform end.

*Done when:* ownership is tested — a user cannot read or mutate another user's profile, AI,
submission or game — and the audit log records every privileged action.

### B4 — The AI registry

The official champion (`models/champion_az.pt`, `gen6-gilded-beacon`) and the built-in agents
become registry rows with checksums and contract signatures. Board art and `/api/geometry` are
served from the installed package with `ETag` and `Cache-Control` — 1.64 MB currently shipping
with no caching headers at all.

Freeze **content, not paths**: anchors live at `anchors/<sha256>.pt` and every rating row names
the anchor's hash. `models/champion_az.json` records three promotions on one day; ranking two
submissions against "the champion" on either side of that is two different opponents.

### B5 — One online game, server-authoritative

The session wraps `interfaces.web.api.Game`. The browser sends an action index; the backend
validates and applies it through the engine; the response is the seat-filtered payload from A4.

**Persistence is the seed and the moves**, exactly as `recorder.py` already does it. Reconnect,
API restart and worker crash all recover the same way: replay. The seed is 128 bits from
`SystemRandom` and is **never** in a payload while the game is live — audit §3 recovered a seed
from the tiles block of an observation in 360 ms, and the board must be shown to anyone playing.

**AI moves are computed in the worker**, never in the API process. The request is
`(seed, ruleset, moves so far)` and nothing else, because the engine is deterministic — so the
worker needs no shared memory and no session affinity, and any worker can answer any game.
Models load once per worker, keyed by checksum. WebSocket frames carry server-generated
sequence numbers; duplicate client messages are idempotent; a disconnect does not end the game.

*Done when:* the PDF's central loop runs — create account, choose AI, play, finish, result
saved, statistics update — and an integration test proves the WebSocket never emits seat 2's
hand, the dev deck order, the dice deck or the seed.

### B6 — Frontend

Next.js + TypeScript. The board component consumes `/api/geometry` and the `view` payload and
**works nothing out** — no rules, no board generation, no scoring, no legality. Pages: Home,
Play, Login/Register, Account.

The audit is specific here: extract the drawing out of `app.js` (bound to a module-level
`state` and fixed DOM ids) *before* a second page exists. Copying 190 lines of SVG into a
second file is the second-implementation problem this project already removed from JavaScript
once.

### B7 — Submissions

Upload → quarantine bucket → validation worker calling A3's `validate()` → checksum → registry.
The state machine (`DRAFT · UPLOADED · VALIDATING · VALIDATED · BENCHMARKING · ARENA_READY ·
ACTIVE · ARCHIVED · REJECTED`) lives in PostgreSQL, not in folder names. Accepted artifacts are
immutable and addressed by SHA-256. The validation worker holds no production database
credentials and no outbound network.

**Weights only.** No submitter code executes anywhere, ever — which is what makes the safety
claim trivially true rather than defended. Audit B1/B2 proved that in-process isolation of
CPython is not achievable: `sys._getframe(1).f_locals` reaches `env` from inside an agent's
`__call__`, and `import catan.rules; catan.rules.apply = evil` is equally available. Weights-
only is not a stepping stone to running code; it is the reason none of that matters.

State the restriction honestly on the Submit AI page: this is a **weights competition, not an
agent competition** — everyone submits into one architecture.

### B8 — Rating and the leaderboard

Batch Bradley-Terry MAP over the whole append-only game log, refit from scratch on every
update, so the table is order-independent and a pure function of its input. `HeuristicAgent(0)`
pinned at 0. The seat term fitted as a free parameter and published as a monitored row — it is
49.8% [48.7, 50.9] over 8,000 games today, and if a rules change moves it the table should say
so rather than absorb it.

Rating identity is **`(weights, simulations, budget)`**, not weights alone. Submitters choose
their own simulation count, and the ladder therefore ranks compute spend as well as skill; say
that on the page instead of pretending otherwise.

Truncation is scored on `public_scores` at the truncation ply — ahead wins, level is a loss for
both. Catan has no draws, so dropping truncated games from the denominator (which `arena.py`
and `evaluate.py` both do) makes refusing to build a free non-loss. **Fix this before the first
row is published**; changing it later renumbers every result.

### B9 — Arena workers and Arena TV

Scheduler, Redis match queue, workers with a concurrency limit — one ranked match at a time by
default, from the ~7 GB measurement. CPU-time budgets (`RLIMIT_CPU`), not wall-clock: a
wall-clock deadline on a shared box manufactures forfeits, and forfeits are not missing at
random. A rung whose forfeit rate exceeds a threshold is discarded and re-run, not published.

Arena TV reuses the same `view` serializer through a `Replay` that duck-types the nine
attributes it reads. The only new UI is a scrubber; cache an `env.clone()` every 20 plies so
seeking is not O(n) per frame.

### B10-B11 — Tutorials, creator dashboard, payments, admin, production

In the PDF's order, and after the free loop is reliable. Entitlement is a backend decision, not
a frontend variable; creator revenue is an immutable ledger.

---

## The order this actually ships in

**Go-live MVP is A1-A5 then B1-B6** — the PDF's First Sprint, which proves the one thing worth
proving early: the two repositories work together without merging their responsibilities. B7
onward is the community loop and follows it.

## Definition of done for the split

- Somebody clones **CatanIA**, installs nothing, and plays or trains locally.
- Somebody clones **CatanIA-Web** and runs the whole stack with `docker compose up`.
- CatanIA-Web contains no Catan rules, no observation encoder, no legality check and no
  hidden-information filter of its own.
- A CatanIA version is pinned, upgraded and rolled back deliberately.
- A user submits weights and never code, and the platform validates without trusting them.
- One runtime powers local play, online play and Arena matches.
- Every published number names its engine, its ruleset, its anchor and a checksum.
