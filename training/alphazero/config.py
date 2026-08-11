"""The run's settings, and where they are read from.

The guide specifies ``configs/train.yaml``, and that is what this reads — but PyYAML is not
a dependency of this project and adding one for a file of twenty ``key: value`` lines is a
poor trade. So :func:`load_config` parses the subset the guide's example actually uses: flat
scalars, ``#`` comments, and nothing nested. A file that uses more than that is **rejected
with the offending line**, rather than half-parsed into a config that is quietly wrong — a
training run misconfigured by a silently dropped key is a wasted afternoon that looks like a
bad idea.

Defaults live in :data:`DEFAULTS`, so a missing key is never a crash and the YAML is a set of
overrides rather than a required document.
"""

import pathlib

#: Everything the pipeline reads, with the value used when the config does not say.
#:
#: The numbers that differ from the guide's recommended configuration are marked, because
#: each is a deliberate response to *this* machine — a 20-core CPU with no GPU — rather than
#: a preference. See ``docs/decisions/0023-alphazero-self-play.md``.
DEFAULTS = {
    # --- self-play ------------------------------------------------------------------- #
    "self_play_workers": 14,        # guide: 16. Leaves cores for the trainer and the web UI.
    "envs_per_worker": 12,          # games in flight per worker; sets the evaluation batch.
                                    # Games in a worker advance in lockstep, one simulation
                                    # each per round, so they also *finish* together: the
                                    # width is how long the pipeline takes to deliver its
                                    # first samples. At 24 that is ~65s of worker time and
                                    # samples then arrive in one lump; at 12 it is half that,
                                    # for about 6% more time per network evaluation.
    "mcts_simulations": 96,         # guide: 200. Not 200 because that is ~4 games/sec across
                                    # the whole machine; not 48 because at 48 the labels are
                                    # only four points better than the policy already is.
                                    # See configs/train.yaml for the measurement.
    "max_turns": 400,
    "temperature": 1.0,
    "temperature_final": 0.15,
    "temperature_opening_turns": 20,
    "c_puct": 1.5,
    "fpu": 0.25,
    "dirichlet_alpha": 0.5,
    "dirichlet_weight": 0.10,  # AlphaZero uses 0.25 at 800 sims; at 96 it flips 24% of labels

    # --- the Gumbel root — implemented, measured, and OFF ------------------------------ #
    #
    # ⚠️ This was the highest-leverage change on paper and it measured **worse**, badly.
    # Agreement of the recorded label with a clean 400-simulation search, paired, 250
    # positions, against a raw policy that already agrees 72.8%:
    #
    #     PUCT 96, noise 0.10        73.2%    +0.4   p=1.00
    #     Gumbel, c_visit 0.5        59.2%   -13.6   p=0.0019
    #     Gumbel, c_visit 2          57.2%   -15.6   p=0.0002
    #     Gumbel, c_visit 5          54.4%   -18.4   p<0.0001
    #     Gumbel, c_visit 12         55.2%   -17.6   p<0.0001
    #     Gumbel, c_visit 50 (paper) 49.2%   -23.6   p<0.0001
    #
    # Monotone in the scale, and the small-scale limit is not an escape: as sigma goes to
    # zero the target degenerates to softmax(logits), which is the prior, which scores 72.8%
    # — still no better than PUCT. There is no setting where it wins.
    #
    # **Why, and it is not a bug in the implementation.** Gumbel's improvement guarantee
    # rests on the Q estimates carrying signal. Here every simulation resamples a *different*
    # determinized world — a fresh guess at the opponent's hand and the dice deck — and
    # scores it with a value head measured at 14% of held-out outcome variance. At 96
    # simulations over ~9.6 legal moves each Q is a handful of samples of a noisy quantity.
    # sigma(q) then amplifies differences that are mostly determinization variance, and
    # visit counts, which lean on Q far more weakly, survive it better.
    #
    # Kept in the code, off, with tests: it is correct, it is cheap to re-enable, and it
    # would very likely pay off once the value head is worth leaning on. See 0026.
    "gumbel": False,
    "gumbel_actions": 16,      # sampled without replacement; usually the whole legal set

    # --- playout cap randomization ---------------------------------------------------- #
    #
    # KataGo measures 1.37x. Here the reason is the second-order one: a game contributes a
    # quarter as many rows, so the same buffer holds about four times as many distinct
    # games — and ~900 games was what let the value head memorise board identity.
    "playout_cap_probability": 0.25,   # chance a move gets the full budget and is recorded
    "playout_cap_fast": 24,            # simulations for the moves that are not recorded

    # --- the opening, which the cap and PUCT both handle badly ------------------------ #
    # An opening settlement is 54 legal moves against a normal turn's six, and a player's two
    # placements are seven plies apart, so this is where the *pair* is decided. Measured:
    # setup is 4.6% of searchable decisions and the cap recorded a quarter of them, so about
    # one placement per game reached the buffer. Setup is now always recorded (see self_play),
    # and these two settings are what make the extra rows worth having. What PUCT reaches at
    # this root on its own, and what the floor changes, is measured once in Search's
    # `root_min_visits` docstring and argued in
    # docs/decisions/0028-the-opening-is-fifty-four-moves-wide.md.
    "setup_simulations": 400,          # at a settlement only; the road has three options
    "setup_root_min_visits": 4,        # visits every spot gets before PUCT may concentrate.
                                       # 4 x 54 = 216 of the 400, which fits with room over;
                                       # 8 x 54 = 432 does not, and degenerates silently.
                                       # The sweep is stripped from the target by
                                       # _discretionary_counts. 0 restores plain PUCT.

    # --- replay ---------------------------------------------------------------------- #
    "replay_buffer_size": 180_000,  # guide: 2,000,000. The observation is 2,503 floats,
                                    # so this is ~0.9 GB in float16; see replay_buffer.
    "min_buffer": 4_000,            # positions before the first gradient step

    # --- learning -------------------------------------------------------------------- #
    "batch_size": 512,
    "training_batches": 60,        # guide: 1,000, paired with 20,000 games per iteration
    "generate_seconds": 35,         # per iteration. A clock, not a sample count: see
                                    # Generator.run for why a count wastes most of the pool.
    "positions_per_iteration": 0,   # 0 means "use the clock". Kept for benchmarks and tests.
    "learning_rate": 7e-5,          # guide: 1e-3. Warm-started, so the low end. 2e-4 was
                                    # above the 3e-5..1e-4 band CLAUDE.md gives for
                                    # fine-tuning, and a run at 2e-4 peaked and then drifted
                                    # back down over the next 40 iterations.
    "weight_decay": 1e-4,
    "value_weight": 1.0,
    "grad_clip": 1.0,

    # --- what the value head is actually trained on ----------------------------------- #
    "root_value_weight": 0.5,   # target = 0.5*outcome + 0.5*search root value
    "owner_weight": 0.15,       # final ownership per vertex and road, cross-entropy
    "margin_weight": 0.15,      # final victory-point margin, MSE
    "max_per_game": 8,          # rows one game may contribute to a 512-row batch

    # --- evaluation and promotion ---------------------------------------------------- #
    #
    # The in-loop check is deliberately cheap and deliberately measures the *network*, not
    # the searcher. `play_match` is sequential and a searching agent evaluates the network
    # once per simulation at batch 1, so 200 games at 32 simulations costs about thirteen
    # minutes — which, every six iterations, is most of a training run. At 0 simulations the
    # agent plays its raw policy, which is one forward pass a move and takes about 40
    # seconds; that still answers the only question the loop needs answered, which is whether
    # the network is getting better.
    #
    # Whether the *search* helps is a different question, and it is answered once, properly,
    # by the promotion gate at `promotion_simulations`.
    #
    # ⚠️ `evaluate_every: 0` switches the in-loop check **off**, and that is the default,
    # because the number it produces is one CLAUDE.md already says not to act on. It scores
    # the *raw policy*, which has twice been measured moving in the opposite direction to
    # search-ranked strength — one run read 62.5 -> 60.4 -> 50.5 -> 51.0 and looked finished,
    # while the arena put its last checkpoint first and its "peak" last.
    #
    # It is not free. Measured from five runs' own metrics.jsonl, by differencing the
    # unaccounted residual on evaluating iterations against non-evaluating ones:
    #
    #     az_run_1h        4 evaluations   21.0 s each   2.33% of the run
    #     az_run_1h_b      4               16.8 s        1.87%
    #     az_run_2h        9               19.2 s        2.40%
    #     az_stage1_run    4               19.9 s        3.62%
    #     az_stage2_run    5               29.2 s        4.38%
    #
    # and it is *idle* time: the parent plays 200 sequential games at batch 1 while all
    # fourteen workers wait. Off, the run snapshots instead (`Trainer.snapshot`) and
    # `training/alphazero/arena.py` ranks the snapshots afterwards with search, which is the
    # only ranking this repository considers evidence.
    #
    # Set it to 25 to get the smoke alarm back — it is a real smoke alarm, it just costs
    # 2-4% and cannot be read as progress.
    "evaluation_games": 200,        # 100 gave +-10 points, which cannot see a real change
    "promotion_games": 400,
    "promotion_threshold": 0.55,
    "evaluate_every": 0,            # iterations; 0 = off, and snapshot instead
    "eval_simulations": 0,          # 0 = the raw policy; see above
    #: What the in-loop check plays against. The reigning AlphaZero champion is the yardstick
    #: that matters — "did this run produce a better player than the one we have" — and unlike
    #: the heuristic it is not affected by rule changes. Both sides run at `eval_simulations`,
    #: so at 0 this is an honest network-against-network comparison; it is still the raw
    #: policy, and CLAUDE.md's warning that the policy column moves opposite to search-ranked
    #: strength still applies. The authority remains `training/alphazero/arena.py`.
    "evaluation_opponent": "champion_az",   # or "heuristic"

    # --- housekeeping ---------------------------------------------------------------- #
    "checkpoint_interval_minutes": 15,
    "seed": 0,
    "warm_start": "models/champion_az.pt",
    "run_directory": "checkpoints/alphazero",
}

CONFIG_PATH = pathlib.Path("configs/train.yaml")


class Config(dict):
    """A dict that refuses unknown keys, so a typo is not a silently ignored setting."""

    def __init__(self, values=None):
        super().__init__(DEFAULTS)
        if values:
            unknown = set(values) - set(DEFAULTS)
            if unknown:
                raise KeyError(
                    f"unknown setting(s) {sorted(unknown)} — known settings are "
                    f"{sorted(DEFAULTS)}"
                )
            self.update(values)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def describe(self):
        """The settings that differ from the defaults, for a run's header line."""
        return {k: v for k, v in self.items() if DEFAULTS[k] != v}


def parse(text):
    """The flat subset of YAML the guide's configuration uses.

    Raises:
        ValueError: on nesting, lists, or anything else outside ``key: scalar``.
    """
    values = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line[0] in " \t-":
            raise ValueError(
                f"{CONFIG_PATH}:{number}: only flat 'key: value' lines are supported, "
                f"got {raw.strip()!r}"
            )
        if ":" not in line:
            raise ValueError(f"{CONFIG_PATH}:{number}: expected 'key: value', got {raw!r}")
        key, _, value = line.partition(":")
        values[key.strip()] = _scalar(value.strip())
    return values


def _scalar(text):
    if text in ("true", "True", "yes"):
        return True
    if text in ("false", "False", "no"):
        return False
    if text in ("null", "~", ""):
        return None
    body = text.replace("_", "")
    try:
        return int(body)
    except ValueError:
        pass
    try:
        return float(body)
    except ValueError:
        pass
    return text.strip("'\"")


def load_config(path=CONFIG_PATH, **overrides):
    """The configuration, from ``path`` if it exists, with keyword overrides on top."""
    path = pathlib.Path(path)
    values = parse(path.read_text(encoding="utf-8")) if path.is_file() else {}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Config(values)
