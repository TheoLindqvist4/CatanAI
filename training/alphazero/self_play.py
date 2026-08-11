"""Generating games, and the positions learned from them.

One self-play game is a loop of "search, play the most-visited move, repeat", and the naive
version of that runs the network once per simulation at batch 1. Measured on this machine
the structured network does 3,133 evaluations a second at batch 1 and 22,391 at batch 32 —
the same arithmetic, seven times the throughput, because a batch-1 matmul spends its life in
call overhead rather than in FLOPs.

So a worker does **not** play one game at a time. :class:`Generator` keeps ``width`` games in
flight, asks each of their searches for a position to evaluate, evaluates all of them in one
forward pass, and hands the answers back. The games are otherwise completely independent —
different boards, different seeds, no shared tree — and they drift out of step with each
other immediately, which is fine and in fact helps: a batch then contains positions from
every stage of a game rather than 32 openings.

**What is recorded.** One sample per *searched* decision: the observation from the mover's
point of view, the legality mask, the visit-count distribution as the policy target, and —
once the game ends — that mover's result as the value target. Forced moves are skipped: their
policy target is a one-hot over a single option and carries no information, and they are about
a third of all decisions.

**Both seats are recorded from every game.** Catan's env reports a reward for the player who
just acted and the winner is always that player (``CLAUDE.md``), so a learner that used the
returned reward would train on winners only. Value targets here are written from
``info["winner"]`` against each sample's own mover.

**Temperature.** The guide's schedule: 1.0 for the opening, then near-greedy. Sampling in the
opening is what produces different games from the same network; without it a deterministic
policy plays one game per board and the buffer fills with copies.
"""

import random
import time

import numpy as np

from catan import encoder, rules
from catan.env import CatanEnv
from catan.rulesets import RANKED_1V1
from catan.state import NO_OWNER, Phase
from catan.topology import NUM_ROADS, NUM_VERTICES
from training.alphazero import replay_buffer
from training.alphazero.determinize import determinize
from training.alphazero.mcts import Search

#: Turns after which a training game is abandoned as a draw. Matches the PPO trainer's
#: ``TRAINING_MAX_TURNS`` so the two techniques' games are the same length.
TRAINING_MAX_TURNS = 400


class Sample:
    """One recorded decision, waiting for the game to tell it whether it was a win."""

    __slots__ = ("obs", "mask", "index", "probability", "player", "root_value")

    def __init__(self, obs, mask, index, probability, player, root_value=0.0):
        self.obs = obs
        self.mask = mask
        self.index = index
        self.probability = probability
        self.player = player
        #: The search's own opinion of this position, in this mover's frame. A second value
        #: target beside the game result: the result is one bit shared by ~195 decisions,
        #: which is why the value head was measured explaining 14% of held-out variance.
        self.root_value = root_value


def final_ownership(state, player):
    """Who owns each vertex and each road at the end, in ``player``'s frame.

    ``0`` nobody, ``1`` mine, ``2`` theirs — 54 vertices then 72 roads, as one int8 row.
    An auxiliary target: dense, per-element, and impossible to answer by recognising which
    board this is, which is exactly the failure the plain outcome target permits.
    """
    owners = np.zeros(NUM_VERTICES + NUM_ROADS, dtype=np.int8)
    for vertex in range(1, NUM_VERTICES + 1):
        owner = state.vertex_owner[vertex]
        owners[vertex - 1] = 0 if owner == NO_OWNER else (1 if owner == player else 2)
    for road in range(1, NUM_ROADS + 1):
        owner = state.edge_owner[road]
        owners[NUM_VERTICES + road - 1] = (
            0 if owner == NO_OWNER else (1 if owner == player else 2))
    return owners


def final_margin(state, player, target):
    """Final victory-point difference in ``player``'s frame, in units of the target score."""
    others = [p for p in state.player_order if p != player]
    mine = rules.victory_points(state, player)
    theirs = max(rules.victory_points(state, p) for p in others) if others else 0
    return float(np.clip((mine - theirs) / target, -1.0, 1.0))


class Game:
    """One self-play game in progress, advanced one simulation at a time.

    The protocol is the same as :class:`~training.alphazero.mcts.Search`'s, lifted to a whole
    game: :meth:`request` returns a position to evaluate or ``None`` when the game is over,
    and :meth:`deliver` supplies the answer. Between the two, this class does everything that
    is not a network call — starting searches, playing moves, resetting.
    """

    def __init__(self, seed, config, rng):
        self.config = config
        self.rng = rng
        self.env = CatanEnv(num_players=2, ruleset=RANKED_1V1,
                            max_turns=config.get("max_turns", TRAINING_MAX_TURNS))
        self.seed = seed
        self.samples = []
        self.finished = []          # completed games' samples, drained by the caller
        self.results = []           # (winner, turns, decisions) per completed game
        self.search = None
        self._start()

    # ------------------------------------------------------------------ #

    def _start(self):
        obs, self.info = self.env.reset(seed=self.seed)
        self.info["obs"] = obs
        self.samples = []
        self.decisions = 0
        self.searched = 0
        self._searches = 0
        #: Which game a sample came from, so the trainer can cap how many rows one game
        #: contributes to a batch. Seeds are distinct per game and per worker, so this is
        #: unique across the pool without any shared counter.
        self.game_id = int(self.seed)
        self.recording = True

    def _temperature(self):
        opening = self.config.get("temperature_opening_turns", 20)
        if self.env.state.turn_number < opening:
            return self.config.get("temperature", 1.0)
        return self.config.get("temperature_final", 0.15)

    def _open_search(self):
        """Begin a search at the current position, on a world the mover is entitled to.

        The search gets its **own** generator rather than the environment's. Sharing would
        make the real game's dice depend on how many simulations were spent thinking about
        the position before them, so changing the simulation count would change the games a
        seed produces — and "deterministic with a fixed seed" is on the guide's pre-flight
        checklist for a reason: it is the only way a regression can be reproduced.
        """
        player = self.info["player"]
        self._searches += 1
        stream = random.Random(self.seed * 1_000_003 + self._searches)
        world = determinize(self.env.state, player, rng=stream)

        # Playout cap randomization (KataGo, 1.37x measured). Most moves get a cheap search
        # that is played but never recorded; a minority get the full budget and are the only
        # ones written to the buffer. Two effects, both wanted here: the average move costs
        # far less, and — because a game contributes a quarter as many rows — the same buffer
        # holds about four times as many distinct games. The second is the point. The value
        # head was memorising board identity from ~900 games; this is the cheapest way to
        # multiply that number without generating more data.
        cap = self.config.get("playout_cap_probability", 0.0)
        self.recording = cap <= 0 or self.rng.random() < cap
        budget = (self.config.get("simulations", 48) if self.recording
                  else self.config.get("playout_cap_fast", 24))

        # --- the opening, which is neither cheap nor typical ------------------------- #
        # A settlement placement offers 54 moves against a normal turn's six, and the two
        # placements a player makes are seven plies apart, so this is where "which pair of
        # spots do I end up with" is decided. Two things are wrong with treating it as one
        # more decision. It is 4.6% of searchable decisions and the playout cap records a
        # quarter of those, so about one placement per game reached the buffer. And PUCT
        # tuned for six moves looks at a handful of the 54, so the choice was made among
        # spots the prior already liked. `root_min_visits` fixes the second, and it is not
        # paid for in time. The measurement is written down once, in Search's
        # `root_min_visits` docstring, and argued in
        # docs/decisions/0028-the-opening-is-fifty-four-moves-wide.md.
        phase = self.env.state.phase
        if phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
            # Always recorded, including the road: pointing the opening road at what you
            # intend to expand toward is part of the placement, and there are only three of
            # them, so it costs almost nothing to search and is worth learning.
            self.recording = True
            budget = self.config.get("simulations", 48)
        floor = 0
        if phase is Phase.SETUP_SETTLEMENT:
            budget = self.config.get("setup_simulations", budget)
            floor = self.config.get("setup_root_min_visits", 0)

        self.search = Search(
            world,
            budget=budget,
            rng=self.rng,
            root_min_visits=floor,
            c_puct=self.config.get("c_puct", 1.5),
            fpu=self.config.get("fpu", 0.25),
            # A fast search is never trained on, so root exploration would only add variance
            # to a move that is actually played. Exploration belongs to the recorded ones.
            noise=self.config.get("dirichlet_weight", 0.25) if self.recording else 0.0,
            alpha=self.config.get("dirichlet_alpha", 0.5),
            max_turns=self.env.max_turns,
            gumbel=self.config.get("gumbel", False),
            gumbel_actions=self.config.get("gumbel_actions", 16),
        )

    def request(self):
        """A position to evaluate, or ``None`` when this game has nothing to ask.

        Plays out every forced move and every finished search inside the loop, so the caller
        only ever sees real work.
        """
        while True:
            if self.info["done"]:
                self._finish()
                self.seed += 7_919                  # a prime, so seeds do not collide
                self._start()
                continue
            if self.search is None:
                self._open_search()
            if not self.search.searchable:
                forced = self.search.forced
                if forced is None:
                    # A chance node at the root: the env rolls for us on the next step, and
                    # a state with no decision cannot be reached through step() anyway.
                    raise RuntimeError(
                        f"no decision at {self.env.state.phase.name} — the environment "
                        f"should have advanced past it"
                    )
                self.search = None
                self._play(forced, record=False)
                continue
            pending = self.search.request()
            if pending is not None:
                return pending
            self._commit()

    def deliver(self, probabilities, value):
        self.search.deliver(probabilities, value)

    # ------------------------------------------------------------------ #

    def _commit(self):
        """The search is finished: record the target and play the move."""
        action = self.search.best_action(self._temperature())
        if self.recording:
            # `policy_target` rather than `visit_counts`: under the Gumbel root the visits
            # are spread deliberately evenly by Sequential Halving, so the counts carry much
            # less than the completed-Q distribution does. For a plain PUCT search the two
            # are the same thing.
            actions, weights = self.search.policy_target()
            index, probability = replay_buffer.sparse_policy(actions, weights)
            self.samples.append(Sample(
                obs=np.fromiter(self.info["obs"], dtype=np.float16, count=encoder.SIZE),
                mask=replay_buffer.pack_mask(self.info["mask"]),
                index=index, probability=probability,
                player=self.info["player"],
                root_value=self.search.root_value(),
            ))
            self.searched += 1
        self.search = None
        self._play(action, record=True)

    def _play(self, action, record):
        obs, _, _, _, info = self.env.step(action)
        # step() already returns the observation for whoever must act next, which is exactly
        # what the next sample needs. Re-encoding here would cost 0.07 ms per decision for
        # an identical array.
        info["obs"] = obs
        self.info = info
        self.decisions += 1

    def _finish(self):
        """Stamp the outcome and the auxiliary targets onto every sample, and retire it."""
        winner = self.info["winner"]
        state = self.env.state
        target = self.env.ruleset.victory_points_to_win
        # One ownership row and one margin per *seat*, not per sample: every sample from a
        # seat shares them, and computing them 200 times per game would cost more than the
        # search that produced the samples.
        by_seat = {}
        for sample in self.samples:
            if sample.player not in by_seat:
                by_seat[sample.player] = (final_ownership(state, sample.player),
                                          final_margin(state, sample.player, target))
        for sample in self.samples:
            # A truncated game is a draw for both seats. Scoring it as a loss for whoever
            # was on move would teach the other seat that stalling is worth something.
            outcome = 0 if winner is None else (1 if sample.player == winner else -1)
            owners, margin = by_seat[sample.player]
            self.finished.append((sample, outcome, owners, margin, self.game_id))
        self.results.append({
            "winner": winner,
            "turns": self.env.state.turn_number,
            "decisions": self.decisions,
            "searched": self.searched,
            "samples": len(self.samples),
        })
        self.samples = []


# --------------------------------------------------------------------------- #

class Generator:
    """``width`` self-play games sharing one batched evaluator.

    Args:
        evaluate: ``(obs_batch, mask_batch) -> (probabilities, values)``. Supplied by the
            caller so this class never imports torch — which is what lets the whole
            generation path be tested with a stub network in milliseconds.
        width: games in flight. The batch a forward pass sees is at most this, and in
            practice a little less because some games are between searches.
    """

    def __init__(self, evaluate, config, seed=0, width=24):
        self.evaluate = evaluate
        self.config = config
        self.width = width
        self.rng = np.random.default_rng(seed)
        self.games = [
            Game(seed=seed * 7_919 + i * 1_000_003, config=config,
                 rng=np.random.default_rng(seed * 104_729 + i))
            for i in range(width)
        ]

    def run(self, positions=None, games=None, seconds=None):
        """Generate until a stopping condition. Returns ``(samples, results)``.

        Games still in flight are *kept*: the generator is reused across iterations, so their
        work is not thrown away. This is the same reasoning as the PPO collector's
        persistence, and the same trap — a benchmark that does not warm up measures a
        pipeline that is still filling.

        **Prefer ``seconds`` when running a pool.** Samples only bank when a game *finishes*,
        and a worker holding ``width`` games started together finishes them in a cohort: one
        long quiet stretch, then several thousand positions at once. Stopping on a sample
        count therefore takes a wildly variable time — measured at 5s for one worker and 46s
        for another on the same request — and a pool that waits for all of them leaves most
        of them idle for the difference. Measured across 14 workers, stopping on a count gave
        172 positions/sec against a single worker's 57; stopping on a clock gives every worker
        the same amount of *work* rather than the same amount of luck.
        """
        if positions is None and games is None and seconds is None:
            raise ValueError("run() needs a stopping condition: positions, games or seconds")
        deadline = None if seconds is None else time.perf_counter() + seconds

        collected, results = [], []
        while True:
            pending, owners = [], []
            for game in self.games:
                item = game.request()
                if item is not None:
                    pending.append(item)
                    owners.append(game)
                collected.extend(game.finished)
                game.finished.clear()
                results.extend(game.results)
                game.results.clear()

            # A round that has been *started* is always finished, before the stopping
            # condition is even looked at. Returning here instead would leave every game
            # holding an undelivered request, and the next call would walk into
            # Search.request's "a previous request has not been delivered" — which is exactly
            # how this was found, and is the reason that guard exists rather than a silent
            # overwrite of the pending leaf.
            if pending:
                # Both are already numpy rows, so this is a stack rather than a conversion:
                # np.asarray over 24 Python lists of 1,884 floats measured at 824 us, and
                # np.stack over the same data as arrays at 14.5.
                obs = np.stack([item[0] for item in pending])
                masks = np.stack([item[1] for item in pending])
                probabilities, values = self.evaluate(obs, masks)
                for row, game in enumerate(owners):
                    game.deliver(probabilities[row], float(values[row]))

            if positions is not None and len(collected) >= positions:
                return collected, results
            if games is not None and len(results) >= games:
                return collected, results
            if deadline is not None and time.perf_counter() >= deadline:
                return collected, results
            if not pending:
                return collected, results


#: How many arrays :func:`to_arrays` produces and :meth:`ReplayBuffer.add` consumes. Written
#: down because the pool stitches them by index and a mismatch is silent: the arrays would
#: still concatenate, just into the wrong columns.
NUM_ARRAYS = 9


def to_arrays(samples):
    """``[(Sample, outcome, owners, margin, game_id)]`` -> the arrays the buffer takes.

    Order is the argument order of :meth:`ReplayBuffer.add` and is load-bearing: the pool
    stitches the workers' arrays by index, so a column swapped here would concatenate
    cleanly and train on the wrong target.
    """
    if not samples:
        empty = np.zeros((0,), dtype=np.float32)
        return tuple(empty for _ in range(NUM_ARRAYS))
    return (
        np.stack([s.obs for s, _, _, _, _ in samples]),
        np.stack([s.index for s, _, _, _, _ in samples]),
        np.stack([s.probability for s, _, _, _, _ in samples]),
        np.stack([s.mask for s, _, _, _, _ in samples]),
        np.asarray([outcome for _, outcome, _, _, _ in samples], dtype=np.int8),
        np.asarray([s.root_value for s, _, _, _, _ in samples], dtype=np.float16),
        np.stack([owners for _, _, owners, _, _ in samples]),
        np.asarray([margin for _, _, _, margin, _ in samples], dtype=np.float16),
        np.asarray([game_id for _, _, _, _, game_id in samples], dtype=np.int64),
    )
