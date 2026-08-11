"""Self-play across processes.

The engine is pure Python, so one process uses one core of twenty. This module is the same
answer :mod:`training.parallel` gives the PPO trainer, and it inherits the same three traps —
recorded here rather than referred to, because each of them fails quietly.

**Windows spawns rather than forks.** Every child re-imports this module, so the worker
functions must be module-level and everything crossing the boundary must be picklable.
Nothing is inherited: each child builds its own network from a config and its own generator
from a seed.

**Thread oversubscription is the usual way to make this slower.** Torch sizes its OpenMP pool
at *import*, and reports 14 threads on this machine, so fourteen workers would ask for 196
threads on 20 cores and spend their lives in the scheduler. Children are pinned to one thread
each by setting the environment before the pool is created — a child inherits ``os.environ``
at spawn time, and the parent has already built its own pool by then, so the parent keeps the
wide setting it wants for the gradient step.

**Generators are persistent inside each worker.** Games are kept in flight between
iterations. A worker that dropped its unfinished games would throw away most of its work,
because a game is 100-plus decisions and an iteration is a few thousand positions per worker.
The consequence is that the *first* iteration is slower than the rest — the pipeline is still
filling — so a benchmark that does not warm up measures the wrong thing. ``CLAUDE.md`` records
an earlier version of this project reporting 4 workers as faster than 8 for exactly that
reason.

Only the learner's weights cross the boundary each iteration: 780 KB, against the tens of
megabytes of samples coming back.
"""

import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from training.alphazero.self_play import NUM_ARRAYS, Generator, to_arrays

#: Per-process state, built once by :func:`_configure` and reused for the whole run.
_WORKER = {}


def _configure(settings, identities):
    """Runs once in each child: claim an identity, then build the network and the generator.

    The identity matters more than it looks. Every worker runs the same code from the same
    config, so without it all fourteen would seed their games identically and the pool would
    generate fourteen copies of the same twenty-four games — at full CPU cost, with a replay
    buffer that looks healthy and contains a fourteenth of the data it claims. Identities are
    claimed from a queue rather than derived from the process id so that the *set* of games a
    seed produces is the same on every run; which process claims which is arbitrary and does
    not matter, because the games are independent.
    """
    import torch

    index = identities.get()
    seed = settings["seed"] * 104_729 + index
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    from training.net import build

    net = build(settings["config"])
    net.eval()

    def evaluate(obs, masks):
        """Masked probabilities and values for a batch of positions."""
        with torch.no_grad():
            logits, value = net(torch.from_numpy(obs))
            logits = net._apply_mask(logits, torch.from_numpy(masks))
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.numpy(), value.numpy()

    _WORKER.clear()
    _WORKER.update({
        "net": net,
        "settings": settings,
        "index": index,
        "generator": Generator(
            evaluate,
            config=settings["play"],
            seed=seed,
            width=settings["width"],
        ),
    })


def _generate(payload):
    """One worker's share of an iteration. Returns pickled arrays, never tensors."""
    weights, positions, seconds = pickle.loads(payload)
    state = _WORKER
    state["net"].load_state_dict(weights)

    samples, results = state["generator"].run(positions=positions, seconds=seconds)
    return pickle.dumps((to_arrays(samples), results), protocol=pickle.HIGHEST_PROTOCOL)


class ParallelSelfPlay:
    """A pool of self-play workers sharing one policy.

    Args:
        net: the network being trained. Only its weights are sent.
        config: a :class:`~training.alphazero.config.Config`.
    """

    def __init__(self, net, config):
        self.net = net
        self.config = config
        self.workers = int(config["self_play_workers"])

        play = {
            "simulations": config["mcts_simulations"],
            "max_turns": config["max_turns"],
            "temperature": config["temperature"],
            "temperature_final": config["temperature_final"],
            "temperature_opening_turns": config["temperature_opening_turns"],
            "c_puct": config["c_puct"],
            "fpu": config["fpu"],
            "dirichlet_alpha": config["dirichlet_alpha"],
            "dirichlet_weight": config["dirichlet_weight"],
            "gumbel": config["gumbel"],
            "gumbel_actions": config["gumbel_actions"],
            "playout_cap_probability": config["playout_cap_probability"],
            "playout_cap_fast": config["playout_cap_fast"],
            "setup_simulations": config["setup_simulations"],
            "setup_root_min_visits": config["setup_root_min_visits"],
        }

        self._manager = multiprocessing.Manager()
        identities = self._manager.Queue()
        for index in range(self.workers):
            identities.put(index)

        previous = {}
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            previous[name] = os.environ.get(name)
            os.environ[name] = "1"
        try:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_configure,
                initargs=({
                    "config": net.config(),
                    "play": play,
                    "width": config["envs_per_worker"],
                    "seed": config["seed"],
                }, identities),
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    # ------------------------------------------------------------------ #

    def generate(self, positions=None, seconds=None):
        """Run every worker for the same slice of work, and stitch what comes back.

        Returns ``(arrays, results)``: the five arrays :meth:`ReplayBuffer.add` takes, and one
        dict per completed game.

        ``seconds`` is the mode a training run should use. ``pool.map`` does not return until
        the *slowest* worker does, so a share expressed as a sample count leaves the fast
        workers idle for the difference — and the difference is large here, because samples
        bank in cohorts when games finish. See :meth:`Generator.run`.
        """
        if positions is None and seconds is None:
            raise ValueError("generate() needs positions or seconds")
        share = None if positions is None else max(64, positions // self.workers)
        weights = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        payload = pickle.dumps((weights, share, seconds), protocol=pickle.HIGHEST_PROTOCOL)

        parts = [pickle.loads(blob)
                 for blob in self.pool.map(_generate, [payload] * self.workers)]
        return _stitch(parts)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self._manager.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _stitch(parts):
    """Join the workers' arrays and concatenate their game results."""
    arrays, results = [], []
    for piece, games in parts:
        if len(piece[0]):
            arrays.append(piece)
        results.extend(games)
    if not arrays:
        empty = np.zeros((0,), dtype=np.float32)
        return tuple(empty for _ in range(NUM_ARRAYS)), results
    joined = tuple(np.concatenate([piece[i] for piece in arrays], axis=0)
                   for i in range(NUM_ARRAYS))
    return joined, results
