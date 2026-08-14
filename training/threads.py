"""How many threads torch is allowed, and why the answer is almost always one.

Everything this project asks a network for is a **batch of one**: one position, one forward
pass, a hundred times a second. A batch-1 matmul has nothing to parallelise, so every extra
thread adds a barrier to wait on and nothing to do inside it. The numbers are not marginal —
same machine, same agent, same 32 simulations:

    default (14 threads)          median 1,758 ms    p90 17,983 ms    max 22,444 ms
    torch.set_num_threads(1)      median   179 ms                     max    237 ms

Ten times the median and about seventy-five times the tail, for one line. The training code
learned this and pins threads in a dozen places; the web process never did, so a person
playing in the browser was waiting on an OpenMP barrier and concluding the AI was slow. See
``docs/audit-2026-08-05-public-arena.md`` §7.

⚠️ **Order matters, and getting it wrong fails silently.** Torch sizes its OpenMP pool when
it is *imported*, so ``OMP_NUM_THREADS`` set afterwards changes nothing — the pool already
exists. :func:`pin` therefore reports whether it was called in time, and any process that
means to be single-threaded should call it as the first thing it does:

    from training.threads import pin
    pin()                       # before `import torch`, and before anything that imports it

For a worker pool the variables are set in the **parent** just before spawning, because
children inherit ``os.environ`` at spawn — that pins the children without narrowing the
parent, which is what :func:`for_children` is for.
"""

import os
import sys

#: The environment variables that decide the pool size, across the BLAS libraries torch may
#: have been built against. Setting one and not the others is a common way to half-fix this.
ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin(threads=1, force=False):
    """Hold torch to ``threads``. Returns whether it was called before torch was imported.

    Args:
        threads: how many. One, unless you are doing something with a real batch in it.
        force: overwrite the variables even if the environment already sets them. Off by
            default, so an operator who exported ``OMP_NUM_THREADS=4`` deliberately is not
            overruled by a library.

    Returns:
        bool: ``True`` if torch had not yet been imported, so the environment variables will
        actually take effect. ``False`` means the pool was already sized and only
        ``torch.set_num_threads`` — which is applied here too, and is the weaker of the two —
        is doing any work. A caller that cares can assert on it.
    """
    in_time = "torch" not in sys.modules

    for name in ENV_VARS:
        if force or name not in os.environ:
            os.environ[name] = str(threads)

    if not in_time:
        import torch

        torch.set_num_threads(threads)
    return in_time


class for_children:
    """Pin the threads of processes spawned inside this block, and restore afterwards.

    Children inherit ``os.environ`` at spawn, and the parent has already sized its own pool,
    so setting the variables here narrows the workers and leaves the parent alone. That idiom
    was written out three times — in ``parallel.py``, ``alphazero/workers.py`` and
    ``alphazero/arena.py`` — with the variable list copied each time, which is exactly how a
    fourth BLAS variable comes to be handled in two places out of three.

        with for_children():
            pool = ProcessPoolExecutor(max_workers=n, initializer=...)
    """

    def __init__(self, threads=1):
        self.threads = threads
        self._previous = {}

    def __enter__(self):
        for name in ENV_VARS:
            self._previous[name] = os.environ.get(name)
            os.environ[name] = str(self.threads)
        return self

    def __exit__(self, *_):
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False
