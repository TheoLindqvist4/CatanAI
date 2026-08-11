"""Train, select, promote, repeat — until a wall-clock deadline.

    python -m training.alphazero.chain --until "2026-08-06 09:00"

One stage is: train for a while, rank the snapshots that stage produced *with search*, and
offer the best one to the promotion gate. Then the next stage starts from where the last one
ended. Written for the case where nobody is watching, so every decision it makes is one the
repository already makes by hand, and every failure is survivable.

**Training continues from the previous stage, not from the champion.** This is the one
choice worth arguing about. Resetting to the champion each stage would throw away a stage
that was learning but had not yet overtaken — and ``CLAUDE.md`` records two runs that looked
like failures for their first ~100 iterations and then reached 77.6%. So the *network* moves
forward continuously and ``models/champion_az.pt`` only ever moves *up*, through the gate.
A stage the gate refuses costs nothing but its own hours.

**Every model is kept.** Each stage writes to its own directory and nothing is ever deleted,
because a losing checkpoint is still a rung somebody else can start from.

**Stages run as subprocesses.** A stage that dies takes its own hours with it and nothing
else: the chain logs the failure, and carries on from the last checkpoint that exists. In
one process a bad iteration would take the whole night.
"""

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

#: Wall-clock reserved for the ranking and the gate at the end of a stage. Measured on this
#: machine: ranking five candidates over 120 games is ~9 minutes, and the gate as it stood
#: then — 400 head-to-head plus 200 against the heuristic — was ~20.
#:
#: That gate can no longer happen. ``--baseline-games`` defaults to 0 and the chain passes
#: ``--ppo-games 0``, so ``promote`` plays one 400-game match and nothing else: two-thirds of
#: the games, so about 13 minutes of the 20. That is arithmetic on the old measurement rather
#: than a fresh one. 9 + 13 is 22, and 35 keeps the margin the number was chosen to have —
#: the margin is what a stage forfeits when a match runs long, not slack to be reclaimed.
#: A stage will not start unless this much plus a worthwhile amount of training fits before
#: the deadline.
EVALUATION_MINUTES = 35

#: The shortest stage worth starting. Below this the evaluation costs more than the training.
MINIMUM_STAGE_MINUTES = 45


def parse_deadline(text):
    """``"2026-08-06 09:00"`` or ``"9h"`` -> a datetime."""
    text = text.strip()
    if text.endswith("h"):
        return datetime.datetime.now() + datetime.timedelta(hours=float(text[:-1]))
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%H:%M"):
        try:
            when = datetime.datetime.strptime(text, pattern)
        except ValueError:
            continue
        if pattern == "%H:%M":
            today = datetime.datetime.now()
            when = today.replace(hour=when.hour, minute=when.minute, second=0, microsecond=0)
            if when <= today:
                when += datetime.timedelta(days=1)
        return when
    raise ValueError(f"could not read {text!r} as a time; try '2026-08-06 09:00' or '9h'")


def use_utf8_console():
    """Make this process's own console unable to raise on a character.

    Belt to :func:`run`'s braces. ``run`` asks its children for UTF-8, but nothing at all
    should be able to kill an overnight chain by printing something — the log is allowed to
    be ugly and is not allowed to be fatal. The log *file* is opened as UTF-8 already; this
    is the console half.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):   # not a real stream, or already detached
            pass


def run(command, log):
    """Run a subprocess, streaming nothing, returning ``(ok, tail)``.

    The child's stdout is a pipe, so Python encodes it with the locale codec — cp1252 here,
    which maps the em-dash in ``champion.promote``'s own summary line onto the single byte
    ``0x97``. Decoding that as UTF-8 below yields ``U+FFFD``, which cp1252 cannot encode on
    the way back out, so :func:`log` raised and a ten-hour chain died at 02:39 relaying
    someone else's punctuation. Tell the child to use UTF-8 so both ends agree.
    """
    log(f"    $ {' '.join(str(part) for part in command)}")
    finished = subprocess.run(command, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    output = (finished.stdout or "") + (finished.stderr or "")
    return finished.returncode == 0, output


def latest_network(directory):
    """The newest usable checkpoint a stage left behind, or ``None``.

    ``latest.pt`` normally, but a stage killed between snapshots may not have written one,
    and in that case its last snapshot is still a perfectly good place to continue from.
    """
    directory = pathlib.Path(directory)
    latest = directory / "latest.pt"
    if latest.is_file():
        return latest
    snapshots = sorted((directory / "snapshots").glob("iter_*.pt"))
    return snapshots[-1] if snapshots else None


def candidates(directory, keep=5):
    """The checkpoints worth ranking: the last few snapshots, plus ``latest``.

    Not all of them. A run's best checkpoint is usually near its end, and ranking twelve
    candidates costs more than the gate that follows it. Everything is still *kept* — this
    only decides what is measured tonight.
    """
    directory = pathlib.Path(directory)
    snapshots = sorted((directory / "snapshots").glob("iter_*.pt"))[-keep:]
    latest = directory / "latest.pt"
    if latest.is_file():
        snapshots.append(latest)
    return snapshots


def rank(paths, simulations, games, workers, log):
    """Play each candidate against the champion; return them best-first."""
    from training.alphazero.arena import compete

    champion = {"kind": "mcts", "path": "models/champion_az.pt",
                "simulations": simulations}
    scored = []
    for path in paths:
        result = compete({"kind": "mcts", "path": str(path), "simulations": simulations},
                         champion, games=games, seed=90_000, workers=workers)
        low, high = result["ci"]
        log(f"    {path.name:<18} {100 * result['win_rate']:>5.1f}%  "
            f"[{100 * low:.1f}, {100 * high:.1f}]")
        scored.append((result["win_rate"], str(path)))
    scored.sort(reverse=True)
    return scored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--until", required=True,
                        help="stop by this time: '2026-08-06 09:00', '09:00' or '9h'")
    parser.add_argument("--config", default="configs/train_v2.yaml")
    parser.add_argument("--stage-hours", type=float, default=2.5,
                        help="training per stage before it is measured")
    parser.add_argument("--from-network", default=None,
                        help="what the first stage continues from; defaults to the champion")
    parser.add_argument("--prefix", default="checkpoints/az_night",
                        help="run directories are <prefix>_1, _2, ...")
    parser.add_argument("--gate-games", type=int, default=400,
                        help="head-to-head games in the promotion gate")
    parser.add_argument("--baseline-games", type=int, default=0,
                        help="heuristic games in the gate, recorded and never a veto. 0, the "
                             "default, does not play it — the gate is the champion alone")
    parser.add_argument("--rank-games", type=int, default=120,
                        help="games per candidate when choosing what to submit")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--wait-for-running", action="store_true",
                        help="wait for an existing training process to finish first")
    arguments = parser.parse_args(argv)

    deadline = parse_deadline(arguments.until)
    log_path = pathlib.Path(f"{arguments.prefix}_chain.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    use_utf8_console()

    def log(message):
        stamped = f"[{datetime.datetime.now():%H:%M:%S}] {message}"
        print(stamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    log(f"chain until {deadline:%Y-%m-%d %H:%M} "
        f"({(deadline - datetime.datetime.now()).total_seconds() / 3600:.1f} h from now)")

    if arguments.wait_for_running:
        from training.alphazero.chain import _training_is_running
        while _training_is_running():
            time.sleep(60)
        log("the run that was already going has finished")

    source = arguments.from_network or "models/champion_az.pt"
    stage = 0
    summary = []

    while True:
        remaining = (deadline - datetime.datetime.now()).total_seconds() / 60
        trainable = remaining - EVALUATION_MINUTES
        if trainable < MINIMUM_STAGE_MINUTES:
            log(f"{remaining:.0f} min left — not enough for another stage, stopping")
            break

        stage += 1
        hours = min(arguments.stage_hours, trainable / 60)
        directory = f"{arguments.prefix}_{stage}"
        log(f"stage {stage}: {hours:.2f} h into {directory}, continuing from {source}")

        ok, output = run([
            sys.executable, "-u", "-m", "training.alphazero.train",
            "--config", arguments.config, "--hours", f"{hours:.4f}",
            "--run-directory", directory, "--warm-start", str(source),
        ], log)
        if not ok:
            log(f"    stage {stage} FAILED; last output:\n{output[-1500:]}")
            if latest_network(directory) is None:
                log("    nothing usable was written — stopping rather than looping on a "
                    "broken stage")
                break
        for line in output.splitlines()[-3:]:
            log(f"    | {line}")

        network = latest_network(directory)
        if network is None:
            log("    no checkpoint from this stage; stopping")
            break
        source = network                       # the next stage continues from here, always

        paths = candidates(directory)
        if not paths:
            log("    no candidates to rank")
            continue
        log(f"    ranking {len(paths)} candidates against the champion, "
            f"{arguments.rank_games} games each")
        scored = rank(paths, arguments.simulations, arguments.rank_games,
                      arguments.workers, log)
        best_rate, best = scored[0]
        log(f"    best: {best} at {100 * best_rate:.1f}%")

        # Submit it even when the screen says it lost: 120 games is +-9 points, and the
        # gate is the thing that is allowed to decide. It refuses cheaply.
        command = [sys.executable, "-u", "-m", "training.alphazero.champion", "promote",
                   best, "--games", str(arguments.gate_games),
                   "--simulations", str(arguments.simulations),
                   "--baseline-games", str(arguments.baseline_games),
                   "--ppo-games", "0"]
        promoted, output = run(command, log)
        for line in output.splitlines()[-4:]:
            log(f"    | {line}")
        summary.append({"stage": stage, "directory": directory, "candidate": best,
                        "screen_win_rate": best_rate, "promoted": promoted})
        log(f"    stage {stage}: {'PROMOTED' if promoted else 'gate refused it'}")

    log("chain finished")
    for entry in summary:
        log(f"  stage {entry['stage']:>2}  {entry['directory']:<28} "
            f"screen {100 * entry['screen_win_rate']:>5.1f}%  "
            f"{'promoted' if entry['promoted'] else 'refused'}")
    pathlib.Path(f"{arguments.prefix}_chain.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def _training_is_running():
    """Whether a ``training.alphazero.train`` process is alive. Windows-safe."""
    try:
        finished = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*alphazero.train*' }).Count"],
            capture_output=True, text=True, timeout=60)
        return int((finished.stdout or "0").strip() or 0) > 0
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
