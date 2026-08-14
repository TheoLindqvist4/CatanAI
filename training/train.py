"""Self-play training.

    python -m training.train --iterations 400
    python -m training.train --resume checkpoints/latest.pt --iterations 200
    python -m training.train --smoke                 # 60-second pipeline check

Each iteration: sample an opponent, collect a rollout, run the PPO update, log. Every
``--eval-every`` iterations the policy is measured against the *fixed* heuristic, because
that is the only number that means anything — a self-play win rate is 50% by construction.

Everything is logged to ``metrics.jsonl`` as it goes, so a run that goes wrong can be
diagnosed from the record instead of re-run. That matters more than usual here: PPO's
failure modes are silent, and by the time a run looks wrong the interesting part is hours
in the past.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from catan import action_space, encoder
from catan.agents import HeuristicAgent
from training.evaluate import evaluate, format_result
from training.loading import load_training_state
from training.net import PolicyValueNet, build
from training.pool import OpponentPool
from training.ppo import PPO
from training.parallel import ParallelCollector
from training.rollout import SelfPlayCollector

DEFAULT_CHECKPOINTS = pathlib.Path("checkpoints")


def build_from_args(args):
    if args.net == "structured":
        from training.structured_net import StructuredPolicyValueNet

        net = StructuredPolicyValueNet()
    else:
        net = PolicyValueNet(
            obs_size=encoder.SIZE,
            num_actions=action_space.NUM_ACTIONS,
            hidden=tuple(args.hidden),
        )
    ppo = PPO(
        net,
        lr=args.lr,
        clip=args.clip,
        epochs=args.epochs,
        minibatch=args.minibatch,
        entropy_coef=args.entropy,
        value_coef=args.value_coef,
        target_kl=args.target_kl,
    )
    return net, ppo


def save(path, net, ppo, pool, iteration, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": net.config(),
        "weights": net.state_dict(),
        "optimizer": ppo.optimizer.state_dict(),
        "pool": pool.snapshot(),
        "iteration": iteration,
        "history": history,
    }, path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net, ppo = build_from_args(args)
    pool = OpponentPool(
        capacity=args.pool_size,
        self_play=args.self_play,
        heuristic=args.heuristic_share,
        seed=args.seed,
    )
    start_iteration, history = 0, []

    if args.resume:
        checkpoint = load_training_state(args.resume)
        net = build(checkpoint["config"])
        net.load_state_dict(checkpoint["weights"])
        ppo = PPO(net, lr=args.lr, clip=args.clip, epochs=args.epochs,
                  minibatch=args.minibatch, entropy_coef=args.entropy,
                  value_coef=args.value_coef, target_kl=args.target_kl)
        ppo.optimizer.load_state_dict(checkpoint["optimizer"])
        pool.restore(checkpoint.get("pool"))
        start_iteration = checkpoint["iteration"]
        history = checkpoint.get("history", [])
        print(f"resumed from {args.resume} at iteration {start_iteration}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"

    print(f"{net!r}")
    print(f"{args.steps:,} transitions/iteration x {args.iterations} iterations "
          f"= {args.steps * args.iterations:,} learner steps")
    print(f"{'iter':>5} {'opponent':>12} {'games':>6} {'win%':>6} {'len':>5} "
          f"{'entropy':>8} {'KL':>7} {'clip':>6} {'EV':>7} {'value':>8} {'sec':>6}")

    best_rate = -1.0
    total_steps = 0
    started = time.perf_counter()

    # One collector for the whole run. Games still in progress when a rollout fills carry
    # over to the next iteration instead of being discarded — rebuilding it per iteration
    # measured as 7.5x the work at 128 environments, and made *more* environments slower
    # rather than faster.
    clock = {"iteration": start_iteration}
    if args.workers > 1:
        # Measured 1,879 transitions/sec on one process against 23,599 across sixteen.
        # Each worker keeps its own mirror of the opponent pool, built on the same schedule
        # from the same weights, so only the learner's 5.4 MB crosses the boundary.
        collector = ParallelCollector(
            net,
            workers=args.workers,
            envs=args.envs,
            gamma=args.gamma, lam=args.lam, shaping=args.shaping,
            max_turns=args.max_turns,
            self_play=args.self_play, heuristic=args.heuristic_share,
            pool_size=args.pool_size,
            seed=args.seed,
        )
    else:
        collector = SelfPlayCollector(
            net,
            num_envs=args.envs,
            opponent=lambda: pool.sample(net, clock["iteration"]),
            gamma=args.gamma,
            lam=args.lam,
            shaping=args.shaping,
            max_turns=args.max_turns,
            seed=args.seed * 7919,
        )

    for iteration in range(start_iteration, start_iteration + args.iterations):
        tick = time.perf_counter()
        clock["iteration"] = iteration
        if args.workers > 1:
            rollout = collector.collect(
                args.steps, iteration=iteration,
                snapshot=(iteration + 1) % args.pool_every == 0,
            )
        else:
            rollout = collector.collect(args.steps)
        total_steps += len(rollout)

        if args.anneal_lr:
            progress = (iteration - start_iteration) / max(1, args.iterations)
            ppo.set_lr(args.lr * (1.0 - progress))

        stats = rollout.stats
        diagnostics = ppo.update(rollout)

        decided = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / decided if decided else float("nan")
        mean_length = float(np.mean(stats["lengths"])) if stats["lengths"] else 0.0
        elapsed = time.perf_counter() - tick

        record = {
            "iteration": iteration,
            "opponents": stats["opponents"],
            "steps": len(rollout),
            "total_steps": total_steps,
            "games": stats["games"],
            "win_rate": win_rate,
            "truncated": stats["truncated"],
            "mean_episode_length": mean_length,
            "mean_turns": float(np.mean(stats["turns"])) if stats["turns"] else 0.0,
            "forced_fraction": stats["forced"] / max(1, stats["decisions"] + stats["forced"]),
            "seconds": elapsed,
            "steps_per_second": len(rollout) / elapsed,
            **diagnostics,
        }

        label = "+".join(f"{k}:{v}" for k, v in sorted(stats["opponents"].items()))[:12]
        print(f"{iteration:>5} {label:>12} {stats['games']:>6} {100 * win_rate:>5.1f} "
              f"{mean_length:>5.0f} {diagnostics['entropy']:>8.3f} {diagnostics['kl']:>7.4f} "
              f"{diagnostics['clip_fraction']:>6.3f} {diagnostics['explained_variance']:>7.3f} "
              f"{diagnostics['value_loss']:>8.4f} {elapsed:>6.1f}")

        if (iteration + 1) % args.eval_every == 0 or iteration == start_iteration:
            from training.agent import PolicyAgent
            result = evaluate(PolicyAgent(net), games=args.eval_games,
                              seed=10_000 + iteration, max_turns=args.max_turns * 2)
            record["eval"] = {k: v for k, v in result.items() if k != "ci"}
            record["eval"]["ci"] = list(result["ci"])
            print(f"      -> vs {format_result('heuristic', result)}")
            net.train()

            if result["win_rate"] > best_rate:
                best_rate = result["win_rate"]
                save(out / "best.pt", net, ppo, pool, iteration, history)
                print(f"         new best ({100 * best_rate:.1f}%), saved")

        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if (iteration + 1) % args.pool_every == 0:
            pool.add(net, iteration)

        if (iteration + 1) % args.save_every == 0:
            save(out / "latest.pt", net, ppo, pool, iteration + 1, history)

    save(out / "latest.pt", net, ppo, pool, start_iteration + args.iterations, history)
    if hasattr(collector, "close"):
        collector.close()
    total = time.perf_counter() - started
    print(f"\n{total / 60:.1f} min, {total_steps:,} learner transitions "
          f"({total_steps / total:,.0f}/sec). Best vs heuristic: {100 * best_rate:.1f}%")
    return net


def smoke(args):
    """A 60-second end-to-end check: does the pipeline run and do the numbers move?

    Worth more than it looks. Every expensive PPO failure is one that ran happily for an
    hour before anyone noticed, and most of them are visible in the first two iterations —
    an entropy that does not fall, a KL of exactly 0, a value loss that is NaN.
    """
    args.iterations = 3
    args.steps = 1_500
    args.envs = 16
    args.eval_every = 3
    args.eval_games = 20
    args.max_turns = 200
    print("smoke test\n" + "-" * 100)
    net = train(args)
    print("-" * 100)
    print("pipeline runs.")
    return net


def parse(argv=None):
    parser = argparse.ArgumentParser(
        description="PPO self-play for Catan 1v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--steps", type=int, default=8_192,
                        help="learner transitions per iteration")
    parser.add_argument("--envs", type=int, default=48,
                        help="games stepped in lockstep per worker; batches the forward pass")
    parser.add_argument("--workers", type=int, default=1,
                        help="rollout processes. 1 keeps everything in-process; higher is "
                             "measurably faster (1,879/sec -> 23,599/sec at 16 workers)")
    parser.add_argument("--max-turns", type=int, default=400,
                        help="games are adjudicated on victory points beyond this")

    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--net", choices=["flat", "structured"], default="flat",
                        help="structured shares weights across board elements: measured "
                             "80.3%% held-out cloning agreement against flat's 69.6%%, with a "
                             "2.2-point train/test gap against 13.9 — but a worse value head")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="1.0: the game pays out once, at the end")
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--shaping", type=float, default=0.3,
                        help="potential-based victory-point shaping; 0 disables")

    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--pool-every", type=int, default=10)
    parser.add_argument("--self-play", type=float, default=0.6)
    parser.add_argument("--heuristic-share", type=float, default=0.15)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_CHECKPOINTS))
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Windows spawns rather than forks, so every worker re-imports this module. The guard
    # above is what stops each of them starting its own training run.
    import multiprocessing

    multiprocessing.freeze_support()
    arguments = parse()
    (smoke if arguments.smoke else train)(arguments)
