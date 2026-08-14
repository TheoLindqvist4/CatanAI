"""Carry a champion onto a different architecture.

    python -m training.alphazero.distil --source models/champion_az.pt \
        --out checkpoints/distilled.pt --games 600

``network.graft`` widens the layers whose input is the *observation*, which is what an
encoder change needs. It cannot help when ``width`` or ``depth`` changes: then every weight
in the network has a different shape and there is no column-wise correspondence to preserve.

So the new shape is taught to imitate the old one. This is behaviour cloning with the
champion as the teacher, and it has one property that matters more than its accuracy: the
teacher can be queried on **exactly** the positions the student will meet, because the
teacher generates them. Record 0021's cloning had to accept the heuristic's distribution;
this does not.

**Why warm-start at all.** Record 0023: at the simulation counts a CPU affords, MCTS is a
modest improvement over its prior, so a run that starts from random play spends its budget
rediscovering that settlements go on high-pip vertices. The champion already knows that.

**What is not carried.** The auxiliary heads, because the teacher has none. They start small
and are learned from self-play, which is the only place their targets exist.

**Labels are the teacher's policy, not its move.** Record 0021 measured that cloning a
sampled action caps the student at the teacher's self-agreement. The teacher's full
distribution is free here — it is the same forward pass — so the student learns the
distribution and the games are *played* with sampling, for coverage.
"""

import argparse
import os
import pathlib
import pickle
import time

import numpy as np
import torch
from torch import nn

from catan import action_space, encoder
from catan.env import CatanEnv
from catan.rulesets import RANKED_1V1
from training.loading import load_model
from training.net import build
from training.structured_net import StructuredPolicyValueNet

#: Per-process state for the position generators.
_WORKER = {}


def _configure(payload):
    import torch as _torch
    _torch.set_num_threads(1)
    settings = pickle.loads(payload)
    checkpoint = load_model(settings["source"])
    net = build(checkpoint["config"])
    net.load_state_dict(checkpoint["weights"])
    net.eval()
    _WORKER.clear()
    _WORKER.update({"net": net, "max_turns": settings["max_turns"],
                    "temperature": settings["temperature"]})


def _positions(task):
    """Play ``games`` teacher-against-itself games; return the observations and masks.

    Only the inputs come back. The teacher's policy and value are recomputed in the parent
    in large batches, which is several times faster than one batch-of-one call per move and
    keeps the workers doing nothing but engine work.
    """
    index, games, seed = task
    net = _WORKER["net"]
    rng = np.random.default_rng(seed * 7919 + index)
    observations, masks = [], []
    temperature = _WORKER["temperature"]

    for game in range(games):
        env = CatanEnv(num_players=2, ruleset=RANKED_1V1, max_turns=_WORKER["max_turns"])
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        while not info["done"]:
            legal = info["legal"]
            if len(legal) == 1:
                obs, _, _, _, info = env.step(legal[0])
                continue
            row = np.asarray(obs, dtype=np.float32)
            flags = np.frombuffer(bytes(info["mask"]), dtype=np.uint8).astype(bool)
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(row[None]))
                logits = net._apply_mask(logits, torch.from_numpy(flags[None]))
                probabilities = torch.softmax(logits / temperature, dim=-1)[0].numpy()
            observations.append(row.astype(np.float16))
            masks.append(np.packbits(flags))
            total = probabilities.sum()
            action = (int(rng.choice(len(probabilities), p=probabilities / total))
                      if total > 0 else int(legal[0]))
            obs, _, _, _, info = env.step(action)
    if not observations:
        return None
    return pickle.dumps((np.stack(observations), np.stack(masks)),
                        protocol=pickle.HIGHEST_PROTOCOL)


def generate(source, games=600, workers=None, seed=11, max_turns=400, temperature=1.0,
             log=print):
    """Teacher self-play positions, across processes. Returns ``(observations, masks)``."""
    from concurrent.futures import ProcessPoolExecutor

    workers = max(1, (os.cpu_count() or 4) - 2) if workers is None else workers
    per = max(1, games // workers)
    payload = pickle.dumps({"source": str(source), "max_turns": max_turns,
                            "temperature": temperature},
                           protocol=pickle.HIGHEST_PROTOCOL)

    previous = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        previous[name] = os.environ.get(name)
        os.environ[name] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_configure,
                                 initargs=(payload,)) as pool:
            parts = list(pool.map(_positions, [(i, per, seed) for i in range(workers)]))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    pieces = [pickle.loads(blob) for blob in parts if blob is not None]
    observations = np.concatenate([p[0] for p in pieces])
    masks = np.concatenate([p[1] for p in pieces])
    log(f"  {len(observations):,} positions from {workers * per} games")
    return observations, masks


def teach(teacher, observations, masks, batch=4096):
    """The teacher's masked policy and value on every position."""
    policies = np.zeros((len(observations), action_space.NUM_ACTIONS), dtype=np.float16)
    values = np.zeros(len(observations), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(observations), batch):
            rows = slice(start, start + batch)
            obs = torch.from_numpy(observations[rows].astype(np.float32))
            flags = torch.from_numpy(
                np.unpackbits(masks[rows], axis=1,
                              count=action_space.NUM_ACTIONS).astype(bool))
            logits, value = teacher(obs)
            logits = teacher._apply_mask(logits, flags)
            policies[rows] = torch.softmax(logits, dim=-1).numpy().astype(np.float16)
            values[rows] = value.numpy()
    return policies, values


def distil(student, observations, masks, policies, values, epochs=6, batch=512, lr=1e-3,
           value_weight=1.0, holdout=0.05, seed=0, log=print):
    """Train ``student`` to reproduce the teacher's distribution. Returns the history."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(observations), generator=generator)
    split = int(len(order) * (1 - holdout))
    train_rows, test_rows = order[:split], order[split:]
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    history = []

    def batches(rows, size):
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size].numpy()
            yield (torch.from_numpy(observations[chunk].astype(np.float32)),
                   torch.from_numpy(np.unpackbits(
                       masks[chunk], axis=1,
                       count=action_space.NUM_ACTIONS).astype(bool)),
                   torch.from_numpy(policies[chunk].astype(np.float32)),
                   torch.from_numpy(values[chunk]))

    for epoch in range(epochs):
        student.train()
        shuffled = train_rows[torch.randperm(len(train_rows), generator=generator)]
        total = steps = 0.0
        for obs, flags, target, value in batches(shuffled, batch):
            logits, predicted = student(obs)
            logits = student._apply_mask(logits, flags)
            policy_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = policy_loss + value_weight * nn.functional.mse_loss(predicted, value)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total += float(policy_loss.detach())
            steps += 1

        student.eval()
        agree = seen = 0
        error = 0.0
        with torch.no_grad():
            for obs, flags, target, value in batches(test_rows, 4096):
                logits, predicted = student(obs)
                logits = student._apply_mask(logits, flags)
                agree += int((logits.argmax(-1) == target.argmax(-1)).sum())
                error += float((predicted - value).abs().sum())
                seen += len(obs)
        record = {"epoch": epoch, "policy_loss": total / max(steps, 1),
                  "agreement": agree / max(seen, 1), "value_mae": error / max(seen, 1)}
        history.append(record)
        log(f"  epoch {epoch}  loss {record['policy_loss']:.4f}  "
            f"agrees with teacher {100 * record['agreement']:.1f}%  "
            f"value MAE {record['value_mae']:.4f}")
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="models/champion_az.pt")
    parser.add_argument("--out", default="checkpoints/distilled.pt")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="teacher sampling temperature while generating; >1 widens "
                             "coverage, which is the point of generating rather than reusing")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--seed", type=int, default=11)
    # the student's shape
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--road-width", type=int, default=64)
    parser.add_argument("--context", type=int, default=192)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--trunk", type=int, default=192)
    parser.add_argument("--no-aux", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    print(f"teacher: {args.source}")
    observations, masks = generate(args.source, games=args.games, workers=args.workers,
                                   seed=args.seed, temperature=args.temperature)

    torch.set_num_threads(args.threads)
    checkpoint = load_model(args.source)
    teacher = build(checkpoint["config"])
    teacher.load_state_dict(checkpoint["weights"])
    teacher.eval()
    policies, values = teach(teacher, observations, masks)

    student = StructuredPolicyValueNet(
        obs_size=encoder.SIZE, num_actions=action_space.NUM_ACTIONS,
        width=args.width, road_width=args.road_width, context=args.context,
        hops=args.hops, depth=args.depth, rounds=0, trunk=args.trunk,
        value_activation="tanh", aux=not args.no_aux)
    print(f"student: {student!r}")
    torch.manual_seed(args.seed)
    history = distil(student, observations, masks, policies, values,
                     epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": student.config(),
        "weights": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "lineage": "alphazero",
        "distilled_from": str(args.source),
        "distil_history": history,
    }, path)
    print(f"wrote {path} in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
