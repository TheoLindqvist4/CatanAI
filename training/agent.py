"""A trained policy, wearing the same interface as every other agent.

    from training.agent import PolicyAgent
    agent = PolicyAgent.load("checkpoints/latest.pt")
    play_match({1: agent, 2: HeuristicAgent(0)}, games=100)

This is the whole payoff of having kept the agent interface to
``(observation, info) -> index``: a network drops into ``interfaces.web.api.OPPONENTS`` and
``interfaces.cli.AGENTS`` as one more entry, and neither interface changes.
"""

import random

import numpy as np
import torch

from catan import action_space, encoder, rules
from catan.actions import ActionType
from training.loading import load_model, load_training_state
from training.net import PolicyValueNet, build


class PolicyAgent:
    """Wraps a :class:`~training.net.PolicyValueNet` as a callable agent.

    Args:
        net: the network.
        temperature: 0 plays the argmax — strongest, and completely predictable, which makes
            it a poor sparring partner for a human. Above 0 samples, so the same position
            does not always get the same reply.
        seed: for sampling.
    """

    def __init__(self, net, temperature=0.0, seed=None):
        self.net = net.eval()
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path, temperature=0.0, seed=None, map_location="cpu"):
        checkpoint = load_model(path, map_location=map_location)
        net = build(checkpoint["config"])
        net.load_state_dict(checkpoint["weights"])
        agent = cls(net, temperature=temperature, seed=seed)
        agent.metadata = {k: v for k, v in checkpoint.items()
                          if k not in ("weights", "optimizer")}
        return agent

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(self, observation, info):
        mask = info["mask"]
        legal = info["legal"]
        if len(legal) == 1:
            return legal[0]

        obs = torch.as_tensor(np.asarray([observation], dtype=np.float32))
        mask_t = torch.as_tensor(np.asarray([mask], dtype=bool))
        logits, _ = self.net(obs)
        logits = self.net._apply_mask(logits, mask_t)[0].numpy()

        if self.temperature <= 0:
            return int(np.argmax(logits))

        scaled = logits / self.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))

    @torch.no_grad()
    def value(self, observation):
        """The critic's opinion of a position, in [-1, 1]. Useful for a UI, and for debugging."""
        obs = torch.as_tensor(np.asarray([observation], dtype=np.float32))
        _, value = self.net(obs)
        return float(value[0])

    def __repr__(self):
        return f"PolicyAgent(temperature={self.temperature}, {self.net!r})"


def export(checkpoint, path):
    """Write a slim checkpoint: the network, and nothing else.

    A training checkpoint carries the optimiser state, the frozen opponent pool and the
    full per-iteration history, which is 70 MB and exactly the right thing to keep for
    resuming a run. None of it is needed to *play*, and shipping it would mean the web
    interface loading fourteen networks to use one.
    """
    import pathlib
    source = load_training_state(checkpoint)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": source["config"],
        "weights": source["weights"],
        "iteration": source.get("iteration"),
        "trained_at": source.get("history", [])[-1] if source.get("history") else None,
    }, path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a slim, playable checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    arguments = parser.parse_args()
    written = export(arguments.checkpoint, arguments.out)
    size = written.stat().st_size / 1e6
    print(f"wrote {written} ({size:.1f} MB)")


#: Action types whose effect on the game is **fully determined by public information**.
#:
#: This list is a leak boundary, not an optimisation. `GameState.clone` copies the dev deck,
#: the dice deck and opponents' hands *verbatim* — correct for a point-in-time copy, and
#: exactly why a naive lookahead cheats. Applying `BUY_DEV_CARD` to a clone draws the real
#: next card; applying `MOVE_ROBBER` performs the real steal and so reveals a card from the
#: victim's hand; `END_TURN` rolls the real next die from the 36-card balanced deck;
#: `PLAY_MONOPOLY` reads what opponents actually hold. Searching over any of those would let
#: the agent see what a player may not.
#:
#: What remains is most of what positional judgement is *for*: where to build, what to trade.
DETERMINISTIC_TYPES = frozenset({
    ActionType.BUILD_ROAD,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.TRADE_WITH_BANK,
    ActionType.DISCARD,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
})


class LookaheadAgent(PolicyAgent):
    """A policy that checks where each candidate move actually leads.

    For every legal action whose effect is public and deterministic, the position after it is
    built and scored by the value head. The action is then chosen by

        log pi(a)  +  weight * (V(s') - V(s))

    — the policy's own preference, corrected by how much better the resulting position looks.
    Actions that are not in :data:`DETERMINISTIC_TYPES` keep their prior alone, which is the
    neutral assumption rather than a penalty.

    One ply, no tree. Deeper search would need the opponent's reply, which needs their hand,
    which is hidden — that is belief sampling, and it is deliberately not built here.

    Cost is one extra batched forward pass per decision, over at most a few dozen candidates.

    Args:
        weight: how much to trust the value head against the policy. 0 reproduces
            :class:`PolicyAgent` exactly.
    """

    def __init__(self, net, weight=1.0, temperature=0.0, seed=None):
        super().__init__(net, temperature=temperature, seed=seed)
        self.weight = weight

    def __repr__(self):
        return f"LookaheadAgent(weight={self.weight}, temperature={self.temperature})"

    @torch.no_grad()
    def __call__(self, observation, info):
        legal = info["legal"]
        if len(legal) == 1:
            return legal[0]

        view = info.get("view")
        if view is None or self.weight == 0.0:
            return super().__call__(observation, info)

        state = view._state
        me = view.me

        obs = torch.as_tensor(np.asarray([observation], dtype=np.float32))
        mask_t = torch.as_tensor(np.asarray([info["mask"]], dtype=bool))
        logits, here = self.net(obs)
        scores = self.net._apply_mask(logits, mask_t)[0].numpy().astype(np.float64)
        scores -= scores.max()

        futures, indices = [], []
        for index in legal:
            action = action_space.decode(index)
            if action.type not in DETERMINISTIC_TYPES:
                continue
            try:
                ahead = state.clone(rng=random.Random(0))
                rules.apply(ahead, action)
            except Exception:
                continue                       # never let search break a legal move
            futures.append(encoder.encode(ahead, me))
            indices.append(index)

        if futures:
            _, values = self.net(torch.as_tensor(np.asarray(futures, dtype=np.float32)))
            gain = values.numpy() - float(here[0])
            for position, index in enumerate(indices):
                scores[index] += self.weight * float(gain[position])

        if self.temperature <= 0:
            return int(np.argmax(scores))
        scaled = scores / self.temperature
        scaled -= scaled.max()
        probabilities = np.exp(scaled)
        probabilities[~np.asarray(info["mask"], dtype=bool)] = 0.0
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))
