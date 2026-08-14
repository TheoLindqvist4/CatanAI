"""The AlphaZero policy, wearing the same interface as every other agent.

    from training.alphazero.agent import MCTSAgent
    agent = MCTSAgent.load("models/champion_az.pt", simulations=64)
    play_match({1: agent, 2: HeuristicAgent(0)}, games=100)

``(observation, info) -> index``, so it drops into ``interfaces.web.api.OPPONENTS`` and
``interfaces.cli.AGENTS`` as one more entry and neither interface changes.

**It searches from ``info["view"]``, not from a state it was handed.** The environment gives
an agent a :class:`~catan.view.PublicView`, whose ``_state`` is the real game — including the
opponent's hand. Reading it would work, would never crash, and would be cheating, so the state
is passed through :func:`~training.alphazero.determinize.determinize` first and the tree is
built on a world consistent with what this player can see. That is the same boundary
:data:`training.agent.DETERMINISTIC_TYPES` draws for the one-ply lookahead agent, drawn once
here instead of per action type, which is why this search can go deeper than one ply.

**Two knobs, and they mean different things.** ``simulations`` is strength and costs time
linearly. ``temperature`` is unpredictability: 0 plays the most-visited move, which is
strongest and completely repeatable — a poor sparring partner for a person, who will find one
line that works and play it forever. The interfaces use a small positive temperature for the
same reason they do with the PPO champion.

A search of 64 simulations costs about 50 ms on one torch thread at an opening settlement, and
about 100 ms at a mid-game decision — ``champion.CHAMPION_SIMULATIONS`` carries that one, and
the reason 64 is the number. Both are well inside what a web interface can spend between a
click and a reply. Pin the threads: these are batch-1 forward passes and more threads make
them *slower*, not faster — the 400-simulation setup search measured 0.34 s at one thread and
0.79 s at twenty.
"""

import random

import numpy as np
import torch

from catan import action_space
from catan.state import Phase
from training.alphazero.determinize import determinize
from training.alphazero.mcts import Search
from training.loading import load_model
from training.net import build

#: Simulations per move when nobody says. Chosen for latency in the web interface rather than
#: for strength; the evaluator and the promotion gate pass their own.
DEFAULT_SIMULATIONS = 64

#: Simulations at an opening settlement, where 54 spots are legal against a normal turn's six.
#: Spent here rather than everywhere because there are only two such decisions per player in a
#: game: 400 simulations costs ~0.33 s at one thread, which a person will not notice once, and
#: it is the decision they will judge the agent by.
DEFAULT_SETUP_SIMULATIONS = 400

#: Visits every opening spot gets before PUCT may concentrate. Without it the search looks at
#: a handful of the 54; with it, all 54, and marginally cheaper. The measurement is written
#: down once, in :class:`~training.alphazero.mcts.Search`'s ``root_min_visits`` docstring, and
#: argued in ``docs/decisions/0028-the-opening-is-fifty-four-moves-wide.md``.
DEFAULT_SETUP_ROOT_MIN_VISITS = 4


class MCTSAgent:
    """A network plus a search, callable as an agent.

    Args:
        net: a policy/value network over the current observation and action space.
        simulations: PUCT simulations per decision. 0 falls back to the raw policy, which
            makes this exactly a :class:`~training.agent.PolicyAgent` and is useful for
            measuring what the search is worth.
        temperature: 0 plays the most-visited move.
        seed: for the sampling and for the determinization.
        max_turns: positions past this are scored as draws inside the search.
        setup_simulations: simulations at an opening settlement, where the root is 54 wide.
            ``None`` uses ``simulations``, which is what every caller did before this existed.
        setup_root_min_visits: visits every opening spot is guaranteed before PUCT may
            concentrate. 0 restores the old behaviour of choosing among about three of them.
    """

    def __init__(self, net, simulations=DEFAULT_SIMULATIONS, temperature=0.0, seed=None,
                 max_turns=400, setup_simulations=DEFAULT_SETUP_SIMULATIONS,
                 setup_root_min_visits=DEFAULT_SETUP_ROOT_MIN_VISITS):
        self.net = net.eval()
        self.simulations = simulations
        self.setup_simulations = setup_simulations
        self.setup_root_min_visits = setup_root_min_visits
        self.temperature = temperature
        self.max_turns = max_turns
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._stream = random.Random(seed)
        self.metadata = {}

    @classmethod
    def load(cls, path, simulations=DEFAULT_SIMULATIONS, temperature=0.0, seed=None,
             map_location="cpu", setup_simulations=DEFAULT_SETUP_SIMULATIONS,
             setup_root_min_visits=DEFAULT_SETUP_ROOT_MIN_VISITS):
        checkpoint = load_model(path, map_location=map_location)
        net = build(checkpoint["config"])
        net.load_state_dict(checkpoint["weights"])
        agent = cls(net, simulations=simulations, temperature=temperature, seed=seed,
                    setup_simulations=setup_simulations,
                    setup_root_min_visits=setup_root_min_visits)
        agent.metadata = {k: v for k, v in checkpoint.items()
                          if k not in ("weights", "optimizer")}
        return agent

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _evaluate(self, obs, mask):
        batch = torch.as_tensor(np.asarray([obs], dtype=np.float32))
        flags = torch.as_tensor(np.asarray([np.frombuffer(bytes(mask), dtype=np.uint8)],
                                           dtype=np.uint8), dtype=torch.bool)
        logits, value = self.net(batch)
        logits = self.net._apply_mask(logits, flags)
        return torch.softmax(logits, dim=-1)[0].numpy(), float(value[0])

    @torch.no_grad()
    def __call__(self, observation, info):
        legal = info["legal"]
        if len(legal) == 1:
            return legal[0]

        view = info.get("view")
        if view is None or self.simulations <= 0:
            return self._policy_only(observation, info)

        world = determinize(view._state, view.me, rng=self._pick_stream())
        opening = view._state.phase is Phase.SETUP_SETTLEMENT
        search = Search(
            world,
            budget=(self.setup_simulations or self.simulations) if opening
            else self.simulations,
            rng=self.rng,
            noise=0.0,                  # exploration noise belongs to training, not to play
            root_min_visits=self.setup_root_min_visits if opening else 0,
            max_turns=self.max_turns,
        )
        if not search.searchable:
            return self._policy_only(observation, info)

        while (pending := search.request()) is not None:
            search.deliver(*self._evaluate(*pending))

        action = search.best_action(self.temperature)
        # The search ran on a *resampled* world, so in principle it could return a move that
        # is legal there and not here. It cannot in practice — legality at the root depends
        # only on public facts and on this player's own hand, neither of which determinization
        # touches — but the cost of being wrong is an exception mid-game, and the cost of
        # checking is one array lookup.
        return action if info["mask"][action] else self._policy_only(observation, info)

    def _policy_only(self, observation, info):
        probabilities, _ = self._evaluate(observation, info["mask"])
        if self.temperature <= 0:
            return int(np.argmax(probabilities))
        scaled = probabilities ** (1.0 / self.temperature)
        total = scaled.sum()
        if total <= 0:
            return int(info["legal"][0])
        return int(self.rng.choice(len(scaled), p=scaled / total))

    def _pick_stream(self):
        return random.Random(self._stream.randrange(1 << 30))

    @torch.no_grad()
    def value(self, observation):
        """The critic's opinion of a position, in ``[-1, 1]``. For a UI, and for debugging."""
        _, value = self.net(torch.as_tensor(np.asarray([observation], dtype=np.float32)))
        return float(value[0])

    def __repr__(self):
        return (f"MCTSAgent(simulations={self.simulations}, "
                f"temperature={self.temperature}, {self.net!r})")
