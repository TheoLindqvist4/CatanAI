"""A Gymnasium-style environment over the engine.

    env = CatanEnv(num_players=2)
    obs, info = env.reset(seed=0)
    while not info["done"]:
        action = agent(obs, info["mask"])
        obs, reward, terminated, truncated, info = env.step(action)

Three things this layer decides, none of which belong in the rules:

**The dice are rolled for you.** :func:`catan.rules.roll_dice` is environment
stochasticity, not a move, so ``step`` rolls automatically whenever the game is waiting on
one — after applying an action and before returning the next observation. An agent never
sees a state where its only option is "roll". The one exception is a Knight played *before*
the roll: that is a real choice — it decides which tile pays out this turn — so if it is
available the environment stops and offers it.

**Whoever must act is the observer.** Catan is not strictly alternating: during a discard the
decision belongs to whoever is over the hand limit, usually an opponent. So every step
returns ``info["player"]`` and the observation is built from *that* player's view. A self-play
loop should read it rather than assume turn order — this is the usual source of bugs in
multi-agent environments.

**Rewards are terminal and zero-sum.** ``+1`` for the winner, ``-1`` for everyone else, 0
while the game runs. Reward is attributed to the player who *acted*, which for a
zero-sum-at-the-end game is all the signal there is; shaping is a training decision, not an
environment one, so it is left to the caller.

Truncation is separate from termination: a game stopped at ``max_turns`` reports
``truncated=True`` with no winner, which a learner must treat differently from a loss.
"""

import random

from catan import action_space, encoder, rules
from catan.state import GameState, Phase
from catan.view import PublicView

#: Games are stopped rather than run forever. Random play reaches 15 points in a median 435
#: turns, so this is generous — a real agent finishes far sooner.
DEFAULT_MAX_TURNS = 5_000

WIN_REWARD = 1.0
LOSS_REWARD = -1.0


class CatanEnv:
    """One game of Catan, stepped by flat action indices.

    Attributes:
        state: the live :class:`~catan.state.GameState`. Read it freely; mutate it and the
            environment's guarantees no longer hold.
        observation_size: length of every observation.
        num_actions: size of the action space, the same in every state.
    """

    observation_size = encoder.SIZE
    num_actions = action_space.NUM_ACTIONS

    def __init__(self, num_players=2, ruleset=None, max_turns=DEFAULT_MAX_TURNS):
        self.num_players = num_players
        self.ruleset = ruleset
        self.max_turns = max_turns
        self.state = None
        self._rng = None

    # ------------------------------------------------------------------ #
    # RESET                                                              #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, randomize_order=False):
        """Start a new game.

        Args:
            seed: reproduces the board, the decks and every roll.
            randomize_order: shuffle turn order before setup. Off by default so a seed
                alone pins the whole game.

        Returns:
            ``(observation, info)`` for the player who must act first.
        """
        self._rng = random.Random(seed)
        self.state = GameState(
            num_players=self.num_players,
            rng=self._rng,
            ruleset=self.ruleset,
        )
        if randomize_order:
            self.state.randomize_order()

        self.state.events = []
        self._advance_to_decision()
        return self._observe()

    # ------------------------------------------------------------------ #
    # STEP                                                               #
    # ------------------------------------------------------------------ #

    def step(self, action):
        """Apply one action index and hand back the next decision.

        Args:
            action: an index into :mod:`catan.action_space`. Must be legal — check
                ``info["mask"]``. An illegal index raises rather than being ignored,
                because silently substituting a legal move would teach an agent that its
                choice does not matter.

        Returns:
            ``(observation, reward, terminated, truncated, info)``. ``reward`` belongs to
            the player who just acted.
        """
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        if self.state.phase is Phase.GAME_OVER:
            raise RuntimeError("the game is over; call reset()")

        actor = self.state.current_player
        # Events accumulate across everything this step does — the action itself, and any
        # dice rolled on the way to the next decision.
        self.state.events = []
        rules.apply(self.state, action_space.decode(_check_index(action)))
        self._advance_to_decision()

        terminated = self.state.phase is Phase.GAME_OVER
        truncated = not terminated and self.state.turn_number >= self.max_turns
        reward = self._reward(actor) if terminated else 0.0

        observation, info = self._observe(terminated or truncated)
        return observation, reward, terminated, truncated, info

    def _reward(self, player):
        return WIN_REWARD if self.state.winner == player else LOSS_REWARD

    # ------------------------------------------------------------------ #
    # INTERNALS                                                          #
    # ------------------------------------------------------------------ #

    def _advance_to_decision(self):
        """Roll the dice while nobody has a choice to make.

        A loop rather than a single roll: rolling can lead straight to another roll for the
        next player when nobody is over the hand limit and no card is playable.
        """
        while (
            self.state.phase is Phase.ROLL
            and self.state.turn_number < self.max_turns
            and not rules.legal_actions(self.state)
        ):
            rules.roll_dice(self.state)

    def _observe(self, done=False):
        state = self.state
        player = state.current_player
        mask = action_space.legal_mask(state)
        finished = done or state.phase is Phase.GAME_OVER
        info = {
            "player": player,
            # What this player may see of the board, for agents that reason about
            # positions rather than vectors. An explicit allow-list, so an agent cannot
            # read the opponent's cards even by accident — see catan.view.
            "view": PublicView(state, player),
            "mask": mask,
            "legal": [i for i, flag in enumerate(mask) if flag],
            "phase": state.phase,
            "turn": state.turn_number,
            "last_roll": state.last_roll,
            "scores": _scores(state, player, finished),
            "public_scores": {
                p: rules.public_victory_points(state, p) for p in state.players
            },
            "winner": state.winner,
            "events": list(state.events),
            "done": finished,
        }
        return encoder.encode(state, player), info

    # ------------------------------------------------------------------ #
    # CONVENIENCE                                                        #
    # ------------------------------------------------------------------ #

    def observe(self, player):
        """The observation ``player`` is entitled to, whether or not it is their decision.

        Self-play algorithms that keep a per-player trajectory need the losing side's view
        of the final position too.
        """
        return encoder.encode(self.state, player)

    def clone(self):
        """A copy for search, sharing the RNG stream so rollouts diverge.

        ⚠️ Sharing the stream is not enough on its own: the development deck and the
        Balanced Dice deck are copied verbatim, so a clone replays the same draws. To
        sample genuinely different futures, reshuffle the unseen parts —
        see ``docs/decisions/0013-ranked-1v1-ruleset.md``.
        """
        other = CatanEnv(
            num_players=self.num_players,
            ruleset=self.ruleset,
            max_turns=self.max_turns,
        )
        other.state = self.state.clone(rng=self.state.rng)
        other._rng = self._rng
        return other

    def __repr__(self):
        if self.state is None:
            return f"CatanEnv(players={self.num_players}, not started)"
        return f"CatanEnv({self.state!r})"


def _scores(state, me, finished):
    """``{player: victory points}``, filtered to what ``me`` is entitled to know.

    A Victory Point card is held, never played, and stays hidden until its owner wins — so an
    opponent's *true* score is hidden information, and the difference between it and their
    public score is the exact number of VP cards they hold. That is the fact deciding whether
    they are one build from winning.

    ⚠️ This used to be ``rules.scores(state)`` unconditionally, sitting next to
    ``public_scores``, so one subtraction gave it away. Not a sandbox escape and not something
    the existing leak tests could catch: they point at the observation and at
    :class:`~catan.view.PublicView`, and this is a leak in the *documented* interface,
    readable by a well-behaved agent using only public keys. See
    ``docs/audit-2026-08-05-public-arena.md`` §B3.

    Two things are still revealed, both of them entitlements rather than concessions. You see
    your **own** true score, because you can see your own hand. And once the game is
    ``finished`` everyone's is revealed, because the winner's cards are turned over — which is
    also what ``interfaces.web.stats`` already keys its ``reveal`` on, and what the recorder
    needs to write down a final margin.
    """
    return {
        player: (rules.victory_points(state, player)
                 if finished or player == me
                 else rules.public_victory_points(state, player))
        for player in state.players
    }


def _check_index(action):
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError(f"action must be an int index, got {type(action).__name__}")
    return action
