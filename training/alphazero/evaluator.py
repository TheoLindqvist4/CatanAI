"""Measuring an AlphaZero candidate, and deciding whether it is better.

Reuses :mod:`training.evaluate` rather than restating it: the Wilson interval, the seat
swapping and the truncation count are the same problem for both techniques, and two
implementations of "is 54% of 400 games meaningful" would eventually disagree.

What is added here is the **ladder**: opponents named by a string rather than passed in as
objects. A caller then does not have to know how to build the previous technique's champion,
and a rung this checkout cannot build — no file, or a file whose observation does not fit —
is *skipped and reported* rather than raising in the middle of a training run. Three names
exist, and :func:`_opponent` is the whole list:

``heuristic``
    The fixed yardstick, playing its best: ``HeuristicAgent(0)`` is seed 0, and its noise
    defaults to 0. A figure against it is only comparable within one version of the rules,
    which is why ``CLAUDE.md`` says not to change it.

``ppo_champion``
    ``models/champion.pt`` — the other lineage — grafted onto the current observation, so the
    two champions can meet and "which one should the interface offer" has an answer.

``champion``
    ``models/champion_az.pt``, the reigning champion of this lineage.

**This module measures; it decides nothing.** Which rungs are worth playing, which of them
may refuse a candidate, and how many games each is worth are the caller's business, and the
callers answer differently. :func:`training.alphazero.champion.promote` is the gate — one
rung decides, and ``docs/decisions/0030-one-rung.md`` is why — and its per-opponent defaults
are documented on it. :meth:`training.alphazero.trainer.Trainer.evaluate` plays a single
opponent named by its config. :mod:`training.alphazero.chain` then overrides the gate's
defaults on the command line, so what an overnight chain run plays is neither the gate's
default nor this module's. Read each of the three where it lives, and do not restate them
here: this docstring used to recite the gate's defaults, accurately, directly above a
:func:`ladder` whose own defaults are the opposite of them.

The gate does not call :func:`ladder` at all. It builds its matches through
:mod:`training.alphazero.arena`, because they run across processes — a sequential match is
tens of minutes per rung, and :data:`training.alphazero.champion.PROMOTION_GAMES` carries the
measurement.

What is evaluated is an *agent*, not a network, and in this package that agent searches. So a
win rate is a property of ``(weights, simulations)`` and both belong in the record;
``simulations=0`` is a legitimate value, means the raw policy, and is a different player from
the same weights at 64. The ``ppo_champion`` rung is deliberately not searched — the graft
resets its value head, so searching it would measure the graft rather than the model. See
:func:`training.alphazero.champion.load_previous_technique`.
"""

from catan.agents import HeuristicAgent
from catan.rulesets import RANKED_1V1
from training.evaluate import confidence_interval, evaluate, format_result

#: How many games both functions below play unless told otherwise, and the same number the
#: in-loop check carries as ``config["evaluation_games"]``. +-6.9 points at 50%: enough to
#: notice a collapse, not enough to promote on, which is what
#: :data:`training.alphazero.champion.PROMOTION_GAMES` is for.
EVALUATION_GAMES = 200


def evaluate_agent(agent, opponent=None, games=EVALUATION_GAMES, seed=10_000,
                   max_turns=800):
    """One matchup. Thin wrapper so every call site uses the same ruleset and turn cap.

    ``opponent`` defaults to ``None``, which :func:`training.evaluate.evaluate` reads as the
    fixed heuristic — so a bare ``evaluate_agent(agent)`` is a yardstick measurement and
    never self-play. The ruleset is pinned to :data:`~catan.rulesets.RANKED_1V1` and the turn
    cap to 800, tighter than that function's 1,000; both are fixed here rather than left to
    the caller so that two numbers produced by this package are comparable.
    """
    return evaluate(agent, opponent, games=games, seed=seed, ruleset=RANKED_1V1,
                    max_turns=max_turns)


def ladder(agent, games=EVALUATION_GAMES, seed=10_000, include=("heuristic",), log=None):
    """Play ``agent`` against every rung named in ``include``. Returns ``{name: result}``.

    ``include`` defaults to ``("heuristic",)`` — the fixed yardstick, alone. Neither champion
    is played unless a caller names it. **That default belongs to this function and to
    nothing else**: the promotion gate does not come through here, and chooses its own
    opponents. ``games`` is per rung rather than in total, and each rung is played at
    ``seed + offset`` so the rungs do not all replay the same games.

    A rung this checkout cannot build is recorded as ``None`` and skipped, so an evaluation
    between iterations cannot kill a training run over a missing file —
    ``tests/test_alphazero.py::test_evaluator_ladder_skips_a_missing_rung`` pins that. An
    unrecognised *name* still raises, from :func:`_opponent`.

    Nothing in the package calls this today: the gate builds its matches across processes and
    :meth:`training.alphazero.trainer.Trainer.evaluate` calls :func:`evaluate_agent` with the
    one opponent it wants. So this is a convenience for a caller who wants several rungs in
    one go, and its default says nothing about how anything is actually measured.
    """
    results = {}
    for offset, name in enumerate(include):
        opponent = _opponent(name)
        if opponent is None:
            results[name] = None
            continue
        results[name] = evaluate_agent(agent, opponent, games=games, seed=seed + offset)
        if log is not None:
            log("  " + format_result(name, results[name]))
    return results


def _opponent(name):
    """Build a named rung, or ``None`` when this checkout has no such opponent.

    ``None`` and the ``ValueError`` mean different things on purpose: a rung that will not
    build is a fact about this machine and this encoder, which a caller mid-run should be
    able to skip past, while a name that does not exist is a typo and cannot be skipped past
    into anything meaningful.
    """
    if name == "heuristic":
        return HeuristicAgent(0)
    if name == "ppo_champion":
        from training.alphazero import champion

        return champion.load_previous_technique()
    if name == "champion":
        from training.alphazero import champion

        return champion.load()
    raise ValueError(f"unknown evaluation rung {name!r}")


def better(result, threshold=0.5):
    """Whether a result shows the candidate is *better*, not merely ahead.

    The Wilson lower bound has to clear ``threshold``. A candidate that won 52% of 400 games
    has shown nothing — the interval covers 47% to 57% — and promoting on the point estimate
    is how a ladder climbs while the player gets worse.
    """
    low, _ = result["ci"]
    return low > threshold


__all__ = ["EVALUATION_GAMES", "better", "confidence_interval", "evaluate_agent",
           "format_result", "ladder"]
