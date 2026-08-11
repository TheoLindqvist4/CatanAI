"""The AlphaZero model the game plays against.

A second champion, beside the first. ``models/champion.pt`` is the PPO lineage and is not
touched by anything in this package; ``models/champion_az.pt`` is this one. Two files rather
than one because they are trained by different methods and the interesting question — which
of them a person should be playing — is answered by a match, not by whichever finished last.

    checkpoints/alphazero/   scratch. A run owns it, rewrites it, may ruin it. Not in git.
    models/champion_az.pt    the champion. Changes only by promotion. In git.

The interfaces read ``models/`` and never ``checkpoints/``, so a training run cannot disturb
a game in progress. Installation is a rename over a fully written file, so a game that loads
the champion mid-promotion gets either the old one or the new one and never half of either.

**One rung decides: the reigning champion.** A candidate is installed when it beats
``models/champion_az.pt`` with its Wilson lower bound above 50% over ``games``, and for no
other reason. The heuristic is played only when ``--baseline-games`` asks for it, and even
then it is recorded rather than consulted — the number is history, not a veto. The argument
that used to justify a heuristic rung is still sound (self-play is non-transitive, so a
candidate can climb the ladder by learning the champion's habits while getting worse at the
game); it was removed on purpose, so that "promoted" means exactly one measurable thing.

**The first promotion of a lineage is refused rather than waved through, which is why this
module exists rather than reusing :mod:`training.champion`.** ``CLAUDE.md`` records the hole
in the older gate: when no champion loads, ``promote`` takes its ``reigning is None`` branch
and installs *immediately*, then writes the baseline every later candidate is measured
against. It fires exactly when the encoder has changed, which is exactly when nobody is
watching. With the heuristic rung gone there is nothing left to gate a first candidate on, so
it is refused outright and installing one is an explicit ``--force --reason`` that the record
carries forever.
"""

import argparse
import datetime
import hashlib
import json
import pathlib

from catan import action_space, encoder

MODELS = pathlib.Path("models")

#: This lineage's champion. Deliberately not ``champion.pt``.
CHAMPION = MODELS / "champion_az.pt"
RECORD = MODELS / "champion_az.json"

#: The other lineage, read-only from here.
PPO_CHAMPION = MODELS / "champion.pt"

#: Games in a promotion match. 400 gives a Wilson interval of about +-4.9 points, so a
#: candidate has to win around 55% for the lower bound to clear 50% — the guide's threshold,
#: arrived at from the statistics rather than chosen.
#:
#: This was 300 while matches were sequential. A searching agent runs the network once per
#: simulation at batch 1, so a game costs 5.2 seconds at 32 simulations against 0.29 without
#: search, and 400 games was half an hour *per rung* with three rungs then being played —
#: which is how a gate stops being run. Only one rung decides now and the heuristic is off by
#: default, so the gate plays two matches at most.
#: :mod:`training.alphazero.arena` plays the match across processes and
#: brings a rung back to a couple of minutes, so the number is set by what makes a good gate
#: rather than by what fits in an afternoon.
PROMOTION_GAMES = 400

# --------------------------------------------------------------------------- #
# Naming a champion                                                           #
# --------------------------------------------------------------------------- #
#
# Every promotion overwrites ``models/champion_az.pt``, so "the champion" means a different
# player every few hours and a sentence like "the champion scores 74.7%" rots the moment the
# next one lands. A name fixes the reference.
#
# The name is **derived from the weights**, not allocated. Two consequences, both wanted:
# the same weights always produce the same name, so a checkpoint copied to another machine
# or promoted twice keeps its identity; and two different networks cannot collide onto one
# name by accident, because the collision would require a SHA-256 prefix collision rather
# than a counter being reset. The generation number is what makes it *ordered*; the codename
# is what makes it memorable.

#: Words the codename is built from. 32 x 32 is 1,024 pairs, which is far more than this
#: project will ever promote, and the generation prefix disambiguates regardless.
_ADJECTIVES = (
    "amber", "ashen", "bold", "bronze", "clever", "coastal", "copper", "crimson",
    "dusty", "eager", "fallow", "fertile", "gilded", "granite", "hardy", "hollow",
    "inland", "iron", "keen", "lucky", "northern", "patient", "quiet", "rugged",
    "sable", "shrewd", "steady", "stony", "thrifty", "verdant", "wary", "windward",
)
_NOUNS = (
    "anchor", "badger", "beacon", "cairn", "cedar", "clay", "crane", "delta",
    "ember", "falcon", "ferry", "forge", "granary", "harbour", "heron", "keep",
    "lantern", "marten", "meadow", "mill", "orchard", "otter", "quarry", "raven",
    "ridge", "sable", "sheaf", "spire", "thicket", "vale", "warren", "wheat",
)


def fingerprint(weights):
    """A stable SHA-256 over a state dict's keys and raw bytes.

    Sorted by key so the digest does not depend on insertion order, and taken over the
    tensor bytes rather than a repr so it does not depend on formatting. This is the
    identity of a set of weights.
    """
    digest = hashlib.sha256()
    for key in sorted(weights):
        tensor = weights[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def codename(digest):
    """``adjective-noun`` from a hex digest. Deterministic, and stable forever."""
    value = int(digest[:16], 16)
    return f"{_ADJECTIVES[value % len(_ADJECTIVES)]}-{_NOUNS[(value // len(_ADJECTIVES)) % len(_NOUNS)]}"


def champion_name(generation, digest):
    """The full name: ``gen5-amber-otter``.

    The generation orders them and the codename identifies them. Either alone is not
    enough — a bare counter says nothing about *which* weights, and a bare codename does
    not say which came first.
    """
    return f"gen{generation}-{codename(digest)}"


def name(default="unnamed"):
    """The reigning champion's name, from the record. For the interfaces to display."""
    return record().get("name", default)

#: Simulations the champion is measured at **and played at**. A win rate is a property of the
#: ``(weights, simulations)`` pair, so measuring at one number and playing at another would
#: publish a figure for a player nobody faces. The interfaces import this rather than keeping
#: their own copy.
#:
#: 64, raised from 32, because search still buys strength at this network size — measured
#: against the fixed heuristic over 200 games apiece, on the champion of the day rather than
#: on this one (the 74.4% at 32 simulations is the figure in gen2's record):
#:
#:      0 sims (raw policy)  64.8%      64 sims   78.9%
#:     16 sims               73.4%     128 sims   81.9%
#:     32 sims               74.4%
#:
#: The shape of that table is the claim, not its level. The reigning champion,
#: gen6-gilded-beacon, carries ``beat_heuristic: null`` — it has never played the heuristic,
#: because the gate stopped asking.
#:
#: Still climbing at 128, so this is a *latency* choice rather than a strength one. One
#: decision costs 52 ms at 32, 101 ms at 64 and 207 ms at 128 on one thread, and 207 ms does
#: not fit inside the 200 ms pace a watched game is played back at — so 64 takes most of the
#: available gain and stays comfortably inside the budget. Raise it if you do not care about
#: watch mode, and re-measure the record if you do: the number in `champion_az.json` belongs
#: to the simulation count it was measured at.
CHAMPION_SIMULATIONS = 64

# There is deliberately no baseline-regression limit here any more. The gate is the reigning
# champion and nothing else; the heuristic is measured only when asked for, and never vetoes.
# `training.champion` — the PPO lineage — still has one, and is unchanged. Plain `#`, not
# `#:`: a doc-comment left where its constant used to be documents whatever follows it, and
# what follows it is `load`.


def load(path=None, simulations=CHAMPION_SIMULATIONS, temperature=0.0, seed=None):
    """The AlphaZero champion as a playable agent, or ``None`` if there is not a usable one.

    Never raises. A missing file, a missing PyTorch, or a model built for a different
    observation or action space all mean the same thing to a caller: offer something else.
    A checkpoint from before an encoder change loads perfectly well and then fails on the
    first move, which is the failure this exists to prevent.

    ``path`` defaults to :data:`CHAMPION` but is resolved *when called*, not when this
    function was defined. Written ``path=CHAMPION`` it would capture the module constant at
    import, so redirecting the champion — which a test does on every run, and which anyone
    debugging a candidate will try — would silently keep reading the real one.
    """
    path = pathlib.Path(CHAMPION if path is None else path)
    if not path.is_file():
        return None
    from training.alphazero.agent import MCTSAgent

    agent = None
    try:
        agent = MCTSAgent.load(path, simulations=simulations, temperature=temperature,
                               seed=seed)
    except Exception:
        # Falls through to the graft. The network's constructor *raises* when `obs_size`
        # does not match the encoder, so an observation change lands here rather than
        # producing a loaded-but-wrong-shaped model — which is why the graft cannot be
        # attempted only after a successful load.
        agent = None

    if agent is None or agent.net.obs_size != encoder.SIZE:
        # The observation changed under a champion promoted against the old one. That used
        # to mean "offer the heuristic instead", and it is how both interfaces silently lost
        # their learned opponent once already. It does not have to: every change to this
        # observation appends, so `network.graft` gives the new columns zero weight and the
        # result computes *exactly* the function that was measured. Verified rather than
        # asserted — the champion of the time, the 1,884-float one, scored 74.7% before the
        # change and 75.6% after, over 400 and 200 games. Several promotions have happened
        # since, so that pair is evidence about the graft and about nothing else; it is not a
        # statement about how strong the champion is today.
        # See ``docs/decisions/0024-what-a-placement-can-see.md``.
        try:
            from training.alphazero.network import load_for_alphazero

            net, _ = load_for_alphazero(path)
            agent = MCTSAgent(net, simulations=simulations, temperature=temperature,
                              seed=seed)
        except Exception:
            return None

    if agent.net.num_actions != action_space.NUM_ACTIONS:
        return None                       # the action space moved; nothing can save that
    if agent.net.obs_size != encoder.SIZE:
        return None
    return agent


def load_previous_technique(temperature=0.0, seed=None):
    """The PPO champion, grafted onto the current observation if it predates it.

    ``training.champion.load`` deliberately refuses a checkpoint whose observation does not
    match, and that is right for *its* callers — a stale model plays nonsense. But the
    AlphaZero ladder needs to play the previous technique to know whether it has beaten it,
    and the graft is exact: the new observation columns are given zero weight, so the grafted
    network computes the same function on the same position. See
    :func:`training.alphazero.network.graft`.

    Returns ``None`` when there is no PPO champion or it cannot be reconciled.
    """
    if not PPO_CHAMPION.is_file():
        return None
    try:
        from training.agent import PolicyAgent
        from training.alphazero.network import load_for_alphazero
        import torch

        checkpoint = torch.load(PPO_CHAMPION, map_location="cpu", weights_only=False)
        if checkpoint["config"].get("obs_size") == encoder.SIZE:
            return PolicyAgent.load(PPO_CHAMPION, temperature=temperature, seed=seed)
        # Grafted, and kept as a *policy* agent: the value head was reset by the graft, and
        # a PolicyAgent never reads it. Playing it as an MCTSAgent would search with a value
        # head that has never been trained, which measures the graft rather than the model.
        net, _ = load_for_alphazero(PPO_CHAMPION, value_activation="linear")
        return PolicyAgent(net, temperature=temperature, seed=seed)
    except Exception:
        return None


def record():
    """What is known about the reigning AlphaZero champion, or ``{}``."""
    if not RECORD.is_file():
        return {}
    try:
        return json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def describe():
    """One line about the champion, for a startup message or a CLI.

    ``beat_champion`` comes first because it is the only rung that decides — and since
    ``baseline_games`` began defaulting to 0 it is often the only one measured at all, so a
    version of this that reported the other two reported nothing but a date.
    """
    if load() is None:
        return "no AlphaZero champion yet"
    info = record()
    parts = [f"AlphaZero champion from {info.get('promoted_at', 'an unknown date')}"]
    if info.get("beat_champion") is not None:
        beat = f"{100 * info['beat_champion']:.1f}% vs the champion it replaced"
        # A forced promotion did not clear the gate, and the number must not read as if it
        # had. The record carries the reason forever; this is the one-line version.
        parts.append(f"{beat} (forced)" if info.get("forced") else beat)
    if info.get("beat_heuristic") is not None:
        parts.append(f"{100 * info['beat_heuristic']:.1f}% vs the heuristic")
    if info.get("beat_ppo_champion") is not None:
        parts.append(f"{100 * info['beat_ppo_champion']:.1f}% vs the PPO champion")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Promotion                                                                   #
# --------------------------------------------------------------------------- #

def promote(candidate_path, games=PROMOTION_GAMES, seed=41_000,
            simulations=CHAMPION_SIMULATIONS, force=False, reason=None,
            baseline_games=None, ppo_games=None, log=print):
    """Install ``candidate_path`` as the AlphaZero champion if it earns the place.

    Args:
        games: the head-to-head rung against the reigning champion. This is the one that
            decides, so it is the one that should stay large.
        baseline_games: games against the fixed heuristic, **recorded and never a veto**.
            Defaults to 0, which does not play it at all.

            It used to be the anti-overfitting tripwire, and the argument for it is still
            true: self-play is non-transitive, so a candidate can beat the champion by
            learning its habits while getting worse at the game, and a fixed external
            opponent is the only thing that notices. It was removed deliberately — the gate
            is now strictly "did it beat the champion" — so that one rung decides and the
            promotion means exactly one thing. Pass a positive number to record the yardstick
            for the history; nothing will refuse a promotion on it.
        ppo_games: games against the PPO champion, recorded but never a veto. **Defaults to
            ``games``**, unlike ``baseline_games`` above — so a bare ``promote`` plays two
            matches whenever a PPO champion loads, and only the first of them can refuse.
            0 skips it, which is what :mod:`training.alphazero.chain` passes.
        force: install without requiring the head-to-head rung. The record then carries
            ``"forced": true`` and ``"forced_reason"``, because a promotion that did not pass
            the gate must never be mistaken for one that did — and a year later the only
            thing that distinguishes them is what was written down at the time.
        reason: why the gate was overridden. **Required** when ``force`` is set. A forced
            promotion with no stated reason is indistinguishable from a mistake.

    Returns ``(promoted, reason)``.
    """
    if force and not reason:
        return False, ("a forced promotion has to say why — pass --reason. Overriding the "
                       "gate silently is how a gate stops meaning anything")
    from training.alphazero.arena import compete
    from training.alphazero.evaluator import better, format_result

    if load(candidate_path, simulations=simulations) is None:
        return False, f"{candidate_path} is not a usable model for this engine"

    MODELS.mkdir(parents=True, exist_ok=True)
    reigning_exists = load(simulations=simulations) is not None

    # Matches run across processes. Sequentially, a 300-game match at 32 simulations is 26
    # minutes *per rung*, which is how a gate stops being run — see
    # :mod:`training.alphazero.arena`.
    me = {"kind": "mcts", "path": str(candidate_path), "simulations": simulations}
    baseline_games = 0 if baseline_games is None else int(baseline_games)
    ppo_games = games if ppo_games is None else int(ppo_games)

    log(f"candidate: {candidate_path} at {simulations} simulations/move")
    if baseline_games:
        log(f"{baseline_games} games against the heuristic:")
        against_baseline = compete(me, {"kind": "heuristic", "noise": 0},
                                   games=baseline_games, seed=seed)
        log("  " + format_result("heuristic", against_baseline))
    else:
        # The default. The record must not carry a stale `beat_heuristic` from the previous
        # champion as though it were measured for this one, so it is written as null.
        log("not playing the heuristic — the gate is the champion and nothing else")
        against_baseline = None

    against_ppo = None
    if ppo_games and load_previous_technique() is not None:
        log(f"{ppo_games} games against the PPO champion:")
        against_ppo = compete(me, {"kind": "ppo_champion"}, games=ppo_games, seed=seed + 1)
        log("  " + format_result("ppo", against_ppo))

    results = {
        "beat_heuristic": None if against_baseline is None else against_baseline["win_rate"],
        "beat_ppo_champion": None if against_ppo is None else against_ppo["win_rate"],
        "beat_champion": None,
        "games": games,
        "baseline_games": baseline_games,
        "simulations": simulations,
    }

    if force:
        # The head-to-head rung is still *played*, even though it cannot refuse: the number
        # is the whole context for the override, and a record that omits it cannot be
        # argued with later. Skipping it was the first version, and it wrote
        # `beat_champion: null` onto the one promotion where that number mattered most.
        if reigning_exists:
            log(f"{games} games against the reigning AlphaZero champion "
                f"(recorded, but not a veto because this is forced):")
            against_champion = compete(
                me, {"kind": "mcts", "path": str(CHAMPION), "simulations": simulations},
                games=games, seed=seed + 2)
            log("  " + format_result("champion", against_champion))
            results["beat_champion"] = against_champion["win_rate"]
        _install(candidate_path, {**results, "forced": True, "forced_reason": reason})
        return True, f"forced: {reason}"

    if not reigning_exists:
        # The gate is the reigning champion and nothing else, so with no reigning champion
        # there is no gate. The old code filled the hole with the heuristic; that rung is
        # gone, and the one thing this must not become is the silent auto-install
        # `CLAUDE.md` records — which fires exactly when `encoder.SIZE` changed, which is
        # exactly when nobody is watching. So it refuses, and installing the first champion
        # of a lineage is now an explicit, recorded act.
        return False, (
            "there is no reigning champion to measure against, and the head-to-head rung is "
            "the whole gate — so this would install unmeasured. Pass --force --reason with "
            "whatever you did measure by hand; the record will say it was forced."
        )

    log(f"{games} games against the reigning AlphaZero champion:")
    against_champion = compete(
        me, {"kind": "mcts", "path": str(CHAMPION), "simulations": simulations},
        games=games, seed=seed + 2)
    log("  " + format_result("champion", against_champion))
    results["beat_champion"] = against_champion["win_rate"]

    if not better(against_champion):
        low, high = against_champion["ci"]
        return False, (f"beat the champion {100 * against_champion['win_rate']:.1f}% but the "
                       f"interval [{100 * low:.1f}, {100 * high:.1f}] includes 50% — "
                       f"not shown better")

    _install(candidate_path, results)
    return True, (f"promoted: {100 * against_champion['win_rate']:.1f}% against the champion "
                  f"(lower bound {100 * against_champion['ci'][0]:.1f}%)")


def _install(candidate_path, results):
    """Replace the champion atomically, so a game in progress never sees half a file."""
    import torch

    MODELS.mkdir(parents=True, exist_ok=True)
    source = torch.load(candidate_path, map_location="cpu", weights_only=False)

    staging = CHAMPION.with_suffix(".incoming")
    torch.save({
        "config": source["config"],
        "weights": source["weights"],
        "iteration": source.get("iteration"),
        "lineage": "alphazero",
    }, staging)
    staging.replace(CHAMPION)

    previous = record()
    history = (previous.get("history", []) + [
        {k: v for k, v in previous.items() if k != "history"}
    ])[-10:] if previous else []

    digest = fingerprint(source["weights"])
    generation = len(history) + 1
    entry = {
        "name": champion_name(generation, digest),
        "generation": generation,
        "fingerprint": digest[:16],
        "promoted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lineage": "alphazero",
        "source": str(candidate_path),
        "observation": encoder.SIZE,
        "actions": action_space.NUM_ACTIONS,
        **results,
    }
    entry["history"] = history
    RECORD.write_text(json.dumps(entry, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The AlphaZero model the game plays against",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="what the AlphaZero champion is")
    show.set_defaults(func=lambda a: print(describe()) or
                      print(json.dumps(record(), indent=2) if record() else ""))

    named = sub.add_parser("name", help="the reigning champion's name, and the lineage")
    named.set_defaults(func=lambda a: _name_command())

    run = sub.add_parser("promote", help="install a candidate if it is measurably better")
    run.add_argument("candidate")
    run.add_argument("--games", type=int, default=PROMOTION_GAMES,
                     help="the head-to-head rung, which is the one that decides")
    run.add_argument("--baseline-games", type=int, default=None, dest="baseline_games",
                     help="games against the fixed heuristic, recorded for the history and "
                          "never a veto. Defaults to 0: the gate is the champion alone")
    run.add_argument("--ppo-games", type=int, default=None, dest="ppo_games",
                     help="games against the PPO champion, recorded but never a veto. "
                          "Defaults to --games, so this rung IS played unless you pass 0")
    run.add_argument("--seed", type=int, default=41_000)
    run.add_argument("--simulations", type=int, default=CHAMPION_SIMULATIONS)
    run.add_argument("--force", action="store_true",
                     help="install even if the head-to-head rung does not clear 50%%; "
                          "requires --reason and is recorded as forced")
    run.add_argument("--reason", default=None,
                     help="why the gate is being overridden; required with --force")
    run.set_defaults(func=_promote_command)

    arguments = parser.parse_args(argv)
    return arguments.func(arguments)


def _name_command():
    """The lineage, newest first. Unnamed entries predate naming and say so."""
    current = record()
    if not current:
        print("no champion")
        return 1
    rows = [current] + list(reversed(current.get("history", [])))
    total = len(rows)
    print(f"{'name':<24} {'promoted':<18} {'vs champion':>12} {'vs heuristic':>13}")
    for offset, entry in enumerate(rows):
        generation = entry.get("generation", total - offset)
        label = entry.get("name") or f"gen{generation}-(unnamed)"
        beat = entry.get("beat_champion")
        against = entry.get("beat_heuristic")
        print(f"{label:<24} {entry.get('promoted_at', '?'):<18} "
              # ASCII on purpose, though not for the reason this comment used to give:
              # cp1252 encodes an em dash perfectly well, at 0x97. What it cannot encode is
              # the U+FFFD that a pipe's decode makes of that byte — `chain.run`'s docstring
              # has the round trip, and it killed a ten-hour chain once. A listing that
              # raises is worse than a plain one. (train.py's banner avoids the *warning
              # sign* for a different and simpler reason: cp1252 has no such character.)
              f"{('-' if beat is None else f'{100 * beat:.1f}%'):>12} "
              f"{('-' if against is None else f'{100 * against:.1f}%'):>13}"
              + ("   (forced)" if entry.get("forced") else ""))
    return 0


def _promote_command(arguments):
    import torch

    torch.set_num_threads(4)
    promoted, reason = promote(arguments.candidate, games=arguments.games,
                               seed=arguments.seed, simulations=arguments.simulations,
                               force=arguments.force, reason=arguments.reason,
                               baseline_games=arguments.baseline_games,
                               ppo_games=arguments.ppo_games)
    print(("PROMOTED — " if promoted else "kept the current champion — ") + reason)
    print(describe())
    return 0 if promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
