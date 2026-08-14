"""The network AlphaZero trains, and how it is started.

Not a new architecture. :class:`~training.structured_net.StructuredPolicyValueNet` already
produces exactly what AlphaZero needs — one logit per action laid out by board element, and
one value — and it is already the structure-aware model the guide asks for: 19 tiles, 54
vertices and 72 roads embedded by shared per-entity MLPs, with neighbourhood information
carried by constant incidence matrices from :mod:`catan.topology`. That *is* a graph network
over the Catan graph; it simply exploits the graph being the same in every game to do the
message passing as a matmul against a constant instead of a gather.

Two things are different here.

**The value head is bounded.** AlphaZero's target is exactly ``+1``/``-1`` and the loss is a
mean square error, so ``tanh`` is applied — see ``value_activation`` on the network.

**The run is warm-started, not started.** A cold AlphaZero net plays randomly, and at the
simulation counts a 20-core CPU can afford, random self-play produces a policy target barely
better than its own prior. The repository already contains a policy that beats the heuristic
about 71% of the time. :func:`graft` carries those weights onto the current observation, so
search begins as *policy improvement over something that already plays*, which is the regime
AlphaZero is strong in. Recorded as a deliberate departure from "learns entirely from
self-play" in ``docs/decisions/0023-alphazero-self-play.md``.

**Why a graft is needed at all.** ``models/champion.pt`` was promoted when
``encoder.SIZE`` was 1868. The affordability block added 16 floats at offset 1773, so the
encoder is now 1884 and the champion no longer loads — ``training.champion.load()`` returns
``None`` today, which is why the interfaces currently offer no learned opponent. Only one
tensor is the wrong shape: the first layer of ``context_mlp``, whose input is the
un-positional tail of the observation. Every other parameter is untouched by the change,
because the tiles/vertices/roads encoders read fixed-width rows and the affordability block
is not one of them.

The 16 new columns are inserted **as zeros**. That is the whole reason this is safe rather
than a guess: a zero column contributes nothing, so the grafted network computes the same
function as the original on the same position, and the new features start neutral and are
learned. :func:`graft` returns the number of columns it inserted so a caller can refuse a
checkpoint it does not understand instead of silently mangling one.
"""

import pathlib

import torch

from catan import action_space, encoder
from training.alphazero import layouts
from training.loading import load_model
from training.net import build
from training.structured_net import StructuredPolicyValueNet

def new_network(width=128, road_width=64, context=192, hops=1, depth=3, rounds=0,
                trunk=192, aux=True):
    """A fresh AlphaZero network at the current observation and action space.

    The defaults changed in record 0026, from ``64/32/128, depth 2, trunk 256`` (200,379
    parameters) to ``128/64/192, depth 3, trunk 192`` (374,331). Three measurements, not a
    preference:

    * A capacity sweep over nine shapes on 176,342 decisions — matched to the replay buffer
      size on purpose — put ``depth`` 2 -> 3 as the single largest effect, halving value
      error (MAE 0.1121 -> 0.0554) for +37k parameters, while policy agreement barely moved.
    * 57.4% of the old parameters sat in one layer, the pooled trunk, against 9.4% in the
      per-entity encoders that do the board reasoning. Cutting ``trunk`` and widening the
      entities buys the depth almost for free.
    * The whole change costs 11% of self-play throughput, because the network is only ~20%
      of a searched decision — the Python engine is the other 80%.

    ``aux`` defaults on: the auxiliary heads are the fix for a value head measured
    explaining 14% of held-out outcome variance. See :class:`StructuredPolicyValueNet`.
    """
    return StructuredPolicyValueNet(
        obs_size=encoder.SIZE,
        num_actions=action_space.NUM_ACTIONS,
        width=width, road_width=road_width, context=context,
        hops=hops, depth=depth, rounds=rounds, trunk=trunk,
        value_activation="tanh", aux=aux,
    )


# --------------------------------------------------------------------------- #
# Carrying a checkpoint across an observation change                          #
# --------------------------------------------------------------------------- #

def _segments(hops):
    """How each observation-width layer's input is assembled, in order.

    A block name means "one row of that block"; an integer means that many constant geometry
    columns; ``"context"`` means the whole un-positional tail. Read straight off
    :meth:`~training.structured_net.StructuredPolicyValueNet.forward`'s three ``torch.cat``
    calls, and checked against a real network at test time so the two cannot drift.
    """
    tile, vertex, road = ["tiles", 1], ["vertices", 2], ["roads", 1]
    if hops >= 1:
        tile += ["vertices"]
        vertex += ["tiles", "vertices", "roads"]
        road += ["vertices"]
    if hops >= 2:
        tile += ["tiles"]
        vertex += ["tiles", "vertices"]
        road += ["tiles"]
    return {
        "tile_embed.weight": tile,
        "vertex_embed.weight": vertex,
        "road_embed.weight": road,
        "context_mlp.0.weight": ["context"],
    }


def _positional(layout):
    return sum(layout[name][0] * layout[name][1]
               for name in ("tiles", "vertices", "roads"))


def _input_map(segments, old, new):
    """For each input column of a layer, the old column it came from, or ``-1``.

    Walks the segments once, keeping a cursor into the *old* layer's inputs. That is the
    whole trick: the new layer is wider, but the segments appear in the same order, so the
    old position of every surviving column is just "how much old input came before it".
    """
    mapping, cursor = [], 0
    for segment in segments:
        if isinstance(segment, int):                       # constant geometry columns
            mapping.extend(range(cursor, cursor + segment))
            cursor += segment
        elif segment == "context":
            columns = layouts.column_map(old, new)
            old_start = _positional(old)
            mapping.extend(-1 if c < 0 else c - old_start + cursor
                           for c in columns[_positional(new):])
            cursor += layouts.total(old) - old_start
        else:                                              # one row of a repeated block
            old_features = old.get(segment, (0, 0))[1]
            mapping.extend(cursor + f if f < old_features else -1
                           for f in range(new[segment][1]))
            cursor += old_features
    return mapping


def _gather(weight, mapping):
    """``weight`` rebuilt so column *i* is old column ``mapping[i]``, or zeros.

    Zeros are what make this safe rather than a guess: a zero column contributes nothing, so
    the rebuilt layer computes exactly the function it did on the features it already had,
    and the new ones start neutral and are learned.
    """
    out = torch.zeros(weight.shape[0], len(mapping),
                      dtype=weight.dtype, device=weight.device)
    keep = [i for i, source in enumerate(mapping) if source >= 0]
    if keep:
        out[:, keep] = weight[:, [mapping[i] for i in keep]]
    return out


def graft(state_dict, config):
    """Widen a checkpoint's observation-width layers to the current layout, with zero columns.

    Handles blocks that were **appended to** — which is the only shape a change to this
    observation is allowed to take, per the "append, never insert" rule in ``CLAUDE.md``.
    A block whose features were reordered would need a different and much more careful
    function; this refuses instead of pretending.

    Args:
        state_dict: the stored weights.
        config: the stored config. Carries ``layout`` on checkpoints written since that
            existed, and otherwise is looked up by ``obs_size`` in
            :data:`training.alphazero.layouts.HISTORICAL`.

    Returns:
        ``(state_dict, inserted)`` — weights safe to load at the current
        :data:`catan.encoder.SIZE`, and how many columns were added in total. ``inserted``
        is 0 when the checkpoint already matches.

    Raises:
        ValueError: when the old layout cannot be known, or a block shrank. Loud rather than
            approximate: a checkpoint bent into the wrong shape loads, runs, and plays
            nonsense.
    """
    if config.get("obs_size") == encoder.SIZE:
        return dict(state_dict), 0
    if config.get("kind") != "structured":
        raise ValueError(f"cannot graft a {config.get('kind', 'flat')!r} checkpoint: only "
                         f"the structured network has observation-width layers this "
                         f"function knows how to widen")

    old = layouts.resolve(config)
    if old is None:
        raise ValueError(
            f"checkpoint was trained at obs_size={config.get('obs_size')} and carries no "
            f"layout, and that size is not in layouts.HISTORICAL — so which block grew, and "
            f"by how much, is unknown"
        )
    new = layouts.signature()

    grafted = dict(state_dict)
    inserted = 0
    for name, segments in _segments(config.get("hops", 1)).items():
        weight = grafted.get(name)
        if weight is None:
            continue
        mapping = _input_map(segments, old, new)
        if len(mapping) == weight.shape[1]:
            continue                                       # this layer did not change
        if weight.shape[1] != sum(1 for m in mapping if m >= 0):
            raise ValueError(
                f"{name} has {weight.shape[1]} inputs but the recorded layout accounts for "
                f"{sum(1 for m in mapping if m >= 0)} — the layout does not describe this "
                f"checkpoint"
            )
        grafted[name] = _gather(weight, mapping)
        inserted += sum(1 for m in mapping if m < 0)
    return grafted, inserted


def load_for_alphazero(path, value_activation="tanh"):
    """Load any structured checkpoint as an AlphaZero network at the current observation.

    Grafts if needed, and switches the value head to ``tanh``. Switching the activation
    changes what the *existing* value head means — an unbounded head trained by PPO produces
    numbers outside ``[-1, 1]`` that ``tanh`` will saturate — so the head is re-initialised
    small rather than carried over. The policy, which is what warm-starting is for, is kept
    in full.

    Returns:
        ``(net, notes)`` where ``notes`` is a dict describing what was done, for the record.
    """
    path = pathlib.Path(path)
    checkpoint = load_model(path)
    config = dict(checkpoint["config"])
    weights, inserted = graft(checkpoint["weights"], config)

    config["obs_size"] = encoder.SIZE
    config["value_activation"] = value_activation
    net = build(config)

    if value_activation != checkpoint["config"].get("value_activation", "linear"):
        # A PPO critic is a return predictor on a different scale. Keeping it would start
        # the run with a confident, wrong, saturated value head, and AlphaZero's search is
        # steered by the value. Reset it; it is 257 numbers.
        torch.nn.init.orthogonal_(net.value_head.weight, gain=0.01)
        torch.nn.init.constant_(net.value_head.bias, 0.0)
        weights = {k: v for k, v in weights.items() if not k.startswith("value_head.")}

    missing, unexpected = net.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint has parameters this network does not: {unexpected}")
    return net, {
        "source": str(path),
        "trained_at_obs_size": checkpoint["config"].get("obs_size"),
        "grafted_columns": inserted,
        "reset": sorted(missing),
    }
