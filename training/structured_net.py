"""A structure-aware policy/value network.

The flat :class:`~training.net.PolicyValueNet` treats the observation as an unordered bag of
numbers. It is not one. ``catan.encoder`` lays it out as regular blocks — 19 tiles x 19,
54 vertices x 27, 72 roads x 6, then the un-positional remainder — and
``catan.action_space`` lays the actions out the same way: 275 of them (84.6%) name a board
element.

Three things follow, and each is measured rather than assumed.

**Weight sharing.** What "a vertex with three high-pip tiles and nobody built next door"
means is the same fact at vertex 3 and at vertex 41. A flat first layer learns it 54 times,
once per column group, from 54 times less data each. Here one small MLP is applied to all 54
rows, so every vertex of every game is a training example for it. The same argument retires
the tiles block's biggest defect: 361 floats, 20% of the input, *constant for a whole game*,
which a flat layer is free to spend on board identity instead of on play.

**Positional heads.** The logit for "build a settlement at vertex 23" is ``Linear(d, 2)``
applied to vertex 23's own embedding, not a column of a 512x324 dense layer. The head holds
two weights per output rather than 512, and they are the *same* weights for all 54 vertices,
so a vertex that was never built on in training still gets a sensibly ranked logit.

**Neighbourhood information, bought cheaply.** What a vertex knows about its own tiles has
moved twice since this was written: record 0024 gave it the expected cards it makes of *each*
resource, and record 0029 retired the resource-blind pip total that used to be the only
placement signal there was. What a vertex still does not know is whether the robber is sitting
on one of its tiles, or anything at all about the corners next to it; a road knows who owns it
and nothing about its endpoints. Those are one hop away in ``catan.topology``.

The obvious way to fetch them — gather each entity's neighbours' *embeddings* and pool
them — was built and measured first, and it is far too slow: one gather-and-sum of the
vertex-from-road relation at width 64 costs 3.8 ms at batch 512, where the entire flat MLP
trunk costs 6.6 ms. It materialises a ``(512, 162, 64)`` intermediate to add up 162 numbers.

So the aggregation is done on the **raw** features instead, through constant row-normalised
incidence matrices: the same relation as one dense matmul at width 6-19 costs 0.33 ms.
Because the graph is fixed and the aggregation is linear, *k* hops compose into a single
precomputed matrix — ``N_vv @ N_vt`` is "the tiles two steps away" — so a second hop costs
one more small matmul rather than a second round of anything. Non-linear message passing
between embeddings is available (``rounds``) and is measured, but it is not the default:
it costs more than the whole rest of the network and, on the behaviour-cloning benchmark,
buys nothing.

The adjacency is board *geometry*, generated in ``catan.topology`` from ``ROW_LENGTHS``
alone. It is identical on every board, in every game, for every player, and it is the same
information already visible in the rendered board. It carries no play information and
therefore cannot leak any: this module reads nothing but the observation the encoder has
already sanctioned, and ``tests/test_encoder.py``'s leak detectors bound what that contains.

Everything else — the masking contract, ``act``, ``evaluate``, ``config`` — is deliberately
identical to :class:`~training.net.PolicyValueNet`, so this is a drop-in.
"""

import torch
from torch import nn

from catan import action_space, encoder
from catan.actions import ActionType
from catan.state import MAX_PLAYERS
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_TILES,
    ROAD_VERTICES,
    TILE_ADJACENCY,
    TILE_VERTICES,
    VERTEX_NEIGHBOURS,
    VERTEX_ROADS,
    VERTEX_TILES,
)
from training.net import MASK_FILL, _layer


def _layout_signature():
    """The encoder's block shapes, for recording in a checkpoint. Imported lazily so this
    module does not depend on the AlphaZero package."""
    return {
        name: encoder.SHAPES.get(
            name, (1, encoder.LAYOUT[name].stop - encoder.LAYOUT[name].start))
        for name in encoder.LAYOUT
    }

# --------------------------------------------------------------------------- #
# Where each block sits. Read from the encoder rather than written down, so a  #
# change to the layout cannot silently misalign the reshape.                   #
# --------------------------------------------------------------------------- #

TILE_SPAN, VERTEX_SPAN, ROAD_SPAN = (
    encoder.LAYOUT[name] for name in ("tiles", "vertices", "roads")
)
#: Players and global are both un-positional, so they are treated as one context vector.
CONTEXT_START = encoder.LAYOUT["players"].start
CONTEXT_FEATURES = encoder.SIZE - CONTEXT_START

_T_ROWS, _T_FEAT = encoder.SHAPES["tiles"]
_V_ROWS, _V_FEAT = encoder.SHAPES["vertices"]
_R_ROWS, _R_FEAT = encoder.SHAPES["roads"]

#: The action blocks produced from a *position* rather than from the pooled trunk.
_ROAD_SLICE = action_space.SLICES[ActionType.BUILD_ROAD]
_SETTLEMENT_SLICE = action_space.SLICES[ActionType.BUILD_SETTLEMENT]
_CITY_SLICE = action_space.SLICES[ActionType.BUILD_CITY]
_ROBBER_SLICE = action_space.SLICES[ActionType.MOVE_ROBBER]
#: MOVE_ROBBER is tile-major: 19 tiles x (nobody, or one of MAX_PLAYERS victims).
_VICTIMS = MAX_PLAYERS + 1

#: Everything left over is produced from the pooled embedding: END_TURN, the 20 bank
#: trades, and the contiguous DISCARD..PLAY_MONOPOLY run.
NUM_GLOBAL_ACTIONS = action_space.NUM_ACTIONS - (
    NUM_ROADS + 2 * NUM_VERTICES + NUM_TILES * _VICTIMS
)


# --------------------------------------------------------------------------- #
# The board's fixed incidence structure, as constant matrices                 #
# --------------------------------------------------------------------------- #

def incidence(table, out_ids, num_sources):
    """Row-normalised ``(len(out_ids), num_sources)`` incidence matrix for one relation.

    ``M @ X`` is then "the mean of my neighbours' feature rows". Row-normalised rather than
    summed so that a coastal vertex touching one tile and an inland one touching three
    produce values on the same scale; the *degree* those means throw away is handed back
    separately by :func:`geometry`, which is cheaper than letting the magnitudes vary.

    Ids in ``catan.topology`` are 1-based, so column ``j`` is source id ``j + 1``.
    """
    matrix = torch.zeros(len(out_ids), num_sources)
    for row, out_id in enumerate(out_ids):
        sources = table[out_id]
        if sources:
            matrix[row, [s - 1 for s in sources]] = 1.0 / len(sources)
    return matrix


def geometry(tables, out_ids, scales):
    """Static per-entity degree features, in ``[0, 1]``.

    Row-normalising the incidence matrices discards how many neighbours an entity has, and
    that is a real feature — a one-tile corner vertex, a coastal road and a border tile are
    all different places to play. Restored here as two or three constant columns rather
    than by letting the aggregation magnitudes carry it.
    """
    return torch.tensor(
        [[len(table[i]) / scale for table, scale in zip(tables, scales)] for i in out_ids],
        dtype=torch.float32,
    ).unsqueeze(0)


class StructuredPolicyValueNet(nn.Module):
    """``observation -> (324 logits, value)``, computed per board element.

    Args:
        obs_size: must equal ``catan.encoder.SIZE``; an argument only so the constructor
            signature matches :class:`~training.net.PolicyValueNet`.
        num_actions: ``catan.action_space.NUM_ACTIONS``.
        width: embedding width for tiles and vertices.
        road_width: embedding width for roads. There are 72 of them and each produces a
            single logit, so they do not need the vertices' width; halving it is most of
            the difference between this network and the flat one on a forward pass.
        context: width of the embedding of the un-positional part of the observation.
        hops: how far each entity sees through the fixed topology, 0-2. Free-ish: one
            extra constant matmul at raw feature width per relation.
        depth: shared per-entity layers after the neighbourhood concat.
        rounds: rounds of *non-linear* message passing between embeddings, after the
            shared encoders. Measured, expensive, and off by default.
        trunk: width of the pooled head producing the value and the 49 non-positional logits.
        value_activation: ``"linear"`` (the default, and what every PPO checkpoint was
            trained with) or ``"tanh"``. AlphaZero's value target is exactly +-1 and its
            loss is a mean square error, so an unbounded head spends its first epochs
            learning the range instead of the position; ``tanh`` puts the output in
            ``[-1, 1]`` by construction. Kept as an option rather than a second class so
            there is one network, and defaulted to ``"linear"`` so a checkpoint written
            before this field existed rebuilds byte-identically.
        aux: add the auxiliary prediction heads — final ownership per vertex and per road,
            and the final victory-point margin. Off by default so a checkpoint written
            before this field existed rebuilds byte-identically.

            These exist to fix a *measured* failure, not for completeness. The value head
            trained on outcomes alone scores MSE 0.83 on held-out positions against 0.97
            for predicting nothing — it explains 14% of the variance — while scoring 0.28
            on the replay buffer it was fitted to. The mechanism is that the board layout
            is constant within a game and is in the observation, the buffer holds only
            about 900 distinct games, and every position in a game shares one label, so the
            head can identify the board and memorise its result. Ownership and margin are
            dense, per-element, and *cannot* be answered by recognising the board, so they
            force the shared trunk to encode play. This is the largest single item in
            KataGo's own ablation table (1.65x training time to remove), and it costs
            about 1% of the parameters here. See ``docs/decisions/0026``.
    """

    def __init__(self, obs_size=encoder.SIZE, num_actions=action_space.NUM_ACTIONS,
                 width=64, road_width=32, context=128, hops=1, depth=2, rounds=0,
                 trunk=256, value_activation="linear", aux=False):
        super().__init__()
        if value_activation not in ("linear", "tanh"):
            raise ValueError(f"value_activation must be 'linear' or 'tanh', "
                             f"got {value_activation!r}")
        self.value_activation = value_activation
        self.aux = bool(aux)
        if obs_size != encoder.SIZE:
            raise ValueError(f"this network is tied to the encoder layout: expected "
                             f"{encoder.SIZE} floats, got {obs_size}")
        if num_actions != action_space.NUM_ACTIONS:
            raise ValueError(f"expected {action_space.NUM_ACTIONS} actions, got {num_actions}")
        if not 0 <= hops <= 2:
            raise ValueError(f"hops must be 0, 1 or 2, got {hops}")
        self.obs_size, self.num_actions = obs_size, num_actions
        self.width, self.road_width, self.context = width, road_width, context
        self.hops, self.depth, self.rounds, self.trunk_width = hops, depth, rounds, trunk

        tiles = range(1, NUM_TILES + 1)
        vertices = range(1, NUM_VERTICES + 1)
        roads = range(1, NUM_ROADS + 1)
        gain = 2 ** 0.5

        # --- the fixed board structure, as constant buffers --------------------------
        # one hop
        n_tv = incidence(TILE_VERTICES, tiles, NUM_VERTICES)      # tile   <- its 6 corners
        n_vt = incidence(VERTEX_TILES, vertices, NUM_TILES)       # vertex <- its 1-3 tiles
        n_vv = incidence(VERTEX_NEIGHBOURS, vertices, NUM_VERTICES)
        n_vr = incidence(VERTEX_ROADS, vertices, NUM_ROADS)       # vertex <- its 2-3 roads
        n_rv = incidence(ROAD_VERTICES, roads, NUM_VERTICES)      # road   <- its 2 ends
        for name, matrix in (("n_tv", n_tv), ("n_vt", n_vt), ("n_vv", n_vv),
                             ("n_vr", n_vr), ("n_rv", n_rv)):
            self.register_buffer(name, matrix, persistent=False)

        # two hops. Because the graph is fixed and the aggregation is linear, the second
        # hop is a *product of constants*, not a second pass over anything.
        self.register_buffer("n_tt", incidence(TILE_ADJACENCY, tiles, NUM_TILES), False)
        self.register_buffer("n_vt2", n_vv @ n_vt, False)         # tiles two vertices away
        self.register_buffer("n_vv2", n_vv @ n_vv, False)
        self.register_buffer("n_rt", incidence(ROAD_TILES, roads, NUM_TILES), False)

        # --- static degree features ---------------------------------------------------
        self.register_buffer("g_t", geometry((TILE_ADJACENCY,), tiles, (6,)), False)
        self.register_buffer(
            "g_v", geometry((VERTEX_TILES, VERTEX_NEIGHBOURS), vertices, (3, 3)), False)
        self.register_buffer("g_r", geometry((ROAD_TILES,), roads, (2,)), False)

        # --- how wide each entity's input is after the neighbourhood concat -----------
        tile_in = _T_FEAT + 1
        vertex_in = _V_FEAT + 2
        road_in = _R_FEAT + 1
        if hops >= 1:
            tile_in += _V_FEAT
            vertex_in += _T_FEAT + _V_FEAT + _R_FEAT
            road_in += _V_FEAT
        if hops >= 2:
            tile_in += _T_FEAT
            vertex_in += _T_FEAT + _V_FEAT
            road_in += _T_FEAT

        # --- the un-positional half: my hand, the bank, the phase, the score ----------
        self.context_mlp = nn.Sequential(_layer(CONTEXT_FEATURES, context, gain), nn.Tanh())
        # one matmul makes a broadcast bias for all three entity types, which is how a
        # per-position embedding gets to be conditional on what I can afford right now
        self.context_bias = _layer(context, 2 * width + road_width, gain=1.0)

        # --- shared per-entity encoders ------------------------------------------------
        self.tile_embed = _layer(tile_in, width, gain)
        self.vertex_embed = _layer(vertex_in, width, gain)
        self.road_embed = _layer(road_in, road_width, gain)

        def stack(size, layers):
            body = []
            for _ in range(layers):
                body += [nn.ReLU(), _layer(size, size, gain)]
            return nn.Sequential(*body)

        # ReLU, not Tanh: there are 6,976 entity activations per row against the flat
        # net's 1,024, and tanh over that many costs 2.8 ms at batch 512 where relu costs
        # 0.9 ms. It is the single largest elementwise cost in the network.
        self.tile_body = stack(width, depth - 1)
        self.vertex_body = stack(width, depth - 1)
        self.road_body = stack(road_width, depth - 1)

        # --- optional non-linear message passing between embeddings --------------------
        self.tile_round = nn.ModuleList(
            _layer(width + width, width, gain) for _ in range(rounds))
        self.vertex_round = nn.ModuleList(
            _layer(3 * width + road_width, width, gain) for _ in range(rounds))
        self.road_round = nn.ModuleList(
            _layer(road_width + width, road_width, gain) for _ in range(rounds))

        # --- positional heads ----------------------------------------------------------
        # tiny gain, as in the flat net: a policy that starts confident explores nothing
        self.road_logit = _layer(road_width, 1, gain=0.01)
        self.vertex_logit = _layer(width, 2, gain=0.01)          # settlement, city
        self.robber_logit = _layer(width, _VICTIMS, gain=0.01)   # per tile, one per victim

        # --- the leftovers, and the critic ---------------------------------------------
        pooled = context + 4 * width + 2 * road_width            # mean and max of each type
        self.head = nn.Sequential(_layer(pooled, trunk, gain), nn.Tanh())
        self.policy_head = _layer(trunk, NUM_GLOBAL_ACTIONS, gain=0.01)
        self.value_head = _layer(trunk, 1, gain=1.0)

        # --- auxiliary heads, off unless asked for --------------------------------------
        # Registered as None rather than omitted so `state_dict` has a stable set of keys
        # for a given config, and a checkpoint cannot half-load.
        if self.aux:
            # three classes, in the *mover's* frame: nobody, mine, theirs
            self.vertex_owner = _layer(width, 3, gain=0.01)
            self.road_owner = _layer(road_width, 3, gain=0.01)
            # final victory-point margin, in units of the target score
            self.margin_head = _layer(trunk, 1, gain=0.01)

    # ------------------------------------------------------------------ #

    def _embed(self, obs):
        """The three per-entity embeddings and the context vector.

        Split out of :meth:`forward` so the auxiliary heads can read the same features
        instead of recomputing them — they are 99% of the network's cost, and an auxiliary
        head that re-ran the trunk would make the extra targets cost more than they are
        worth. The body is exactly what ``forward`` used to do, moved rather than rewritten.
        """
        batch = obs.shape[0]
        tiles = obs[:, TILE_SPAN].reshape(batch, _T_ROWS, _T_FEAT)
        vertices = obs[:, VERTEX_SPAN].reshape(batch, _V_ROWS, _V_FEAT)
        roads = obs[:, ROAD_SPAN].reshape(batch, _R_ROWS, _R_FEAT)

        one = (batch, -1, -1)
        tile_in = [tiles, self.g_t.expand(one)]
        vertex_in = [vertices, self.g_v.expand(one)]
        road_in = [roads, self.g_r.expand(one)]
        if self.hops >= 1:
            tile_in.append(self.n_tv @ vertices)
            vertex_in += [self.n_vt @ tiles, self.n_vv @ vertices, self.n_vr @ roads]
            road_in.append(self.n_rv @ vertices)
        if self.hops >= 2:
            tile_in.append(self.n_tt @ tiles)
            vertex_in += [self.n_vt2 @ tiles, self.n_vv2 @ vertices]
            road_in.append(self.n_rt @ tiles)

        g = self.context_mlp(obs[:, CONTEXT_START:])
        bias = self.context_bias(g).unsqueeze(1)
        d, dr = self.width, self.road_width
        t = self.tile_body(self.tile_embed(torch.cat(tile_in, -1)) + bias[..., :d])
        v = self.vertex_body(self.vertex_embed(torch.cat(vertex_in, -1)) + bias[..., d:2 * d])
        r = self.road_body(self.road_embed(torch.cat(road_in, -1)) + bias[..., 2 * d:])

        for i in range(self.rounds):
            t, v, r = (
                torch.relu(self.tile_round[i](torch.cat([t, self.n_tv @ v], -1))),
                torch.relu(self.vertex_round[i](torch.cat(
                    [v, self.n_vt @ t, self.n_vv @ v, self.n_vr @ r], -1))),
                torch.relu(self.road_round[i](torch.cat([r, self.n_rv @ v], -1))),
            )
        return t, v, r, g

    def _decide(self, t, v, r, g, batch):
        """``(logits, value, hidden)`` from the embeddings. The other half of ``forward``."""
        vertex_logits = self.vertex_logit(v)
        hidden = self.head(torch.cat([
            g,
            t.mean(1), t.amax(1),
            v.mean(1), v.amax(1),
            r.mean(1), r.amax(1),
        ], dim=-1))
        other = self.policy_head(hidden)

        # assembled in index order, so this cat *is* the action space's layout
        logits = torch.cat([
            other[:, :1],                                     # END_TURN
            self.road_logit(r).squeeze(-1),                    # BUILD_ROAD        x 72
            vertex_logits[..., 0],                             # BUILD_SETTLEMENT  x 54
            vertex_logits[..., 1],                             # BUILD_CITY        x 54
            other[:, 1:21],                                    # TRADE_WITH_BANK   x 20
            self.robber_logit(t).reshape(batch, -1),           # MOVE_ROBBER       x 95
            other[:, 21:],                                     # DISCARD..MONOPOLY x 28
        ], dim=-1)
        value = self.value_head(hidden).squeeze(-1)
        if self.value_activation == "tanh":
            value = torch.tanh(value)
        return logits, value, hidden

    def forward(self, obs):
        t, v, r, g = self._embed(obs)
        logits, value, _ = self._decide(t, v, r, g, obs.shape[0])
        return logits, value

    def forward_aux(self, obs):
        """``(logits, value, vertex_owner, road_owner, margin)``.

        ``vertex_owner`` is ``(batch, 54, 3)`` and ``road_owner`` is ``(batch, 72, 3)``,
        both as raw logits over (nobody, mine, theirs) in the *mover's* frame — the same
        frame the observation and the value target use, so no seat bookkeeping is needed.
        ``margin`` is the final victory-point difference in units of the target score.

        Raises:
            RuntimeError: if this network was not built with ``aux=True``. Loud rather than
                returning zeros, because a silently absent auxiliary loss is a change that
                trains fine and measures as "the idea did not work".
        """
        if not self.aux:
            raise RuntimeError("this network was built without auxiliary heads: rebuild it "
                               "with aux=True, or do not ask for them")
        t, v, r, g = self._embed(obs)
        logits, value, hidden = self._decide(t, v, r, g, obs.shape[0])
        return (logits, value, self.vertex_owner(v), self.road_owner(r),
                self.margin_head(hidden).squeeze(-1))

    # ------------------------------------------------------------------ #
    # The masking contract, identical to PolicyValueNet's.               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_mask(logits, mask):
        return logits.masked_fill(~mask, MASK_FILL)

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False, generator=None):
        logits, value = self.forward(obs)
        logits = self._apply_mask(logits, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        elif generator is None:
            action = distribution.sample()
        else:
            probabilities = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        return action, distribution.log_prob(action), value

    def evaluate(self, obs, mask, action):
        logits, value = self.forward(obs)
        logits = self._apply_mask(logits, mask)
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action), distribution.entropy(), value

    # ------------------------------------------------------------------ #

    def config(self):
        return {
            "kind": "structured",
            "obs_size": self.obs_size,
            "num_actions": self.num_actions,
            "width": self.width,
            "road_width": self.road_width,
            "context": self.context,
            "hops": self.hops,
            "depth": self.depth,
            "rounds": self.rounds,
            "trunk": self.trunk_width,
            "value_activation": self.value_activation,
            "aux": self.aux,
            # Not a constructor argument: the *shape of the observation* this checkpoint was
            # trained against, recorded so a later encoder change can reconcile it without
            # guessing which block grew. `obs_size` alone cannot say that — 1,884 floats
            # could be many layouts. See training/alphazero/layouts.py.
            "layout": {name: list(shape) for name, shape in _layout_signature().items()},
        }

    #: Config keys that describe the checkpoint rather than construct the network.
    NOT_ARGUMENTS = frozenset({"kind", "layout"})

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k not in cls.NOT_ARGUMENTS})

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        tail = "" if self.value_activation == "linear" else f", value={self.value_activation}"
        return (f"StructuredPolicyValueNet(width={self.width}/{self.road_width}, "
                f"hops={self.hops}, depth={self.depth}, rounds={self.rounds}{tail}, "
                f"{self.num_parameters():,} params)")


def _validate():
    """The reshapes and the head layout must agree with what they read, at import.

    ``python -O`` strips these; the point is that a change to ``encoder.LAYOUT`` or to the
    order of ``action_space.ACTIONS`` cannot silently misalign a positional head — which
    would not crash, and would train to a merely-bad policy.
    """
    assert TILE_SPAN.stop - TILE_SPAN.start == _T_ROWS * _T_FEAT
    assert VERTEX_SPAN.stop - VERTEX_SPAN.start == _V_ROWS * _V_FEAT
    assert ROAD_SPAN.stop - ROAD_SPAN.start == _R_ROWS * _R_FEAT
    assert (_T_ROWS, _V_ROWS, _R_ROWS) == (NUM_TILES, NUM_VERTICES, NUM_ROADS)

    # The three positional spans and the context vector must between them cover the whole
    # observation. A block added anywhere in the gap would be read by *nothing*: the spans
    # above are looked up by name so they still resolve, and CONTEXT_START still points at
    # the players block, so the new floats are simply never fed to the network — no
    # exception, no misalignment, just a feature that silently does not exist.
    assert ROAD_SPAN.stop == CONTEXT_START, (
        "an observation block fell between the positional blocks and the context vector, "
        "where the network reads neither"
    )

    # the positional blocks must be contiguous and in the order the final cat assumes
    assert _ROAD_SLICE.start == 1 and _ROAD_SLICE.stop == _SETTLEMENT_SLICE.start
    assert _SETTLEMENT_SLICE.stop == _CITY_SLICE.start
    assert _ROBBER_SLICE.stop - _ROBBER_SLICE.start == NUM_TILES * _VICTIMS

    # The final cat interleaves positional blocks with slices of `other`, so what must hold
    # is that the non-positional indices are exactly the ones `other` is sliced into, in
    # order. Checked by construction rather than against written-down offsets: the first
    # version of this assertion hardcoded "28 actions after MOVE_ROBBER" and broke the
    # moment one was appended, which is a fact about the assertion, not about the network.
    positional = set(range(_ROAD_SLICE.start, _CITY_SLICE.stop))
    positional |= set(range(_ROBBER_SLICE.start, _ROBBER_SLICE.stop))
    non_positional = [i for i in range(action_space.NUM_ACTIONS) if i not in positional]

    assert len(non_positional) == NUM_GLOBAL_ACTIONS
    assert non_positional[0] == 0                                   # END_TURN
    assert non_positional[1:21] == list(range(_CITY_SLICE.stop, _ROBBER_SLICE.start))
    assert non_positional[21:] == list(
        range(_ROBBER_SLICE.stop, action_space.NUM_ACTIONS)
    )

    # BUILD_ROAD/SETTLEMENT/CITY must run 1..n in id order, or a head lands on the wrong place
    for i, position in enumerate(range(1, NUM_ROADS + 1)):
        assert action_space.decode(_ROAD_SLICE.start + i).position == position
    for i, position in enumerate(range(1, NUM_VERTICES + 1)):
        assert action_space.decode(_SETTLEMENT_SLICE.start + i).position == position
        assert action_space.decode(_CITY_SLICE.start + i).position == position
    # MOVE_ROBBER must be tile-major for the (tiles, victims) reshape to land correctly
    for tile in range(1, NUM_TILES + 1):
        for victim in range(_VICTIMS):
            action = action_space.decode(_ROBBER_SLICE.start + (tile - 1) * _VICTIMS + victim)
            assert action.position == tile and action.extra == victim


_validate()
