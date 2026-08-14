"""CatanIA engine.

Layers, innermost first:

* :mod:`catan.topology` — immutable geometry, generated from the row structure.
* :mod:`catan.resources` — the five resources and what things cost.
* :mod:`catan.board` — one board layout: numbers, resources, production index.
  **Immutable after construction**, so clones share it.
* :mod:`catan.state` — :class:`~catan.state.GameState`: everything mutable, and
  nothing else. Cheap :meth:`~catan.state.GameState.clone`.
* :mod:`catan.actions` — what a player can do, as data.
* :mod:`catan.rules` — the only legality authority: ``legal_actions`` and ``apply``.

Nothing in this package performs I/O or touches the global ``random`` module.
"""

#: The engine's version — the rules, the geometry and the state model together.
#:
#: It is what :mod:`catan.contract` publishes as ``engine_version``, and therefore what every
#: recorded win rate, rating and match result names as the world it was measured in. Bump it
#: when the game changes; ``catan.contract.rules_digest`` and ``tests/test_contract.py`` are
#: what make forgetting to fail a test rather than quietly invalidate a published number.
__version__ = "0.1.0"

from catan.actions import Action, ActionType
from catan.board import Board, Production
from catan.resources import (
    CITY_COST,
    DEV_CARD_COST,
    NUM_RESOURCES,
    ROAD_COST,
    SETTLEMENT_COST,
    Resource,
)
from catan.state import GameState, Phase, Piece

__all__ = [
    "Action",
    "ActionType",
    "Board",
    "Production",
    "Resource",
    "NUM_RESOURCES",
    "ROAD_COST",
    "SETTLEMENT_COST",
    "CITY_COST",
    "DEV_CARD_COST",
    "GameState",
    "Phase",
    "Piece",
]
