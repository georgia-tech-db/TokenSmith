"""Shared base for all KG pipeline components.

Every component (Divider, Extractor, Linker, Persister) carries a
``metadata`` dict populated during execution and a ``get_config()``
method for run introspection.  Concrete base classes (``BaseDivider``
etc.) add their own abstract interface on top of this.
"""

from abc import ABC
from typing import Any


class BasePipelineComponent(ABC):
    """Mixin providing ``metadata`` storage and ``get_config()`` for pipeline components."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Return a dict describing this component's configuration.

        Subclasses should call ``super().get_config()`` and update the result
        with their own parameters.
        """
        return {"class": self.__class__.__name__}
