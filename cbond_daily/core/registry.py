from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Type


@dataclass
class Registry:
    items: Dict[str, Type] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[Type], Type]:
        def _decorator(cls: Type) -> Type:
            if name in self.items:
                raise ValueError(f"duplicate: {name}")
            self.items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> Type:
        if name not in self.items:
            raise KeyError(f"not found: {name}")
        return self.items[name]
