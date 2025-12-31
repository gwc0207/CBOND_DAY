from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Factor(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FactorRegistry:
    _items: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def _decorator(factor_cls: type):
            if name in cls._items:
                raise ValueError(f"duplicate factor: {name}")
            cls._items[name] = factor_cls
            return factor_cls

        return _decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._items:
            raise KeyError(f"factor not registered: {name}")
        return cls._items[name]
