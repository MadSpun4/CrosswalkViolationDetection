from __future__ import annotations

from abc import ABC, abstractmethod


class Alerter(ABC):
    @abstractmethod
    def notify(self, message: str) -> None:
        raise NotImplementedError
