from __future__ import annotations

import datetime

from .base import Alerter


class ConsoleAlerter(Alerter):
    def notify(self, message: str) -> None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"[{ts}] {message}")
