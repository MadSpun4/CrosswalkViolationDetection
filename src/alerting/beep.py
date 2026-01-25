from __future__ import annotations

import sys

from .base import Alerter


class BeepAlerter(Alerter):
    """Simple terminal beep. In Docker this may be inaudible; kept as a stub."""

    def notify(self, message: str) -> None:
        sys.stdout.write("\a")  # terminal bell
        sys.stdout.flush()
        print(message)
