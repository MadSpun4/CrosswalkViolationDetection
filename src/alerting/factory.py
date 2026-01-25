from __future__ import annotations

from ..config import Settings
from .console import ConsoleAlerter
from .beep import BeepAlerter
from .base import Alerter


def build_alerter(settings: Settings) -> Alerter:
    mode = (settings.alert_mode or "console").strip().lower()
    if mode == "beep":
        return BeepAlerter()
    return ConsoleAlerter()
