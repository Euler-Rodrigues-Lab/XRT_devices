"""Common snapshot envelope; actions retain the established SEW dictionary schema."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceFrame:
    source: str
    sequence: int
    received_at: float  # local monotonic seconds, not headset wall time
    action: dict[str, Any]
    source_timestamp: float | None = None
