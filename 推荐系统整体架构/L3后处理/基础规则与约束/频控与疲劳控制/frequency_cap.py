from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class InMemoryExposureStore:
    timestamps_by_key: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def reserve(
        self,
        user_id: str,
        scope: str,
        now: int,
        window_seconds: int,
        max_exposures: int,
    ) -> tuple[bool, str]:
        key = (user_id, scope)
        window_start = now - window_seconds
        timestamps = [
            timestamp
            for timestamp in self.timestamps_by_key[key]
            if timestamp > window_start
        ]
        self.timestamps_by_key[key] = timestamps
        if len(timestamps) >= max_exposures:
            return False, "frequency_cap"
        timestamps.append(now)
        return True, "reserved"