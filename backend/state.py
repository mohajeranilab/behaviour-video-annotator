"""Persistent state management for the annotation controller."""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict


STATE_FILE = "controller_state.json"


@dataclass
class ControllerState:
    """Tracks the controller's progress across restarts."""

    wave_number: int = 0
    processed_prefixes: list = field(default_factory=list)
    pending_wave_task_ids: list = field(default_factory=list)
    buffer_wave_task_ids: list = field(default_factory=list)
    threshold_history: list = field(default_factory=list)
    
    # Track generated intervals per video prefix to avoid overlaps: {video_prefix: [[start_f, end_f], ...]}
    generated_intervals: dict = field(default_factory=dict)

    # Current thresholds (will be updated by the threshold updater)
    current_day_thresholds: dict = field(default_factory=lambda: {
        'disp_thresh': 8.7546,
        'roi_thresh': 4.9532,
        'area_thresh': 0.0162,
    })
    current_night_thresholds: dict = field(default_factory=lambda: {
        'disp_thresh': 0.3784,
        'roi_thresh': 0.8821,
        'area_thresh': 0.0057,
    })

    def save(self, state_dir: str | Path):
        """Persist state to JSON."""
        path = Path(state_dir) / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, state_dir: str | Path) -> "ControllerState":
        """Load state from JSON, or return defaults if not found."""
        path = Path(state_dir) / STATE_FILE
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def record_threshold_update(self, wave: int, day_thresh: dict, night_thresh: dict):
        """Record a threshold change in history."""
        self.threshold_history.append({
            'wave': wave,
            'day': day_thresh.copy(),
            'night': night_thresh.copy(),
        })
        self.current_day_thresholds = day_thresh.copy()
        self.current_night_thresholds = night_thresh.copy()
