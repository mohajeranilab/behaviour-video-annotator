import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PipelineConfig:
    """All tunable parameters for the awake/sleep detection pipeline."""

    # --- Detection model resolution ---
    detection_w: int = 512
    detection_h: int = 384

    # --- Interpolation ---
    interp_limit: int = 10

    # --- Median filter (glitch removal) ---
    median_win: int = 7

    # --- ROI crop padding ---
    crop_pad_ratio: float = 0.15
    crop_pad_min_px: float = 10.0

    # --- Gaussian blur on ROI crop ---
    blur_ksize: int = 5

    # --- Outlier capping ---
    disp_cap_quantile: float = 0.99
    area_cap_quantile: float = 0.99

    # --- Smoothing ---
    smooth_sec: float = 2.0

    # --- Adaptive threshold estimation ---
    baseline_quantile: float = 0.10
    baseline_n_sigma: float = 3.0

    # --- Activity thresholds ---
    disp_thresh: float = 2.0
    roi_thresh: float = None
    area_thresh: float = None

    # --- Minimum threshold floors ---
    roi_thresh_min: float = None
    area_thresh_min: float = None

    # --- Morphological filtering ---
    noise_close_sec: float = 3.0
    noise_open_sec: float = 3.0
    context_close_sec: float = 45.0
    context_open_sec: float = 45.0

    # --- Clip extraction ---
    clip_duration_sec: float = 3.0
    max_clips: int = 4

    def to_json(self, path: str):
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        """Load configuration from a JSON file, allowing partial overrides."""
        with open(path, 'r') as f:
            overrides = json.load(f)

        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}

        filtered_overrides = {}
        for k, v in overrides.items():
            if k in valid_keys:
                filtered_overrides[k] = v
            else:
                print(f"Warning: Ignoring unexpected config field '{k}' in {path}")

        return cls(**filtered_overrides)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Load configuration from a dictionary, ignoring unknown keys."""
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def summary(self):
        """Print a formatted summary of all parameters."""
        print("=" * 50)
        print(" Pipeline Configuration")
        print("=" * 50)
        for k, v in asdict(self).items():
            print(f"  {k:30s} = {v}")
        print("=" * 50)


@dataclass
class DayNightConfig:
    """Wraps separate day and night PipelineConfigs with a time-of-day switch.

    Day:   day_start_hour (default 08:00) to night_start_hour (default 20:00)
    Night: night_start_hour (default 20:00) to day_start_hour (default 08:00)
    """

    day_start_hour: int = 8
    night_start_hour: int = 20

    day_config: PipelineConfig = field(default_factory=lambda: PipelineConfig(
        disp_thresh=8.7546,
        roi_thresh=2.515,
        area_thresh=0.0162,
        roi_thresh_min=2.515,
        area_thresh_min=0.0047,
    ))

    night_config: PipelineConfig = field(default_factory=lambda: PipelineConfig(
        disp_thresh=0.7576,
        roi_thresh=1.5247,
        area_thresh=0.0165,
        roi_thresh_min=0.8821,
        area_thresh_min=0.0057,
    ))

    def get_config_for_hour(self, hour: int) -> PipelineConfig:
        """Return the appropriate PipelineConfig for the given hour (0-23)."""
        if self.day_start_hour <= hour < self.night_start_hour:
            return self.day_config
        return self.night_config

    def get_config_for_video(self, video_prefix: str) -> PipelineConfig:
        """Parse the hour from a video prefix like '20260410_130000_xxx' and return config."""
        parts = video_prefix.split('_')
        if len(parts) >= 2:
            try:
                hour = int(parts[1][:2])
                return self.get_config_for_hour(hour)
            except (ValueError, IndexError):
                pass
        print(f"Warning: Could not parse hour from '{video_prefix}', using day config.")
        return self.day_config

    def to_json(self, path: str):
        """Save to JSON."""
        data = {
            'day_start_hour': self.day_start_hour,
            'night_start_hour': self.night_start_hour,
            'day_config': asdict(self.day_config),
            'night_config': asdict(self.night_config),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DayNightConfig":
        """Load from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        day_cfg = PipelineConfig.from_dict(data.get('day_config', {}))
        night_cfg = PipelineConfig.from_dict(data.get('night_config', {}))

        return cls(
            day_start_hour=data.get('day_start_hour', 8),
            night_start_hour=data.get('night_start_hour', 20),
            day_config=day_cfg,
            night_config=night_cfg,
        )
