"""
Active-learning annotation controller.

Main loop with double-wave buffering and feedback-driven threshold adjustments.
"""

import os
import sys
import time
import random
import logging
import cv2
import numpy as np
from pathlib import Path

# Ensure pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.state import ControllerState
from backend import ls_client
from backend.threshold_updater import update_thresholds_from_annotations
from pipeline.config import PipelineConfig, DayNightConfig
from pipeline.feature_extractor import load_or_extract_features
from pipeline.classifier import classify_active
from pipeline.clip_generator import generate_clips


# ---------------------------------------------------------------------------
# Logging Setup (Console + File)
# ---------------------------------------------------------------------------
STATE_DIR = os.environ.get("STATE_DIR", "/app/state")
log_path = Path(STATE_DIR) / "controller.log"

logger = logging.getLogger("controller")
logger.setLevel(logging.INFO)

# Console handler
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

# File handler
try:
    f_handler = logging.FileHandler(str(log_path), mode='a')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.info(f"Logging initialized. Log file at: {log_path}")
except Exception as e:
    logger.error(f"Failed to initialize file logger: {e}")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")
POLL_INTERVAL_SEC = int(os.environ.get("POLL_INTERVAL_SEC", "30"))
CLIPS_PER_WAVE = int(os.environ.get("CLIPS_PER_WAVE", "20"))
SLEEP_CLIPS_PER_WAVE = int(os.environ.get("SLEEP_CLIPS_PER_WAVE", "3"))
DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CLIPS_DIR = os.environ.get("CLIPS_DIR", "/app/clips")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "")


def scan_available_videos(data_dir: str) -> list[str]:
    data_path = Path(data_dir)
    prefixes = set()
    for mp4 in data_path.glob("*.compressed.mp4"):
        if mp4.name.startswith('.'):
            continue
        prefix = mp4.name.replace(".compressed.mp4", "")
        cache = data_path / f"{prefix}.features.parquet"
        parquet = data_path / f"{prefix}.detection_v1.parquet"
        if cache.exists() or parquet.exists():
            prefixes.add(prefix)
    return sorted(prefixes)


def intervals_overlap(start1, end1, start2, end2):
    return not (end1 <= start2 or end2 <= start1)


def generate_wave(
    video_prefixes: list[str],
    state: ControllerState,
    dn_config: DayNightConfig,
    data_dir: str,
    clips_dir: str,
    awake_clips_target: int,
    sleep_clips_target: int,
) -> list[dict]:
    """Generate a wave of non-overlapping clips from the available video prefixes."""
    all_clips = []

    # Copy list and shuffle to randomly select different videos first
    available = video_prefixes[:]
    random.shuffle(available)

    generated_awake = 0
    generated_sleep = 0

    for prefix in available:
        if generated_awake >= awake_clips_target and generated_sleep >= sleep_clips_target:
            break

        cfg = dn_config.get_config_for_video(prefix)

        try:
            feat_df, video_fps = load_or_extract_features(prefix, data_dir, cfg)
        except Exception as e:
            logger.error(f"Failed to generate features for video {prefix}: {e}")
            continue

        if len(feat_df) == 0:
            continue

        feat_df, _thresholds_used = classify_active(feat_df, video_fps, cfg)
        mp4_path = Path(data_dir) / f"{prefix}.compressed.mp4"

        # Determine clips to generate
        video_intervals = state.generated_intervals.get(prefix, [])

        # Filter candidate segments in classification df
        for is_active, label, target, current_count in [
            (True, "awake", awake_clips_target, generated_awake),
            (False, "sleeping", sleep_clips_target, generated_sleep)
        ]:
            if current_count >= target:
                continue

            # Identify candidates
            mask = feat_df['is_active'].values == is_active
            diff_mask = np.diff(mask.astype(int))
            starts = np.where(diff_mask == 1)[0] + 1
            ends = np.where(diff_mask == -1)[0] + 1

            if mask[0]:
                starts = np.insert(starts, 0, 0)
            if mask[-1]:
                ends = np.append(ends, len(mask) - 1)

            clip_frames_req = int(cfg.clip_duration_sec * video_fps)
            candidates = []

            for s_idx, e_idx in zip(starts, ends):
                start_frame = int(feat_df.iloc[s_idx]['frame_index'])
                end_frame = int(feat_df.iloc[min(e_idx, len(feat_df) - 1)]['frame_index'])

                curr_f = start_frame
                while curr_f + clip_frames_req <= end_frame:
                    c_end_f = curr_f + clip_frames_req
                    
                    # Verify no overlap with past clips on this video
                    overlap = False
                    for p_start, p_end in video_intervals:
                        if intervals_overlap(curr_f, c_end_f, p_start, p_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        candidates.append((curr_f, c_end_f))
                    curr_f += clip_frames_req

            # Randomly select at most 3 clips from this video
            random.shuffle(candidates)
            selected_candidates = candidates[:10]

            # Generate selected
            if selected_candidates:
                cap = cv2.VideoCapture(str(mp4_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                for sf, ef in selected_candidates:
                    if label == "awake" and generated_awake >= awake_clips_target:
                        break
                    if label == "sleeping" and generated_sleep >= sleep_clips_target:
                        break

                    filename = f"{prefix}_{label}_{sf:06d}_{ef:06d}.mp4"
                    out_path = Path(clips_dir) / filename
                    temp_path = out_path.with_suffix('.temp.mp4')

                    writer = cv2.VideoWriter(str(temp_path), fourcc, video_fps, (width, height))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

                    for _ in range(sf, ef):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        writer.write(frame)
                    writer.release()

                    # Convert to browser-compatible H.264 using ffmpeg
                    try:
                        import subprocess
                        cmd = [
                            'ffmpeg', '-y', '-i', str(temp_path),
                            '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p',
                            '-loglevel', 'error',
                            str(out_path)
                        ]
                        subprocess.run(cmd, check=True)
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception as fe:
                        logger.error(f"FFmpeg conversion failed for {filename}: {fe}")
                        if temp_path.exists():
                            temp_path.rename(out_path)

                    # Save to state intervals
                    video_intervals.append((sf, ef))
                    state.generated_intervals[prefix] = video_intervals

                    all_clips.append({
                        'filename': filename,
                        'start_frame': sf,
                        'end_frame': ef,
                        'start_sec': round(sf / video_fps, 2),
                        'end_sec': round(ef / video_fps, 2),
                        'predicted_label': label,
                        'video_prefix': prefix,
                    })

                    if label == "awake":
                        generated_awake += 1
                    else:
                        generated_sleep += 1

                cap.release()

    logger.info(f"Wave generated: {len(all_clips)} total clips ({generated_awake} awake, {generated_sleep} sleeping)")
    return all_clips


def main():
    logger.info("=" * 60)
    logger.info(" Active-Learning Annotation Controller Starting")
    logger.info("=" * 60)

    state = ControllerState.load(STATE_DIR)

    dn_config = DayNightConfig()
    for k, v in state.current_day_thresholds.items():
        setattr(dn_config.day_config, k, v)
    for k, v in state.current_night_thresholds.items():
        setattr(dn_config.night_config, k, v)

    all_prefixes = scan_available_videos(DATA_DIR)
    if not all_prefixes:
        logger.error("No source videos found in DATA_DIR. Exiting.")
        return

    client = None
    project_id = None
    for attempt in range(10):
        try:
            client = ls_client.get_client(LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY)
            project_id = ls_client.ensure_project(client)
            break
        except Exception as e:
            logger.warning(f"Waiting for Label Studio... (attempt {attempt + 1}/10): {e}")
            time.sleep(10)

    if client is None or project_id is None:
        logger.error("Could not connect to Label Studio or fetch project. Exiting. Check your LABEL_STUDIO_API_KEY in .env")
        return

    awake_count = CLIPS_PER_WAVE - SLEEP_CLIPS_PER_WAVE
    sleep_count = SLEEP_CLIPS_PER_WAVE

    # Validate task IDs against current project to avoid stale state from old/recreated projects
    stale = False
    for tid_list in [state.pending_wave_task_ids, state.buffer_wave_task_ids]:
        if tid_list:
            try:
                client.tasks.get(id=tid_list[0])
            except Exception:
                stale = True
                break
    if stale:
        logger.warning(
            "Detected stale tasks in controller_state (tasks do not exist in current Label Studio project). "
            "Resetting wave task queues."
        )
        state.pending_wave_task_ids = []
        state.buffer_wave_task_ids = []
        state.wave_number = 0
        state.save(STATE_DIR)

    # Setup initial waves
    if not state.pending_wave_task_ids:
        logger.info(f"Generating Wave {state.wave_number} (active)...")
        clips = generate_wave(all_prefixes, state, dn_config, DATA_DIR, CLIPS_DIR, awake_count, sleep_count)
        if clips:
            state.pending_wave_task_ids = ls_client.push_clips(client, project_id, clips, state.wave_number)
            state.save(STATE_DIR)

    if not state.buffer_wave_task_ids:
        logger.info(f"Generating Wave {state.wave_number + 1} (buffer)...")
        clips = generate_wave(all_prefixes, state, dn_config, DATA_DIR, CLIPS_DIR, awake_count, sleep_count)
        if clips:
            state.buffer_wave_task_ids = ls_client.push_clips(client, project_id, clips, state.wave_number + 1)
            state.save(STATE_DIR)

    logger.info("Entering main poll loop...")
    while True:
        try:
            if state.pending_wave_task_ids and ls_client.check_wave_complete(client, project_id, state.pending_wave_task_ids):
                logger.info(f"Wave {state.wave_number} completed.")

                # 1. Retrieve annotations
                annotations = ls_client.get_completed_annotations(client, project_id, state.pending_wave_task_ids)
                
                # 2. Update thresholds via P/R feedback loop
                if annotations:
                    day_thresh, night_thresh = update_thresholds_from_annotations(
                        annotations,
                        state.current_day_thresholds,
                        state.current_night_thresholds,
                        dn_config.day_start_hour,
                        dn_config.night_start_hour
                    )
                    state.record_threshold_update(state.wave_number, day_thresh, night_thresh)
                    
                    # Update config
                    for k, v in day_thresh.items():
                        setattr(dn_config.day_config, k, v)
                    for k, v in night_thresh.items():
                        setattr(dn_config.night_config, k, v)

                # 3. Promote buffer -> active
                state.pending_wave_task_ids = state.buffer_wave_task_ids
                state.buffer_wave_task_ids = []
                state.wave_number += 1

                # 4. Generate next wave
                logger.info(f"Generating Wave {state.wave_number + 1} (buffer)...")
                clips = generate_wave(all_prefixes, state, dn_config, DATA_DIR, CLIPS_DIR, awake_count, sleep_count)
                if clips:
                    state.buffer_wave_task_ids = ls_client.push_clips(client, project_id, clips, state.wave_number + 1)

                state.save(STATE_DIR)
                logger.info(f"State saved. Waiting on wave {state.wave_number}.")

        except Exception as e:
            logger.error(f"Error in poll loop: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
