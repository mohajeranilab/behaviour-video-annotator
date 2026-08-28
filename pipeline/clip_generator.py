"""Clip generator — extracts short MP4 clips from source videos."""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from .config import PipelineConfig


def generate_clips(
    video_path: str | Path,
    feat_df: pd.DataFrame,
    video_fps: float,
    cfg: PipelineConfig,
    output_dir: str | Path,
    video_prefix: str,
    predicted_label: str = "awake",
    max_clips: int = None,
) -> list[dict]:
    """Generate short MP4 clips from segments matching the predicted label.

    Args:
        video_path: Path to the source .compressed.mp4.
        feat_df: DataFrame with 'is_active' column (from classifier).
        video_fps: Video FPS.
        cfg: PipelineConfig for clip duration.
        output_dir: Where to write the MP4 clips.
        video_prefix: e.g. '20260410_130000_dca632e87112'.
        predicted_label: 'awake' extracts active segments, 'sleeping' extracts inactive segments.
        max_clips: Maximum number of clips to extract (None = all).

    Returns:
        List of clip metadata dicts: {filename, start_frame, end_frame, start_sec, end_sec,
                                       predicted_label, video_prefix}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select segments based on predicted_label
    if predicted_label == "awake":
        mask = feat_df['is_active'].values.astype(bool)
    else:
        mask = ~feat_df['is_active'].values.astype(bool)

    # Find contiguous segment boundaries
    diff_mask = np.diff(mask.astype(int))
    starts = np.where(diff_mask == 1)[0] + 1
    ends = np.where(diff_mask == -1)[0] + 1

    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask) - 1)

    clip_frames_req = int(cfg.clip_duration_sec * video_fps)
    candidate_clips = []

    for s_idx, e_idx in zip(starts, ends):
        start_frame = int(feat_df.iloc[s_idx]['frame_index'])
        end_frame = int(feat_df.iloc[min(e_idx, len(feat_df) - 1)]['frame_index'])

        curr_f = start_frame
        while curr_f + clip_frames_req <= end_frame:
            c_end_f = curr_f + clip_frames_req
            candidate_clips.append({
                'start_frame': curr_f,
                'end_frame': c_end_f,
                'start_sec': round(curr_f / video_fps, 2),
                'end_sec': round(c_end_f / video_fps, 2),
            })
            curr_f += clip_frames_req

    # Sample if max_clips specified
    if max_clips is not None and len(candidate_clips) > max_clips:
        np.random.shuffle(candidate_clips)
        candidate_clips = candidate_clips[:max_clips]

    if not candidate_clips:
        return []

    # Write clips
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    results = []
    for clip in candidate_clips:
        sf, ef = clip['start_frame'], clip['end_frame']
        filename = f"{video_prefix}_{predicted_label}_{sf:06d}_{ef:06d}.mp4"
        out_path = output_dir / filename

        writer = cv2.VideoWriter(str(out_path), fourcc, video_fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

        for _ in range(sf, ef):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()

        results.append({
            'filename': filename,
            'start_frame': sf,
            'end_frame': ef,
            'start_sec': clip['start_sec'],
            'end_sec': clip['end_sec'],
            'predicted_label': predicted_label,
            'video_prefix': video_prefix,
        })

    cap.release()
    return results
