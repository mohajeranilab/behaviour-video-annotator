"""Threshold-based classification with morphological filtering."""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_opening
from .config import PipelineConfig


def classify_active(
    feat_df: pd.DataFrame,
    video_fps: float,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, dict]:
    """Apply smoothing, thresholding, and morphological filtering.

    Returns:
        (feat_df, thresholds) — DataFrame with 'is_active' column, and dict of thresholds used.
    """
    # Cap outliers
    disp_cap = feat_df['disp'].quantile(cfg.disp_cap_quantile)
    feat_df['disp_capped'] = feat_df['disp'].clip(upper=disp_cap)
    area_cap = feat_df['area_change'].quantile(cfg.area_cap_quantile)
    feat_df['area_change_capped'] = feat_df['area_change'].clip(upper=area_cap)

    detection_fps = video_fps / 2.0
    win_steps = int(cfg.smooth_sec * detection_fps) | 1

    feat_df['disp_smooth'] = feat_df['disp_capped'].rolling(window=win_steps, center=True, min_periods=1).mean()
    feat_df['roi_diff_smooth'] = feat_df['roi_diff'].rolling(window=win_steps, center=True, min_periods=1).mean()
    feat_df['area_change_smooth'] = feat_df['area_change_capped'].rolling(window=win_steps, center=True, min_periods=1).mean()

    # Compute thresholds
    disp_thresh = cfg.disp_thresh

    if cfg.roi_thresh is not None:
        roi_thresh = cfg.roi_thresh
    else:
        cutoff = feat_df['disp_smooth'].quantile(cfg.baseline_quantile)
        baseline = feat_df.loc[feat_df['disp_smooth'] < cutoff, 'roi_diff_smooth']
        roi_thresh = max(float(baseline.mean() + cfg.baseline_n_sigma * baseline.std()), cfg.roi_thresh_min)

    if cfg.area_thresh is not None:
        area_thresh = cfg.area_thresh
    else:
        cutoff = feat_df['disp_smooth'].quantile(cfg.baseline_quantile)
        baseline = feat_df.loc[feat_df['disp_smooth'] < cutoff, 'area_change_smooth']
        area_thresh = max(float(baseline.mean() + cfg.baseline_n_sigma * baseline.std()), cfg.area_thresh_min)

    # Raw classification
    is_moving = feat_df['disp_smooth'] > disp_thresh
    is_in_place = feat_df['roi_diff_smooth'] > roi_thresh
    is_shape_changing = feat_df['area_change_smooth'] > area_thresh
    raw_active = (is_moving | is_in_place | is_shape_changing).values

    # Morphological filtering with edge-replication padding
    noise_close_steps = int(cfg.noise_close_sec * detection_fps) | 1
    noise_open_steps = int(cfg.noise_open_sec * detection_fps) | 1
    context_close_steps = int(cfg.context_close_sec * detection_fps) | 1
    context_open_steps = int(cfg.context_open_sec * detection_fps) | 1

    pad_size = max(noise_close_steps, noise_open_steps, context_close_steps, context_open_steps)
    padded = np.pad(raw_active, pad_width=pad_size, mode='edge')

    p1 = binary_closing(padded, structure=np.ones(noise_close_steps))
    p1 = binary_opening(p1, structure=np.ones(noise_open_steps))
    p2 = binary_closing(p1, structure=np.ones(context_close_steps))
    final = binary_opening(p2, structure=np.ones(context_open_steps))

    feat_df['is_active'] = final[pad_size:-pad_size]
    feat_df['is_moving'] = is_moving
    feat_df['is_in_place'] = is_in_place
    feat_df['is_shape_changing'] = is_shape_changing

    thresholds = {
        'disp_thresh': disp_thresh,
        'roi_thresh': roi_thresh,
        'area_thresh': area_thresh,
    }

    return feat_df, thresholds
