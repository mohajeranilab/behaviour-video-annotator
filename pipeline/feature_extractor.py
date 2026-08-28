"""Feature extraction — loads cached .features.parquet or extracts from video+parquet."""

import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from .config import PipelineConfig


def load_or_extract_features(
    video_prefix: str,
    data_dir: str | Path,
    cfg: PipelineConfig,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, float]:
    """Load cached features or extract them from video + detection parquet.

    Returns:
        (feat_df, video_fps) — raw feature DataFrame and the video FPS.
    """
    data_path = Path(data_dir)
    parquet_path = data_path / f"{video_prefix}.detection_v1.parquet"
    mp4_path = data_path / f"{video_prefix}.compressed.mp4"
    cache_path = data_path / f"{video_prefix}.features.parquet"

    cap = cv2.VideoCapture(str(mp4_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    scale_x = width / cfg.detection_w
    scale_y = height / cfg.detection_h

    if cache_path.exists() and not force_recompute:
        feat_df = pd.read_parquet(cache_path)
        cap.release()
        return feat_df, video_fps

    # Full extraction
    df = pd.read_parquet(parquet_path)
    df_single = df.sort_values('confidence', ascending=False).groupby('frame_index').first().reset_index()
    df_single = df_single.sort_values('frame_index').reset_index(drop=True)

    max_frame = int(df_single['frame_index'].max())
    full_frame_indices = np.arange(0, max_frame + 1, 2)
    full_df = pd.DataFrame({'frame_index': full_frame_indices})
    df_full = pd.merge(full_df, df_single, on='frame_index', how='left')

    for col in ['bbox_cx', 'bbox_cy', 'bbox_width', 'bbox_height']:
        df_full[col] = df_full[col].interpolate(method='linear', limit=cfg.interp_limit).bfill().ffill()
    df_full['confidence'] = df_full['confidence'].fillna(0.0)

    raw_cxs = df_full['bbox_cx'].values * scale_x
    raw_cys = df_full['bbox_cy'].values * scale_y
    raw_ws = df_full['bbox_width'].values * scale_x
    raw_hs = df_full['bbox_height'].values * scale_y
    med_cxs = pd.Series(raw_cxs).rolling(window=cfg.median_win, center=True, min_periods=1).median().values
    med_cys = pd.Series(raw_cys).rolling(window=cfg.median_win, center=True, min_periods=1).median().values
    med_ws = pd.Series(raw_ws).rolling(window=cfg.median_win, center=True, min_periods=1).median().values
    med_hs = pd.Series(raw_hs).rolling(window=cfg.median_win, center=True, min_periods=1).median().values

    records = []
    prev_crop = None
    prev_cx, prev_cy = None, None
    blur_k = (cfg.blur_ksize, cfg.blur_ksize)

    t0 = time.time()
    for i, (idx, row) in enumerate(df_full.iterrows()):
        f_idx = int(row['frame_index'])
        if f_idx >= total_video_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        cx, cy = med_cxs[i], med_cys[i]
        w, h = med_ws[i], med_hs[i]
        area = w * h

        pad_w = max(cfg.crop_pad_min_px, w * cfg.crop_pad_ratio)
        pad_h = max(cfg.crop_pad_min_px, h * cfg.crop_pad_ratio)
        x1, y1 = max(0, int(cx - w / 2 - pad_w)), max(0, int(cy - h / 2 - pad_h))
        x2, y2 = min(frame.shape[1], int(cx + w / 2 + pad_w)), min(frame.shape[0], int(cy + h / 2 + pad_h))

        crop = frame[y1:y2, x1:x2]
        crop_gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), blur_k, 0) if crop.size > 0 else None

        roi_diff_mean, disp, area_change_rate = 0.0, 0.0, 0.0

        if prev_crop is not None and crop_gray is not None:
            prev_resized = cv2.resize(prev_crop, (crop_gray.shape[1], crop_gray.shape[0])) if prev_crop.shape != crop_gray.shape else prev_crop
            roi_diff_mean = float(np.mean(cv2.absdiff(crop_gray, prev_resized)))

        if prev_cx is not None:
            disp = float(np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2))

        if i > 0:
            prev_area = med_ws[i - 1] * med_hs[i - 1]
            if prev_area > 0:
                area_change_rate = abs(area - prev_area) / prev_area

        records.append({
            'frame_index': f_idx,
            'time_sec': f_idx / video_fps,
            'disp': disp,
            'roi_diff': roi_diff_mean,
            'area_change': area_change_rate,
            'confidence': row['confidence'],
            'bbox_cx': cx, 'bbox_cy': cy,
            'bbox_w': w, 'bbox_h': h,
        })
        prev_crop, prev_cx, prev_cy = crop_gray, cx, cy

    cap.release()

    feat_df = pd.DataFrame(records)
    elapsed = time.time() - t0
    print(f"  Extracted features for {video_prefix} in {elapsed:.1f}s ({len(feat_df)} steps)")

    # Save cache
    try:
        raw_cols = ['frame_index', 'time_sec', 'disp', 'roi_diff', 'area_change',
                    'confidence', 'bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h']
        feat_df[raw_cols].to_parquet(cache_path)
    except Exception as e:
        print(f"  Warning: Could not cache features: {e}")

    return feat_df, video_fps
