"""Threshold updater — adjusts thresholds based on precision/recall feedback loop."""

import logging
import re
from pathlib import Path

logger = logging.getLogger("controller")

def update_thresholds_from_annotations(
    annotations: list[dict],
    current_day_thresholds: dict,
    current_night_thresholds: dict,
    day_start_hour: int,
    night_start_hour: int,
) -> tuple[dict, dict]:
    """Slightly modify thresholds based on precision/recall of user annotations."""
    
    day_annotations = []
    night_annotations = []

    for ann in annotations:
        prefix = ann.get('video_prefix', '')
        if not prefix and 'filename' in ann:
            fn = ann['filename'].split('/')[-1]
            parts = fn.split('_')
            if len(parts) >= 3:
                prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"

        parts = prefix.split('_') if prefix else []
        is_day = True
        if len(parts) >= 2:
            try:
                hour = int(parts[1][:2])
                is_day = day_start_hour <= hour < night_start_hour
            except (ValueError, IndexError):
                pass
        
        if is_day:
            day_annotations.append(ann)
        else:
            night_annotations.append(ann)

    new_day = adjust_thresholds_for_group(day_annotations, current_day_thresholds, "day")
    new_night = adjust_thresholds_for_group(night_annotations, current_night_thresholds, "night")

    return new_day, new_night

def adjust_thresholds_for_group(annotations: list[dict], current_thresholds: dict, group_name: str) -> dict:
    """Adjust thresholds based on precision & recall metrics."""
    new_thresholds = current_thresholds.copy()
    if not annotations:
        logger.info(f"No annotations for {group_name} wave. Keeping current thresholds.")
        return new_thresholds

    tp = fp = fn = tn = 0
    for ann in annotations:
        pred = ann.get('predicted_label', '').lower()
        annotated = ann.get('annotated_label', '').capitalize()

        if pred == 'awake':
            if annotated == 'Awake':
                tp += 1
            else:
                fp += 1
        else:
            if annotated == 'Sleeping':
                tn += 1
            else:
                fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    logger.info(f"[{group_name}] Performance - Recall: {recall:.2f}, Precision: {precision:.2f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")

    # Feedback loop adjustments
    adjustment = 1.0
    if recall < 0.90:
        adjustment = 0.95
        logger.info(f"[{group_name}] Recall is low (< 90%). Decreasing thresholds by 5% to capture more awake clips.")
    elif precision < 0.90:
        adjustment = 1.05
        logger.info(f"[{group_name}] Precision is low (< 90%). Increasing thresholds by 5% to reduce false awake predictions.")
    else:
        logger.info(f"[{group_name}] Performance satisfies targets. Keeping thresholds unchanged.")

    if adjustment != 1.0:
        new_thresholds['disp_thresh'] = max(0.01, round(new_thresholds.get('disp_thresh', 2.0) * adjustment, 4))
        new_thresholds['roi_thresh'] = max(0.1, round(new_thresholds.get('roi_thresh', 1.5) * adjustment, 4))
        new_thresholds['area_thresh'] = max(0.001, round(new_thresholds.get('area_thresh', 0.02) * adjustment, 4))
        
    return new_thresholds
