"""
Pupil segmentation / pupil localization (FAIR-documented).

Acceptance rule (important)
---------------------------
A pupil is accepted only if the eye crop looks like an *open eye*
(bright sclera / sufficient dynamic range) AND a dark blob is found.

Covering an eye with a hand usually removes the bright sclera and flattens
the intensity range — those crops are rejected, even if a dark spot exists.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import EyeSample, FaceSample, Point

MIN_PUPIL_CONFIDENCE = 0.32


def is_valid_pupil(eye: EyeSample, min_confidence: float = MIN_PUPIL_CONFIDENCE) -> bool:
    """Return True when an eye has an accepted pupil detection."""
    return eye.pupil is not None and float(eye.pupil_confidence) >= min_confidence


def count_valid_pupils(
    face: FaceSample,
    min_confidence: float = MIN_PUPIL_CONFIDENCE,
) -> int:
    """Count eyes with a pupil above the confidence threshold."""
    return sum(1 for eye in face.eyes if is_valid_pupil(eye, min_confidence))


def _open_eye_evidence(gray: np.ndarray) -> Tuple[bool, float]:
    """
    Estimate whether an eye crop shows an open eye.

    Open eye cues
    -------------
    - Non-trivial intensity range (pupil dark, sclera/skin brighter)
    - A non-negligible fraction of relatively bright pixels (sclera)
    - Not a nearly uniform patch (hand / closed lid)

    Returns
    -------
    is_open, openness_score
    """
    if gray.size < 16:
        return False, 0.0

    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    dynamic = float(p90 - p10)
    contrast = float(np.std(gray))
    mean_intensity = float(np.mean(gray))

    # Bright pixels relative to the crop (sclera proxy).
    bright_thresh = max(float(p50 + 15.0), float(p90 - 5.0))
    bright_ratio = float(np.mean(gray >= bright_thresh))

    # Dark pixels (pupil / iris proxy).
    dark_thresh = min(float(p50 - 10.0), float(p10 + 8.0))
    dark_ratio = float(np.mean(gray <= dark_thresh))

    # Hard rejects: flat / covered.
    if contrast < 8.0 or dynamic < 18.0:
        return False, 0.0
    # Hand over eye: often mid-tone, little bright "white", little structured dark hole.
    if bright_ratio < 0.04 and dynamic < 35.0:
        return False, 0.0
    if bright_ratio < 0.02:
        return False, 0.0
    # Almost no dark core => no pupil visible.
    if dark_ratio < 0.02:
        return False, 0.0
    # Very dark uniform cover.
    if mean_intensity < 50.0 and bright_ratio < 0.06:
        return False, 0.0

    openness = float(
        np.clip(
            0.35 * min(1.0, dynamic / 60.0)
            + 0.30 * min(1.0, contrast / 35.0)
            + 0.20 * min(1.0, bright_ratio / 0.15)
            + 0.15 * min(1.0, dark_ratio / 0.12),
            0.0,
            1.0,
        )
    )
    return openness >= 0.34, openness


def _find_pupil_blob(gray: np.ndarray) -> Tuple[Optional[Point], float, np.ndarray]:
    """Find the best dark circular blob; no 'darkest pixel' invention."""
    h, w = gray.shape
    roi_area = float(h * w)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = cv2.bitwise_not(blur)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    border = max(1, min(h, w) // 10)
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    fg = float(np.count_nonzero(mask)) / roi_area
    if fg < 0.005 or fg > 0.55:
        return None, 0.0, mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pt: Optional[Point] = None
    best_conf = 0.0
    mean_intensity = float(np.mean(gray))

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.01 * roi_area or area > 0.35 * roi_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.30:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        x1, x2 = max(0, int(cx) - 2), min(w, int(cx) + 3)
        y1, y2 = max(0, int(cy) - 2), min(h, int(cy) + 3)
        local = gray[y1:y2, x1:x2]
        if local.size == 0:
            continue
        pupil_level = float(np.mean(local))
        # Must be darker than the eye-crop average.
        if pupil_level > mean_intensity - 4.0:
            continue

        center_dist = np.hypot(cx - w / 2.0, cy - h / 2.0) / (0.5 * np.hypot(w, h) + 1e-6)
        if center_dist > 0.85:
            continue

        darkness = float(np.clip((mean_intensity - pupil_level) / 40.0, 0.0, 1.0))
        conf = float(
            np.clip(
                0.45 * circularity
                + 0.30 * darkness
                + 0.25 * max(0.0, 1.0 - center_dist),
                0.0,
                1.0,
            )
        )
        if conf > best_conf:
            best_conf = conf
            best_pt = (cx, cy)

    return best_pt, best_conf, mask


def _pupil_from_roi(eye_bgr: np.ndarray) -> Tuple[Optional[Point], float, np.ndarray]:
    """
    Locate the pupil center inside an eye crop.

    Returns ``(None, 0, mask)`` when the crop does not look like an open eye
    or when no credible pupil blob is found.
    """
    empty = np.zeros((1, 1), dtype=np.uint8)
    if eye_bgr.size == 0 or eye_bgr.shape[0] < 8 or eye_bgr.shape[1] < 8:
        return None, 0.0, empty

    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    is_open, openness = _open_eye_evidence(gray)
    if not is_open:
        # Covered / closed / no sclera visible — do not invent a pupil.
        return None, 0.0, empty

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    pupil, conf, mask = _find_pupil_blob(enhanced)
    if pupil is None:
        return None, 0.0, mask

    # Blend blob confidence with openness so weak open-eye cues are down-weighted.
    final_conf = float(np.clip(0.65 * conf + 0.35 * openness, 0.0, 1.0))
    if final_conf < MIN_PUPIL_CONFIDENCE:
        return None, final_conf, mask
    return pupil, final_conf, mask


def segment_eyes_for_face(
    frame: np.ndarray,
    face: FaceSample,
    min_confidence: float = MIN_PUPIL_CONFIDENCE,
) -> FaceSample:
    """
    Enrich a ``FaceSample`` with pupil centers for each eye ROI.

    Also applies a cross-eye consistency check: if one eye crop looks covered
    relative to the other (much lower brightness range), clear its pupil.
    """
    updated_eyes: List[EyeSample] = []
    crop_stats = []

    for eye in face.eyes:
        x1, y1, x2, y2 = eye.bbox
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
        if gray is not None and gray.size:
            dynamic = float(np.percentile(gray, 90) - np.percentile(gray, 10))
            bright_ratio = float(np.mean(gray >= np.percentile(gray, 50) + 15))
        else:
            dynamic, bright_ratio = 0.0, 0.0
        crop_stats.append((dynamic, bright_ratio))

        pupil_local, confidence, _ = _pupil_from_roi(crop)
        pupil_global: Optional[Point] = None
        kept_conf = 0.0
        if pupil_local is not None and confidence >= min_confidence:
            pupil_global = (pupil_local[0] + x1, pupil_local[1] + y1)
            kept_conf = float(confidence)

        updated_eyes.append(
            replace(
                eye,
                pupil=pupil_global,
                pupil_confidence=kept_conf,
            )
        )

    # If both eyes exist and one is clearly flatter / less bright than the other,
    # treat the weaker one as occluded (hand often covers only one eye).
    if len(updated_eyes) == 2:
        (d0, b0), (d1, b1) = crop_stats
        # Relative occlusion: one eye much less open-looking than the other.
        if d0 > 25 and d1 < 0.45 * d0 and b1 < 0.5 * max(b0, 1e-6):
            updated_eyes[1] = replace(updated_eyes[1], pupil=None, pupil_confidence=0.0)
        elif d1 > 25 and d0 < 0.45 * d1 and b0 < 0.5 * max(b1, 1e-6):
            updated_eyes[0] = replace(updated_eyes[0], pupil=None, pupil_confidence=0.0)

    return replace(face, eyes=updated_eyes)


def draw_eye_segmentation(frame: np.ndarray, face: FaceSample) -> None:
    """Overlay eye boxes and *valid* pupil centers on ``frame`` (in-place)."""
    for eye in face.eyes:
        x1, y1, x2, y2 = eye.bbox
        valid = is_valid_pupil(eye)
        color = (255, 80, 80) if eye.side == "left" else (80, 80, 255)
        box_color = color if valid else (90, 90, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)

        label = f"{eye.side[0].upper()}:OK {eye.pupil_confidence:.2f}" if valid else f"{eye.side[0].upper()}:--"
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            box_color,
            1,
            cv2.LINE_AA,
        )

        if valid and eye.pupil is not None:
            px, py = int(eye.pupil[0]), int(eye.pupil[1])
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1, cv2.LINE_AA)


def make_eye_debug_panel(frame: np.ndarray, face: Optional[FaceSample], cell: int = 120) -> np.ndarray:
    """Return a horizontal panel with left/right eye crops for debugging."""
    panel = np.zeros((cell, cell * 2 + 10, 3), dtype=np.uint8)
    eyes = face.eyes if face is not None else []
    if not eyes:
        cv2.putText(panel, "No eyes", (20, cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return panel

    for i, eye in enumerate(eyes[:2]):
        x1, y1, x2, y2 = eye.bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (cell, cell))
        status = "OK" if is_valid_pupil(eye) else "NO"
        if is_valid_pupil(eye) and eye.pupil is not None:
            lx = (eye.pupil[0] - x1) / max(1, x2 - x1) * cell
            ly = (eye.pupil[1] - y1) / max(1, y2 - y1) * cell
            cv2.circle(resized, (int(lx), int(ly)), 4, (0, 255, 255), -1)
        x0 = i * (cell + 10)
        panel[:, x0 : x0 + cell] = resized
        cv2.putText(
            panel,
            f"{eye.side}:{status}",
            (x0 + 6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return panel
