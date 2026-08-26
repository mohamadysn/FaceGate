#!/usr/bin/env python3
"""
Real-time face detection with InsightFace.

FAIR compliance notes
---------------------
Findable
    Module ``face-recognition``; outputs (FPS, latency, scores) are overlaid and logged
    so detection quality is easy to inspect.
Accessible
    CLI entry point; defaults to ``CPUExecutionProvider`` for broad hardware
    support, with optional CUDA.
Interoperable
    Optionally consumes camera-calibration ``camera_matrix.yml`` (OpenCV ``K`` / ``D``) for
    undistortion; landmarks follow InsightFace's 5-point convention.
Reusable
    Core helpers (ROI from landmarks, app factory, drawing) are documented
    functions that later stages (eye segmentation / gaze) can import.

Example
-------
::

    python insightface_camera.py --provider CPUExecutionProvider
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

_EYE_TRACKING_ROOT = Path(__file__).resolve().parents[2]
if str(_EYE_TRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_EYE_TRACKING_ROOT))

from common.calibration_io import load_camera_coefficients  # noqa: E402
from common.camera_io import (  # noqa: E402
    draw_text_lines,
    open_camera,
    should_quit,
    smooth_fps_update,
    window_was_closed,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parents[1] / "camera-calibration" / "camera_matrix.yml"

# InsightFace 5-point landmark order:
#   0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
LEFT_EYE_IDX = 0
RIGHT_EYE_IDX = 1
BBox = Tuple[int, int, int, int]


def build_face_app(
    model_name: str,
    provider: str,
    device: int,
    det_size: int,
    min_score: float,
) -> FaceAnalysis:
    """
    Construct and prepare an InsightFace detection application.

    Parameters
    ----------
    model_name :
        Model pack name, e.g. ``buffalo_l``.
    provider :
        ONNX Runtime provider string (``CPUExecutionProvider`` or
        ``CUDAExecutionProvider``).
    device :
        CUDA device id when using a CUDA provider.
    det_size :
        Square detector input size; ``<= 0`` keeps the library default.
    min_score :
        Detection score threshold passed to ``prepare``.

    Returns
    -------
    FaceAnalysis
        Ready-to-use detector (detection module only).
    """
    providers: List[str] = [provider]
    # Graceful degradation: if CUDA is requested, also list CPU as fallback.
    if "CUDA" in provider.upper():
        providers.append("CPUExecutionProvider")

    app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection"],
        providers=providers,
    )
    det_size_value = None if det_size <= 0 else (det_size, det_size)
    ctx_id = int(device) if "CUDA" in provider.upper() else -1
    app.prepare(ctx_id=ctx_id, det_size=det_size_value, det_thresh=min_score)
    return app


def eye_roi_from_landmarks(
    landmarks: np.ndarray,
    frame_shape: Sequence[int],
    scale: float = 1.8,
) -> List[BBox]:
    """
    Estimate axis-aligned eye ROIs from 5 facial landmarks.

    This is a lightweight precursor to pupil segmentation:
    ROI size scales with inter-ocular distance so it adapts to face scale.

    Parameters
    ----------
    landmarks :
        Array of shape ``(5, 2)`` in image coordinates.
    frame_shape :
        Frame ``(H, W, ...)`` used for clipping.
    scale :
        Multiplier controlling ROI size relative to eye distance.

    Returns
    -------
    list of (x1, y1, x2, y2)
        Clipped bounding boxes for left then right eye (when valid).
    """
    height, width = frame_shape[:2]
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    half = max(8.0, eye_distance * 0.22 * scale)

    rois: List[BBox] = []
    for eye in (left_eye, right_eye):
        x1 = int(max(0, eye[0] - half))
        y1 = int(max(0, eye[1] - half * 0.7))
        x2 = int(min(width - 1, eye[0] + half))
        y2 = int(min(height - 1, eye[1] + half * 0.7))
        if x2 > x1 and y2 > y1:
            rois.append((x1, y1, x2, y2))
    return rois


def draw_face(frame: np.ndarray, face, draw_eyes: bool = True) -> None:
    """
    Draw bbox, confidence, landmarks, and optional eye ROIs on ``frame``.

    Parameters
    ----------
    frame :
        BGR image modified in-place.
    face :
        InsightFace face object with ``bbox``, ``det_score``, optional ``kps``.
    draw_eyes :
        If True, overlay approximate eye rectangles from landmarks.
    """
    x1, y1, x2, y2 = face.bbox.astype(int)
    score = float(face.det_score)
    color = (0, 220, 0) if score >= 0.8 else (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{score * 100:.1f}%"
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

    if getattr(face, "kps", None) is None:
        return

    for point in face.kps.astype(int):
        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 200, 0), -1, cv2.LINE_AA)

    if draw_eyes:
        for ex1, ey1, ex2, ey2 in eye_roi_from_landmarks(face.kps, frame.shape):
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 80, 80), 1)


def make_preview_panel(frame: np.ndarray, face, size: int = 180) -> np.ndarray:
    """
    Build a small side panel with face crop + zoomed eye crops.

    Parameters
    ----------
    frame :
        Full BGR frame (already mirrored / undistorted if those modes are on).
    face :
        Best face object, or ``None`` when no detection is available.
    size :
        Pixel size of each square cell in the panel.

    Returns
    -------
    np.ndarray
        BGR panel image suitable for ``cv2.imshow``.
    """
    panel = np.zeros((size, size * 2 + 8, 3), dtype=np.uint8)
    if face is None:
        cv2.putText(
            panel,
            "No face",
            (20, size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        return panel

    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return panel

    face_crop = frame[y1:y2, x1:x2]
    panel[:, :size] = cv2.resize(face_crop, (size, size))

    if getattr(face, "kps", None) is not None:
        rois = eye_roi_from_landmarks(face.kps, frame.shape, scale=2.2)
        eye_panel = np.zeros((size, size, 3), dtype=np.uint8)
        for i, (ex1, ey1, ex2, ey2) in enumerate(rois[:2]):
            eye = frame[ey1:ey2, ex1:ex2]
            if eye.size == 0:
                continue
            target_h = size // 2 - 4
            target_w = size - 8
            eye = cv2.resize(eye, (target_w, target_h))
            y0 = 4 + i * (target_h + 4)
            eye_panel[y0 : y0 + target_h, 4 : 4 + target_w] = eye
        panel[:, size + 8 :] = eye_panel
        cv2.putText(panel, "face", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(panel, "eyes", (size + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return panel


def run_camera_detection(
    camera_index: int = 0,
    model_name: str = "buffalo_l",
    provider: str = "CPUExecutionProvider",
    det_size: int = 640,
    device: int = 0,
    max_fps: float = 0.0,
    min_score: float = 0.5,
    width: int = 1280,
    height: int = 720,
    mirror: bool = True,
    calib_path: Optional[Path] = DEFAULT_CALIB,
) -> None:
    """
    Run the live detection loop and display performance metrics.

    Controls
    --------
    q / Esc : quit
    m       : toggle horizontal mirror
    u       : toggle undistortion (requires camera calibration YAML)
    """
    app = build_face_app(model_name, provider, device, det_size, min_score)
    camera_matrix, dist_coeffs = load_camera_coefficients(calib_path) if calib_path else (None, None)
    if camera_matrix is not None:
        print(f"Loaded calibration: {calib_path}")

    cap = open_camera(camera_index, width, height)
    window_name = "Face detection"
    preview_name = "Preview face / eyes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)

    print("Click the video window. q/Esc=quit, m=mirror, u=undistort")

    previous_time = time.time()
    fps = 0.0
    latency_ms = 0.0
    last_proc_time = 0.0
    target_dt = 0.0 if max_fps <= 0 else 1.0 / float(max_fps)
    last_faces = []
    use_undistort = False
    use_mirror = mirror

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read camera frame.")
            break

        # Geometric corrections before detection keep landmarks in display space.
        if use_undistort and camera_matrix is not None and dist_coeffs is not None:
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        if use_mirror:
            frame = cv2.flip(frame, 1)

        fps, previous_time = smooth_fps_update(fps, previous_time)

        now = time.time()
        do_detect = target_dt <= 0.0 or (now - last_proc_time) >= target_dt
        if do_detect:
            t0 = time.time()
            last_faces = app.get(frame)
            latency_ms = (time.time() - t0) * 1000.0
            last_proc_time = now

        faces = [face for face in last_faces if float(face.det_score) >= min_score]
        for face in faces:
            draw_face(frame, face, draw_eyes=True)

        top_face = max(faces, key=lambda item: item.det_score) if faces else None
        preview = make_preview_panel(frame, top_face)

        hud = [
            f"FPS: {fps:.1f}  |  latency: {latency_ms:.0f} ms",
            f"Faces: {len(faces)}",
            "q: quit  m: mirror  u: undistort",
        ]
        flags = []
        if use_mirror:
            flags.append("mirror")
        if use_undistort:
            flags.append("undistort")
        if flags:
            hud.append(" | ".join(flags))
        draw_text_lines(frame, hud, color=(40, 255, 120))

        cv2.imshow(window_name, frame)
        cv2.imshow(preview_name, preview)

        key = cv2.waitKey(15) & 0xFF
        if should_quit(key) or window_was_closed(window_name):
            break
        if key in (ord("m"), ord("M")):
            use_mirror = not use_mirror
            last_faces = []  # invalidate stale boxes after geometry change
        if key in (ord("u"), ord("U")):
            if camera_matrix is None:
                print("No calibration YAML found — undistort unavailable.")
            else:
                use_undistort = not use_undistort
                last_faces = []

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for live face detection."""
    parser = argparse.ArgumentParser(
        description="Real-time face detection (InsightFace), FAIR-documented CLI.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--model", type=str, default="buffalo_l", help="InsightFace model pack.")
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX provider: CPUExecutionProvider or CUDAExecutionProvider.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Throttle detection FPS (0 = unlimited).")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum detection confidence.")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size.")
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height.")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirror at startup.")
    parser.add_argument(
        "--calib",
        type=str,
        default=str(DEFAULT_CALIB),
        help="Path to camera_matrix.yml (OpenCV K/D).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for face-recognition live detection."""
    args = parse_args()
    run_camera_detection(
        camera_index=args.camera,
        model_name=args.model,
        provider=args.provider,
        det_size=args.det_size,
        device=args.device,
        max_fps=args.max_fps,
        min_score=args.min_score,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        calib_path=Path(args.calib) if args.calib else None,
    )


if __name__ == "__main__":
    main()
