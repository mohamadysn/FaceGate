"""
Reusable camera I/O helpers (FAIR: Accessible + Reusable).

These utilities wrap OpenCV capture/display patterns shared by capture / recognition modules so
each feature module does not re-implement the same boilerplate.

Works on Linux, Windows, and macOS (camera backend chosen per platform).
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

# Key codes used across interactive viewers.
KEY_QUIT = {ord("q"), ord("Q"), 27}  # q / Q / Esc


def preferred_camera_backends() -> List[int]:
    """Return OpenCV capture backends to try, in order, for this OS."""
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform == "win32":
        # DirectShow is often more reliable for USB webcams than MSMF alone.
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(
    index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> cv2.VideoCapture:
    """
    Open a webcam and request a capture resolution / frame rate.

    Uses a platform-appropriate backend, then MJPEG + a short buffer when
    available so USB webcams stay responsive instead of queuing stale frames.

    Parameters
    ----------
    index :
        Device index passed to ``cv2.VideoCapture`` (0 = default webcam).
    width, height :
        Requested frame size. The driver may return a different size.
    fps :
        Requested FPS. Use ``0`` to keep the camera default.

    Returns
    -------
    cv2.VideoCapture
        Opened capture object.

    Raises
    ------
    RuntimeError
        If the device cannot be opened.
    """
    cap = None
    for backend in preferred_camera_backends():
        try:
            candidate = cv2.VideoCapture(index, backend)
        except Exception:
            continue
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()

    if cap is None or not cap.isOpened():
        # Last resort without an explicit backend.
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index={index} on {sys.platform}")

    # MJPEG reduces USB bandwidth on many webcams (Linux/Windows); ignore failures.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        try:
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        except Exception:
            pass
    # Drop buffered frames so the latest image is shown (less "laggy" feel).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


def grab_latest_frame(cap: cv2.VideoCapture, flush: int = 2):
    """
    Read the newest camera frame, optionally flushing a few queued frames.

    Returns
    -------
    ok, frame
        Same contract as ``cap.read()``.
    """
    ok, frame = False, None
    for _ in range(max(1, flush)):
        ok, frame = cap.read()
        if not ok:
            break
    return ok, frame


def resize_for_inference(
    frame: np.ndarray,
    max_side: int = 640,
) -> Tuple[np.ndarray, float]:
    """
    Downscale a frame for faster model inference.

    Returns
    -------
    small_frame, scale
        ``scale`` maps coordinates from ``small_frame`` back to ``frame``
        (``x_full = x_small * scale``).
    """
    height, width = frame.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame, 1.0
    scale = longest / float(max_side)
    new_w = max(1, int(round(width / scale)))
    new_h = max(1, int(round(height / scale)))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, scale


def get_screen_size(fallback: Tuple[int, int] = (1280, 720)) -> Tuple[int, int]:
    """
    Best-effort screen size for fitting OpenCV windows.

    Returns
    -------
    width, height
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback


def create_fitted_window(
    window_name: str,
    frame_width: int,
    frame_height: int,
    max_screen_fraction: float = 0.85,
) -> Tuple[int, int]:
    """
    Create a resizable OpenCV window that fits on the screen.

    Parameters
    ----------
    window_name :
        Window title.
    frame_width, frame_height :
        Native content size.
    max_screen_fraction :
        Maximum portion of the screen the window may occupy.

    Returns
    -------
    win_w, win_h
        Applied window size in pixels.
    """
    screen_w, screen_h = get_screen_size()
    max_w = max(320, int(screen_w * max_screen_fraction))
    max_h = max(240, int(screen_h * max_screen_fraction))

    scale = min(max_w / max(1, frame_width), max_h / max(1, frame_height), 1.0)
    win_w = max(320, int(frame_width * scale))
    win_h = max(240, int(frame_height * scale))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, win_w, win_h)
    # Keep a bit of margin from the top-left so the title bar stays visible.
    cv2.moveWindow(window_name, 40, 40)
    return win_w, win_h


def fit_frame_to_window(frame: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    """
    Resize a frame to fit inside the window while keeping aspect ratio.

    Useful when the source image (e.g. a large JPEG) is bigger than the screen.
    """
    h, w = frame.shape[:2]
    scale = min(win_w / max(1, w), win_h / max(1, h), 1.0)
    if scale >= 0.999:
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_text_lines(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (12, 28),
    line_height: int = 28,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """
    Draw multi-line HUD text with a dark outline for readability.

    Parameters
    ----------
    frame :
        BGR image modified in-place.
    lines :
        Text rows to render.
    origin :
        Pixel ``(x, y)`` of the first baseline.
    line_height :
        Vertical spacing between lines in pixels.
    color :
        BGR fill color for the foreground text.
    """
    x, y = origin
    for line in lines:
        # Outline then fill: keeps text readable on bright/dark frames.
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += line_height


def should_quit(key: int, quit_keys: Iterable[int] = KEY_QUIT) -> bool:
    """
    Return True when ``key`` is a quit key (q / Q / Esc by default).

    Parameters
    ----------
    key :
        Value from ``cv2.waitKey(...) & 0xFF``.
    quit_keys :
        Set/list of accepted quit key codes.
    """
    return key in quit_keys


def window_was_closed(window_name: str) -> bool:
    """
    Return True if the named OpenCV window was closed by the user.

    On some backends, querying a destroyed window raises ``cv2.error``.
    """
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def smooth_fps_update(previous_fps: float, previous_time: float, alpha: float = 0.1) -> Tuple[float, float]:
    """
    Update an exponentially smoothed FPS estimate.

    Parameters
    ----------
    previous_fps :
        Last smoothed FPS value (use ``0.0`` on the first call).
    previous_time :
        Timestamp (``time.time()``) of the previous frame.
    alpha :
        Smoothing factor in ``(0, 1]``. Higher = more reactive.

    Returns
    -------
    smoothed_fps, now
        Updated FPS and the current timestamp for the next iteration.
    """
    now = time.time()
    dt = max(now - previous_time, 1e-6)
    instant = 1.0 / dt
    if previous_fps <= 0.0:
        return instant, now
    return (1.0 - alpha) * previous_fps + alpha * instant, now
