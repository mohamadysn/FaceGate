"""
Interactive image editor for enrollment photos.

Supports zoom, pan, crop, rotate, and reset before accepting the image.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from .platform_util import ui_font
    from .theme import ACCENT, ACCENT_TEXT, BG, CARD, CARD_BORDER, MUTED, PANEL, PANEL_HOVER, SIDE, TEXT
except ImportError:
    from app.desktop.platform_util import ui_font  # type: ignore
    from app.desktop.theme import ACCENT, ACCENT_TEXT, BG, CARD, CARD_BORDER, MUTED, PANEL, PANEL_HOVER, SIDE, TEXT


@dataclass
class EditedImage:
    """One prepared enrollment image (BGR) with a display label."""

    label: str
    bgr: np.ndarray


class ImageEditorDialog(tk.Toplevel):
    """
    Modal editor: zoom (wheel / buttons), pan (drag), crop (draw box), rotate.

    Call ``result`` after ``wait_window`` — list of :class:`EditedImage` or ``None`` if cancelled.
    """

    def __init__(
        self,
        master,
        images: List[Tuple[str, np.ndarray]],
        title: str = "Edit images",
    ) -> None:
        super().__init__(master)
        self.title(title)
        self.configure(bg=BG)
        self.geometry("1000x720")
        self.minsize(760, 540)
        self.transient(master)
        self.grab_set()
        self.result: Optional[List[EditedImage]] = None

        if not images:
            raise ValueError("No images to edit")

        self._sources: List[Tuple[str, np.ndarray]] = [
            (label, frame.copy()) for label, frame in images
        ]
        self._working: List[np.ndarray] = [frame.copy() for _, frame in self._sources]
        self._index = 0

        # View transform: zoom + pan offset in canvas pixels
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._drag_mode: Optional[str] = None  # "pan" | "crop"
        self._last_xy: Optional[Tuple[int, int]] = None
        self._crop_start: Optional[Tuple[int, int]] = None
        self._crop_rect: Optional[Tuple[int, int, int, int]] = None  # canvas coords
        self._crop_enabled = tk.BooleanVar(value=False)
        self._photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._fit_to_view()
        self._redraw()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.bind("<Return>", lambda _e: self._accept_all())

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=16, pady=10)

        self._title_var = tk.StringVar()
        tk.Label(
            top,
            textvariable=self._title_var,
            bg=BG,
            fg=TEXT,
            font=ui_font(13, bold=True),
        ).pack(side=tk.LEFT)

        nav = tk.Frame(top, bg=BG)
        nav.pack(side=tk.RIGHT)
        for label, cmd in (("◀ Prev", self._prev), ("Next ▶", self._next)):
            tk.Button(
                nav,
                text=label,
                command=cmd,
                bg=PANEL,
                fg=TEXT,
                activebackground=PANEL_HOVER,
                relief="flat",
                padx=12,
                pady=6,
                cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        tools = tk.Frame(self, bg=SIDE)
        tools.pack(fill=tk.X, padx=16, pady=(0, 8))

        def btn(text, cmd):
            return tk.Button(
                tools,
                text=text,
                command=cmd,
                bg=PANEL,
                fg=TEXT,
                activebackground=PANEL_HOVER,
                relief="flat",
                padx=12,
                pady=6,
                cursor="hand2",
            )

        btn("Zoom −", lambda: self._zoom_by(0.85)).pack(side=tk.LEFT, padx=2, pady=8)
        btn("Zoom +", lambda: self._zoom_by(1.15)).pack(side=tk.LEFT, padx=2, pady=8)
        btn("Fit", self._fit_to_view).pack(side=tk.LEFT, padx=2, pady=8)
        btn("↺ 90°", lambda: self._rotate(90)).pack(side=tk.LEFT, padx=2, pady=8)
        btn("↻ 90°", lambda: self._rotate(-90)).pack(side=tk.LEFT, padx=2, pady=8)
        btn("Reset", self._reset_current).pack(side=tk.LEFT, padx=2, pady=8)

        tk.Checkbutton(
            tools,
            text="Crop mode",
            variable=self._crop_enabled,
            command=self._on_crop_toggle,
            bg=SIDE,
            fg=TEXT,
            selectcolor=BG,
            activebackground=SIDE,
            activeforeground=ACCENT,
        ).pack(side=tk.LEFT, padx=12)

        btn("Apply crop", self._apply_crop).pack(side=tk.LEFT, padx=2, pady=8)

        hint = tk.Label(
            self,
            text="Scroll = zoom · Drag = pan · Crop: draw a box, then Apply crop",
            bg=BG,
            fg=MUTED,
            font=ui_font(9),
        )
        hint.pack(fill=tk.X, padx=16)

        canvas_frame = tk.Frame(self, bg=BG, highlightthickness=1, highlightbackground=CARD_BORDER)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self._on_wheel_linux(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._on_wheel_linux(e, -1))

        bottom = tk.Frame(self, bg=BG)
        bottom.pack(fill=tk.X, padx=16, pady=12)
        tk.Button(
            bottom,
            text="Cancel",
            command=self._cancel,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL_HOVER,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)
        tk.Button(
            bottom,
            text="Accept all images",
            command=self._accept_all,
            bg=ACCENT,
            fg=ACCENT_TEXT,
            activebackground="#7DD3FC",
            relief="flat",
            padx=16,
            pady=8,
            font=ui_font(10, bold=True),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

    # ----------------------------------------------------------- navigation
    def _update_title(self) -> None:
        label, _ = self._sources[self._index]
        h, w = self._working[self._index].shape[:2]
        self._title_var.set(
            f"{label}  ({self._index + 1}/{len(self._working)})  ·  {w}×{h} px  ·  zoom {self._zoom:.0%}"
        )

    def _prev(self) -> None:
        self._index = (self._index - 1) % len(self._working)
        self._crop_rect = None
        self._fit_to_view()
        self._redraw()

    def _next(self) -> None:
        self._index = (self._index + 1) % len(self._working)
        self._crop_rect = None
        self._fit_to_view()
        self._redraw()

    # -------------------------------------------------------------- transforms
    def _current(self) -> np.ndarray:
        return self._working[self._index]

    def _fit_to_view(self) -> None:
        self.update_idletasks()
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        h, w = self._current().shape[:2]
        self._zoom = min(cw / w, ch / h, 1.5)
        self._offset_x = (cw - w * self._zoom) / 2
        self._offset_y = (ch - h * self._zoom) / 2
        self._crop_rect = None
        self._redraw()

    def _zoom_by(self, factor: float, pivot: Optional[Tuple[int, int]] = None) -> None:
        old = self._zoom
        new = float(np.clip(old * factor, 0.1, 8.0))
        if abs(new - old) < 1e-6:
            return
        if pivot is None:
            pivot = (self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)
        px, py = pivot
        # Keep the image point under the cursor stable.
        img_x = (px - self._offset_x) / old
        img_y = (py - self._offset_y) / old
        self._zoom = new
        self._offset_x = px - img_x * new
        self._offset_y = py - img_y * new
        self._redraw()

    def _rotate(self, angle_ccw: int) -> None:
        frame = self._current()
        if angle_ccw % 360 == 90:
            self._working[self._index] = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle_ccw % 360 == -90 or angle_ccw % 360 == 270:
            self._working[self._index] = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif abs(angle_ccw) % 360 == 180:
            self._working[self._index] = cv2.rotate(frame, cv2.ROTATE_180)
        self._crop_rect = None
        self._fit_to_view()

    def _reset_current(self) -> None:
        self._working[self._index] = self._sources[self._index][1].copy()
        self._crop_rect = None
        self._fit_to_view()

    def _on_crop_toggle(self) -> None:
        self._crop_rect = None
        self.canvas.configure(cursor="crosshair" if self._crop_enabled.get() else "fleur")
        self._redraw()

    def _apply_crop(self) -> None:
        if not self._crop_rect:
            return
        x1, y1, x2, y2 = self._crop_rect
        ix1, iy1 = self._canvas_to_image(min(x1, x2), min(y1, y2))
        ix2, iy2 = self._canvas_to_image(max(x1, x2), max(y1, y2))
        frame = self._current()
        h, w = frame.shape[:2]
        ix1 = int(np.clip(ix1, 0, w - 1))
        iy1 = int(np.clip(iy1, 0, h - 1))
        ix2 = int(np.clip(ix2, ix1 + 1, w))
        iy2 = int(np.clip(iy2, iy1 + 1, h))
        if ix2 - ix1 < 16 or iy2 - iy1 < 16:
            return
        self._working[self._index] = frame[iy1:iy2, ix1:ix2].copy()
        self._crop_rect = None
        self._crop_enabled.set(False)
        self.canvas.configure(cursor="fleur")
        self._fit_to_view()

    # -------------------------------------------------------------- mapping
    def _canvas_to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self._offset_x) / self._zoom, (cy - self._offset_y) / self._zoom

    def _image_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        return ix * self._zoom + self._offset_x, iy * self._zoom + self._offset_y

    # ---------------------------------------------------------------- draw
    def _redraw(self) -> None:
        self.canvas.delete("all")
        frame = self._current()
        h, w = frame.shape[:2]
        tw = max(1, int(round(w * self._zoom)))
        th = max(1, int(round(h * self._zoom)))
        interp = cv2.INTER_AREA if self._zoom < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(frame, (tw, th), interpolation=interp)
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(self._offset_x, self._offset_y, anchor="nw", image=self._photo)
        if self._crop_rect:
            x1, y1, x2, y2 = self._crop_rect
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#38BDF8", width=2, dash=(4, 3))
            # Dim outside crop
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="", fill="", width=0
            )
        self._update_title()

    # -------------------------------------------------------------- events
    def _on_press(self, event) -> None:
        self._last_xy = (event.x, event.y)
        if self._crop_enabled.get():
            self._drag_mode = "crop"
            self._crop_start = (event.x, event.y)
            self._crop_rect = (event.x, event.y, event.x, event.y)
        else:
            self._drag_mode = "pan"

    def _on_drag(self, event) -> None:
        if self._last_xy is None:
            return
        if self._drag_mode == "pan":
            dx = event.x - self._last_xy[0]
            dy = event.y - self._last_xy[1]
            self._offset_x += dx
            self._offset_y += dy
            self._last_xy = (event.x, event.y)
            self._redraw()
        elif self._drag_mode == "crop" and self._crop_start:
            self._crop_rect = (self._crop_start[0], self._crop_start[1], event.x, event.y)
            self._last_xy = (event.x, event.y)
            self._redraw()

    def _on_release(self, _event) -> None:
        self._drag_mode = None
        self._last_xy = None

    def _on_wheel(self, event) -> None:
        # Windows / macOS
        delta = 1 if event.delta > 0 else -1
        self._zoom_by(1.12 if delta > 0 else 0.89, pivot=(event.x, event.y))

    def _on_wheel_linux(self, event, direction: int) -> None:
        self._zoom_by(1.12 if direction > 0 else 0.89, pivot=(event.x, event.y))

    # ----------------------------------------------------------- finish
    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()

    def _accept_all(self) -> None:
        self.result = [
            EditedImage(label=self._sources[i][0], bgr=self._working[i].copy())
            for i in range(len(self._working))
        ]
        self.grab_release()
        self.destroy()


def load_bgr_images(paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """Load image paths as (filename, BGR) pairs; skip unreadable files."""
    out: List[Tuple[str, np.ndarray]] = []
    for raw in paths:
        frame = cv2.imread(raw)
        if frame is None:
            continue
        out.append((PathName(raw), frame))
    return out


def PathName(path: str) -> str:
    from pathlib import Path

    return Path(path).name
