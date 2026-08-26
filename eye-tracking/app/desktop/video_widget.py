"""Tkinter video panel that displays OpenCV BGR frames."""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoPanel(tk.Label):
    """Label that shows the latest BGR frame, scaled to fit the widget."""

    def __init__(self, master, **kwargs) -> None:
        kwargs.setdefault("bg", "#111827")
        kwargs.setdefault("fg", "#9CA3AF")
        kwargs.setdefault("text", "No video")
        kwargs.setdefault("compound", "center")
        super().__init__(master, **kwargs)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_size: Tuple[int, int] = (0, 0)

    def clear(self, message: str = "No video") -> None:
        self._photo = None
        self.configure(image="", text=message)

    def show_bgr(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self.update_idletasks()
        max_w = max(160, self.winfo_width() - 4)
        max_h = max(120, self.winfo_height() - 4)
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        if (tw, th) != (w, h):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self._photo = photo
        self.configure(image=photo, text="")
