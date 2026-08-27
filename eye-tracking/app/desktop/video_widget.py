"""Tkinter video panel that displays OpenCV BGR frames."""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from .theme import BG, CARD, CARD_BORDER, MUTED
from .platform_util import ui_font


class VideoPanel(tk.Frame):
    """Framed video area with placeholder text when idle."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, bg=BG, highlightthickness=0)
        self._shell = tk.Frame(
            self,
            bg=CARD,
            highlightbackground=CARD_BORDER,
            highlightthickness=1,
        )
        self._shell.pack(fill=tk.BOTH, expand=True)

        self._label = tk.Label(
            self._shell,
            bg=CARD,
            fg=MUTED,
            text="No video",
            compound="center",
            font=ui_font(11),
        )
        self._label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_size: Tuple[int, int] = (0, 0)

    def clear(self, message: str = "No video") -> None:
        self._photo = None
        self._label.configure(image="", text=message)

    def show_bgr(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self.update_idletasks()
        max_w = max(160, self._label.winfo_width() - 8)
        max_h = max(120, self._label.winfo_height() - 8)
        if max_w < 20 or max_h < 20:
            max_w, max_h = 640, 480
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        if (tw, th) != (w, h):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self._photo = photo
        self._label.configure(image=photo, text="")
