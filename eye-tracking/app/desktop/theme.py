"""FaceGate desktop UI theme — colors, ttk styles, and reusable widgets."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from .platform_util import ui_font

ASSETS = Path(__file__).resolve().parent / "assets"

# Palette — deep slate + sky accent
BG = "#0B1120"
BG_ELEVATED = "#0F172A"
SIDE = "#0E1628"
SIDE_BORDER = "#1E293B"
PANEL = "#1A2332"
PANEL_HOVER = "#243044"
CARD = "#151D2E"
CARD_BORDER = "#2A3548"
ACCENT = "#38BDF8"
ACCENT_SOFT = "#0EA5E9"
ACCENT_TEXT = "#082F49"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
MUTED_DARK = "#64748B"
OK = "#34D399"
WARN = "#FBBF24"
DANGER = "#F87171"
NAV_ACTIVE = "#1E3A5F"
NAV_ACTIVE_BORDER = "#38BDF8"

NAV_ITEMS: List[Tuple[str, str, str]] = [
    ("live", "Live recognition", "◉"),
    ("enroll", "Camera enrollment", "◎"),
    ("enroll_img", "Photo enrollment", "▣"),
    ("recognize_img", "Recognize image", "⌕"),
    ("gallery", "Gallery", "☰"),
    ("settings", "Settings", "⚙"),
    ("legal", "Privacy / GDPR", "◈"),
]


def apply_ttk_theme(root: tk.Misc) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Main.TFrame", background=BG)
    style.configure("Side.TFrame", background=SIDE)
    style.configure("Card.TFrame", background=CARD)
    style.configure("Toolbar.TFrame", background=BG)

    style.configure("TLabel", background=BG, foreground=TEXT, font=ui_font(10))
    style.configure("Title.TLabel", background=BG, foreground=TEXT, font=ui_font(22, bold=True))
    style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=ui_font(10))
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=ui_font(9))
    style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=ui_font(10))
    style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=ui_font(9))
    style.configure("StatValue.TLabel", background=CARD, foreground=ACCENT, font=ui_font(20, bold=True))
    style.configure("StatLabel.TLabel", background=CARD, foreground=MUTED, font=ui_font(9))

    style.configure(
        "TButton",
        background=PANEL,
        foreground=TEXT,
        borderwidth=0,
        focusthickness=0,
        padding=(14, 8),
        font=ui_font(10),
    )
    style.map(
        "TButton",
        background=[("active", PANEL_HOVER), ("disabled", SIDE)],
        foreground=[("disabled", MUTED_DARK)],
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground=ACCENT_TEXT,
        padding=(16, 9),
        font=ui_font(10, bold=True),
    )
    style.map("Accent.TButton", background=[("active", "#7DD3FC")])

    style.configure(
        "Ghost.TButton",
        background=BG,
        foreground=MUTED,
        padding=(12, 8),
    )
    style.map("Ghost.TButton", background=[("active", PANEL)], foreground=[("active", TEXT)])

    style.configure(
        "Danger.TButton",
        background="#3F1D2B",
        foreground=DANGER,
        padding=(12, 8),
    )
    style.map("Danger.TButton", background=[("active", "#5C2438")])

    style.configure("TEntry", fieldbackground="#0B1220", foreground=TEXT, bordercolor=CARD_BORDER)
    style.configure("TCombobox", fieldbackground="#0B1220", foreground=TEXT, bordercolor=CARD_BORDER)
    style.configure("TRadiobutton", background=BG, foreground=TEXT, font=ui_font(10))
    style.configure("TCheckbutton", background=BG, foreground=TEXT, font=ui_font(10))
    style.configure("Card.TRadiobutton", background=CARD, foreground=TEXT, font=ui_font(10))
    style.configure("Card.TCheckbutton", background=CARD, foreground=TEXT, font=ui_font(10))
    style.configure("Card.TCombobox", fieldbackground="#0B1220", foreground=TEXT, bordercolor=CARD_BORDER)
    style.configure("Card.TSpinbox", fieldbackground="#0B1220", foreground=TEXT, bordercolor=CARD_BORDER)
    style.configure("Card.TEntry", fieldbackground="#0B1220", foreground=TEXT, bordercolor=CARD_BORDER)

    style.configure(
        "Treeview",
        background=CARD,
        foreground=TEXT,
        fieldbackground=CARD,
        borderwidth=0,
        rowheight=32,
        font=ui_font(10),
    )
    style.configure(
        "Treeview.Heading",
        background=SIDE,
        foreground=TEXT,
        font=ui_font(10, bold=True),
        relief="flat",
    )
    style.map("Treeview", background=[("selected", NAV_ACTIVE)], foreground=[("selected", TEXT)])

    style.configure(
        "Horizontal.TProgressbar",
        background=ACCENT,
        troughcolor=CARD_BORDER,
        borderwidth=0,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
    )

    return style


class NavRail:
    """Sidebar navigation with active-state highlighting."""

    def __init__(self, parent: tk.Misc, on_select: Callable[[str], None]) -> None:
        self._on_select = on_select
        self._buttons: Dict[str, tk.Button] = {}
        self._active = "live"

        self.frame = tk.Frame(parent, bg=SIDE, width=248)
        self.frame.pack_propagate(False)

        header = tk.Frame(self.frame, bg=SIDE)
        header.pack(fill=tk.X, padx=16, pady=(20, 8))

        brand_row = tk.Frame(header, bg=SIDE)
        brand_row.pack(anchor="w")
        logo_path = ASSETS / "face_recog_48.png"
        if logo_path.is_file():
            try:
                img = Image.open(logo_path)
                self._logo = ImageTk.PhotoImage(img)
                tk.Label(brand_row, image=self._logo, bg=SIDE).pack(side=tk.LEFT, padx=(0, 10))
            except Exception:
                self._logo = None
        else:
            self._logo = None

        title_col = tk.Frame(brand_row, bg=SIDE)
        title_col.pack(side=tk.LEFT)
        tk.Label(
            title_col,
            text="FaceGate",
            bg=SIDE,
            fg=ACCENT,
            font=ui_font(18, bold=True),
        ).pack(anchor="w")

        tk.Label(
            title_col,
            text="Face recognition suite",
            bg=SIDE,
            fg=MUTED,
            font=ui_font(9),
        ).pack(anchor="w", pady=(2, 0))

        tk.Frame(self.frame, bg=SIDE_BORDER, height=1).pack(fill=tk.X, padx=16, pady=(12, 8))

        nav_box = tk.Frame(self.frame, bg=SIDE)
        nav_box.pack(fill=tk.X, padx=8)

        for key, label, icon in NAV_ITEMS:
            self._add_btn(nav_box, key, label, icon)

        self._status_var = tk.StringVar(value="Ready")
        footer = tk.Frame(self.frame, bg=SIDE)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)

        tk.Label(footer, text="Status", bg=SIDE, fg=MUTED_DARK, font=ui_font(8)).pack(anchor="w")
        tk.Label(
            footer,
            textvariable=self._status_var,
            bg=SIDE,
            fg=MUTED,
            font=ui_font(9),
            wraplength=210,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(4, 0))

    def status_var(self) -> tk.StringVar:
        return self._status_var

    def _add_btn(self, parent: tk.Misc, key: str, label: str, icon: str) -> None:
        btn = tk.Button(
            parent,
            text=f"  {icon}   {label}",
            anchor="w",
            bg=SIDE,
            fg=TEXT,
            activebackground=NAV_ACTIVE,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=11,
            font=ui_font(11),
            cursor="hand2",
            command=lambda k=key: self._select(k),
        )
        btn.pack(fill=tk.X, pady=2)
        self._buttons[key] = btn

    def _select(self, key: str) -> None:
        self.set_active(key)
        self._on_select(key)

    def set_active(self, key: str) -> None:
        self._active = key
        for k, btn in self._buttons.items():
            if k == key:
                btn.configure(bg=NAV_ACTIVE, fg=ACCENT, font=ui_font(11, bold=True))
            else:
                btn.configure(bg=SIDE, fg=TEXT, font=ui_font(11))


def page_header(
    parent: tk.Misc,
    title: str,
    subtitle: str,
    *,
    badge: Optional[str] = None,
) -> ttk.Frame:
    """Title block used at the top of each page."""
    wrap = ttk.Frame(parent, style="Main.TFrame")
    wrap.pack(fill=tk.X, pady=(0, 14))

    row = ttk.Frame(wrap, style="Main.TFrame")
    row.pack(fill=tk.X)

    ttk.Label(row, text=title, style="Title.TLabel").pack(side=tk.LEFT, anchor="w")
    if badge:
        badge_lbl = tk.Label(
            row,
            text=badge,
            bg=NAV_ACTIVE,
            fg=ACCENT,
            font=ui_font(9, bold=True),
            padx=10,
            pady=4,
        )
        badge_lbl.pack(side=tk.LEFT, padx=(12, 0))

    ttk.Label(wrap, text=subtitle, style="Subtitle.TLabel").pack(anchor="w", pady=(4, 0))
    return wrap


def toolbar(parent: tk.Misc) -> ttk.Frame:
    bar = ttk.Frame(parent, style="Toolbar.TFrame")
    bar.pack(fill=tk.X, pady=(0, 12))
    return bar


def card(parent: tk.Misc, *, pad: int = 14) -> tk.Frame:
    """Elevated panel with subtle border."""
    outer = tk.Frame(parent, bg=BG)
    outer.pack(fill=tk.BOTH, expand=True)
    box = tk.Frame(outer, bg=CARD, highlightbackground=CARD_BORDER, highlightthickness=1)
    box.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
    inner = tk.Frame(box, bg=CARD)
    inner.pack(fill=tk.BOTH, expand=True, padx=pad, pady=pad)
    return inner


def stat_card(parent: tk.Misc, label: str, value: str) -> tk.Frame:
    box = tk.Frame(parent, bg=CARD, highlightbackground=CARD_BORDER, highlightthickness=1)
    box.pack(side=tk.LEFT, padx=(0, 12), pady=0)
    inner = tk.Frame(box, bg=CARD)
    inner.pack(padx=18, pady=14)
    val = tk.Label(inner, text=value, bg=CARD, fg=ACCENT, font=ui_font(22, bold=True))
    val.pack(anchor="w")
    tk.Label(inner, text=label, bg=CARD, fg=MUTED, font=ui_font(9)).pack(anchor="w", pady=(2, 0))
    return box


def info_strip(parent: tk.Misc, text_var: tk.StringVar) -> tk.Label:
    strip = tk.Frame(parent, bg=PANEL, highlightbackground=CARD_BORDER, highlightthickness=1)
    strip.pack(fill=tk.X, pady=(0, 10))
    lbl = tk.Label(
        strip,
        textvariable=text_var,
        bg=PANEL,
        fg=MUTED,
        font=ui_font(9),
        anchor="w",
        padx=12,
        pady=8,
    )
    lbl.pack(fill=tk.X)
    return lbl
