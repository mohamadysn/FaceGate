"""
Desktop application — face recognition suite.

Pages
-----
- Live recognition
- Enroll (webcam NEAR + FAR)
- Enroll from images
- Recognize image
- Gallery management
- Settings
"""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.camera_io import grab_latest_frame, open_camera, smooth_fps_update
from common.quality import assess_face_quality, is_frontal_enough

from .image_editor import ImageEditorDialog, load_bgr_images
from .platform_util import ui_font
from .services import AppServices, embed_one
from .video_widget import VideoPanel

# Visual theme
BG = "#0F172A"
SIDE = "#111827"
PANEL = "#1E293B"
ACCENT = "#38BDF8"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
OK = "#22C55E"
WARN = "#F59E0B"
DANGER = "#EF4444"


class FaceRecogApp:
    """Main desktop window with sidebar navigation."""

    def __init__(self) -> None:
        self.services = AppServices()
        self.root = tk.Tk()
        self.root.title("FaceGate")
        self.root.geometry("1180x720")
        self.root.minsize(960, 600)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._page = tk.StringVar(value="live")
        self._status = tk.StringVar(value="Ready")
        self._busy = False
        self._stop_flags = {"live": False, "enroll": False}
        self._live_thread: Optional[threading.Thread] = None
        self._enroll_thread: Optional[threading.Thread] = None
        self._cap = None
        self._engine = None

        self._build_style()
        self._build_layout()
        self._show_page("live")
        self._refresh_gallery_list()

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------ UI
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Side.TFrame", background=SIDE)
        style.configure("Main.TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure(
            "Nav.TButton",
            background=SIDE,
            foreground=TEXT,
            borderwidth=0,
            focusthickness=0,
            padding=(14, 10),
            font=ui_font(11),
        )
        style.map("Nav.TButton", background=[("active", PANEL)])
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#0F172A",
            padding=(12, 8),
            font=ui_font(10, bold=True),
        )
        style.map("Accent.TButton", background=[("active", "#7DD3FC")])
        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground=TEXT,
            padding=(10, 6),
        )
        style.configure("TLabel", background=BG, foreground=TEXT, font=ui_font(10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=ui_font(18, bold=True))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=ui_font(9))
        style.configure("Card.TLabel", background=PANEL, foreground=TEXT, font=ui_font(10))
        style.configure("Status.TLabel", background=SIDE, foreground=MUTED, font=ui_font(9))
        style.configure("TEntry", fieldbackground="#0B1220", foreground=TEXT)
        style.configure("TCombobox", fieldbackground="#0B1220", foreground=TEXT)
        style.configure(
            "Treeview",
            background=PANEL,
            foreground=TEXT,
            fieldbackground=PANEL,
            borderwidth=0,
            rowheight=28,
        )
        style.configure("Treeview.Heading", background=SIDE, foreground=TEXT, font=ui_font(10, bold=True))

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, style="Main.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)

        side = ttk.Frame(shell, style="Side.TFrame", width=220)
        side.pack(side=tk.LEFT, fill=tk.Y)
        side.pack_propagate(False)

        brand = tk.Label(
            side,
            text="FaceGate",
            bg=SIDE,
            fg=ACCENT,
            font=ui_font(14, bold=True),
            pady=18,
        )
        brand.pack(fill=tk.X, padx=12)

        nav_items = [
            ("live", "Live recognition"),
            ("enroll", "Camera enrollment"),
            ("enroll_img", "Photo enrollment"),
            ("recognize_img", "Recognize image"),
            ("gallery", "Gallery"),
            ("settings", "Settings"),
        ]
        for key, label in nav_items:
            btn = tk.Button(
                side,
                text=label,
                anchor="w",
                bg=SIDE,
                fg=TEXT,
                activebackground=PANEL,
                activeforeground=ACCENT,
                relief="flat",
                bd=0,
                padx=16,
                pady=10,
                font=ui_font(11),
                cursor="hand2",
                command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill=tk.X, padx=8, pady=2)

        tk.Label(
            side,
            textvariable=self._status,
            bg=SIDE,
            fg=MUTED,
            font=ui_font(9),
            wraplength=190,
            justify="left",
            anchor="sw",
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=12)

        self.content = ttk.Frame(shell, style="Main.TFrame")
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.pages = {}
        self.pages["live"] = self._build_live_page(self.content)
        self.pages["enroll"] = self._build_enroll_page(self.content)
        self.pages["enroll_img"] = self._build_enroll_img_page(self.content)
        self.pages["recognize_img"] = self._build_recognize_img_page(self.content)
        self.pages["gallery"] = self._build_gallery_page(self.content)
        self.pages["settings"] = self._build_settings_page(self.content)

    def _show_page(self, name: str) -> None:
        if name != "live":
            self._stop_live()
        if name != "enroll":
            self._stop_enroll()
        self._page.set(name)
        for key, frame in self.pages.items():
            if key == name:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()
        if name == "gallery":
            self._refresh_gallery_list()
        if name == "enroll_img":
            self._refresh_enroll_name_choices()

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self._status.set(text))

    # -------------------------------------------------------------- Live
    def _build_live_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Live recognition", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text="Detect every frame · identity in the background",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        controls = ttk.Frame(page, style="Main.TFrame")
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(controls, text="▶ Start", style="Accent.TButton", command=self._start_live).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(controls, text="■ Stop", command=self._stop_live).pack(side=tk.LEFT)

        self.live_info = tk.StringVar(value="Press Start")
        ttk.Label(controls, textvariable=self.live_info, style="Muted.TLabel").pack(side=tk.LEFT, padx=16)

        self.live_video = VideoPanel(page)
        self.live_video.pack(fill=tk.BOTH, expand=True)
        return page

    def _start_live(self) -> None:
        if self._live_thread and self._live_thread.is_alive():
            return
        gallery = self.services.gallery()
        if len(gallery) == 0:
            messagebox.showwarning("Empty gallery", "Enroll someone first (camera or photos).")
            return
        self._stop_flags["live"] = False
        self._set_status("Loading model…")
        self._live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self._live_thread.start()

    def _stop_live(self) -> None:
        self._stop_flags["live"] = True

    def _live_loop(self) -> None:
        cap = None
        engine = None
        try:
            profile = self.services.profile
            engine = self.services.make_engine()
            engine.start()
            self._engine = engine
            cap = open_camera(
                self.services.camera_index,
                profile.width,
                profile.height,
                fps=30,
            )
            self._cap = cap
            self._set_status(f"Live · {profile.name} · galerie {len(self.services.gallery())}")
            fps = 0.0
            prev = time.time()
            while not self._stop_flags["live"]:
                ok, frame = grab_latest_frame(cap, flush=1)
                if not ok or frame is None:
                    break
                if self.services.mirror:
                    frame = cv2.flip(frame, 1)
                tracks = engine.update_from_frame(frame)
                engine.submit_frame(frame)
                stats = engine.get_stats()
                fps, prev = smooth_fps_update(fps, prev)
                for tr in tracks:
                    x1, y1, x2, y2 = tr.bbox
                    if tr.identity is not None:
                        color = (0, 220, 0)
                        label = f"{tr.identity} ({tr.identity_score:.2f})"
                    elif tr.pending_name and tr.pending_hits > 0:
                        color = (0, 200, 255)
                        label = f"... {tr.pending_name}?"
                    else:
                        color = (0, 165, 255)
                        label = f"Unknown ({tr.identity_score:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                info = (
                    f"FPS {fps:.1f}  ·  det {stats.det_ms:.0f} ms  ·  "
                    f"ID {stats.infer_ms:.0f} ms  ·  faces {len(tracks)}"
                )
                self.root.after(0, lambda f=frame.copy(), i=info: self._update_live_ui(f, i))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Live error", str(exc)))
            self._set_status(f"Error: {exc}")
        finally:
            if engine is not None:
                engine.stop()
            if cap is not None:
                cap.release()
            self._engine = None
            self._cap = None
            self.root.after(0, lambda: self.live_video.clear("Stopped"))
            self._set_status("Ready")

    def _update_live_ui(self, frame, info: str) -> None:
        self.live_video.show_bgr(frame)
        self.live_info.set(info)

    # ------------------------------------------------------------- Enroll cam
    def _build_enroll_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Camera enrollment", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text="NEAR then FAR phases — face the camera",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(page, style="Main.TFrame")
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Name:").pack(side=tk.LEFT)
        self.enroll_name = tk.StringVar()
        ttk.Entry(row, textvariable=self.enroll_name, width=28).pack(side=tk.LEFT, padx=8)
        ttk.Button(row, text="▶ Enroll", style="Accent.TButton", command=self._start_enroll).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(row, text="■ Cancel", command=self._stop_enroll).pack(side=tk.LEFT)

        self.enroll_info = tk.StringVar(value="Enter a name, then start")
        ttk.Label(page, textvariable=self.enroll_info, style="Muted.TLabel").pack(anchor="w", pady=(0, 6))

        self.enroll_video = VideoPanel(page)
        self.enroll_video.pack(fill=tk.BOTH, expand=True)
        return page

    def _start_enroll(self) -> None:
        name = self.enroll_name.get().strip()
        if not name:
            messagebox.showwarning("Name required", "Enter the person's name.")
            return
        if self._enroll_thread and self._enroll_thread.is_alive():
            return
        self._stop_live()
        self._stop_flags["enroll"] = False
        self._set_status("Enrolling…")
        self._enroll_thread = threading.Thread(target=self._enroll_loop, args=(name,), daemon=True)
        self._enroll_thread.start()

    def _stop_enroll(self) -> None:
        self._stop_flags["enroll"] = True

    def _enroll_loop(self, name: str) -> None:
        cap = None
        try:
            recognizer, detector = self.services.ensure_models(det_size=320)
            # Light detector for smooth preview
            from common.face_detector import FaceDetector

            detector = FaceDetector(
                backend="insightface",
                model_name="buffalo_l",
                provider=self.services.provider,
                min_score=0.45,
                det_size=320,
            )
            cap = open_camera(self.services.camera_index, 640, 480, fps=30)
            near_emb, near_w = self._capture_phase(
                cap, detector, recognizer, name, "NEAR", 12, 90, 30.0, 0.45,
                "NEAR: move closer, keep face sharp",
            )
            if self._stop_flags["enroll"]:
                raise KeyboardInterrupt()
            far_emb, far_w = self._capture_phase(
                cap, detector, recognizer, name, "FAR", 12, 40, 12.0, 0.35,
                "FAR: step back until the face is small",
            )
            embeddings = near_emb + far_emb
            weights = near_w + far_w
            gallery = self.services.gallery()
            entry = gallery.enroll(name, embeddings, replace=True, weights=weights)
            self.services.invalidate_models()
            msg = (
                f'Enrolled "{entry.name}" ({entry.n_samples} samples, '
                f"near={len(near_emb)} far={len(far_emb)})"
            )
            self.root.after(0, lambda: messagebox.showinfo("Success", msg))
            self._set_status(msg)
        except KeyboardInterrupt:
            self._set_status("Enrollment cancelled")
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Enrollment error", str(exc)))
            self._set_status(f"Error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.root.after(0, lambda: self.enroll_video.clear("Ready"))
            self.root.after(0, self._refresh_gallery_list)

    def _capture_phase(
        self,
        cap,
        detector,
        recognizer,
        name: str,
        phase: str,
        needed: int,
        min_face_px: int,
        min_sharpness: float,
        min_score: float,
        hint: str,
    ):
        embeddings: List = []
        weights: List[float] = []
        rejected = 0
        last_capture = 0.0
        refine_small = phase == "FAR"
        while len(embeddings) < needed and not self._stop_flags["enroll"]:
            ok, frame = grab_latest_frame(cap, flush=1)
            if not ok or frame is None:
                break
            if self.services.mirror:
                frame = cv2.flip(frame, 1)
            detections = detector.detect(frame, max_side=480)
            status = "no face"
            color = (0, 165, 255)
            good = False
            if detections:
                det = detections[0]
                x1, y1, x2, y2 = det.bbox
                short = min(x2 - x1, y2 - y1)
                report = assess_face_quality(
                    frame,
                    det,
                    min_face_px=min_face_px,
                    min_det_score=min_score,
                    min_sharpness=min_sharpness,
                )
                frontal = is_frontal_enough(det)
                good = report.ok and frontal
                if phase == "FAR" and short > 180:
                    good = False
                    status = "move farther (face still too large)"
                elif phase == "NEAR" and short < 100:
                    good = False
                    status = "move closer (face too small)"
                elif good:
                    status = f"READY {short}px — capturing…"
                    color = (0, 255, 0)
                else:
                    status = f"reject:{report.reason if not report.ok else 'not_frontal'}"
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 0) if good else (0, 165, 255),
                    2,
                )
                now = time.time()
                if good and (now - last_capture) >= 0.35:
                    face = embed_one(recognizer, frame, refine_small=refine_small)
                    if face is not None and face.embedding is not None:
                        q = assess_face_quality(
                            frame,
                            face,
                            min_face_px=min_face_px,
                            min_det_score=min_score,
                            min_sharpness=min_sharpness,
                        )
                        if q.ok and is_frontal_enough(face):
                            embeddings.append(face.embedding.copy())
                            weights.append(max(0.05, q.score))
                            last_capture = now
                            status = f"SAVED {len(embeddings)}/{needed}"
                            color = (0, 255, 0)
                        else:
                            rejected += 1
                    else:
                        rejected += 1
            hud = f"{name} [{phase}]  {len(embeddings)}/{needed}  |  {hint}  |  {status}"
            cv2.putText(frame, hud[:90], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            info = f"{phase}: {len(embeddings)}/{needed}  rejected={rejected}  — {status}"
            self.root.after(0, lambda f=frame.copy(), i=info: self._update_enroll_ui(f, i))
        if self._stop_flags["enroll"]:
            raise KeyboardInterrupt()
        return embeddings, weights

    def _update_enroll_ui(self, frame, info: str) -> None:
        self.enroll_video.show_bgr(frame)
        self.enroll_info.set(info)

    # -------------------------------------------------------- Enroll images
    def _build_enroll_img_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Photo enrollment", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text="New person, or add photos to someone already in the gallery",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(page, style="Main.TFrame")
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Name:").pack(side=tk.LEFT)
        self.img_enroll_name = tk.StringVar()
        self.img_enroll_name_combo = ttk.Combobox(
            row, textvariable=self.img_enroll_name, width=26, values=[]
        )
        self.img_enroll_name_combo.pack(side=tk.LEFT, padx=8)
        ttk.Button(row, text="↻ names", command=self._refresh_enroll_name_choices).pack(side=tk.LEFT)

        mode_row = ttk.Frame(page, style="Main.TFrame")
        mode_row.pack(fill=tk.X, pady=4)
        self.img_enroll_mode = tk.StringVar(value="merge")
        ttk.Radiobutton(
            mode_row,
            text="Add to person (if already enrolled)",
            variable=self.img_enroll_mode,
            value="merge",
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Radiobutton(
            mode_row,
            text="Replace completely",
            variable=self.img_enroll_mode,
            value="replace",
        ).pack(side=tk.LEFT)

        row2 = ttk.Frame(page, style="Main.TFrame")
        row2.pack(fill=tk.X, pady=4)
        ttk.Button(row2, text="Choose images…", command=self._pick_enroll_images).pack(side=tk.LEFT)
        ttk.Button(row2, text="Edit (crop / zoom)…", command=self._edit_enroll_images).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(row2, text="Enroll", style="Accent.TButton", command=self._run_enroll_images).pack(
            side=tk.LEFT, padx=8
        )

        self.img_enroll_files: List[str] = []
        self.img_enroll_edited: List = []
        self.img_enroll_info = tk.StringVar(value="No image selected")
        ttk.Label(page, textvariable=self.img_enroll_info, style="Muted.TLabel").pack(anchor="w", pady=8)

        self.img_enroll_preview = VideoPanel(page)
        self.img_enroll_preview.pack(fill=tk.BOTH, expand=True)
        self._refresh_enroll_name_choices()
        return page

    def _refresh_enroll_name_choices(self) -> None:
        names = self.services.gallery().names()
        if hasattr(self, "img_enroll_name_combo"):
            self.img_enroll_name_combo.configure(values=names)

    def _prepare_add_photos_for(self, name: str) -> None:
        """Jump to photo enroll with an existing gallery identity selected (merge mode)."""
        self.img_enroll_name.set(name)
        self.img_enroll_mode.set("merge")
        self._refresh_enroll_name_choices()
        self._show_page("enroll_img")
        self._set_status(f"Adding photos for \"{name}\"")

    def _pick_enroll_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Images for enrollment",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.img_enroll_files = list(paths)
        self.img_enroll_edited = []
        loaded = load_bgr_images(self.img_enroll_files)
        if not loaded:
            messagebox.showerror("Images", "Could not read the selected files.")
            return
        self._open_image_editor(loaded)

    def _edit_enroll_images(self) -> None:
        if self.img_enroll_edited:
            sources = [(e.label, e.bgr) for e in self.img_enroll_edited]
        elif self.img_enroll_files:
            sources = load_bgr_images(self.img_enroll_files)
        else:
            messagebox.showwarning("Images", "Choose images first.")
            return
        if not sources:
            messagebox.showerror("Images", "No readable images.")
            return
        self._open_image_editor(sources)

    def _open_image_editor(self, sources) -> None:
        dialog = ImageEditorDialog(self.root, sources)
        self.root.wait_window(dialog)
        if not dialog.result:
            if not self.img_enroll_edited and self.img_enroll_files:
                preview = cv2.imread(self.img_enroll_files[0])
                if preview is not None:
                    self.img_enroll_preview.show_bgr(preview)
                self.img_enroll_info.set(
                    f"{len(self.img_enroll_files)} image(s) — click Edit for crop / zoom"
                )
            return
        self.img_enroll_edited = dialog.result
        self.img_enroll_info.set(
            f"{len(self.img_enroll_edited)} image(s) ready (edited) — re-edit or Enroll"
        )
        self.img_enroll_preview.show_bgr(self.img_enroll_edited[0].bgr)
        self._set_status(f"{len(self.img_enroll_edited)} photo(s) prepared")

    def _run_enroll_images(self) -> None:
        name = self.img_enroll_name.get().strip()
        if not name:
            messagebox.showwarning("Name required", "Enter the person's name (or pick it from the list).")
            return
        if self.img_enroll_edited:
            frames = [(e.label, e.bgr) for e in self.img_enroll_edited]
        elif self.img_enroll_files:
            frames = load_bgr_images(self.img_enroll_files)
        else:
            messagebox.showwarning("Images", "Choose at least one image.")
            return
        if not frames:
            messagebox.showwarning("Images", "No usable images.")
            return

        merge = self.img_enroll_mode.get() == "merge"
        gallery = self.services.gallery()
        exists = name in gallery.names()
        if exists and merge:
            confirm = messagebox.askyesno(
                "Add photos",
                f'"{name}" is already in the gallery '
                f"({next(e.n_samples for e in gallery.entries if e.name == name)} samples).\n\n"
                "Add these new images to their profile?",
            )
            if not confirm:
                return
        elif exists and not merge:
            confirm = messagebox.askyesno(
                "Replace",
                f'"{name}" already exists.\n\nReplace their profile completely with these images?',
            )
            if not confirm:
                return

        def work():
            try:
                self._set_status("Enrolling from photos…")
                n, rejected = self.services.enroll_from_frames(name, frames, merge=merge)
                if n == 0:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Failed",
                            "No usable face.\n" + "\n".join(rejected[:8]),
                        ),
                    )
                    return
                g = self.services.gallery()
                total = next((e.n_samples for e in g.entries if e.name == name), n)
                action = "updated (merged)" if merge and exists else "enrolled"
                msg = f'"{name}" {action} with {n} image(s). Total samples: {total}.'
                if rejected:
                    msg += f"\n{len(rejected)} rejected."
                self.root.after(0, lambda: messagebox.showinfo("Success", msg))
                self._set_status(msg.split(".")[0])
                self.root.after(0, self._refresh_gallery_list)
                self.root.after(0, self._refresh_enroll_name_choices)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    # ----------------------------------------------------- Recognize image
    def _build_recognize_img_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Recognize an image", style="Title.TLabel").pack(anchor="w")
        ttk.Label(page, text="Match a photo against the gallery", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        row = ttk.Frame(page, style="Main.TFrame")
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="Open image…", command=self._pick_recognize_image).pack(side=tk.LEFT)
        ttk.Button(row, text="Analyze", style="Accent.TButton", command=self._run_recognize_image).pack(
            side=tk.LEFT, padx=8
        )

        self.recognize_path: Optional[str] = None
        self.recognize_info = tk.StringVar(value="No image")
        ttk.Label(page, textvariable=self.recognize_info, style="Muted.TLabel").pack(anchor="w", pady=8)

        self.recognize_video = VideoPanel(page)
        self.recognize_video.pack(fill=tk.BOTH, expand=True)
        return page

    def _pick_recognize_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Image to recognize",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.recognize_path = path
        self.recognize_info.set(Path(path).name)
        frame = cv2.imread(path)
        if frame is not None:
            self.recognize_video.show_bgr(frame)

    def _run_recognize_image(self) -> None:
        if not self.recognize_path:
            messagebox.showwarning("Image", "Open an image first.")
            return

        def work():
            try:
                self._set_status("Analyse image…")
                annotated, labels = self.services.recognize_image(Path(self.recognize_path))
                text = " · ".join(labels) if labels else "No face"
                self.root.after(0, lambda: self.recognize_video.show_bgr(annotated))
                self.root.after(0, lambda: self.recognize_info.set(text))
                self._set_status(text)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------- Gallery
    def _build_gallery_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Identity gallery", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text="Stored embeddings (no photos) — gallery.json + embeddings.npy",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        row = ttk.Frame(page, style="Main.TFrame")
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="Refresh", command=self._refresh_gallery_list).pack(side=tk.LEFT)
        ttk.Button(
            row,
            text="Add photos…",
            style="Accent.TButton",
            command=self._gallery_add_photos,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(row, text="Delete selected", style="Danger.TButton", command=self._delete_selected).pack(
            side=tk.LEFT, padx=8
        )

        self.gallery_path_label = tk.StringVar()
        ttk.Label(page, textvariable=self.gallery_path_label, style="Muted.TLabel").pack(anchor="w", pady=6)

        cols = ("name", "samples")
        self.gallery_tree = ttk.Treeview(page, columns=cols, show="headings", height=16)
        self.gallery_tree.heading("name", text="Name")
        self.gallery_tree.heading("samples", text="Samples")
        self.gallery_tree.column("name", width=280)
        self.gallery_tree.column("samples", width=120)
        self.gallery_tree.pack(fill=tk.BOTH, expand=True)
        return page

    def _refresh_gallery_list(self) -> None:
        if not hasattr(self, "gallery_tree"):
            return
        for item in self.gallery_tree.get_children():
            self.gallery_tree.delete(item)
        gallery = self.services.gallery()
        self.gallery_path_label.set(f"Path: {gallery.root}  ·  {len(gallery)} identities")
        for entry in gallery.entries:
            self.gallery_tree.insert("", tk.END, values=(entry.name, entry.n_samples))

    def _gallery_add_photos(self) -> None:
        sel = self.gallery_tree.selection()
        if not sel:
            messagebox.showinfo(
                "Gallery",
                "Select a person, then Add photos…\n"
                "Or open Photo enrollment and pick their name from the list.",
            )
            return
        name = self.gallery_tree.item(sel[0], "values")[0]
        self._prepare_add_photos_for(name)

    def _delete_selected(self) -> None:
        sel = self.gallery_tree.selection()
        if not sel:
            return
        name = self.gallery_tree.item(sel[0], "values")[0]
        if not messagebox.askyesno("Confirm", f'Delete "{name}" from the gallery?'):
            return
        if self.services.remove_identity(name):
            self._refresh_gallery_list()
            self._set_status(f"Deleted: {name}")

    # ------------------------------------------------------------- Settings
    def _build_settings_page(self, parent) -> ttk.Frame:
        page = ttk.Frame(parent, style="Main.TFrame")
        ttk.Label(page, text="Settings", style="Title.TLabel").pack(anchor="w")
        ttk.Label(page, text="Profile, camera, ONNX provider", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 12)
        )

        grid = ttk.Frame(page, style="Main.TFrame")
        grid.pack(anchor="w")

        self.var_profile = tk.StringVar(value=self.services.profile_name)
        self.var_provider = tk.StringVar(value=self.services.provider)
        self.var_camera = tk.IntVar(value=self.services.camera_index)
        self.var_mirror = tk.BooleanVar(value=self.services.mirror)
        self.var_max_faces = tk.IntVar(value=self.services.max_faces)
        self.var_threshold = tk.StringVar(value="")

        def row(r, label, widget):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=6, padx=(0, 12))
            widget.grid(row=r, column=1, sticky="w", pady=6)

        row(0, "Profile", ttk.Combobox(grid, textvariable=self.var_profile, values=["fast", "balanced", "accurate"], width=18, state="readonly"))
        row(1, "Provider", ttk.Combobox(grid, textvariable=self.var_provider, values=["CPUExecutionProvider", "CUDAExecutionProvider"], width=24, state="readonly"))
        row(2, "Camera index", ttk.Spinbox(grid, from_=0, to=8, textvariable=self.var_camera, width=8))
        row(3, "Mirror", ttk.Checkbutton(grid, variable=self.var_mirror))
        row(4, "Max faces", ttk.Spinbox(grid, from_=1, to=10, textvariable=self.var_max_faces, width=8))
        row(5, "Threshold (empty = profile)", ttk.Entry(grid, textvariable=self.var_threshold, width=10))

        ttk.Button(page, text="Save settings", style="Accent.TButton", command=self._save_settings).pack(
            anchor="w", pady=16
        )
        return page

    def _save_settings(self) -> None:
        self.services.profile_name = self.var_profile.get()
        self.services.provider = self.var_provider.get()
        self.services.camera_index = int(self.var_camera.get())
        self.services.mirror = bool(self.var_mirror.get())
        self.services.max_faces = int(self.var_max_faces.get())
        raw = self.var_threshold.get().strip()
        if raw:
            try:
                self.services.threshold_override = float(raw)
            except ValueError:
                messagebox.showerror("Threshold", "Threshold must be a number (e.g. 0.42).")
                return
        else:
            self.services.threshold_override = None
        self.services.invalidate_models()
        self._set_status(
            f"Settings OK · {self.services.profile_name} · cam {self.services.camera_index}"
        )
        messagebox.showinfo("Settings", "Settings saved.")

    # --------------------------------------------------------------- Close
    def _on_close(self) -> None:
        self._stop_live()
        self._stop_enroll()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        self.root.destroy()
