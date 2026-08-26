"""
Face recognition gallery + matcher (FAIR: Findable / Interoperable / Reusable).

Workflow
--------
1. Enroll: capture several embeddings per person and average them.
2. Store gallery as JSON metadata + ``.npz`` embedding matrix.
3. Live match: cosine similarity against the gallery; accept above a threshold.

Artifact layout (Findable)
--------------------------
::

    gallery/
      gallery.json      # names, thresholds, embedding dim
      embeddings.npy    # shape (N, D), L2-normalized rows
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .face_detector import clip_bbox, eye_rois_from_landmarks
from .types import EyeSample, FaceSample

PathLike = Union[str, Path]

# Typical working threshold for InsightFace buffalo_l cosine similarity.
DEFAULT_MATCH_THRESHOLD = 0.35


def l2_normalize(vectors: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize vectors along ``axis`` (safe for zero rows)."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings (already L2-normalized preferred)."""
    a = l2_normalize(np.asarray(a, dtype=np.float32).reshape(1, -1))[0]
    b = l2_normalize(np.asarray(b, dtype=np.float32).reshape(1, -1))[0]
    return float(np.dot(a, b))


@dataclass
class GalleryEntry:
    """One enrolled identity."""

    name: str
    embedding: np.ndarray  # (D,)
    n_samples: int = 1


class FaceGallery:
    """
    Persistent set of known face embeddings.

    Parameters
    ----------
    path :
        Directory that will hold ``gallery.json`` and ``embeddings.npy``.
    """

    def __init__(self, path: PathLike) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: List[GalleryEntry] = []
        self.match_threshold = DEFAULT_MATCH_THRESHOLD
        self._load_if_exists()

    @property
    def json_path(self) -> Path:
        return self.root / "gallery.json"

    @property
    def npy_path(self) -> Path:
        return self.root / "embeddings.npy"

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> List[str]:
        """Return enrolled identity names in gallery order."""
        return [entry.name for entry in self.entries]

    def _load_if_exists(self) -> None:
        if not self.json_path.is_file() or not self.npy_path.is_file():
            return
        meta = json.loads(self.json_path.read_text(encoding="utf-8"))
        matrix = np.load(self.npy_path)
        self.match_threshold = float(meta.get("match_threshold", DEFAULT_MATCH_THRESHOLD))
        people = meta.get("people", [])
        if len(people) != len(matrix):
            raise ValueError(
                f"Gallery corrupt: {len(people)} names vs {len(matrix)} embeddings in {self.root}"
            )
        self.entries = [
            GalleryEntry(
                name=str(person["name"]),
                embedding=np.asarray(matrix[i], dtype=np.float32),
                n_samples=int(person.get("n_samples", 1)),
            )
            for i, person in enumerate(people)
        ]

    def save(self) -> None:
        """Write JSON metadata + embedding matrix to disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.entries:
            matrix = np.stack([e.embedding for e in self.entries], axis=0)
        else:
            matrix = np.zeros((0, 512), dtype=np.float32)
        np.save(self.npy_path, matrix.astype(np.float32))
        payload = {
            "match_threshold": self.match_threshold,
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 512,
            "people": [
                {"name": e.name, "n_samples": e.n_samples, "index": i}
                for i, e in enumerate(self.entries)
            ],
        }
        self.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def enroll(
        self,
        name: str,
        embeddings: Sequence[np.ndarray],
        replace: bool = True,
        weights: Optional[Sequence[float]] = None,
        merge: bool = False,
    ) -> GalleryEntry:
        """
        Add or update a person from one or more embeddings.

        Parameters
        ----------
        name :
            Display identity (case-sensitive as provided).
        embeddings :
            Raw or normalized vectors; quality-weighted then L2-normalized.
        replace :
            If True and ``merge`` is False, overwrite an existing identity.
        weights :
            Optional per-sample quality weights (same length as embeddings).
        merge :
            If True and the identity already exists, blend the new mean into
            the stored embedding (weighted by sample counts) instead of
            replacing it. Useful to add more photos to someone already enrolled.
        """
        name = name.strip()
        if not name:
            raise ValueError("Identity name must be non-empty.")
        if not embeddings:
            raise ValueError("Need at least one embedding to enroll.")

        stacked = l2_normalize(
            np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in embeddings], axis=0)
        )
        if weights is None:
            w = np.ones((stacked.shape[0],), dtype=np.float32)
        else:
            w = np.asarray(weights, dtype=np.float32).reshape(-1)
            if w.shape[0] != stacked.shape[0]:
                raise ValueError("weights length must match embeddings")
            w = np.maximum(w, 1e-3)
        w = w / float(w.sum())
        mean_vec = l2_normalize((stacked * w[:, None]).sum(axis=0, keepdims=True))[0]
        n_new = len(embeddings)
        entry = GalleryEntry(name=name, embedding=mean_vec, n_samples=n_new)

        existing = next((i for i, e in enumerate(self.entries) if e.name == name), None)
        if existing is not None:
            if merge:
                old = self.entries[existing]
                total = max(1, int(old.n_samples) + n_new)
                blended = l2_normalize(
                    (
                        old.embedding.astype(np.float32) * float(old.n_samples)
                        + mean_vec.astype(np.float32) * float(n_new)
                    ).reshape(1, -1)
                )[0]
                entry = GalleryEntry(name=name, embedding=blended, n_samples=total)
                self.entries[existing] = entry
            elif not replace:
                raise ValueError(f"Identity already enrolled: {name}")
            else:
                self.entries[existing] = entry
        else:
            self.entries.append(entry)

        self.save()
        return entry

    def remove(self, name: str) -> bool:
        """Remove an identity by name. Returns True if something was deleted."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.name != name]
        if len(self.entries) != before:
            self.save()
            return True
        return False

    def match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        margin: float = 0.05,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best gallery match for ``embedding``.

        Uses a margin check against the 2nd best score to reduce false accepts
        when two identities are close.

        Returns
        -------
        name, score
            Matched name (or ``None``) and best cosine similarity.
        """
        if not self.entries:
            return None, -1.0

        thr = self.match_threshold if threshold is None else float(threshold)
        query = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        matrix = np.stack([e.embedding for e in self.entries], axis=0)
        scores = matrix @ query
        order = np.argsort(scores)[::-1]
        best_idx = int(order[0])
        best_score = float(scores[best_idx])
        second = float(scores[int(order[1])]) if len(order) > 1 else -1.0

        if best_score < thr:
            return None, best_score
        if len(order) > 1 and (best_score - second) < float(margin):
            # Ambiguous: two people too close -> Unknown is safer.
            return None, best_score
        return self.entries[best_idx].name, best_score


class FaceRecognizer:
    """
    InsightFace detector + recognizer that also fills ``FaceSample.embedding``.

    Parameters
    ----------
    gallery :
        Optional :class:`FaceGallery` used to assign ``identity`` labels.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        provider: str = "CPUExecutionProvider",
        device: int = 0,
        det_size: int = 320,
        min_score: float = 0.5,
        gallery: Optional[FaceGallery] = None,
        match_threshold: Optional[float] = None,
        match_margin: float = 0.05,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required for face recognition. "
                "Install with: pip install insightface onnxruntime"
            ) from exc

        # Limit ORT CPU oversubscription which often hurts latency.
        try:
            import os

            os.environ.setdefault("OMP_NUM_THREADS", "4")
            os.environ.setdefault("ORT_NUM_THREADS", "4")
        except Exception:
            pass

        providers = [provider]
        if "CUDA" in provider.upper():
            providers.append("CPUExecutionProvider")

        # detection + recognition (embedding) modules
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        size = None if det_size <= 0 else (det_size, det_size)
        ctx_id = int(device) if "CUDA" in provider.upper() else -1
        self._app.prepare(ctx_id=ctx_id, det_size=size, det_thresh=min_score)
        self.min_score = min_score
        self.gallery = gallery
        self.match_threshold = match_threshold
        self.match_margin = float(match_margin)
        self.det_size = det_size
        self.model_name = model_name

    def _build_sample(
        self,
        face,
        scale: float,
        frame_shape,
        identify: bool,
    ) -> Optional[FaceSample]:
        """Convert one InsightFace result into a ``FaceSample`` in full-frame coords."""
        score = float(face.det_score)
        if score < self.min_score:
            return None
        x1, y1, x2, y2 = (face.bbox.astype(np.float32) * scale).astype(int)
        box = clip_bbox(x1, y1, x2, y2, frame_shape[1], frame_shape[0])
        if box is None:
            return None

        landmarks = None
        eyes: List[EyeSample] = []
        if getattr(face, "kps", None) is not None:
            landmarks = np.asarray(face.kps, dtype=np.float32) * scale
            for side, ebox, landmark in eye_rois_from_landmarks(landmarks, frame_shape):
                eyes.append(EyeSample(side=side, bbox=ebox, landmark=landmark))

        embedding = None
        if getattr(face, "embedding", None) is not None:
            embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32).reshape(1, -1))[0]

        identity = None
        identity_score = 0.0
        if identify and self.gallery is not None and embedding is not None:
            identity, identity_score = self.gallery.match(
                embedding,
                threshold=self.match_threshold,
                margin=self.match_margin,
            )

        return FaceSample(
            bbox=box,
            score=score,
            landmarks=landmarks,
            eyes=eyes,
            embedding=embedding,
            identity=identity,
            identity_score=float(identity_score),
        )

    def _analyze_once(
        self,
        frame: np.ndarray,
        identify: bool,
        max_side: int,
    ) -> List[FaceSample]:
        from .camera_io import resize_for_inference

        infer_frame, scale = resize_for_inference(frame, max_side=max_side)
        faces: List[FaceSample] = []
        for face in self._app.get(infer_frame):
            sample = self._build_sample(face, scale, frame.shape, identify)
            if sample is not None:
                faces.append(sample)
        faces.sort(key=lambda item: item.score, reverse=True)
        return faces

    def _refine_small_face(
        self,
        frame: np.ndarray,
        face: FaceSample,
        identify: bool,
        target_face_px: int = 180,
    ) -> FaceSample:
        """
        Re-run recognition on an upscaled crop around a small/far face.

        Distant faces produce weak embeddings on the full frame; zooming the
        ROI before InsightFace markedly improves match scores.
        """
        import cv2

        x1, y1, x2, y2 = face.bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        short = float(min(fw, fh))
        if short >= target_face_px:
            return face

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        # Context around the face helps alignment landmarks.
        side = max(fw, fh) * 1.8
        half = side * 0.5
        xx1 = int(max(0, cx - half))
        yy1 = int(max(0, cy - half))
        xx2 = int(min(frame.shape[1], cx + half))
        yy2 = int(min(frame.shape[0], cy + half))
        crop = frame[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            return face

        scale_up = float(target_face_px) / max(1.0, short)
        scale_up = float(np.clip(scale_up, 1.25, 4.0))
        big = cv2.resize(
            crop,
            (max(32, int(crop.shape[1] * scale_up)), max(32, int(crop.shape[0] * scale_up))),
            interpolation=cv2.INTER_CUBIC,
        )

        local = self._app.get(big)
        if not local:
            return face

        best = max(local, key=lambda item: float(item.det_score))
        # Map local bbox back to full frame.
        inv = 1.0 / scale_up
        bx1 = int(best.bbox[0] * inv + xx1)
        by1 = int(best.bbox[1] * inv + yy1)
        bx2 = int(best.bbox[2] * inv + xx1)
        by2 = int(best.bbox[3] * inv + yy1)
        box = clip_bbox(bx1, by1, bx2, by2, frame.shape[1], frame.shape[0])
        if box is None:
            return face

        embedding = None
        if getattr(best, "embedding", None) is not None:
            embedding = l2_normalize(np.asarray(best.embedding, dtype=np.float32).reshape(1, -1))[0]

        identity = face.identity
        identity_score = face.identity_score
        if identify and self.gallery is not None and embedding is not None:
            # Keep the same strict threshold (no softening) — wrong IDs are worse.
            identity, identity_score = self.gallery.match(
                embedding,
                threshold=self.match_threshold,
                margin=self.match_margin,
            )

        landmarks = face.landmarks
        eyes = face.eyes
        if getattr(best, "kps", None) is not None:
            landmarks = np.asarray(best.kps, dtype=np.float32) * inv
            landmarks[:, 0] += xx1
            landmarks[:, 1] += yy1
            eyes = []
            for side, ebox, landmark in eye_rois_from_landmarks(landmarks, frame.shape):
                eyes.append(EyeSample(side=side, bbox=ebox, landmark=landmark))

        return FaceSample(
            bbox=box,
            score=max(face.score, float(best.det_score)),
            landmarks=landmarks,
            eyes=eyes,
            embedding=embedding if embedding is not None else face.embedding,
            identity=identity,
            identity_score=float(identity_score),
        )

    def analyze(
        self,
        frame: np.ndarray,
        identify: bool = True,
        max_side: int = 640,
        refine_small: bool = True,
        small_face_px: int = 140,
    ) -> List[FaceSample]:
        """
        Detect faces, extract embeddings, optionally match the gallery.

        For far / small faces, a second zoomed pass improves recognition.

        Parameters
        ----------
        frame :
            BGR image (full display resolution).
        identify :
            If True and a gallery is set, fill ``identity`` / ``identity_score``.
        max_side :
            Longest side used for the first inference pass.
        refine_small :
            If True, re-embed faces smaller than ``small_face_px`` via zoom crop.
        small_face_px :
            Short-side threshold that triggers zoom refinement.
        """
        faces = self._analyze_once(frame, identify=identify, max_side=max_side)

        # If nothing found, retry at higher resolution (helps distant faces).
        if not faces and max_side < 800:
            faces = self._analyze_once(frame, identify=identify, max_side=800)

        # Zoom refine ONLY for genuinely small faces (not for every Unknown).
        # Previously `or identity is None` forced a 2nd InsightFace pass on every
        # close-up Unknown face and blew latency to multi-second on CPU.
        if refine_small:
            out: List[FaceSample] = []
            for face in faces:
                short = min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
                if short < small_face_px:
                    out.append(self._refine_small_face(frame, face, identify=identify))
                else:
                    out.append(face)
            faces = out
            faces.sort(key=lambda item: item.score, reverse=True)

        return faces
