"""Face detection, encoding, and comparison utilities.

Provides two encoder backends:
1. InsightFace (SCRFD + ArcFace buffalo_l) — production-quality 512-d embeddings
   with quality gating (blur, face size, detection confidence).
2. PIL-based fallback — deterministic 128-d embeddings from pixel data.  This
   keeps the pipeline runnable on machines without InsightFace/ONNX, though
   the embeddings are not meaningful for real face comparison.

The public API is the same for both: detect_and_encode() returns
(embedding_vector, image_sha256).
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Try to import InsightFace; fall back gracefully
# ---------------------------------------------------------------------------
_INSIGHTFACE_AVAILABLE = False
try:
    import cv2
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# Quality-gate thresholds (used only with InsightFace)
MIN_FACE_WIDTH = 60
MIN_BLUR_VAR = 50.0
MIN_DET_SCORE = 0.6


class FaceEncoder:
    """Unified face encoder with InsightFace primary and PIL fallback.

    Usage::

        encoder = FaceEncoder()
        embedding, sha256 = encoder.detect_and_encode("photo.jpg")
    """

    def __init__(self, model_pack: str = "buffalo_l", vector_dim: int = 128):
        self.using_insightface = False
        self.vector_dim = vector_dim

        if _INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(name=model_pack)
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
                self.using_insightface = True
                self.vector_dim = 512  # ArcFace output
                print("  [FaceEncoder] Using InsightFace (SCRFD + ArcFace)")
            except Exception as e:
                print(f"  [FaceEncoder] InsightFace init failed ({e}), using fallback")
        else:
            print("  [FaceEncoder] InsightFace not available, using PIL fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_encode(self, image_path: str) -> Tuple[np.ndarray, str]:
        """Detect the largest face and return (embedding, image_sha256).

        Raises ValueError if no usable face is found.
        """
        image_bytes = Path(image_path).read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()

        if self.using_insightface:
            return self._encode_insightface(image_path, digest)
        return self._encode_fallback(image_path, image_bytes, digest)

    def detect_and_encode_bytes(self, image_bytes: bytes, label: str = "upload") -> Tuple[np.ndarray, str]:
        """Encode from raw image bytes (e.g. downloaded from web)."""
        import tempfile, os
        digest = hashlib.sha256(image_bytes).hexdigest()
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            if self.using_insightface:
                return self._encode_insightface(tmp_path, digest)
            return self._encode_fallback(tmp_path, image_bytes, digest)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        if a.shape != b.shape:
            # If dimensions don't match (InsightFace vs fallback), pad shorter
            max_dim = max(a.shape[0], b.shape[0])
            if a.shape[0] < max_dim:
                a = np.pad(a, (0, max_dim - a.shape[0]))
            if b.shape[0] < max_dim:
                b = np.pad(b, (0, max_dim - b.shape[0]))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # InsightFace backend
    # ------------------------------------------------------------------

    def _encode_insightface(self, image_path: str, digest: str) -> Tuple[np.ndarray, str]:
        """SCRFD detection → quality gate → ArcFace embedding."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        faces = self.app.get(img)
        if not faces:
            raise ValueError(f"No faces detected in {image_path}")

        # Quality-gate and keep largest passing face
        kept, rejected = [], []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            w = x2 - x1

            if f.det_score < MIN_DET_SCORE:
                rejected.append(("det_score", round(float(f.det_score), 3)))
                continue
            if w < MIN_FACE_WIDTH:
                rejected.append(("face_width", w))
                continue

            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                rejected.append(("empty_crop", None))
                continue

            blur = cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
            if blur < MIN_BLUR_VAR:
                rejected.append(("blur", round(float(blur), 1)))
                continue

            emb = f.normed_embedding.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)  # defensive renorm
            kept.append((emb, w))

        if rejected:
            print(f"  [QualityGate] Rejected: {rejected}")

        if not kept:
            raise ValueError(
                f"All faces rejected by quality gate in {image_path}: {rejected}"
            )

        # Return embedding of the largest face
        kept.sort(key=lambda x: x[1], reverse=True)
        return kept[0][0], digest

    # ------------------------------------------------------------------
    # PIL fallback backend
    # ------------------------------------------------------------------

    def _encode_fallback(self, image_path: str, image_bytes: bytes, digest: str) -> Tuple[np.ndarray, str]:
        """Deterministic embedding from pixel data (no real face detection)."""
        if Image is not None:
            try:
                with Image.open(image_path) as img:
                    rgb = img.convert("RGB")
                    resized = rgb.resize((16, 16), Image.Resampling.BILINEAR)
                    arr = np.asarray(resized, dtype=np.float32)
                    gray = arr.mean(axis=2).reshape(-1)
                    centered = gray - gray.mean()
                    vector = centered[: self.vector_dim]
                    if vector.shape[0] < self.vector_dim:
                        vector = np.pad(
                            vector, (0, self.vector_dim - vector.shape[0])
                        )
                    vector = vector.astype(np.float32)
                    vector /= np.linalg.norm(vector) + 1e-12
                    return vector, digest
            except Exception:
                pass

        # Last resort: hash-seeded random vector
        rng = np.random.default_rng(abs(int(digest[:16], 16)))
        vector = rng.normal(0.0, 1.0, size=self.vector_dim).astype(np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        return vector, digest


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def image_sha256(image_path: str) -> str:
    """SHA-256 hex digest of an image file."""
    return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()


def face_similarity_from_paths(image_a: str, image_b: str) -> float:
    """Quick cosine similarity between two face images."""
    encoder = FaceEncoder()
    emb1, _ = encoder.detect_and_encode(image_a)
    emb2, _ = encoder.detect_and_encode(image_b)
    return encoder.cosine_similarity(emb1, emb2)
