# src/core/data_cleaner.py
"""
Image deduplication and cosine-diversity sampling engine.

Supports three backends:
    🟢 pdqhash  – PDQHash 256-bit + dihedral 8-orientations, zero-download
    🟡 mobilenet_v3 – MobileNetV3-Small 576d, torchvision (already installed)
    🟠 resnet50  – ResNet-50 2048d, torchvision (already installed)

Architecture:
    ImageHasher (perceptual hash) → BK-tree → Hamming dedup
    FeatureExtractor (CNN/ViT embedding) → hnswlib → cosine dedup
    DiversitySelector → greedy coreset on cosine distance
"""

import os
import json
import time
import hashlib
import threading
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from src.config import t

# ---------------------------------------------------------------------------
# Optional imports – gracefully degrade when not installed
# ---------------------------------------------------------------------------
try:
    import imagehash as _imagehash_lib
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import pdqhash as _pdqhash_lib
    HAS_PDQHASH = True
except ImportError:
    HAS_PDQHASH = False

try:
    # Probe availability WITHOUT importing hnswlib at module-import time.
    # Importing hnswlib AFTER torch has loaded raises an OpenMP access
    # violation (0xC0000005) unless hnswlib wins the load race; keeping the
    # heavy import lazy avoids that at runtime, and also keeps PyInstaller's
    # dependency-analysis subprocess (which imports every collected package
    # into one persistent interpreter) from crashing. PyInstaller still finds
    # the `import hnswlib` below through bytecode analysis.
    from importlib.util import find_spec as _find_spec
    HAS_HNSWLIB = _find_spec("hnswlib") is not None
except Exception:
    HAS_HNSWLIB = False


def _import_hnswlib():
    """Import hnswlib on demand (see HAS_HNSWLIB note above)."""
    import hnswlib
    return hnswlib

try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as _ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class DuplicateGroup:
    """A group of duplicate images."""
    group_id: int
    paths: List[str]
    scores: List[float]           # lower = more similar
    reasons: List[str]            # e.g. "hamming=2", "cosine=0.97", "rotated_90"
    representative: str = ""      # keeper of this group; always paths[0] after scan()
                                  # (see DedupEngine.scan(keep_priority=...))
    layer: str = "L4"             # L1 byte / L2 pixel / L3 geometric / L4 semantic

@dataclass
class DuplicateReport:
    """Result of a dedup scan."""
    backend: str
    total_images: int
    duplicate_groups: List[DuplicateGroup]
    unique_images: List[str]
    blurry_images: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def removed_count(self) -> int:
        return sum(len(g.paths) - 1 for g in self.duplicate_groups)


def _reorder_group_keep_priority(group: "DuplicateGroup",
                                 priority: Callable[[str], int]) -> None:
    """Reorder ``group.paths`` in place so the best keeper comes first.

    ``paths[0]`` is the keeper and ``paths[1:]`` are the copies that get removed,
    so this decides which copy survives. The sort is stable and gives the group's
    existing keeper a tie-break bonus, which preserves the orientation-aware
    choice already made for L3 (rotated / flipped) groups.
    """
    if len(group.paths) < 2:
        return
    current_rep = group.representative or group.paths[0]
    triples = list(zip(group.paths, group.scores, group.reasons))
    triples.sort(key=lambda tr: (-priority(tr[0]), 0 if tr[0] == current_rep else 1))
    group.paths = [tr[0] for tr in triples]
    group.scores = [tr[1] for tr in triples]
    reasons = [tr[2] for tr in triples]
    if "representative" in group.reasons:
        reasons = ["representative"] + [r for r in reasons if r != "representative"]
    group.reasons = reasons
    group.representative = group.paths[0]


@dataclass
class DiversityReport:
    """Result of diversity sampling."""
    backend: str
    selected_indices: List[int]
    selected_paths: List[str]
    cluster_assignments: List[int]      # -1 = noise / unassigned
    representativeness: List[float]     # how central each image is in its cluster
    diversity_score: float              # 0-1, higher = more diverse
    redundancy_ranking: List[Tuple[int, float]]  # (index, avg_similarity) most redundant first
    elapsed_seconds: float = 0.0


# ===========================================================================
# ImageHasher –  🟢 PDQHash with imagehash fallback
# ===========================================================================

class ImageHasher:
    """
    Perceptual image hashing with automatic rotation/flip detection.

    Primary: PDQHash 256-bit via ``pdqhash``  (recommended)
    Fallback: pHash 64-bit via ``imagehash``  (if pdqhash not installed)

    Supports dihedral-group (8 orientations) via ``pdqhash.compute_dihedral()``
    so rotated and flipped variants of the same image collide.
    """

    # Pre-defined dihedral orientation labels for pdqhash
    DIHEDRAL_LABELS = [
        "original", "rotated_90", "rotated_180", "rotated_270",
        "flip_vertical", "flip_horizontal",
        "rotated_90_flip_vertical", "rotated_90_flip_horizontal",
    ]

    # Orientation labels for manual fallback
    MANUAL_ORIENTATIONS = [
        ("original", lambda img: img),
        ("rotated_90", lambda img: img.rotate(90, expand=True)),
        ("rotated_180", lambda img: img.rotate(180, expand=True)),
        ("rotated_270", lambda img: img.rotate(270, expand=True)),
        ("flip_horizontal", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("flip_vertical", lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    ]

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
        self._using_pdq = HAS_PDQHASH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_hashes(self, image_path: str) -> Dict[str, Any]:
        """
        Compute perceptual hashes for an image in all dihedral orientations.

        Returns:
            dict with keys:
                "hashes"      – list of hash vectors (numpy arrays)
                "labels"      – orientation labels
                "quality"     – PDQ quality score (only for pdqhash backend)
                "backend"     – "pdqhash" or "imagehash"
        """
        if self._using_pdq:
            return self._compute_pdq(image_path)
        return self._compute_imagehash(image_path)

    def hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """Bitwise Hamming distance between two hash vectors."""
        if h1.dtype == bool or h1.dtype == np.bool_:
            return int(np.sum(h1 != h2))
        return int(np.sum(np.unpackbits(h1.astype(np.uint8)) !=
                          np.unpackbits(h2.astype(np.uint8))))

    def min_dihedral_distance(self, hashes_a: List[np.ndarray],
                               hashes_b: List[np.ndarray]) -> Tuple[int, str, str]:
        """
        Minimum Hamming distance across all orientation pairs.

        Returns (distance, orientation_a_label, orientation_b_label).
        """
        best_dist = int(1e9)
        best_a, best_b = "", ""
        for i, ha in enumerate(hashes_a):
            for j, hb in enumerate(hashes_b):
                d = self.hamming_distance(ha, hb)
                if d < best_dist:
                    best_dist = d
                    best_a = self._orientation_label(i)
                    best_b = self._orientation_label(j)
        return best_dist, best_a, best_b

    # ------------------------------------------------------------------
    # PDQHash backend
    # ------------------------------------------------------------------

    def _compute_pdq(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hash_vectors, quality = _pdqhash_lib.compute_dihedral(img_rgb)
        return {
            "hashes": [np.asarray(v, dtype=np.uint8) for v in hash_vectors],
            "labels": list(self.DIHEDRAL_LABELS),
            "quality": quality,
            "backend": "pdqhash",
        }

    # ------------------------------------------------------------------
    # imagehash fallback
    # ------------------------------------------------------------------

    def _compute_imagehash(self, image_path: str) -> Dict[str, Any]:
        if not HAS_IMAGEHASH:
            raise ImportError(
                "Neither pdqhash nor imagehash is installed. "
                "Install with: pip install pdqhash  (recommended) "
                "or  pip install imagehash"
            )
        pil_img = Image.open(image_path).convert("L")
        hashes, labels = [], []
        for label, transform in self.MANUAL_ORIENTATIONS:
            transformed = transform(pil_img)
            h = _imagehash_lib.phash(transformed, hash_size=self.hash_size)
            hashes.append(np.asarray(h.hash.flatten(), dtype=bool))
            labels.append(label)
        return {
            "hashes": hashes,
            "labels": labels,
            "quality": None,
            "backend": "imagehash",
        }

    def _orientation_label(self, idx: int) -> str:
        if self._using_pdq:
            return self.DIHEDRAL_LABELS[idx] if idx < len(self.DIHEDRAL_LABELS) else f"orientation_{idx}"
        return self.MANUAL_ORIENTATIONS[idx][0] if idx < len(self.MANUAL_ORIENTATIONS) else f"orientation_{idx}"


# ===========================================================================
# FeatureExtractor –  MobileNetV3 / ResNet-50
# ===========================================================================

class FeatureExtractor:
    """
    CNN feature extraction with batch processing.

    Backends:
        "mobilenet_v3"   – MobileNetV3-Small, 576-d  (fastest on CPU)
        "efficientnet_b0" – EfficientNet-B0, 1280-d (better quality/FLOP)
        "resnet50"       – ResNet-50, 2048-d       (heavier, better semantics)
    """

    # Preprocessing normalization (ImageNet stats)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, backend: str = "mobilenet_v3",
                 device: str = "cpu"):
        if not HAS_TORCH:
            raise ImportError("FeatureExtractor requires PyTorch / torchvision")

        self.backend = backend
        self.device = device
        self._model = None
        self._dim = 0
        self._build_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def extract_batch(self, image_paths: List[str],
                       batch_size: int = 32,
                       progress_cb: Optional[Callable[[int, int], None]] = None
                       ) -> np.ndarray:
        """
        Extract feature vectors for a list of image paths.

        Args:
            image_paths: list of absolute image paths.
            batch_size:  number of images per forward pass.
            progress_cb: optional callback(current, total).

        Returns:
            (N, dim) float32 numpy array, L2-normalized.
        """
        n = len(image_paths)
        features = np.empty((n, self._dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_paths = image_paths[start:end]
            batch_tensor = self._load_batch(batch_paths).to(self.device)
            with torch.no_grad():
                feats = self._model(batch_tensor).cpu().numpy()
            features[start:end] = feats
            if progress_cb:
                progress_cb(end, n)

        # L2-normalize so cosine similarity = dot product
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features /= norms
        return features

    def extract_single(self, image_path: str) -> np.ndarray:
        """Extract feature for a single image."""
        return self.extract_batch([image_path], batch_size=1)[0]

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_model(self):
        builders = {
            "mobilenet_v3": self._build_mobilenet,
            "efficientnet_b0": self._build_efficientnet,
            "resnet50": self._build_resnet,
        }
        builder = builders.get(self.backend)
        if builder is None:
            raise ValueError(f"Unknown backend: {self.backend}. "
                             f"Choose from: {list(builders.keys())}")
        builder()

    def _build_mobilenet(self):
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()   # keep 576-d backbone feature
        model.eval()
        self._model = model.to(self.device)
        self._dim = 576
        self._input_size = 224

    def _build_resnet(self):
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Identity()             # drop classification head
        model.eval()
        self._model = model.to(self.device)
        self._dim = 2048
        self._input_size = 224

    def _build_efficientnet(self):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()     # drop classification head (1280-d)
        model.eval()
        self._model = model.to(self.device)
        self._dim = 1280
        self._input_size = 224



    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_batch(self, paths: List[str]) -> "torch.Tensor":
        """Load images, preprocess, stack into a batch tensor."""
        size = getattr(self, '_input_size', 224)
        tensors = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                # black placeholder for unreadable images
                img = np.zeros((size, size, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            img = img.astype(np.float32) / 255.0
            img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
            tensors.append(torch.from_numpy(img).permute(2, 0, 1))
        return torch.stack(tensors, dim=0)


# ===========================================================================
# DedupEngine —  coordinates scanning across backends
# ===========================================================================

class DedupEngine:
    """
    Orchestrates duplicate detection using either hashing or embeddings.

    Parameters
    ----------
    method : str
        "pdqhash" | "imagehash" | "mobilenet_v3" | "resnet50"
    hash_threshold : int
        Max Hamming distance for hash-based methods (default 5).
    similarity_threshold : float
        Min cosine similarity for embedding-based methods (default 0.95).
    blur_threshold : float
        Laplacian variance below which an image is flagged as blurry (default 100).
    pixel_mae_threshold : float | None
        Second-stage confirmation for hash matches: the mean absolute pixel
        error (0-255, min over the 8 dihedral transforms) must be <= this value
        for a pair to count as a duplicate (default 6.0). Perceptual hashes are
        near-identical for images dominated by a uniform background, so a
        Hamming-only test over-flags such datasets. None disables the gate.
    """

    def __init__(self, method: str = "pdqhash",
                 hash_threshold: int = 5,
                 similarity_threshold: float = 0.95,
                 blur_threshold: float = 100.0,
                 pixel_mae_threshold: Optional[float] = 6.0,
                 shift_search_px: int = 2,
                 shift_search_factor: float = 3.0):
        self.method = method
        self.hash_threshold = hash_threshold
        self.similarity_threshold = similarity_threshold
        self.blur_threshold = blur_threshold
        self.pixel_mae_threshold = pixel_mae_threshold
        # Small-translation search for the pixel gate: a duplicate that was
        # re-cropped / re-saved a few pixels off has a moderate no-shift MAE but
        # a near-zero one once aligned.  Only pairs whose no-shift MAE is within
        # shift_search_factor * pixel_mae_threshold pay for the extra search.
        self.shift_search_px = int(shift_search_px)
        self.shift_search_factor = float(shift_search_factor)
        self._gray_cache: Dict[str, Any] = {}

    def scan(self, image_paths: List[str],
             progress_cb: Optional[Callable[[int, int, str], None]] = None,
             keep_priority: Optional[Callable[[str], int]] = None
             ) -> DuplicateReport:
        """
        Run a full dedup scan over *image_paths*.

        progress_cb(current, total, phase_description)

        keep_priority(path) -> int
            Optional. Higher value = more likely to be KEPT when a group is
            formed. Use it to protect valuable copies, e.g.
            ``lambda p: 1 if os.path.exists(label_path(p)) else 0`` so that an
            annotated image is never discarded in favour of an un-annotated twin.
        """
        t0 = time.time()
        n = len(image_paths)

        # Phase 0: exact byte-hash dedup (always free)
        if progress_cb:
            progress_cb(0, n, t("PHASE_DEDUP"))
        exact_groups = self._exact_dedup(image_paths)

        # From each exact group keep only the first image; keep all non-duplicate images
        l1_paths = set()
        for g in exact_groups:
            for p in g.paths:
                l1_paths.add(p)
        unique = []
        for g in exact_groups:
            unique.append(g.paths[0])  # representative of each L1 group
        for p in image_paths:
            if p not in l1_paths:
                unique.append(p)  # non-duplicate images stay

        # Flatten removed paths for the report
        exact_removed: List[DuplicateGroup] = [g for g in exact_groups if len(g.paths) > 1]

        if self.method in ("pdqhash", "imagehash"):
            report = self._hash_scan(unique, progress_cb)
        else:
            report = self._embedding_scan(unique, progress_cb)

        # Merge exact-duplicate groups into the report
        report.duplicate_groups = exact_removed + report.duplicate_groups

        # Blur detection
        if progress_cb:
            progress_cb(n, n, t("PHASE_BLUR"))
        report.blurry_images = self._detect_blurry(unique)

        report.total_images = len(image_paths)
        report.elapsed_seconds = time.time() - t0

        # Let the caller protect valuable copies (e.g. annotated ones): paths[0]
        # is the keeper, paths[1:] are removed, so reorder before returning.
        if keep_priority is not None:
            for g in report.duplicate_groups:
                _reorder_group_keep_priority(g, keep_priority)

        return report

    # ------------------------------------------------------------------
    # Exact dedup (SHA-256)
    # ------------------------------------------------------------------

    def _exact_dedup(self, paths: List[str]) -> List[DuplicateGroup]:
        """Group byte-identical images by SHA-256."""
        hash_map: Dict[str, List[str]] = {}
        for p in paths:
            try:
                with open(p, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
            except OSError:
                continue
            hash_map.setdefault(digest, []).append(p)

        groups = []
        gid = 0
        for digest, group_paths in hash_map.items():
            if len(group_paths) < 2:
                continue
            groups.append(DuplicateGroup(
                group_id=gid,
                paths=sorted(group_paths),
                scores=[0.0] * len(group_paths),
                reasons=["sha256_match"] * len(group_paths),
                representative=group_paths[0],
                layer="L1",
            ))
            gid += 1
        if groups:
            return groups
        # If no exact dupes, return each path as its own singleton group
        return [DuplicateGroup(
            group_id=i, paths=[p], scores=[0.0],
            reasons=["unique"], representative=p
        ) for i, p in enumerate(paths)]

    # ------------------------------------------------------------------
    # Pixel-level confirmation for hash matches
    # ------------------------------------------------------------------

    def _gray_vec(self, path: str, size: int = 128):
        """Cached small grayscale array used by the pixel-similarity gate."""
        if path in self._gray_cache:
            return self._gray_cache[path]
        try:
            im = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
            vec = np.asarray(im, dtype=np.float32)
        except Exception:
            vec = None
        self._gray_cache[path] = vec
        return vec

    def _pair_mae(self, path_a: str, path_b: str, shift_range: int = 0) -> float:
        """Minimum mean-absolute pixel error between two images over the 8
        dihedral transforms, on the 0-255 scale.

        ``shift_range > 0`` additionally searches small integer translations
        (image b shifted by -shift_range..+shift_range px on both axes) and keeps
        the best match.  This catches duplicates that were re-cropped a few
        pixels off — e.g. NEU-DET's `pitted_surface_292` vs `pitted_surface_171`
        (no-shift MAE 11.3, aligned MAE 1.3).  The cost grows by
        (2*shift_range+1)**2 per transform, so callers should only request it
        for pairs that are already close (see _hash_scan).

        Returns 0.0 when either image cannot be read, so an unreadable file is
        never silently rejected by the gate.
        """
        a = self._gray_vec(path_a)
        b = self._gray_vec(path_b)
        if a is None or b is None:
            return 0.0
        try:
            h, w = a.shape
            rng = max(0, int(shift_range))
            best = 255.0
            for k in range(4):
                r = np.rot90(b, k)
                for v in (r, np.fliplr(r)):
                    if rng == 0:
                        m = float(np.abs(a - v).mean())
                        if m < best:
                            best = m
                    else:
                        for dy in range(-rng, rng + 1):
                            for dx in range(-rng, rng + 1):
                                y0a, y1a = max(0, dy), h + min(0, dy)
                                x0a, x1a = max(0, dx), w + min(0, dx)
                                y0b, y1b = max(0, -dy), h + min(0, -dy)
                                x0b, x1b = max(0, -dx), w + min(0, -dx)
                                m = float(np.abs(
                                    a[y0a:y1a, x0a:x1a] - v[y0b:y1b, x0b:x1b]).mean())
                                if m < best:
                                    best = m
                    if best == 0.0:
                        return 0.0
            return best
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Hash-based scan (PDQHash / imagehash)
    # ------------------------------------------------------------------

    def _hash_scan(self, paths: List[str],
                    progress_cb: Optional[Callable] = None
                    ) -> DuplicateReport:
        n = len(paths)
        hasher = ImageHasher()

        # Compute all hashes
        all_hashes: List[Dict[str, Any]] = []
        for i, p in enumerate(paths):
            try:
                all_hashes.append(hasher.compute_hashes(p))
            except Exception:
                all_hashes.append(None)
            if progress_cb:
                progress_cb(i + 1, n, t("PHASE_HASH", hasher._using_pdq and 'PDQ' or 'pHash'))

        # Pairwise comparison with BK-tree-like early exit
        n_groups = 0
        assigned = set()
        groups: List[DuplicateGroup] = []

        for i in range(n):
            if i in assigned or all_hashes[i] is None:
                continue
            group_paths = [paths[i]]
            group_scores = [0.0]
            group_reasons = ["representative"]
            for j in range(i + 1, n):
                if j in assigned or all_hashes[j] is None:
                    continue
                dist, ori_a, ori_b = hasher.min_dihedral_distance(
                    all_hashes[i]["hashes"], all_hashes[j]["hashes"]
                )
                if dist > self.hash_threshold:
                    continue
                # Second-stage gate: perceptual hashes are near-identical for
                # images dominated by a uniform background (e.g. industrial
                # surface-inspection photos), so confirm with an actual pixel
                # comparison before calling the pair a duplicate.
                if self.pixel_mae_threshold is not None:
                    mae = self._pair_mae(paths[i], paths[j])
                    if mae > self.pixel_mae_threshold:
                        # Second chance for near-misses: a duplicate that was
                        # re-cropped a few pixels off yields a moderate no-shift
                        # MAE but a near-zero one after alignment.  Only pay for
                        # the shift search when the pair is already close.
                        if (self.shift_search_px > 0
                                and mae <= self.pixel_mae_threshold * self.shift_search_factor):
                            mae = self._pair_mae(paths[i], paths[j],
                                                 shift_range=self.shift_search_px)
                        if mae > self.pixel_mae_threshold:
                            continue
                group_paths.append(paths[j])
                group_scores.append(float(dist))
                group_reasons.append(f"hamming={dist} ({ori_a}↔{ori_b})")
                assigned.add(j)

            if len(group_paths) > 1:
                # Classify L2 vs L3 by orientation crossing
                layer = "L2"
                for reason in group_reasons[1:]:
                    if "↔" in reason:
                        _, orient = reason.split("(", 1)
                        oa, ob = orient.rstrip(")").split("↔")
                        if oa != "original" or ob != "original":
                            layer = "L3"
                            break
                # L3 representative: keep the image with 'original' orientation
                rep = group_paths[0]
                if layer == "L3":
                    for k, reason in enumerate(group_reasons):
                        if reason != "representative" and "↔" in reason:
                            _, orient = reason.split("(", 1)
                            oa, ob = orient.rstrip(")").split("↔")
                            if ob == "original" and oa != "original":
                                rep = group_paths[k]  # member k is the original
                                break
                            # oa == original → paths[0] is the original → keep rep default
                            if oa == "original":
                                rep = group_paths[0]
                                break
                groups.append(DuplicateGroup(
                    group_id=n_groups,
                    paths=group_paths,
                    scores=group_scores,
                    reasons=group_reasons,
                    representative=rep,
                    layer=layer,
                ))
                n_groups += 1
            assigned.add(i)

        unique_paths = [p for idx, p in enumerate(paths) if idx not in
                        {idx for g in groups for idx in
                         [paths.index(p2) for p2 in g.paths[1:]]}]
        # More readable: collect all removed paths
        removed_paths = set()
        for g in groups:
            for p in g.paths[1:]:
                removed_paths.add(p)
        unique_paths = [p for p in paths if p not in removed_paths]

        return DuplicateReport(
            backend=hasher._using_pdq and "pdqhash" or "imagehash",
            total_images=n,
            duplicate_groups=groups,
            unique_images=unique_paths,
        )

    # ------------------------------------------------------------------
    # Embedding-based scan (MobileNetV3 / ResNet-50)
    # ------------------------------------------------------------------

    def _embedding_scan(self, paths: List[str],
                         progress_cb: Optional[Callable] = None
                         ) -> DuplicateReport:
        n = len(paths)
        extractor = FeatureExtractor(backend=self.method, device="cpu")

        if progress_cb:
            progress_cb(0, n, t("PHASE_EXTRACT_FEAT", self.method))

        features = extractor.extract_batch(
            paths, batch_size=32,
            progress_cb=lambda cur, tot: progress_cb(cur, tot,
                t("PHASE_EXTRACT", self.method)) if progress_cb else None
        )

        # Build hnswlib index for fast neighbor search
        if progress_cb:
            progress_cb(n, n, t("PHASE_INDEX"))

        if HAS_HNSWLIB:
            groups = self._hnsw_dedup(paths, features, extractor.feature_dim)
        else:
            groups = self._bruteforce_cosine_dedup(paths, features)

        removed = set()
        for g in groups:
            for p in g.paths[1:]:
                removed.add(p)

        return DuplicateReport(
            backend=self.method,
            total_images=n,
            duplicate_groups=groups,
            unique_images=[p for p in paths if p not in removed],
        )

    def _hnsw_dedup(self, paths: List[str], features: np.ndarray,
                     dim: int) -> List[DuplicateGroup]:
        """HNSW-based duplicate clustering."""
        n = len(paths)
        index = _import_hnswlib().Index(space='cosine', dim=dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(features, np.arange(n, dtype=np.int64))
        index.set_ef(50)

        # For each image, find neighbors within similarity threshold
        # Cosine distance = 1 - cosine_similarity
        threshold = 1.0 - self.similarity_threshold + 1e-8

        assigned = set()
        groups: List[DuplicateGroup] = []
        gid = 0

        for i in range(n):
            if i in assigned:
                continue
            labels, distances = index.knn_query(features[i:i+1], k=min(n, 50))
            labels, distances = labels[0], distances[0]

            group_indices = [i]
            group_paths = [paths[i]]
            group_scores = [0.0]
            group_reasons = ["representative"]

            for j, d in zip(labels, distances):
                j = int(j)
                if j == i or j in assigned:
                    continue
                if d <= threshold:
                    group_indices.append(j)
                    group_paths.append(paths[j])
                    group_scores.append(float(1.0 - d))
                    group_reasons.append(f"cosine={1.0 - d:.3f}")
                    assigned.add(j)

            if len(group_paths) > 1:
                groups.append(DuplicateGroup(
                    group_id=gid, paths=group_paths,
                    scores=group_scores, reasons=group_reasons,
                    representative=group_paths[0],
                    layer="L4",
                ))
                gid += 1
            assigned.add(i)

        return groups

    def _bruteforce_cosine_dedup(self, paths: List[str],
                                   features: np.ndarray) -> List[DuplicateGroup]:
        """Fallback: brute-force pairwise cosine comparison."""
        n = len(paths)
        sim = features @ features.T  # cosine = dot product (already L2-normalized)

        assigned = set()
        groups: List[DuplicateGroup] = []
        gid = 0

        for i in range(n):
            if i in assigned:
                continue
            group_paths = [paths[i]]
            group_scores = [1.0]
            group_reasons = ["representative"]
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                if sim[i, j] >= self.similarity_threshold:
                    group_paths.append(paths[j])
                    group_scores.append(float(sim[i, j]))
                    group_reasons.append(f"cosine={sim[i, j]:.3f}")
                    assigned.add(j)
            if len(group_paths) > 1:
                groups.append(DuplicateGroup(
                    group_id=gid, paths=group_paths,
                    scores=group_scores, reasons=group_reasons,
                    representative=group_paths[0],
                    layer="L4",
                ))
                gid += 1
            assigned.add(i)

        return groups

    # ------------------------------------------------------------------
    # Blur detection
    # ------------------------------------------------------------------

    def _detect_blurry(self, paths: List[str]) -> List[str]:
        """Flag images with low Laplacian variance."""
        blurry = []
        for p in paths:
            try:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                var = cv2.Laplacian(img, cv2.CV_64F).var()
                if var < self.blur_threshold:
                    blurry.append(p)
            except Exception:
                continue
        return blurry


# ===========================================================================
# DiversitySelector –  Cosine diversity sampling (greedy coreset)
# ===========================================================================

class DiversitySelector:
    """
    Select a diverse subset of images using cosine distance + greedy coreset.

    Algorithm: Greedy MaxMin (farthest-point sampling), equivalent to
    PatchCore's ``GreedyCoresetSampler``.

    Parameters
    ----------
    features : np.ndarray
        (N, dim) L2-normalized feature matrix.
    """

    def __init__(self, features: np.ndarray):
        if features.ndim != 2:
            raise ValueError("features must be a 2-D array")
        self.features = features
        self.n = features.shape[0]
        # Cosine distance = 1 - dot (since vectors are L2-normalized)
        self._sim = features @ features.T

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def cluster_by_similarity(self, similarity_threshold: float = 0.85) -> DiversityReport:
        """
        Cluster images by cosine-similarity threshold (agglomerative).

        Instead of fixing the number of clusters K, the user picks a similarity
        threshold (e.g. 0.85 = "images at least 85% similar belong to one group").
        Cluster count is determined automatically.

        Representative per cluster = medoid (most central member).
        """
        t0 = time.time()
        if self.n <= 1:
            return DiversityReport(
                backend="similarity_agglomerative",
                selected_indices=[0], selected_paths=[],
                cluster_assignments=[0] * self.n,
                representativeness=[1.0] * self.n,
                diversity_score=1.0, redundancy_ranking=[],
                elapsed_seconds=time.time() - t0,
            )

        try:
            from sklearn.cluster import AgglomerativeClustering
            dist_threshold = max(1.0 - similarity_threshold, 0.0)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(self.features)
        except Exception:
            # Fallback: greedy coreset with a guessed K from similarity
            k_guess = max(2, int(self.n * (1.0 - similarity_threshold)))
            return self.select_diverse(min(k_guess, self.n))

        n_clusters = int(labels.max()) + 1
        selected = []
        reps = []
        assignments = list(labels)

        for c in range(n_clusters):
            members = [i for i in range(self.n) if labels[i] == c]
            if not members:
                continue
            # Medoid: member with max avg similarity to other members
            sub_sim = self._sim[np.ix_(members, members)]
            avg = sub_sim.mean(axis=1)
            medoid = members[int(np.argmax(avg))]
            selected.append(medoid)
            reps.append(float(avg.max()))

        # Diversity score: mean min-pairwise-cosine-distance among representatives
        sel_sim = self._sim[np.ix_(selected, selected)]
        np.fill_diagonal(sel_sim, -1.0)
        div_score = float(np.mean(1.0 - np.max(sel_sim, axis=0))) if len(selected) > 1 else 1.0

        avg_sim = np.mean(self._sim, axis=1)
        redundancy = sorted(
            [(i, float(avg_sim[i])) for i in range(self.n)], key=lambda x: -x[1]
        )

        return DiversityReport(
            backend="similarity_agglomerative",
            selected_indices=list(selected),
            selected_paths=[],
            cluster_assignments=assignments,
            representativeness=[float(r) for r in reps],
            diversity_score=float(div_score),
            redundancy_ranking=redundancy,
            elapsed_seconds=time.time() - t0,
        )
    def _count_clusters_at_threshold(self, threshold: float) -> int:
        """Quick cluster count at a similarity threshold (no report overhead)."""
        if self.n <= 1:
            return 1
        try:
            from sklearn.cluster import AgglomerativeClustering
            dist_threshold = max(1.0 - threshold, 0.0)
            clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=dist_threshold,
                metric="cosine", linkage="average",
            )
            labels = clustering.fit_predict(self.features)
            return int(labels.max()) + 1
        except Exception:
            k_guess = max(2, int(self.n * (1.0 - threshold)))
            return min(k_guess, self.n)

    def find_threshold_for_target_clusters(
        self, target_count: int, search_range=(0.80, 0.98), max_iterations=8
    ) -> float:
        """
        Binary search to find the similarity threshold that produces
        approximately ``target_count`` clusters.

        Parameters
        ----------
        target_count : int
            Desired number of clusters (e.g. from 5% of dataset size).
        search_range : tuple
            (low, high) threshold bounds for binary search.
        max_iterations : int
            Max binary-search iterations (precision ≈ range_width / 2^N).

        Returns
        -------
        float  Best similarity threshold found.
        """
        lo, hi = search_range
        best_threshold = (lo + hi) / 2
        best_diff = float('inf')

        for _ in range(max_iterations):
            mid = (lo + hi) / 2
            n_clusters = self._count_clusters_at_threshold(mid)
            diff = abs(n_clusters - target_count)

            if diff < best_diff:
                best_diff = diff
                best_threshold = mid

            if n_clusters == target_count:
                break
            elif n_clusters > target_count:
                lo = mid  # need higher threshold → fewer clusters
            else:
                hi = mid  # need lower threshold → more clusters

            if abs(hi - lo) < 0.005:
                break

        return round(best_threshold, 3)

    def select_diverse(self, k: int) -> DiversityReport:
        """
        Select *k* most diverse samples.

        Returns a DiversityReport with selected indices and cluster assignments.
        """
        t0 = time.time()
        k = min(k, self.n)
        if k <= 1:
            return DiversityReport(
                backend="greedy_coreset",
                selected_indices=[0],
                selected_paths=[],   # caller fills this
                cluster_assignments=[0] * self.n,
                representativeness=[1.0] * self.n,
                diversity_score=1.0,
                redundancy_ranking=[],
                elapsed_seconds=time.time() - t0,
            )

        # Step 1: greedy farthest-point selection
        selected = self._greedy_coreset(k)

        # Step 2: assign every point to nearest selected center (clustering)
        assignments, distances = self._assign_clusters(selected)

        # Step 3: compute representativeness (1 / (1 + distance_to_center))
        reps = 1.0 / (1.0 + np.array(distances))

        # Step 4: compute overall diversity score
        # average of min pairwise cosine distance among selected
        sel_sim = self._sim[np.ix_(selected, selected)]
        np.fill_diagonal(sel_sim, -1.0)
        div_score = float(np.mean(1.0 - np.max(sel_sim, axis=0)))

        # Step 5: redundancy ranking – images with highest avg similarity to others
        avg_sim = np.mean(self._sim, axis=1)
        redundancy = sorted(
            [(i, float(avg_sim[i])) for i in range(self.n)],
            key=lambda x: -x[1]
        )

        return DiversityReport(
            backend="greedy_coreset",
            selected_indices=list(selected),
            selected_paths=[],
            cluster_assignments=[int(a) for a in assignments],
            representativeness=[float(r) for r in reps],
            diversity_score=float(div_score),
            redundancy_ranking=redundancy,
            elapsed_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _greedy_coreset(self, k: int) -> np.ndarray:
        """Greedy farthest-point sampling."""
        # Start with a random first point (index 0 is fine)
        selected = [0]
        # Distance from each point to the nearest selected point
        min_dists = 1.0 - self._sim[0, :].copy()  # cosine distance

        for _ in range(1, k):
            # Pick the point furthest from all currently selected
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            # Update min distances
            new_dists = 1.0 - self._sim[next_idx, :]
            min_dists = np.minimum(min_dists, new_dists)

        return np.array(selected)

    def _assign_clusters(self, centers: np.ndarray
                          ) -> Tuple[List[int], List[float]]:
        """Assign each point to the nearest center."""
        center_sim = self._sim[:, centers]  # (n, k)
        # Cosine distance from each point to each center
        center_dist = 1.0 - center_sim
        assignments = np.argmin(center_dist, axis=1)
        min_distances = np.min(center_dist, axis=1)
        return [int(a) for a in assignments], [float(d) for d in min_distances]


# ===========================================================================

# ClusterManager –  manages cluster_map.json (cluster → color → status)

# ===========================================================================



class ClusterManager:

    """

    Persists diversity-clustering results to ``<dataset_root>/cluster_map.json``.



    Each cluster gets:

        - a colour (from CLUSTER_COLORS)

        - a representative image (most central / first member)

        - members (image paths)

        - status: "未标注" / "已标代表" / "已AI标注" / "已复核"



    File layout::

        {

            "backend": "mobilenet_v3",

            "k": 8,

            "created": "2026-08-06 12:00:00",

            "clusters": [

                {"cluster_id": 0, "color": "#e74c3c", "representative": "a.jpg",

                 "members": ["a.jpg", "b.jpg"], "status": "未标注"},

                ...

            ]

        }

    """



    DEFAULT_COLORS = [

        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",

        "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#27ae60",

    ]



    def __init__(self, dataset_root: str):

        self.dataset_root = dataset_root

        self.cluster_map_path = os.path.join(dataset_root, "cluster_map.json")

        self.data = None



    # ------------------------------------------------------------------

    # Load / save

    # ------------------------------------------------------------------



    def exists(self) -> bool:

        return os.path.exists(self.cluster_map_path)



    def load(self) -> bool:

        """Load existing cluster map. Returns True if loaded OK."""

        try:

            if self.exists():

                with open(self.cluster_map_path, 'r', encoding='utf-8') as f:

                    self.data = json.load(f)

                return True

        except Exception as e:

            print(f"[ClusterManager] Load error: {e}")

        return False



    def save(self):

        if self.data is None:

            return

        try:

            with open(self.cluster_map_path, 'w', encoding='utf-8') as f:

                json.dump(self.data, f, ensure_ascii=False, indent=2)

        except Exception as e:

            print(f"[ClusterManager] Save error: {e}")



    # ------------------------------------------------------------------

    # Build from DiversityReport + image paths

    # ------------------------------------------------------------------



    def build_from_report(self, div_report, image_paths: List[str],

                          backend: str = "mobilenet_v3", k: int = 8):

        """Create cluster map from a DiversityReport."""

        import datetime

        clusters = []

        n_sel = len(div_report.selected_indices)



        for rank, idx in enumerate(div_report.selected_indices):

            if idx >= len(image_paths):

                continue

            rep_path = image_paths[idx]

            members = [image_paths[j] for j in range(len(image_paths))

                       if div_report.cluster_assignments[j] == rank]

            if not members:

                members = [rep_path]

            clusters.append({

                "cluster_id": rank,

                "color": self.DEFAULT_COLORS[rank % len(self.DEFAULT_COLORS)],

                "representative": rep_path,

                "members": members,

                "status": "未标注",

            })



        self.data = {

            "backend": backend,

            "k": n_sel,
            "rep_target": len(clusters),

            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "clusters": clusters,

        }

        self.save()

    def recluster_paths(
        self, image_paths: List[str], features: np.ndarray,
        similarity_threshold: float, backend: str = "mobilenet_v3"
    ):
        """
        Re-cluster the given paths at a specific similarity threshold,
        rebuilding the cluster_map.json with new representatives.

        Parameters
        ----------
        image_paths : list of str  Full paths of images to cluster.
        features : np.ndarray   (N, dim) L2-normalized feature matrix.
        similarity_threshold : float  e.g. 0.90.
        backend : str  Feature extractor backend name.

        Returns
        -------
        self  (for chaining)
        """
        selector = DiversitySelector(features)
        report = selector.cluster_by_similarity(similarity_threshold)
        self.build_from_report(
            report, image_paths, backend=backend,
            k=len(report.selected_indices)
        )
        return self





    # ------------------------------------------------------------------

    # Queries / mutations

    # ------------------------------------------------------------------



    def get_clusters(self) -> List[dict]:

        if not self.data:

            return []

        return self.data.get("clusters", [])



    def color_of(self, image_path: str) -> str:

        """Return the cluster colour for an image (or None)."""

        norm = os.path.normpath(image_path)

        for cl in self.get_clusters():

            if os.path.normpath(cl["representative"]) == norm:

                return cl.get("color", "#999")

            for m in cl.get("members", []):

                if os.path.normpath(m) == norm:

                    return cl.get("color", "#999")

        return None



    def is_representative(self, image_path: str) -> bool:

        norm = os.path.normpath(image_path)

        for cl in self.get_clusters():

            if os.path.normpath(cl["representative"]) == norm:

                return True

        return False

    def remove_path(self, image_path: str) -> bool:
        """Remove an image from the cluster map (used when it is junked/deleted).
        - Member: dropped from its cluster's members.
        - Representative: the whole cluster is dropped (its members return to
          the unlabeled pool and will be re-clustered in a future round).
        Returns True when anything changed (caller should save())."""
        if not self.data:
            return False
        norm = os.path.normpath(image_path)
        changed = False
        kept = []
        for cl in self.get_clusters():
            rep = cl.get("representative", "")
            if rep and os.path.normpath(rep) == norm:
                changed = True
                continue
            members = cl.get("members", [])
            new_members = [m for m in members if os.path.normpath(m) != norm]
            if len(new_members) != len(members):
                changed = True
                cl["members"] = new_members
            kept.append(cl)
        if changed:
            self.data["clusters"] = kept
        return changed



    def set_cluster_status(self, cluster_id: int, status: str):

        for cl in self.get_clusters():

            if cl["cluster_id"] == cluster_id:

                cl["status"] = status

                self.save()

                return



    def cluster_of(self, image_path: str) -> Optional[int]:

        """Return cluster_id for an image, or None."""

        norm = os.path.normpath(image_path)

        for cl in self.get_clusters():

            if os.path.normpath(cl["representative"]) == norm:

                return cl["cluster_id"]

            for m in cl.get("members", []):

                if os.path.normpath(m) == norm:

                    return cl["cluster_id"]

        return None

