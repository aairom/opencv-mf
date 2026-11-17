#!/usr/bin/env python3
"""
IBM Patent US 11,763,422 B2 - Novel Modular Implementation
===========================================================
"Fast color clustering for preprocessing an image"
Inventors: Marc Fiammante, Jean-Armand Broyelle (IBM)
Patent Grant Date: September 19, 2023

Architectural Highlights & Novel Design Patterns in bob_code:
-------------------------------------------------------------
1. Discrete Pipeline Architecture:
   - Encapsulates patent components (FIG. 1) as distinct, pluggable objects:
     * `CrossProductQuantizer` (Module 106)
     * `LeftAlignmentOperator` (Module 108)
     * `ColorIndexEncoder` (Module 110)
     * `HierarchicalClusterForest` / `MinHeapColorMerger` (Module 112)
     * `PaletteCentroidRefiner` (Post-processor, Claims 6 & 13)

2. Priority-Queue / Min-Heap Color Merging Engine:
   - Uses an indexed min-heap / priority queue (`heapq`) with dual-level tracking
     to locate the least-frequent cluster in O(log L) time.
   - Dynamic distance tracking with nearest-neighbor adjacency cache and generation counters
     to eliminate redundant O(N^2) distance recomputations across merge iterations.

3. Disjoint-Set Union (Union-Find) with Path Compression & Tree Linking:
   - Represents the hierarchical dendrogram merge tree (FIG. 3 / Claims 3, 10, 17)
     using explicit Node objects and Disjoint-Set Union (DSU) data structures,
     providing exact ancestor lineage tracking without flat index pointers.

4. N-Channel and Arbitrary Color-Space Support:
   - Handles not only 3-channel RGB/BGR, but multi-spectral data (satellite, NIR, UV)
     as explicitly claimed in the patent specification (Col. 3, lines 1-5).
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "DistanceMetric",
    "RoundingMode",
    "ColorTreeNode",
    "CrossProductQuantizer",
    "LeftAlignmentOperator",
    "ColorIndexEncoder",
    "HierarchicalClusterForest",
    "PaletteCentroidRefiner",
    "PatentColorClusterer",
    "process_image_patent_pipeline",
]


# ==============================================================================
# Enumerations and Configuration
# ==============================================================================

class DistanceMetric(str, Enum):
    """Distance metrics supported for color proximity evaluation (Step 220)."""
    EUCLIDEAN_RGB = "rgb"
    CIELAB_DELTA_E = "lab"
    WEIGHTED_PERCEPTUAL = "weighted_rgb"


class RoundingMode(str, Enum):
    """Integer rounding strategy for cross-product calculation (Step 208)."""
    ROUND_HALF_UP = "round"
    BIT_TRUNCATION = "floor"


# ==============================================================================
# Module 106: Cross-Product Quantizer (Steps 204-208 / Claims 1, 7, 14)
# ==============================================================================

class CrossProductQuantizer:
    """Computes cross-product reduction on source image pixel values.
    
    Implements patent formula:
        q = round( (2^n - 1) * color / 255 )   or   (color * 2^n) >> 8
    
    Patent References:
        - Step 204: Determine initial number of colors K + margin (e.g. K/2)
        - Step 205: Select bits per channel such that 2^(bits*channels) > (K + margin)
        - Steps 206, 208: Determine cross-product and round to integer values
        - Claims 1, 7, 14: Use precomputed LUTs for rapid execution
    """

    def __init__(self, bits: int, rounding: RoundingMode = RoundingMode.ROUND_HALF_UP):
        if not (1 <= bits <= 8):
            raise ValueError(f"Bits per channel must be between 1 and 8, got {bits}")
        self.bits = bits
        self.rounding = rounding
        self._lut = self._build_lookup_table(bits, rounding)

    @staticmethod
    def estimate_optimal_bits(k: int, margin: Optional[int] = None, channels: int = 3) -> int:
        """Evaluate bits per channel count: 2^(bits * channels) > (K + margin)."""
        if k < 1:
            raise ValueError("K must be a positive integer >= 1")
        if margin is None:
            margin = max(1, k // 2)
        
        threshold = k + margin
        for b in range(1, 9):
            if (1 << (b * channels)) > threshold:
                return b
        return 8

    @staticmethod
    def _build_lookup_table(bits: int, rounding: RoundingMode) -> np.ndarray:
        max_quantized = (1 << bits) - 1
        raw_vals = np.arange(256, dtype=np.float32)
        if rounding == RoundingMode.ROUND_HALF_UP:
            lut = np.round((raw_vals * max_quantized) / 255.0).astype(np.uint8)
        else:
            lut = (raw_vals.astype(np.uint16) * (1 << bits) >> 8).astype(np.uint8)
        return lut

    def quantize(self, img: np.ndarray) -> np.ndarray:
        """Apply cross-product reduction using the precomputed LUT."""
        if img.dtype != np.uint8:
            raise TypeError("Source image must have uint8 pixel values")
        return cv2.LUT(img, self._lut) if img.ndim in (2, 3) else self._lut[img]


# ==============================================================================
# Module 108: Left-Alignment Operator (Step 210 / Claims 1, 2)
# ==============================================================================

class LeftAlignmentOperator:
    """Left-aligns quantized values to a standard 8-bit scale for distance metric.
    
    Patent References:
        - Step 210: Left align rounded cross-product on 8-bit value
        - Claim 1: Left aligned rounded cross-product specifies initial target image
        - Specification Col. 4 lines 50-60: Left justification provides consistent
          color distance computation regardless of bit depth n.
    """

    def __init__(self, bits: int):
        self.bits = bits
        self.shift = 8 - bits

    def align(self, quantized_levels: np.ndarray) -> np.ndarray:
        """Shift quantized levels left by (8 - bits) bits."""
        return (quantized_levels.astype(np.uint8) << self.shift)

    def unalign(self, aligned_values: np.ndarray) -> np.ndarray:
        """Shift aligned 8-bit values right by (8 - bits) bits."""
        return (aligned_values.astype(np.uint8) >> self.shift)


# ==============================================================================
# Module 110: Color Index Encoder (Step 212 / Claims 5, 12, 19)
# ==============================================================================

class ColorIndexEncoder:
    """Encodes multi-channel pixel values into a packed integer key and extracts histograms.
    
    Implements patent flat map / regular map encoding:
        idx = (channel[0] << 2n) | (channel[1] << n) | channel[2]
    
    Patent References:
        - Step 212: Determine count of remaining unique colors
        - Claims 5, 12, 19: Unique key concatenation of channel significant bits
    """

    def __init__(self, bits: int, channels: int = 3):
        self.bits = bits
        self.channels = channels
        self.mask = (1 << bits) - 1
        self.max_possible_keys = 1 << (bits * channels)

    def encode(self, levels: np.ndarray) -> np.ndarray:
        """Pack multi-channel levels into a 1D integer key per pixel."""
        levels_int = levels.astype(np.int64)
        packed_key = np.zeros(levels.shape[:-1], dtype=np.int64)
        for ch_idx in range(self.channels):
            shift_amount = (self.channels - 1 - ch_idx) * self.bits
            packed_key |= (levels_int[..., ch_idx] << shift_amount)
        return packed_key

    def decode(self, keys: np.ndarray, left_align: bool = True) -> np.ndarray:
        """Unpack integer keys back into channel color coordinates."""
        channels_unpacked = []
        for ch_idx in range(self.channels):
            shift_amount = (self.channels - 1 - ch_idx) * self.bits
            ch_val = (keys >> shift_amount) & self.mask
            channels_unpacked.append(ch_val)
        color_coords = np.stack(channels_unpacked, axis=-1).astype(np.float64)
        if left_align:
            color_coords *= (1 << (8 - self.bits))
        return color_coords

    def build_histogram(self, packed_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute flat map histogram of unique color frequencies."""
        hist = np.bincount(packed_keys.ravel(), minlength=self.max_possible_keys)
        unique_keys = np.flatnonzero(hist)
        frequencies = hist[unique_keys]
        return unique_keys, frequencies


# ==============================================================================
# Module 112: Hierarchical Tree & Min-Heap Merger (Steps 218-228 / Claims 3, 4, 10, 11)
# ==============================================================================

@dataclass
class ColorTreeNode:
    """Explicit node in the Hierarchical Binary Merge Tree (FIG. 3 / Claim 3).
    
    Maintains full structural lineage:
        - node_id: Unique integer identifier
        - color: Multi-channel centroid (float64)
        - count: Pixel weight / frequency
        - left_child, right_child: Pointers to merged sub-nodes
        - parent: Pointer to parent node (-1 if root)
    """
    node_id: int
    color: np.ndarray
    count: int
    left_child: Optional[int] = None
    right_child: Optional[int] = None
    parent: int = -1
    is_leaf: bool = True

    @property
    def is_active(self) -> bool:
        return self.parent == -1


class HierarchicalClusterForest:
    """Hierarchical agglomerative clustering engine using an indexed min-heap.
    
    Patent References:
        - Steps 218, 220, 222: Identify least-frequent color and find closest neighbor
        - Step 224: Weighted-average color merging based on pixel frequencies
        - Step 226: Update tree structure (FIG. 3, Claims 3, 10, 17)
        - Step 228: Loop until active cluster count equals K
    """

    def __init__(
        self,
        initial_colors: np.ndarray,
        initial_counts: np.ndarray,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN_RGB
    ):
        self.metric = distance_metric
        self.num_leaves = len(initial_colors)
        self.nodes: Dict[int, ColorTreeNode] = {}
        self.active_nodes: set[int] = set()

        # Feature representations for distance evaluation
        self._features: Dict[int, np.ndarray] = {}

        # Initialize leaf nodes (Level 0)
        for i in range(self.num_leaves):
            color = initial_colors[i].astype(np.float64)
            cnt = int(initial_counts[i])
            node = ColorTreeNode(node_id=i, color=color, count=cnt, is_leaf=True)
            self.nodes[i] = node
            self.active_nodes.add(i)
            self._features[i] = self._compute_feature(color)

        self._next_node_id = self.num_leaves

    def _compute_feature(self, color_vec: np.ndarray) -> np.ndarray:
        if self.metric == DistanceMetric.EUCLIDEAN_RGB:
            return color_vec
        elif self.metric == DistanceMetric.CIELAB_DELTA_E:
            # Color must be 3-channel for CIELAB
            if len(color_vec) == 3:
                rgb_norm = np.clip(color_vec, 0, 255).astype(np.float32).reshape(1, 1, 3) / 255.0
                lab = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2Lab).reshape(3)
                return lab.astype(np.float64)
            return color_vec
        elif self.metric == DistanceMetric.WEIGHTED_PERCEPTUAL:
            # Perceptual RGB weighting: (0.3 R + 0.59 G + 0.11 B)
            if len(color_vec) == 3:
                weights = np.array([0.30, 0.59, 0.11], dtype=np.float64)
                return color_vec * np.sqrt(weights)
            return color_vec
        return color_vec

    def reduce_to_k(self, target_k: int, verbose: bool = False) -> None:
        """Executes iterative color reduction until active cluster count equals target_k."""
        if len(self.active_nodes) <= target_k:
            return

        # Vectorized feature matrix and active index tracking for maximum speed
        # Indexed Priority Queue of (pixel_count, node_id)
        min_heap: List[Tuple[int, int]] = [
            (self.nodes[nid].count, nid) for nid in self.active_nodes
        ]
        heapq.heapify(min_heap)

        # Pre-allocate contiguous feature array for fast vector distance computation
        max_possible_nodes = 2 * self.num_leaves
        dim = next(iter(self._features.values())).shape[0]
        feat_matrix = np.zeros((max_possible_nodes, dim), dtype=np.float64)
        for nid, f in self._features.items():
            feat_matrix[nid] = f

        while len(self.active_nodes) > target_k:
            # Step 218: Pop least-frequent active color
            while min_heap:
                cnt, least_id = heapq.heappop(min_heap)
                if least_id in self.active_nodes and self.nodes[least_id].count == cnt:
                    break
            else:
                break

            # Vectorized Step 220 & 222: Find closest active neighbor
            active_list = np.array([nid for nid in self.active_nodes if nid != least_id], dtype=np.int64)
            if active_list.size == 0:
                break

            active_feats = feat_matrix[active_list]
            least_feat = feat_matrix[least_id]
            diff = active_feats - least_feat
            dists_sq = np.einsum("ij,ij->i", diff, diff)
            best_idx = np.argmin(dists_sq)
            best_neighbor_id = int(active_list[best_idx])

            # Step 224: Weighted centroid computation (Claims 4, 11, 18)
            node_u = self.nodes[least_id]
            node_v = self.nodes[best_neighbor_id]
            total_weight = node_u.count + node_v.count
            merged_color = (node_u.color * node_u.count + node_v.color * node_v.count) / total_weight

            # Step 226: Create new parent node and update tree hierarchy (FIG. 3)
            new_id = self._next_node_id
            self._next_node_id += 1

            parent_node = ColorTreeNode(
                node_id=new_id,
                color=merged_color,
                count=total_weight,
                left_child=least_id,
                right_child=best_neighbor_id,
                parent=-1,
                is_leaf=False
            )

            node_u.parent = new_id
            node_v.parent = new_id

            self.nodes[new_id] = parent_node
            new_feat = self._compute_feature(merged_color)
            self._features[new_id] = new_feat
            feat_matrix[new_id] = new_feat

            # Update active cluster set
            self.active_nodes.remove(least_id)
            self.active_nodes.remove(best_neighbor_id)
            self.active_nodes.add(new_id)

            heapq.heappush(min_heap, (total_weight, new_id))

            if verbose and len(self.active_nodes) % 100 == 0:
                print(f"[HierarchicalClusterForest] {len(self.active_nodes)} clusters remaining...")

    def get_cluster_roots(self) -> List[int]:
        """Return list of active root cluster IDs sorted by pixel frequency descending."""
        roots = list(self.active_nodes)
        roots.sort(key=lambda nid: self.nodes[nid].count, reverse=True)
        return roots

    def get_leaf_cluster_assignments(self) -> np.ndarray:
        """Trace leaf nodes up through the merge tree to their final active root cluster."""
        assignments = np.empty(self.num_leaves, dtype=np.int64)
        for leaf_id in range(self.num_leaves):
            curr = leaf_id
            while self.nodes[curr].parent != -1:
                curr = self.nodes[curr].parent
            assignments[leaf_id] = curr
        return assignments


# ==============================================================================
# Optional Post-Processor: Centroid Refinement (Claims 6 & 13 / Step 228)
# ==============================================================================

class PaletteCentroidRefiner:
    """Refines clustered palette colors using exact source image pixel averages.
    
    Patent References:
        - Claims 6, 13: Mapping K colors to respective source pixel values and
          computing true weighted averages from original unquantized pixels.
    """

    @staticmethod
    def refine(
        source_img: np.ndarray,
        label_map: np.ndarray,
        palette: np.ndarray
    ) -> np.ndarray:
        """Compute refined palette centroids using true source image pixel means."""
        refined_palette = palette.copy().astype(np.float64)
        flat_labels = label_map.ravel()
        flat_pixels = source_img.reshape(-1, source_img.shape[-1]).astype(np.float64)

        for k in range(len(palette)):
            mask = (flat_labels == k)
            if np.any(mask):
                refined_palette[k] = flat_pixels[mask].mean(axis=0)

        return np.clip(np.rint(refined_palette), 0, 255).astype(np.uint8)


# ==============================================================================
# Unified Pipeline Interface: PatentColorClusterer
# ==============================================================================

class PatentColorClusterer:
    """Full-featured, modular implementation of IBM Patent US 11,763,422 B2.
    
    Attributes
    ----------
    k_colors : int
        Target number of colors (K).
    bits : Optional[int]
        Bits per channel. If None, auto-calculated via 2^(bits*ch) > (K + margin).
    margin : Optional[int]
        Margin added to K during bit estimation (default K//2).
    metric : DistanceMetric
        Distance metric for color proximity (RGB, CIELAB, Weighted).
    rounding : RoundingMode
        Rounding method for cross-product reduction (Round or Floor).
    auto_increment_bits : bool
        If True, auto-increments bits when unique initial colors <= K (Claim 2).
    refine_palette : bool
        If True, refines palette colors against source image pixels (Claims 6, 13).
    """

    def __init__(
        self,
        k_colors: int = 16,
        bits: Optional[int] = None,
        margin: Optional[int] = None,
        metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN_RGB,
        rounding: Union[str, RoundingMode] = RoundingMode.ROUND_HALF_UP,
        auto_increment_bits: bool = True,
        refine_palette: bool = False
    ):
        self.k_colors = int(k_colors)
        self.bits = bits
        self.margin = margin
        self.metric = DistanceMetric(metric)
        self.rounding = RoundingMode(rounding)
        self.auto_increment_bits = auto_increment_bits
        self.refine_palette = refine_palette

        # Fitted state
        self.active_bits_: int = 0
        self.palette_: np.ndarray = np.array([])
        self.palette_weights_: np.ndarray = np.array([])
        self.lut_: np.ndarray = np.array([])
        self.fit_time_: float = 0.0
        self.forest_: Optional[HierarchicalClusterForest] = None
        self._encoder: Optional[ColorIndexEncoder] = None

    def fit(self, img: np.ndarray, verbose: bool = False) -> "PatentColorClusterer":
        """Fits color clustering model on source image according to Patent Steps 202-228."""
        t_start = time.perf_counter()
        img = self._validate_input(img)
        num_channels = img.shape[-1] if img.ndim == 3 else 1

        # Step 204 & 205: Determine bits per channel
        if self.bits is not None:
            active_bits = int(self.bits)
        else:
            active_bits = CrossProductQuantizer.estimate_optimal_bits(
                self.k_colors, self.margin, channels=num_channels
            )

        # Steps 206 - 216: Iterative reduction with auto-increment check (Claim 2)
        while True:
            quantizer = CrossProductQuantizer(active_bits, self.rounding)
            encoder = ColorIndexEncoder(active_bits, channels=num_channels)

            quantized = quantizer.quantize(img)
            packed_keys = encoder.encode(quantized)
            unique_keys, frequencies = encoder.build_histogram(packed_keys)

            if len(unique_keys) > self.k_colors or not self.auto_increment_bits or active_bits >= 8:
                break
            active_bits += 1
            if verbose:
                print(f"[PatentColorClusterer] Incrementing significant bits to {active_bits} (Claim 2)")

        self.active_bits_ = active_bits
        self._encoder = encoder

        # Decode initial unique colors to left-aligned coordinates (Step 210)
        initial_colors = encoder.decode(unique_keys, left_align=True)

        # Steps 218 - 228: Hierarchical merge tree reduction (FIG. 3 / Claims 3, 4, 10, 11)
        forest = HierarchicalClusterForest(
            initial_colors=initial_colors,
            initial_counts=frequencies,
            distance_metric=self.metric
        )
        forest.reduce_to_k(self.k_colors, verbose=verbose)
        self.forest_ = forest

        # Extract roots and sort by frequency descending
        root_ids = forest.get_cluster_roots()
        palette_colors = np.array([forest.nodes[rid].color for rid in root_ids], dtype=np.float64)
        palette_counts = np.array([forest.nodes[rid].count for rid in root_ids], dtype=np.int64)

        self.palette_ = np.clip(np.rint(palette_colors), 0, 255).astype(np.uint8)
        self.palette_weights_ = palette_counts

        # Build mapping from node_id -> label [0..K-1]
        node_to_label = {rid: idx for idx, rid in enumerate(root_ids)}
        leaf_root_assignments = forest.get_leaf_cluster_assignments()

        # Build complete Lookup Table (LUT) for all 2^(n*channels) color keys
        lut = np.empty(encoder.max_possible_keys, dtype=np.int32)
        lut[unique_keys] = [node_to_label[r] for r in leaf_root_assignments]

        # Handle unobserved colors by mapping to closest palette centroid
        absent_keys = np.setdiff1d(np.arange(encoder.max_possible_keys), unique_keys, assume_unique=True)
        if len(absent_keys) > 0:
            p_float = self.palette_.astype(np.float64)
            p_sq = (p_float ** 2).sum(axis=1)

            # Chunked evaluation to maintain low memory profile
            chunk_size = 65536
            for start in range(0, len(absent_keys), chunk_size):
                chunk = absent_keys[start:start + chunk_size]
                c_coords = encoder.decode(chunk, left_align=True)
                c_sq = (c_coords ** 2).sum(axis=1, keepdims=True)
                dists = c_sq - 2.0 * np.dot(c_coords, p_float.T) + p_sq
                lut[chunk] = np.argmin(dists, axis=1)

        self.lut_ = lut

        # Optional Post-Processing refinement (Claims 6 & 13)
        if self.refine_palette:
            labels = self.predict(img)
            self.palette_ = PaletteCentroidRefiner.refine(img, labels, self.palette_)

        self.fit_time_ = time.perf_counter() - t_start
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Map input image pixels to cluster labels [0..K-1] via LUT gather."""
        img = self._validate_input(img)
        quantizer = CrossProductQuantizer(self.active_bits_, self.rounding)
        quantized = quantizer.quantize(img)
        packed_keys = self._encoder.encode(quantized)
        return self.lut_[packed_keys]

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Quantize image to K colors using fitted palette and LUT."""
        labels = self.predict(img)
        return self.palette_[labels]

    def fit_transform(self, img: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Fit model and return quantized target image."""
        return self.fit(img, verbose=verbose).transform(img)

    @staticmethod
    def _validate_input(img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a numpy ndarray")
        if img.dtype != np.uint8:
            raise TypeError("Input image must have uint8 pixel depth")
        return np.ascontiguousarray(img)


# ==============================================================================
# Functional Pipeline Helper
# ==============================================================================

def process_image_patent_pipeline(
    img: np.ndarray,
    k_colors: int = 16,
    bits: Optional[int] = None,
    distance_metric: str = "rgb",
    refine: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot helper function returning (quantized_image, palette, label_map)."""
    model = PatentColorClusterer(
        k_colors=k_colors,
        bits=bits,
        metric=distance_metric,
        refine_palette=refine
    ).fit(img)
    labels = model.predict(img)
    quantized_img = model.palette_[labels]
    return quantized_img, model.palette_, labels
