#!/usr/bin/env python3
"""
Fast Color Clustering for Preprocessing an Image
================================================
Python / OpenCV implementation of IBM Patent US 11,763,422 B2
"Fast color clustering for preprocessing an image"
Inventors: Marc Fiammante, Jean-Armand Broyelle
Assignee: International Business Machines Corporation (IBM)
Granted: September 19, 2023

Overview & Patent Architecture
------------------------------
This module implements the complete computer vision and image processing pipeline
described in US 11,763,422 B2 (FIG. 1, FIGS. 2A-2B, FIG. 3, Claims 1-20):

1. Color Clustering System Architecture (FIG. 1):
   - Cross-Product Module (106): Steps 204-208 / Claims 1, 7, 14
     Calculates required significant bits n per channel given target colors K and
     margin (e.g. K/2): 2^(n * channels) > (K + margin). Computes cross-product
     q = round((2^n - 1) * v / 255) (or (v * 2^n) / 256) via fast integer or LUT.
   - Left Alignment Module (108): Step 210 / Claims 1, 2
     Left-justifies reduced channel values to 8 bits: v' = q << (8 - n).
     Ensures consistent scale for distance computation and initial target image.
     If unique colors <= K, automatically increments n and re-reduces (Claim 2).
   - Map Generation Module (110): Step 212 / Claims 1, 5, 12, 19
     Builds flat map (histogram) when colors < 4096 or hash map with unique key
     concatenating significant bits: rrrgggbbb -> idx = (r << 2n) | (g << n) | b.
   - Tree Module (112): Steps 218-228 / Claims 1, 3, 4, 10, 11, 17, 18 / FIG. 3
     Builds a hierarchical binary merge tree. Iteratively:
       a. Identifies least frequent color node (weight = pixel count).
       b. Computes color distance to all remaining active nodes.
       c. Selects closest color node (minimum distance).
       d. Merges nodes via pixel-count weighted average.
       e. Replaces both with parent node and moves children to lower level.
       f. Repeats until exactly K colors remain.
   - Optional Post-Processing (Claims 6, 13):
     Computes exact cluster centroids by averaging original source image pixel
     values mapped to each of the K final clusters.

Performance:
  O(L^2) where L <= 2^(3n) is the number of distinct reduced colors (independent
  of image resolution HxW once histogram is constructed). Extremely fast compared
  to cv2.kmeans while maintaining high PSNR and structural fidelity.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "cross_product_reduce",
    "compute_significant_bits",
    "color_index",
    "index_to_color",
    "MergeTree",
    "FastColorClustering",
    "fast_color_clustering",
    "calculate_psnr",
]


# ==============================================================================
# Helper Functions: Bit calculation, Cross-Product, Left Alignment, Indexing
# ==============================================================================

def compute_significant_bits(k: int, margin: Optional[int] = None, channels: int = 3) -> int:
    """Determine minimum bits per channel according to Step 205 and Claim 1:
    
        2 ^ (bits * channels) > (K + margin)
    
    Parameters
    ----------
    k : int
        Target number of colors (positive integer).
    margin : int, optional
        Predetermined margin (default is K // 2 as described in Step 204).
    channels : int
        Number of image channels (default 3 for RGB/BGR).

    Returns
    -------
    int
        Bits per channel (clamped between 1 and 8).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if margin is None:
        margin = max(1, k // 2)
    
    target_count = k + margin
    for bits in range(1, 9):
        if (1 << (bits * channels)) > target_count:
            return bits
    return 8


def cross_product_reduce(
    img: np.ndarray,
    bits: int,
    rounding: str = "round"
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Steps 206, 208, 210 of Patent US 11,763,422 B2.
    
    Computes cross-product reduction on pixel values and left aligns to 8 bits.

    Formula from Patent Specification (Col. 4, lines 40-55):
        val = (max_colors_number_for_significant_bits * color) / 255.0
        q = int(round(val))
        aligned = q << (8 - bits)

    Parameters
    ----------
    img : np.ndarray
        8-bit image of shape (H, W, C) or (H, W).
    bits : int
        Significant bits per channel (1 <= bits <= 8).
    rounding : str
        "round" (patent specification standard) or "floor" (fast bit-shift).

    Returns
    -------
    levels : np.ndarray
        Quantized levels q in [0, 2^bits - 1], dtype uint8.
    aligned : np.ndarray
        Left-aligned values q << (8 - bits) in [0, 255], dtype uint8.
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in range 1..8, got {bits}")
    if img.dtype != np.uint8:
        raise TypeError("Source image must have uint8 data type")

    max_val = (1 << bits) - 1
    shift = 8 - bits

    if rounding == "round":
        # Patent exact formula: ( (2^n - 1) * color ) / 255.0
        # Implemented using a 256-element LUT for maximum speed and exactness
        lut = np.round((np.arange(256, dtype=np.float32) * max_val) / 255.0).astype(np.uint8)
        levels = cv2.LUT(img, lut) if img.ndim in (2, 3) else lut[img]
    elif rounding == "floor":
        levels = (img >> shift).astype(np.uint8)
    else:
        raise ValueError("rounding must be either 'round' or 'floor'")

    aligned = (levels << shift).astype(np.uint8)
    return levels, aligned


def color_index(levels: np.ndarray, bits: int) -> np.ndarray:
    """Concatenate channel significant bits into a single integer key rrrgggbbb.
    
    According to Patent Col. 5 lines 40-60 and Claim 5:
        colorindex = (ch[0] << (2 * bits)) | (ch[1] << bits) | ch[2]
    
    Parameters
    ----------
    levels : np.ndarray
        Array of shape (..., 3) containing per-channel quantized levels.
    bits : int
        Number of significant bits per channel.

    Returns
    -------
    np.ndarray
        Integer index array of shape (...).
    """
    l = levels.astype(np.int32)
    return (l[..., 0] << (2 * bits)) | (l[..., 1] << bits) | l[..., 2]


def index_to_color(idx: np.ndarray, bits: int, left_align: bool = True) -> np.ndarray:
    """Decode integer index rrrgggbbb back into 3-channel values.
    
    Parameters
    ----------
    idx : np.ndarray
        Array of integer color indices.
    bits : int
        Significant bits per channel.
    left_align : bool
        If True, left aligns to 8 bits [0..255]. Otherwise returns [0..2^bits - 1].

    Returns
    -------
    np.ndarray
        Color array of shape (..., 3), float64.
    """
    mask = (1 << bits) - 1
    c0 = (idx >> (2 * bits)) & mask
    c1 = (idx >> bits) & mask
    c2 = idx & mask
    c = np.stack([c0, c1, c2], axis=-1).astype(np.float64)
    if left_align:
        c *= (1 << (8 - bits))
    return c


# ==============================================================================
# Binary Merge Tree (Tree Module 112 / FIG. 3 / Claims 3, 4, 10, 11, 17, 18)
# ==============================================================================

@dataclass
class MergeTree:
    """Hierarchical binary merge tree representing the color reduction process.
    
    Initial leaves (0..L-1) represent unique colors after cross-product reduction.
    Each merge operation creates a new parent node with branches to the two
    merged nodes, and records the pixel-weighted color and combined frequency.
    """
    colors: np.ndarray             # (N, 3) float64 color values of all nodes
    counts: np.ndarray             # (N,) int64 pixel count of all nodes
    parent: np.ndarray             # (N,) int64 parent index (-1 if root)
    merges: List[Tuple[int, int, int]] = field(default_factory=list) # (least_freq, closest, new_node)
    n_leaves: int = 0

    def get_roots(self) -> np.ndarray:
        """Return indices of all currently active root nodes (clusters)."""
        return np.flatnonzero((self.parent < 0) & (self.counts > 0))

    def get_leaf_to_root_map(self) -> np.ndarray:
        """Compute the final cluster root assigned to each initial leaf color."""
        p = self.parent.copy()
        node = np.arange(len(p), dtype=np.int64)
        r = np.where(p < 0, node, p)
        while True:
            nxt = np.where(p[r] < 0, r, p[r])
            if np.array_equal(nxt, r):
                break
            r = nxt
        return r[:self.n_leaves]


def _build_and_merge_tree(
    initial_colors: np.ndarray,
    initial_counts: np.ndarray,
    target_k: int,
    distance_metric: str = "rgb",
    verbose: bool = False
) -> MergeTree:
    """Execute iterative least-frequent color merging (FIG. 2B, Steps 218-228).
    
    Algorithm:
    ----------
    While remaining_colors > K:
      1. Identify least frequent color i (min pixel count).
      2. Compute pairwise distance from i to all other active colors.
      3. Identify color j with minimum distance to i.
      4. Compute merged color = (color[i]*count[i] + color[j]*count[j]) / (count[i] + count[j]).
      5. Create new node in tree; mark i and j as inactive children of new node.
    """
    L = len(initial_colors)
    if L <= target_k:
        # Already <= target_k colors, no merging needed
        tree = MergeTree(
            colors=initial_colors.copy().astype(np.float64),
            counts=initial_counts.copy().astype(np.int64),
            parent=np.full(L, -1, dtype=np.int64),
            merges=[],
            n_leaves=L
        )
        return tree

    max_nodes = max(2 * L - 1, 1)
    colors = np.zeros((max_nodes, 3), dtype=np.float64)
    counts = np.zeros(max_nodes, dtype=np.int64)
    parent = np.full(max_nodes, -1, dtype=np.int64)

    colors[:L] = initial_colors
    counts[:L] = initial_counts

    tree = MergeTree(colors=colors, counts=counts, parent=parent, merges=[], n_leaves=L)

    def compute_feature(c_rgb: np.ndarray) -> np.ndarray:
        if distance_metric == "rgb":
            return c_rgb
        # Convert RGB/BGR to CIELAB for perceptual distance
        c_clipped = np.clip(c_rgb, 0, 255).astype(np.float32).reshape(-1, 1, 3) / 255.0
        lab = cv2.cvtColor(c_clipped, cv2.COLOR_RGB2Lab)
        return lab.reshape(-1, 3).astype(np.float64)

    features = np.zeros((max_nodes, 3), dtype=np.float64)
    features[:L] = compute_feature(initial_colors)

    alive = np.zeros(max_nodes, dtype=bool)
    alive[:L] = True

    # Priority key for least frequency
    freq_keys = np.where(alive, counts.astype(np.float64), np.inf)

    next_node_id = L
    n_alive = L

    while n_alive > target_k:
        # Step 218: Identify least frequent color
        i = int(np.argmin(freq_keys))

        # Step 220 & 222: Determine Euclidean distances to all other alive colors
        diff = features - features[i]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        dist_sq[~alive] = np.inf
        dist_sq[i] = np.inf

        j = int(np.argmin(dist_sq))

        # Step 224: Generate merged color by pixel-count weighted average
        ci, cj = colors[i], colors[j]
        ni, nj = counts[i], counts[j]
        n_merged = ni + nj
        c_merged = (ci * ni + cj * nj) / n_merged

        new_id = next_node_id
        next_node_id += 1

        colors[new_id] = c_merged
        counts[new_id] = n_merged
        features[new_id] = (
            compute_feature(c_merged.reshape(1, 3))[0]
            if distance_metric != "rgb" else c_merged
        )

        # Step 226: Update tree structure
        parent[i] = new_id
        parent[j] = new_id
        alive[i] = False
        alive[j] = False
        alive[new_id] = True

        freq_keys[i] = np.inf
        freq_keys[j] = np.inf
        freq_keys[new_id] = n_merged

        tree.merges.append((i, j, new_id))
        n_alive -= 1

        if verbose and n_alive % 100 == 0:
            print(f"[FastColorClustering] Merge step: {n_alive} colors remaining")

    return tree


# ==============================================================================
# Main Class: FastColorClustering
# ==============================================================================

class FastColorClustering:
    """OpenCV & Python implementation of IBM Patent US 11,763,422 B2.
    
    Provides model fitting (histogram extraction + hierarchical merging) and
    ultra-fast transformation using precomputed index Lookup Tables (LUT).

    Parameters
    ----------
    n_colors : int, default=16
        Target number of colors (K).
    bits : int, optional
        Number of significant bits per channel (1..8). If None, automatically
        computed using 2^(bits*channels) > (K + margin).
    margin : int, optional
        Margin added to K for bit estimation. Defaults to K // 2.
    auto_increment_bits : bool, default=True
        If True, implements Claim 2: increments bits if initial unique colors <= K.
    rounding : str, default="round"
        Cross-product rounding mode ("round" or "floor").
    distance : str, default="rgb"
        Color distance metric ("rgb" for Euclidean or "lab" for CIELAB DeltaE).
    channel_order : str, default="bgr"
        Channel ordering of input images ("bgr" for cv2.imread, "rgb" for PIL/matplotlib).
    post_process_refine : bool, default=False
        If True, implements Claims 6 & 13: refines palette colors by calculating
        the exact mean of source image pixels mapped to each cluster.
    """

    def __init__(
        self,
        n_colors: int = 16,
        bits: Optional[int] = None,
        margin: Optional[int] = None,
        auto_increment_bits: bool = True,
        rounding: str = "round",
        distance: str = "rgb",
        channel_order: str = "bgr",
        post_process_refine: bool = False
    ):
        self.n_colors = int(n_colors)
        self.margin = margin
        self.bits = bits
        self.auto_increment_bits = auto_increment_bits
        self.rounding = rounding
        self.distance = distance.lower()
        self.channel_order = channel_order.lower()
        self.post_process_refine = post_process_refine

        # Fitted attributes
        self.active_bits_: int = 0
        self.n_initial_colors_: int = 0
        self.palette_: np.ndarray = np.array([])
        self.palette_counts_: np.ndarray = np.array([])
        self.tree_: Optional[MergeTree] = None
        self.lut_: np.ndarray = np.array([])
        self.fit_time_: float = 0.0

    def fit(self, img: np.ndarray, verbose: bool = False) -> "FastColorClustering":
        """Fit color clustering model on source image.
        
        Parameters
        ----------
        img : np.ndarray
            Input image of shape (H, W, 3), uint8.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        self : FastColorClustering
        """
        img = self._validate_image(img)
        t0 = time.perf_counter()

        # Step 204 & 205: Determine initial bits per channel
        if self.bits is not None:
            active_bits = int(self.bits)
        else:
            active_bits = compute_significant_bits(self.n_colors, self.margin, channels=img.shape[2])

        # Step 206 - 216: Cross-product reduction with auto-increment check (Claim 2)
        while True:
            levels, _ = cross_product_reduce(img, active_bits, self.rounding)
            idx = color_index(levels, active_bits)
            total_possible_indices = 1 << (3 * active_bits)
            hist = np.bincount(idx.ravel(), minlength=total_possible_indices)
            present_indices = np.flatnonzero(hist)
            n_unique = len(present_indices)

            if n_unique > self.n_colors or not self.auto_increment_bits or active_bits >= 8:
                break
            # Step 216: Increment bits per channel if remaining unique colors <= K
            active_bits += 1
            if verbose:
                print(f"[FastColorClustering] Unique colors ({n_unique}) <= K ({self.n_colors}), "
                      f"incrementing to {active_bits} bits")

        self.active_bits_ = active_bits
        self.n_initial_colors_ = n_unique

        # Extract initial colors from histogram
        colors0 = index_to_color(present_indices, active_bits, left_align=True)
        counts0 = hist[present_indices]

        # Handle channel order for distance calculation
        if self.distance == "lab" and self.channel_order == "bgr":
            perm = [2, 1, 0]
            tree = _build_and_merge_tree(
                colors0[:, perm], counts0, self.n_colors, distance_metric="lab", verbose=verbose
            )
            # Revert colors back to native order
            tree.colors = tree.colors[:, perm]
        else:
            tree = _build_and_merge_tree(
                colors0, counts0, self.n_colors, distance_metric=self.distance, verbose=verbose
            )

        self.tree_ = tree

        # Identify roots of the merge tree (final clusters)
        roots = tree.get_roots()
        # Sort clusters by frequency descending
        order = np.argsort(-tree.counts[roots])
        roots = roots[order]

        self.palette_ = np.clip(np.rint(tree.colors[roots]), 0, 255).astype(np.uint8)
        self.palette_counts_ = tree.counts[roots]

        # Map merge tree node IDs to [0..K-1] cluster labels
        node_to_label = np.full(len(tree.parent), -1, dtype=np.int64)
        node_to_label[roots] = np.arange(len(roots))

        # Build full Lookup Table (LUT) over all 2^(3n) color indices
        leaf_roots = tree.get_leaf_to_root_map()
        lut = np.empty(total_possible_indices, dtype=np.int32)
        lut[present_indices] = node_to_label[leaf_roots]

        # For colors not observed in the training image, map to nearest cluster in palette
        absent_indices = np.setdiff1d(np.arange(total_possible_indices), present_indices, assume_unique=True)
        if absent_indices.size > 0:
            # Batch distance computation to prevent excessive memory usage or overflow
            p_palette = self.palette_.astype(np.float64)
            p_norm_sq = (p_palette ** 2).sum(axis=1) # (K,)
            
            chunk_size = 65536
            for chunk_start in range(0, len(absent_indices), chunk_size):
                chunk_idx = absent_indices[chunk_start:chunk_start + chunk_size]
                c_absent = index_to_color(chunk_idx, active_bits, left_align=True) # float64
                c_norm_sq = (c_absent ** 2).sum(axis=1, keepdims=True) # (M, 1)
                dists = c_norm_sq - 2.0 * np.dot(c_absent, p_palette.T) + p_norm_sq # (M, K)
                lut[chunk_idx] = np.argmin(dists, axis=1)

        self.lut_ = lut

        # Optional Post-Processing (Claims 6 & 13)
        if self.post_process_refine:
            labels = self.predict(img)
            refined_palette = np.zeros_like(self.palette_, dtype=np.float64)
            for k_idx in range(len(self.palette_)):
                mask = (labels == k_idx)
                if np.any(mask):
                    refined_palette[k_idx] = img[mask].mean(axis=0)
                else:
                    refined_palette[k_idx] = self.palette_[k_idx]
            self.palette_ = np.clip(np.rint(refined_palette), 0, 255).astype(np.uint8)

        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Map source image pixels to cluster labels [0..K-1].
        
        Parameters
        ----------
        img : np.ndarray
            Input image of shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Label map of shape (H, W), dtype int32.
        """
        img = self._validate_image(img)
        levels, _ = cross_product_reduce(img, self.active_bits_, self.rounding)
        indices = color_index(levels, self.active_bits_)
        return self.lut_[indices]

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Quantize image to K colors using fitted palette and LUT.
        
        Parameters
        ----------
        img : np.ndarray
            Input image of shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Clustered target image of shape (H, W, 3), dtype uint8.
        """
        labels = self.predict(img)
        return self.palette_[labels]

    def fit_transform(self, img: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Fit on image and return the quantized target image."""
        return self.fit(img, verbose=verbose).transform(img)

    @staticmethod
    def _validate_image(img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("Expected numpy ndarray for image")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel color image of shape (H, W, 3), got {img.shape}")
        return np.ascontiguousarray(img, dtype=np.uint8)


# ==============================================================================
# Functional Interface & Quality Metrics
# ==============================================================================

def fast_color_clustering(
    img: np.ndarray,
    n_colors: int = 16,
    bits: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Functional one-shot interface for fast color clustering.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (H, W, 3), uint8.
    n_colors : int
        Target number of colors K.
    bits : int, optional
        Significant bits per channel (auto-calculated if None).
    **kwargs :
        Additional arguments passed to FastColorClustering.

    Returns
    -------
    target_image : np.ndarray
        Quantized image of shape (H, W, 3), uint8.
    palette : np.ndarray
        Array of K representative colors (K, 3), uint8.
    label_map : np.ndarray
        Array of cluster assignments (H, W), int32.
    """
    model = FastColorClustering(n_colors=n_colors, bits=bits, **kwargs).fit(img)
    labels = model.predict(img)
    target_img = model.palette_[labels]
    return target_img, model.palette_, labels


def calculate_psnr(img_original: np.ndarray, img_reconstructed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = np.mean((img_original.astype(np.float64) - img_reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


# ==============================================================================
# Command Line Interface (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast Color Clustering for Preprocessing an Image (IBM Patent US 11,763,422 B2)"
    )
    parser.add_argument("input", help="Path to input source image")
    parser.add_argument("-o", "--output", default="clustered_output.png", help="Path for output target image")
    parser.add_argument("-k", "--colors", type=int, default=16, help="Target number of colors K (default: 16)")
    parser.add_argument("-b", "--bits", type=int, default=None, help="Bits per channel (default: auto)")
    parser.add_argument("--rounding", choices=["round", "floor"], default="round", help="Rounding mode")
    parser.add_argument("--distance", choices=["rgb", "lab"], default="rgb", help="Distance metric")
    parser.add_argument("--refine", action="store_true", help="Post-process palette using source pixel averages")
    parser.add_argument("--labels", help="Optional path to save label map as PNG image")
    parser.add_argument("--compare-kmeans", action="store_true", help="Run benchmark comparison against cv2.kmeans")

    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Error: Unable to read image from {args.input}")

    print(f"Loaded image: {args.input} ({img.shape[1]}x{img.shape[0]}, {img.shape[2]} channels)")

    model = FastColorClustering(
        n_colors=args.colors,
        bits=args.bits,
        rounding=args.rounding,
        distance=args.distance,
        channel_order="bgr",
        post_process_refine=args.refine
    )

    t0 = time.perf_counter()
    clustered = model.fit_transform(img)
    dt = time.perf_counter() - t0

    cv2.imwrite(args.output, clustered)
    psnr_val = calculate_psnr(img, clustered)

    print(f"\n--- Fast Color Clustering Results (US 11,763,422 B2) ---")
    print(f"Significant bits/channel : {model.active_bits_}")
    print(f"Initial unique colors    : {model.n_initial_colors_}")
    print(f"Final palette colors (K) : {len(model.palette_)}")
    print(f"Processing time          : {dt * 1000.0:.2f} ms")
    print(f"PSNR quality             : {psnr_val:.2f} dB")
    print(f"Saved clustered image to : {args.output}")

    if args.labels:
        labels = model.predict(img)
        save_labels = labels.astype(np.uint8 if args.colors <= 256 else np.uint16)
        cv2.imwrite(args.labels, save_labels)
        print(f"Saved label map to       : {args.labels}")

    if args.compare_kmeans:
        print(f"\n--- Benchmarking against OpenCV cv2.kmeans ---")
        pixel_data = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        t_km0 = time.perf_counter()
        _, km_labels, km_centers = cv2.kmeans(
            pixel_data, args.colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        dt_km = time.perf_counter() - t_km0
        km_img = np.uint8(np.clip(km_centers, 0, 255))[km_labels.ravel()].reshape(img.shape)
        km_psnr = calculate_psnr(img, km_img)

        print(f"cv2.kmeans time          : {dt_km * 1000.0:.2f} ms")
        print(f"cv2.kmeans PSNR          : {km_psnr:.2f} dB")
        print(f"Speedup factor           : {dt_km / dt:.2f}x faster")


if __name__ == "__main__":
    main()
