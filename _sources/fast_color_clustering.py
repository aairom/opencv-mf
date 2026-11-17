#!/usr/bin/env python3
"""
Fast color clustering for preprocessing an image
================================================
Python / OpenCV reference implementation of US 11,763,422 B2
(M. Fiammante, J.-A. Broyelle - IBM, granted 2023-09-19).

Pipeline (claim 1 / FIGS. 2A-2B):

  1. Receive a source image and a target number of colors K.
  2. Cross-product reduction: for every channel value v (8 bits) and a number
     of bits per channel n, compute  q = v * 2^n / 2^8, rounded to an integer.
  3. Left alignment: v' = q << (8 - n). The result is the initial target image,
     still in the 0..255 range, with at most 2^(3n) distinct colors.
  4. Unique color index "rrrgggbbb": idx = (q0 << 2n) | (q1 << n) | q2,
     so a single np.bincount gives the color frequencies.
  5. Repeat until the color count equals K:
        a. find the least frequent color,
        b. compute its distances to every other remaining color,
        c. take the least distance -> closest color,
        d. merged color = pixel-count-weighted average of the two,
        e. replace both by the merged color (count = sum of counts),
           the new tree node branches to the two merged nodes.
  6. Map every pixel through the merge tree to its final color.

Because the clustering runs on the (small) reduced palette and never on the
pixels, the cost is independent of image size once the histogram is built.
The fitted model is a full lookup table over all 2^(3n) indices, so applying
it to new images (e.g. a whole training set for a neural network) is just
one reduction plus one gather.

Channel order is whatever the array holds (BGR for cv2.imread); the method
is symmetric in the channels.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "cross_product_reduce",
    "color_index",
    "FastColorClustering",
    "fast_color_clustering",
]


# --------------------------------------------------------------------------
# Steps 2-4: cross-product reduction, left alignment, rrrgggbbb index
# --------------------------------------------------------------------------
def cross_product_reduce(img: np.ndarray, bits: int, rounding: str = "round"):
    """Reduce an 8-bit image to `bits` significant bits per channel.

    Returns (levels, aligned):
      levels  : uint8 array, values in [0, 2^bits - 1]   (q)
      aligned : uint8 array, levels left-aligned to 8 bits (q << (8-bits))

    rounding = "round"  -> q = round(v * 2^n / 256), clipped to 2^n - 1
               "floor"  -> q = (v * 2^n) >> 8   (pure integer, == v >> (8-n))
    """
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in 1..8")
    if img.dtype != np.uint8:
        raise TypeError("expected an 8-bit image")
    scale = 1 << bits
    if rounding == "round":
        q = np.rint(img.astype(np.float32) * (scale / 256.0))
        q = np.minimum(q, scale - 1).astype(np.uint8)
    elif rounding == "floor":
        q = ((img.astype(np.uint16) * scale) >> 8).astype(np.uint8)
    else:
        raise ValueError("rounding must be 'round' or 'floor'")
    aligned = (q << (8 - bits)).astype(np.uint8)
    return q, aligned


def color_index(levels: np.ndarray, bits: int) -> np.ndarray:
    """Unique index rrrgggbbb from per-channel levels (H, W, 3) -> (H, W)."""
    l = levels.astype(np.int32)
    return (l[..., 0] << (2 * bits)) | (l[..., 1] << bits) | l[..., 2]


def _index_to_aligned_color(idx: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of color_index, returned left-aligned (float64, 0..255)."""
    m = (1 << bits) - 1
    c = np.stack([(idx >> (2 * bits)) & m, (idx >> bits) & m, idx & m], axis=-1)
    return (c << (8 - bits)).astype(np.float64)


# --------------------------------------------------------------------------
# Step 5: least-frequent-color merging with a merge tree
# --------------------------------------------------------------------------
@dataclass
class MergeTree:
    """Binary merge tree (FIG. 3). Leaves 0..L-1 are the initial colors."""
    colors: np.ndarray   # (N, 3) float  color of every node
    counts: np.ndarray   # (N,)  int64  pixel count of every node
    parent: np.ndarray   # (N,)  int64  parent node, -1 for a root
    merges: list = field(default_factory=list)  # (least_freq, closest, new)
    n_leaves: int = 0

    def roots(self) -> np.ndarray:
        return np.flatnonzero((self.parent < 0) & (self.counts > 0))

    def leaf_roots(self) -> np.ndarray:
        """Root node reached from each leaf (vectorised pointer jumping)."""
        p = self.parent.copy()
        node = np.arange(len(p))
        r = np.where(p < 0, node, p)
        while True:
            nxt = np.where(p[r] < 0, r, p[r])
            if np.array_equal(nxt, r):
                break
            r = nxt
        return r[: self.n_leaves]


def _merge_colors(colors0, counts0, k, distance="rgb", verbose=False):
    L = len(colors0)
    N = max(2 * L - 1, 1)
    colors = np.zeros((N, 3), np.float64)
    counts = np.zeros(N, np.int64)
    parent = np.full(N, -1, np.int64)
    colors[:L] = colors0
    counts[:L] = counts0
    tree = MergeTree(colors, counts, parent, [], L)

    # feature space used for the distance (the merge itself stays in RGB)
    def feat(c):
        if distance == "rgb":
            return c
        lab = cv2.cvtColor(
            np.clip(c, 0, 255).astype(np.float32).reshape(-1, 1, 3) / 255.0,
            cv2.COLOR_RGB2Lab)
        return lab.reshape(-1, 3).astype(np.float64)

    feats = np.zeros((N, 3), np.float64)
    feats[:L] = feat(colors0)

    alive = np.zeros(N, bool)
    alive[:L] = True
    cnt_key = np.where(alive, counts.astype(np.float64), np.inf)
    nxt = L
    n_alive = L
    while n_alive > k:
        # a. least frequent color
        i = int(np.argmin(cnt_key))
        # b./c. distances to all other remaining colors, least distance
        d = np.einsum("ij,ij->i", feats - feats[i], feats - feats[i])
        d[~alive] = np.inf
        d[i] = np.inf
        j = int(np.argmin(d))
        # d. weighted average by pixel counts
        ni, nj = counts[i], counts[j]
        new = nxt
        nxt += 1
        colors[new] = (colors[i] * ni + colors[j] * nj) / (ni + nj)
        counts[new] = ni + nj
        feats[new] = feat(colors[new:new + 1])[0] if distance != "rgb" else colors[new]
        # e. replace both colors by the merged one (new tree node)
        parent[i] = parent[j] = new
        alive[i] = alive[j] = False
        alive[new] = True
        cnt_key[i] = cnt_key[j] = np.inf
        cnt_key[new] = counts[new]
        tree.merges.append((i, j, new))
        n_alive -= 1
        if verbose and n_alive % 500 == 0:
            print(f"  {n_alive} colors left")
    return tree


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
class FastColorClustering:
    """Fit on one image, then transform any image of the same color space.

    Parameters
    ----------
    n_colors : target number of colors K
    bits     : bits per channel kept by the cross-product reduction (1..8)
    rounding : "round" (patent abstract) or "floor"
    distance : "rgb" (Euclidean on channel values) or "lab" (CIE76 dE);
               for "lab", input arrays are assumed RGB order (use
               channel_order="bgr" for cv2.imread images).
    """

    def __init__(self, n_colors=16, bits=4, rounding="round", distance="rgb",
                 channel_order="bgr"):
        self.n_colors = int(n_colors)
        self.bits = int(bits)
        self.rounding = rounding
        self.distance = distance
        self.channel_order = channel_order.lower()

    # ---- fitting -------------------------------------------------------
    def fit(self, img: np.ndarray, verbose=False):
        img = self._check(img)
        t0 = time.perf_counter()
        levels, self.reduced_ = cross_product_reduce(img, self.bits, self.rounding)
        idx = color_index(levels, self.bits)
        hist = np.bincount(idx.ravel(), minlength=1 << (3 * self.bits))
        present = np.flatnonzero(hist)
        self.n_initial_colors_ = len(present)

        colors0 = _index_to_aligned_color(present, self.bits)
        if self.distance == "lab" and self.channel_order == "bgr":
            # distance computed in RGB order, merge stays in native order
            perm = [2, 1, 0]
            tree = _merge_colors(colors0[:, perm], hist[present],
                                 self.n_colors, "lab", verbose)
            tree.colors = tree.colors[:, perm]
        else:
            tree = _merge_colors(colors0, hist[present], self.n_colors,
                                 self.distance, verbose)
        self.tree_ = tree

        # final palette = roots of the merge tree, sorted by frequency
        roots = tree.roots()
        order = np.argsort(-tree.counts[roots])
        roots = roots[order]
        self.palette_ = np.clip(np.rint(tree.colors[roots]), 0, 255).astype(np.uint8)
        self.palette_counts_ = tree.counts[roots]
        node_to_label = np.full(len(tree.parent), -1, np.int64)
        node_to_label[roots] = np.arange(len(roots))

        # full LUT over every possible rrrgggbbb index
        lut = np.empty(1 << (3 * self.bits), np.int32)
        lut[present] = node_to_label[tree.leaf_roots()]
        absent = np.setdiff1d(np.arange(len(lut)), present, assume_unique=True)
        if absent.size:  # colors never seen during fit -> nearest palette color
            c = _index_to_aligned_color(absent, self.bits)
            p = self.palette_.astype(np.float64)
            d = (c ** 2).sum(1)[:, None] - 2 * c @ p.T + (p ** 2).sum(1)[None, :]
            lut[absent] = np.argmin(d, axis=1)
        self.lut_ = lut
        self.fit_time_ = time.perf_counter() - t0
        return self

    # ---- applying ------------------------------------------------------
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Label map (H, W) of palette indices - handy as NN input."""
        img = self._check(img)
        levels, _ = cross_product_reduce(img, self.bits, self.rounding)
        return self.lut_[color_index(levels, self.bits)]

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Target image with at most n_colors colors (same shape/dtype)."""
        return self.palette_[self.predict(img)]

    def fit_transform(self, img, verbose=False):
        return self.fit(img, verbose).transform(img)

    @staticmethod
    def _check(img):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("expected an (H, W, 3) color image")
        return np.ascontiguousarray(img, dtype=np.uint8)


def fast_color_clustering(img, n_colors=16, bits=4, **kw):
    """One-shot helper: returns (target_image, palette, label_map)."""
    m = FastColorClustering(n_colors, bits, **kw).fit(img)
    labels = m.predict(img)
    return m.palette_[labels], m.palette_, labels


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser(description="US11763422 fast color clustering")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="clustered.png")
    ap.add_argument("-k", "--colors", type=int, default=16)
    ap.add_argument("-b", "--bits", type=int, default=4)
    ap.add_argument("--rounding", choices=["round", "floor"], default="round")
    ap.add_argument("--distance", choices=["rgb", "lab"], default="rgb")
    ap.add_argument("--labels", help="also save the label map (uint8/uint16 PNG)")
    ap.add_argument("--compare-kmeans", action="store_true")
    a = ap.parse_args()

    img = cv2.imread(a.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {a.input}")
    m = FastColorClustering(a.colors, a.bits, a.rounding, a.distance, "bgr")
    t0 = time.perf_counter()
    out = m.fit_transform(img)
    dt = time.perf_counter() - t0
    cv2.imwrite(a.output, out)
    print(f"{img.shape[1]}x{img.shape[0]}  initial colors after {a.bits}-bit "
          f"reduction: {m.n_initial_colors_}  ->  {len(m.palette_)} colors")
    print(f"fast clustering: {dt * 1000:.1f} ms   PSNR {_psnr(img, out):.2f} dB")
    if a.labels:
        lab = m.predict(img)
        cv2.imwrite(a.labels, lab.astype(np.uint8 if a.colors <= 256 else np.uint16))
    if a.compare_kmeans:
        Z = img.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        t0 = time.perf_counter()
        _, lbl, ctr = cv2.kmeans(Z, a.colors, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        dk = time.perf_counter() - t0
        km = np.uint8(np.clip(ctr, 0, 255))[lbl.ravel()].reshape(img.shape)
        print(f"cv2.kmeans     : {dk * 1000:.1f} ms   PSNR {_psnr(img, km):.2f} dB")


if __name__ == "__main__":
    main()
