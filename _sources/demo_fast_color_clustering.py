#!/usr/bin/env python3
"""Demo + self-checks for fast_color_clustering.py (US 11,763,422 B2)."""
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from fast_color_clustering import (FastColorClustering, color_index,
                                   cross_product_reduce, _psnr)


def brute_force_reference(img, k, bits):
    """Literal, slow re-implementation of claim 1 on a dict, for checking."""
    q, _ = cross_product_reduce(img, bits)
    idx = color_index(q, bits).ravel()
    ids, cnt = np.unique(idx, return_counts=True)
    m = (1 << bits) - 1
    cols = {int(i): np.array([(i >> 2 * bits) & m, (i >> bits) & m, i & m],
                             float) * (1 << (8 - bits)) for i in ids}
    clusters = {int(i): (cols[int(i)], int(c), {int(i)}) for i, c in zip(ids, cnt)}
    key = len(ids) * 10
    while len(clusters) > k:
        lf = min(clusters, key=lambda c: (clusters[c][1], list(clusters).index(c)))
        cl = min((c for c in clusters if c != lf),
                 key=lambda c: np.sum((clusters[c][0] - clusters[lf][0]) ** 2))
        (c1, n1, s1), (c2, n2, s2) = clusters.pop(lf), clusters.pop(cl)
        clusters[key] = ((c1 * n1 + c2 * n2) / (n1 + n2), n1 + n2, s1 | s2)
        key += 1
    return sorted((tuple(np.rint(c).astype(int)), n) for c, n, _ in clusters.values())


def main():
    # ---------------------------------------------------------- checks
    rng = np.random.default_rng(0)
    v = np.arange(256, dtype=np.uint8).reshape(1, -1, 1).repeat(3, 2)
    q, al = cross_product_reduce(v, 4)
    assert q.max() == 15 and al.max() == 240 and np.all(al % 16 == 0)
    qf, _ = cross_product_reduce(v, 4, "floor")
    assert np.array_equal(qf, v >> 4)          # floor variant == plain shift

    small = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    small[:20, :25] = (200, 30, 30)            # a dominant color
    m = FastColorClustering(8, 3, distance="rgb").fit(small)
    got = sorted((tuple(int(x) for x in c), int(n))
                 for c, n in zip(m.palette_, m.palette_counts_))
    ref = brute_force_reference(small, 8, 3)
    assert got == ref, (got, ref)
    assert m.palette_counts_.sum() == small.shape[0] * small.shape[1]
    assert len(np.unique(m.transform(small).reshape(-1, 3), axis=0)) <= 8
    print("self-checks passed (cross-product, left alignment, merge loop vs "
          "brute-force reference)")

    # ---------------------------------------------------------- demo
    imgs = {"astronaut": data.astronaut(), "coffee": data.coffee(),
            "chelsea": data.chelsea()}
    K, BITS = 16, 4
    rows = []
    fig, ax = plt.subplots(len(imgs), 4, figsize=(16, 4 * len(imgs)))
    for r, (name, rgb) in enumerate(imgs.items()):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _, reduced = cross_product_reduce(bgr, BITS)

        t0 = time.perf_counter()
        model = FastColorClustering(K, BITS).fit(bgr)
        fcc = model.transform(bgr)
        t_fcc = time.perf_counter() - t0

        t0 = time.perf_counter()
        modl = FastColorClustering(K, BITS, distance="lab").fit(bgr)
        fcc_lab = modl.transform(bgr)
        t_lab = time.perf_counter() - t0

        Z = bgr.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(0)
        t0 = time.perf_counter()
        _, lbl, ctr = cv2.kmeans(Z, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        t_km = time.perf_counter() - t0
        km = np.uint8(np.clip(ctr, 0, 255))[lbl.ravel()].reshape(bgr.shape)

        rows.append((name, bgr.shape, model.n_initial_colors_,
                     t_fcc, _psnr(bgr, fcc), t_lab, _psnr(bgr, fcc_lab),
                     t_km, _psnr(bgr, km)))
        ncol = lambda im: len(np.unique(im.reshape(-1, 3), axis=0))
        for c, (im, title) in enumerate([
                (bgr, f"{name} original, {ncol(bgr)} colors"),
                (reduced, f"{BITS}-bit cross-product, "
                          f"{model.n_initial_colors_} colors"),
                (fcc, f"US11763422  {ncol(fcc)} colors  {t_fcc*1e3:.0f} ms  "
                      f"{_psnr(bgr, fcc):.1f} dB"),
                (km, f"cv2.kmeans  {ncol(km)} colors  {t_km*1e3:.0f} ms  "
                     f"{_psnr(bgr, km):.1f} dB")]):
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax[r, c].set_title(title, fontsize=10)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=90)

    print(f"\nK={K}, {BITS} bits/channel")
    print(f"{'image':10s} {'size':>9s} {'init':>5s} | {'FCC rgb':>15s} | "
          f"{'FCC lab':>15s} | {'cv2.kmeans':>15s}")
    for n, s, ic, tf, pf, tl, pl, tk, pk in rows:
        print(f"{n:10s} {s[1]:4d}x{s[0]:<4d} {ic:5d} | {tf*1e3:6.1f}ms "
              f"{pf:5.2f}dB | {tl*1e3:6.1f}ms {pl:5.2f}dB | "
              f"{tk*1e3:6.1f}ms {pk:5.2f}dB")

    # transform of an unseen image reuses the LUT: pure gather
    other = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    for _ in range(20):
        model.transform(other)
    print(f"\ntransform() with fitted LUT on a new image: "
          f"{(time.perf_counter()-t0)/20*1e3:.2f} ms")


if __name__ == "__main__":
    main()
