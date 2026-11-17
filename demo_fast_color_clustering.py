#!/usr/bin/env python3
"""
Comprehensive Demonstration and Verification of IBM Patent US 11,763,422 B2
===========================================================================
Demonstrates and validates:
1. Unit tests and self-checks against exact mathematical definitions in claims.
2. Bit estimation formula 2^(n*channels) > (K + margin).
3. Cross-product reduction, rounding, and left alignment.
4. Auto-increment of significant bits when unique colors <= K (Claim 2).
5. Binary merge tree construction and node linking (Claims 3, 10, 17 / FIG. 3).
6. Frequency-weighted centroid merging (Claims 4, 11, 18).
7. Visual comparison with original and OpenCV cv2.kmeans across multiple test images.
8. Speed benchmarking and PSNR analysis.
"""

from __future__ import annotations

import time
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import data

from fast_color_clustering import (
    FastColorClustering,
    fast_color_clustering,
    compute_significant_bits,
    cross_product_reduce,
    color_index,
    index_to_color,
    calculate_psnr
)


def brute_force_reference(img: np.ndarray, k: int, bits: int):
    """Literal reference implementation of Claim 1 using dictionary structures."""
    q, _ = cross_product_reduce(img, bits)
    idx = color_index(q, bits).ravel()
    ids, cnt = np.unique(idx, return_counts=True)
    mask = (1 << bits) - 1
    cols = {
        int(i): np.array([(i >> (2 * bits)) & mask, (i >> bits) & mask, i & mask], float) * (1 << (8 - bits))
        for i in ids
    }
    clusters = {int(i): (cols[int(i)], int(c), {int(i)}) for i, c in zip(ids, cnt)}
    key = len(ids) * 10
    while len(clusters) > k:
        lf = min(clusters, key=lambda c: (clusters[c][1], list(clusters).index(c)))
        cl = min(
            (c for c in clusters if c != lf),
            key=lambda c: np.sum((clusters[c][0] - clusters[lf][0]) ** 2)
        )
        (c1, n1, s1), (c2, n2, s2) = clusters.pop(lf), clusters.pop(cl)
        clusters[key] = ((c1 * n1 + c2 * n2) / (n1 + n2), n1 + n2, s1 | s2)
        key += 1
    return sorted((tuple(np.rint(c).astype(int)), n) for c, n, _ in clusters.values())


def run_unit_tests():
    print("=" * 70)
    print("RUNNING VERIFICATION TESTS FOR US 11,763,422 B2 IMPLEMENTATION")
    print("=" * 70)

    # 1. Test significant bits computation
    # For K=16, margin=8 -> target=24 -> 2^(1*3)=8 < 24; 2^(2*3)=64 > 24 -> bits=2
    assert compute_significant_bits(16, margin=8, channels=3) == 2
    # For K=256, margin=128 -> target=384 -> 2^(3*3)=512 > 384 -> bits=3
    assert compute_significant_bits(256, margin=128, channels=3) == 3
    print("[PASS] Bit computation formula: 2^(bits*channels) > (K + margin)")

    # 2. Test cross product reduction & left alignment
    ramp = np.arange(256, dtype=np.uint8).reshape(1, -1, 1).repeat(3, axis=2)
    q, al = cross_product_reduce(ramp, 4, rounding="round")
    assert q.max() == 15 and al.max() == 240 and np.all(al % 16 == 0)

    q_floor, al_floor = cross_product_reduce(ramp, 4, rounding="floor")
    assert np.array_equal(q_floor, ramp >> 4)
    print("[PASS] Cross-product reduction, rounding LUT & 8-bit left alignment")

    # 3. Test color indexing (rrrgggbbb)
    levels = np.array([[[1, 2, 3]]], dtype=np.uint8) # bits=3 -> (1<<6)|(2<<3)|3 = 64 + 16 + 3 = 83
    idx = color_index(levels, 3)
    assert idx[0, 0] == 83
    c_recon = index_to_color(idx, 3, left_align=False)
    assert np.all(c_recon[0, 0] == [1, 2, 3])
    print("[PASS] Unique color key concatenation (rrrgggbbb index)")

    # 4. Test exact merge tree results vs brute-force reference
    rng = np.random.default_rng(42)
    synthetic_img = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
    synthetic_img[:15, :20] = [210, 40, 40]
    synthetic_img[15:, 20:] = [40, 210, 40]

    model = FastColorClustering(n_colors=8, bits=3, distance="rgb").fit(synthetic_img)
    got_palette = sorted((tuple(int(x) for x in c), int(n))
                         for c, n in zip(model.palette_, model.palette_counts_))
    expected_ref = brute_force_reference(synthetic_img, 8, 3)
    assert got_palette == expected_ref, f"Mismatch: {got_palette} != {expected_ref}"
    assert model.palette_counts_.sum() == synthetic_img.shape[0] * synthetic_img.shape[1]
    print("[PASS] Binary merge tree least-frequent weighted reduction vs reference")

    # 5. Test auto-increment bits (Claim 2)
    # Image with only 2 unique colors
    flat_img = np.zeros((20, 20, 3), dtype=np.uint8)
    flat_img[:10, :] = [255, 0, 0]
    flat_img[10:, :] = [0, 255, 0]
    model_inc = FastColorClustering(n_colors=16, bits=1, auto_increment_bits=True).fit(flat_img)
    # Should automatically reach 8 bits because unique colors < K
    assert model_inc.active_bits_ == 8
    print("[PASS] Auto-increment of bits per channel when unique colors <= K (Claim 2)")

    # 6. Test post-processing refinement (Claims 6 & 13)
    model_refined = FastColorClustering(n_colors=8, bits=3, post_process_refine=True).fit(synthetic_img)
    target_refined = model_refined.transform(synthetic_img)
    assert len(np.unique(target_refined.reshape(-1, 3), axis=0)) <= 8
    print("[PASS] Post-processing centroid refinement with original source pixels")

    print("\nAll unit tests passed successfully!\n")


def run_benchmark_and_demo():
    print("=" * 70)
    print("RUNNING BENCHMARK & DEMO ON STANDARD TEST IMAGES")
    print("=" * 70)

    images = {
        "Astronaut": data.astronaut(),
        "Coffee": data.coffee(),
        "Chelsea": data.chelsea()
    }

    k_colors = 16
    bits = 4

    results_table = []
    fig, axes = plt.subplots(len(images), 4, figsize=(18, 4.5 * len(images)))

    for row_idx, (name, rgb_img) in enumerate(images.items()):
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        _, reduced_img = cross_product_reduce(bgr_img, bits)

        # 1. Patent Method (RGB Distance)
        t0 = time.perf_counter()
        fcc_model = FastColorClustering(n_colors=k_colors, bits=bits, distance="rgb").fit(bgr_img)
        fcc_rgb = fcc_model.transform(bgr_img)
        t_fcc_rgb = time.perf_counter() - t0
        psnr_fcc_rgb = calculate_psnr(bgr_img, fcc_rgb)

        # 2. Patent Method (CIELAB Perceptual Distance)
        t0 = time.perf_counter()
        fcc_lab_model = FastColorClustering(n_colors=k_colors, bits=bits, distance="lab").fit(bgr_img)
        fcc_lab = fcc_lab_model.transform(bgr_img)
        t_fcc_lab = time.perf_counter() - t0
        psnr_fcc_lab = calculate_psnr(bgr_img, fcc_lab)

        # 3. OpenCV K-Means Baseline
        pixel_data = bgr_img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        cv2.setRNGSeed(0)
        t0 = time.perf_counter()
        _, km_labels, km_centers = cv2.kmeans(
            pixel_data, k_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        t_km = time.perf_counter() - t0
        km_img = np.uint8(np.clip(km_centers, 0, 255))[km_labels.ravel()].reshape(bgr_img.shape)
        psnr_km = calculate_psnr(bgr_img, km_img)

        speedup = t_km / t_fcc_rgb
        results_table.append((
            name, bgr_img.shape, fcc_model.n_initial_colors_,
            t_fcc_rgb * 1000, psnr_fcc_rgb,
            t_fcc_lab * 1000, psnr_fcc_lab,
            t_km * 1000, psnr_km, speedup
        ))

        # Visual layout
        visual_cols = [
            (bgr_img, f"{name} Original\n24-bit TrueColor"),
            (reduced_img, f"Cross-Product ({bits}-bit)\n{fcc_model.n_initial_colors_} initial colors"),
            (fcc_rgb, f"US11763422 (K={k_colors})\n{t_fcc_rgb*1e3:.1f} ms | {psnr_fcc_rgb:.2f} dB"),
            (km_img, f"OpenCV cv2.kmeans (K={k_colors})\n{t_km*1e3:.1f} ms | {psnr_km:.2f} dB")
        ]

        for col_idx, (im, title) in enumerate(visual_cols):
            axes[row_idx, col_idx].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[row_idx, col_idx].set_title(title, fontsize=11)
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig("patent_comparison_results.png", dpi=120)
    print("Saved comparison chart to 'patent_comparison_results.png'")

    # Print summary table
    print("\nBenchmark Summary (K=16 colors, 4 bits/channel):")
    print("-" * 95)
    print(f"{'Image':<12} {'Resolution':<11} {'Initial':<8} | {'US11763422 (RGB)':<18} | {'cv2.kmeans':<18} | {'Speedup':<8}")
    print("-" * 95)
    for row in results_table:
        name, shape, init_c, t_fcc, p_fcc, t_lab, p_lab, t_km, p_km, sp = row
        res_str = f"{shape[1]}x{shape[0]}"
        fcc_str = f"{t_fcc:5.1f} ms ({p_fcc:5.2f} dB)"
        km_str = f"{t_km:5.1f} ms ({p_km:5.2f} dB)"
        print(f"{name:<12} {res_str:<11} {init_c:<8} | {fcc_str:<18} | {km_str:<18} | {sp:5.1f}x")
    print("-" * 95)

    # Test applying precomputed LUT to a stream of new images (inference speed)
    test_img = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    n_frames = 50
    for _ in range(n_frames):
        _ = fcc_model.transform(test_img)
    t_gather = (time.perf_counter() - t0) / n_frames * 1000.0
    print(f"\nInference speed (reusing fitted LUT for preprocessing): {t_gather:.2f} ms / frame ({1000.0/t_gather:.1f} FPS)")


if __name__ == "__main__":
    run_unit_tests()
    run_benchmark_and_demo()
