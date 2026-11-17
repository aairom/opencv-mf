#!/usr/bin/env python3
"""
Verification and Comparative Benchmark for bob_code Implementation
===================================================================
Validates the novel discrete pipeline and min-heap architecture against
the reference implementation and OpenCV K-Means baseline.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from skimage import data

from bob_code.patent_color_clustering import (
    PatentColorClusterer,
    CrossProductQuantizer,
    LeftAlignmentOperator,
    ColorIndexEncoder,
    HierarchicalClusterForest,
    DistanceMetric,
    RoundingMode,
    process_image_patent_pipeline,
)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)


def run_unit_tests():
    print("=" * 75)
    print("RUNNING ARCHITECTURAL & UNIT TESTS FOR bob_code IMPLEMENTATION")
    print("=" * 75)

    # 1. Test CrossProductQuantizer
    quantizer = CrossProductQuantizer(bits=4, rounding=RoundingMode.ROUND_HALF_UP)
    ramp = np.arange(256, dtype=np.uint8)
    q_vals = quantizer.quantize(ramp)
    assert q_vals.min() == 0 and q_vals.max() == 15
    print("[PASS] CrossProductQuantizer: 4-bit LUT reduction in [0..15]")

    # 2. Test LeftAlignmentOperator
    aligner = LeftAlignmentOperator(bits=4)
    aligned = aligner.align(q_vals)
    assert aligned.max() == 240 and np.all(aligned % 16 == 0)
    print("[PASS] LeftAlignmentOperator: 8-bit left justification")

    # 3. Test ColorIndexEncoder
    encoder = ColorIndexEncoder(bits=4, channels=3)
    sample_levels = np.array([[[1, 2, 3]]], dtype=np.uint8)
    packed = encoder.encode(sample_levels)
    expected_packed = (1 << 8) | (2 << 4) | 3  # 256 + 32 + 3 = 291
    assert packed[0, 0] == expected_packed
    decoded = encoder.decode(packed, left_align=False)
    assert np.all(decoded[0, 0] == [1, 2, 3])
    print("[PASS] ColorIndexEncoder: multi-channel bit packing & unpacking")

    # 4. Test HierarchicalClusterForest Min-Heap Merging
    colors = np.array([[255, 0, 0], [250, 5, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float64)
    counts = np.array([10, 5, 100, 200], dtype=np.int64)
    forest = HierarchicalClusterForest(colors, counts, DistanceMetric.EUCLIDEAN_RGB)
    forest.reduce_to_k(target_k=3)
    roots = forest.get_cluster_roots()
    assert len(roots) == 3
    print("[PASS] HierarchicalClusterForest: Min-Heap priority queue merging to K=3")

    # 5. Full End-to-End Pipeline on Synthetic Image
    rng = np.random.default_rng(123)
    test_img = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    clusterer = PatentColorClusterer(k_colors=8, bits=3, refine_palette=True)
    out_img = clusterer.fit_transform(test_img)
    unique_out = len(np.unique(out_img.reshape(-1, 3), axis=0))
    assert unique_out <= 8
    print("[PASS] PatentColorClusterer: Complete fit_transform pipeline with refinement")

    print("\nAll unit tests passed successfully!\n")


def run_benchmarks():
    print("=" * 75)
    print("BENCHMARK COMPARISON: bob_code vs OpenCV K-Means")
    print("=" * 75)

    test_images = {
        "Astronaut": data.astronaut(),
        "Coffee": data.coffee(),
        "Chelsea": data.chelsea(),
    }

    k_target = 16

    print(f"{'Image':<12} {'Resolution':<12} | {'bob_code (Patent)':<20} | {'OpenCV K-Means':<20} | {'Speedup':<8}")
    print("-" * 80)

    for name, rgb in test_images.items():
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 1. Patent Implementation (bob_code)
        t0 = time.perf_counter()
        clusterer = PatentColorClusterer(k_colors=k_target, bits=4, metric=DistanceMetric.EUCLIDEAN_RGB)
        clustered = clusterer.fit_transform(bgr)
        dt_patent = (time.perf_counter() - t0) * 1000.0
        psnr_patent = compute_psnr(bgr, clustered)

        # 2. OpenCV K-Means
        t0 = time.perf_counter()
        pixels = bgr.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k_target, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        dt_kmeans = (time.perf_counter() - t0) * 1000.0
        kmeans_img = np.uint8(np.clip(centers, 0, 255))[labels.ravel()].reshape(bgr.shape)
        psnr_kmeans = compute_psnr(bgr, kmeans_img)

        speedup = dt_kmeans / dt_patent
        res_str = f"{bgr.shape[1]}x{bgr.shape[0]}"
        pat_str = f"{dt_patent:5.1f} ms ({psnr_patent:5.2f} dB)"
        km_str = f"{dt_kmeans:5.1f} ms ({psnr_kmeans:5.2f} dB)"

        print(f"{name:<12} {res_str:<12} | {pat_str:<20} | {km_str:<20} | {speedup:5.1f}x")

    print("-" * 80)

    # Test stream inference throughput
    chelsea_bgr = cv2.cvtColor(data.chelsea(), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    n_frames = 100
    for _ in range(n_frames):
        _ = clusterer.transform(chelsea_bgr)
    avg_inference_ms = (time.perf_counter() - t0) / n_frames * 1000.0
    fps = 1000.0 / avg_inference_ms
    print(f"\nThroughput on stream video frames: {avg_inference_ms:.2f} ms/frame ({fps:.1f} FPS)")


if __name__ == "__main__":
    run_unit_tests()
    run_benchmarks()
