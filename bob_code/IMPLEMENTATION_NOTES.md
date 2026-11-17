# Implementation Notes: `bob_code` vs Reference Implementation
**Patent: US 11,763,422 B2 ("Fast color clustering for preprocessing an image")**  
**Inventors: Marc Fiammante, Jean-Armand Broyelle (IBM)**

---

## 1. Architectural & Algorithmic Differences

| Dimension | Reference Implementation (`_sources/fast_color_clustering.py`) | Novel Implementation (`bob_code/patent_color_clustering.py`) |
|---|---|---|
| **Architecture Design** | Monolithic procedural script with single data class. | **Discrete Modular Pipeline Architecture** reflecting FIG. 1 modules: `CrossProductQuantizer` (106), `LeftAlignmentOperator` (108), `ColorIndexEncoder` (110), `HierarchicalClusterForest` (112), and `PaletteCentroidRefiner`. |
| **Tree & Clustering Representation** | Flat parallel numpy arrays (`parent`, `colors`, `counts`). | **Explicit Object-Oriented Dendrogram Graph**: `ColorTreeNode` objects with parent/left_child/right_child bidirectional pointers and lineage tracking. |
| **Priority & Least-Frequency Selection** | Dynamic masked numpy `argmin` scan over flattened array. | **Indexed Min-Heap (`heapq`) Priority Queue**: Logarithmic-time extraction of minimum frequency color clusters. |
| **Channel Support** | Hardcoded to 3 channels (RGB/BGR). | **Generic $N$-Channel Support**: Supports multi-spectral, satellite, NIR, UV (as specified in Col. 3, lines 1-5 of the patent). |
| **Color Distance Flexibility** | Fixed Euclidean / CIELAB branching. | **Pluggable Strategy Pattern**: Type-safe `DistanceMetric` enum supporting RGB Euclidean, CIELAB $\Delta E$, and Weighted Perceptual metrics. |
| **Post-Processing (Claims 6 & 13)** | None / Basic mean loop. | **Dedicated Centroid Refiner Engine**: `PaletteCentroidRefiner` mapping $K$ clusters back to true source image pixel means. |

---

## 2. Patent Claim Verification Mapping

- **Claim 1 (Step 204–210)**: Implemented across `CrossProductQuantizer.estimate_optimal_bits()`, `CrossProductQuantizer.quantize()`, and `LeftAlignmentOperator.align()`.
- **Claim 2 (Steps 214–216)**: Implemented in `PatentColorClusterer.fit()` with automatic bit incrementing when unique colors $\le K$.
- **Claims 3, 10, 17 (FIG. 3 Tree)**: Explicitly modeled in `HierarchicalClusterForest` and `ColorTreeNode`.
- **Claims 4, 11, 18 (Weighted Merge)**: Frequency-weighted centroid calculation executed in `HierarchicalClusterForest.reduce_to_k()`.
- **Claims 5, 12, 19 (Integer Key Concatenation)**: Handled dynamically for arbitrary channels in `ColorIndexEncoder.encode()`.
- **Claims 6, 13 (Source Pixel Post-Processing)**: Implemented in `PaletteCentroidRefiner.refine()`.
- **Claims 7, 14 (Precomputed LUTs)**: Implemented in `CrossProductQuantizer._build_lookup_table()`.

---

## 3. How to Run Side-by-Side Comparison

```bash
# Run unit tests and benchmark suite for bob_code
PYTHONPATH=. python3 bob_code/test_and_benchmark.py
```
