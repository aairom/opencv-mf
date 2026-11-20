# OpenCV Implementation of IBM Patent US11763422B2
**"Fast color clustering for preprocessing an image"**  
*Inventors: Marc Fiammante, Jean-Armand Broyelle (IBM)*  
*Grant Date: September 19, 2023*  
*Patent Documentation: [US11763422B2 on Google Patents](https://patents.google.com/patent/US11763422B2/en)*

---

> 🆕 **Update to an omptized code generated with a more specific prompt could be found at: https://github.com/aairom/opencv-mf-2**
>
> 🛰️ Link to Marc Fiammante's technical research documentation "**Why fast colour clustering works on tabular data: bit-plane binning, cross-product depth jumping and frequency-ordered merging**": https://github.com/aairom/opencv-mf-2/tree/main/MarcFiammante-ScientificArticle

## 1. Technical Overview & Patent Architecture

IBM Patent US11763422B2 provides a high-speed, precision-preserving color clustering algorithm specifically designed as an ultra-low precision preprocessing stage for machine learning, edge devices, video analytics, and neural network training pipelines.

Unlike classical K-Means clustering (which calculates distances between $K$ centroids and all $W \times H$ pixels across multiple iterations), the patent technique reduces computational complexity by operating entirely on a compressed color histogram space. Once fitted, applying the clustering to any image of arbitrary resolution is reduced to a single cross-product table lookup and direct index gather.

```
+------------------------------------------------------------------------------------------------+
|                                    US 11,763,422 B2 PIPELINE                                   |
+------------------------------------------------------------------------------------------------+

 [Source Image (H x W x 3)] + [Target Colors K]
               |
               v
  (1) Cross-Product Module (106 / Steps 204-208 / Claims 1, 7, 14)
      - Calculate significant bits n: 2^(3n) > (K + margin)
      - Quantize: q = round( (2^n - 1) * color / 255 ) via 256-element LUT
               |
               v
  (2) Left Alignment Module (108 / Step 210 / Claims 1, 2)
      - Left-justify: v' = q << (8 - n)  [maintains standard 0..255 distance scale]
      - Auto-increment bits if unique colors <= K
               |
               v
  (3) Map Generation Module (110 / Step 212 / Claims 1, 5, 12, 19)
      - Unique index key rrrgggbbb: (r << 2n) | (g << n) | b
      - Fast 1D histogram binning (np.bincount)
               |
               v
  (4) Binary Merge Tree Module (112 / Steps 218-228 / Claims 1, 3, 4, 10, 11 / FIG. 3)
      - Loop until remaining color count == K:
          a. Find least-frequent color i (minimum pixel count)
          b. Find closest color j (minimum Euclidean / DeltaE distance)
          c. Merge: c_new = (c_i * n_i + c_j * n_j) / (n_i + n_j)
          d. Add parent node to tree; push i, j to lower level
               |
               v
  (5) Precomputed LUT & Image Transformation (Claims 6, 13)
      - Flatten tree paths from leaves to roots: leaf_root = tree.leaf_roots()
      - Map arbitrary pixel through LUT: O(1) per pixel
```

---

## 2. Key Components & Implementation Mapping

| Patent Section / Claim | Component Name | Description in [`fast_color_clustering.py`](fast_color_clustering.py:1) |
|---|---|---|
| **Claim 1, Step 205** | Bit Estimation | [`compute_significant_bits()`](fast_color_clustering.py:65) determines minimum bits $n$ satisfying $2^{n \cdot \text{channels}} > K + \text{margin}$. |
| **Claims 1, 7, 14 / Step 206-210** | Cross-Product & Left Align | [`cross_product_reduce()`](fast_color_clustering.py:98) performs $q = \text{round}((2^n-1)\cdot v / 255)$ and shifts $q \ll (8-n)$ using fast lookup tables. |
| **Claim 5, Step 212** | Key Concatenation | [`color_index()`](fast_color_clustering.py:149) concatenates channel bits into integer key `rrrgggbbb`. |
| **Claim 2, Steps 214-216** | Auto-Increment Bits | If unique color count $\le K$, bits per channel is automatically incremented. |
| **Claims 3, 10, 17 / FIG. 3** | Binary Merge Tree | [`MergeTree`](fast_color_clustering.py:199) and [`_build_and_merge_tree()`](fast_color_clustering.py:228) maintain hierarchical parent/child tracking. |
| **Claims 4, 11, 18 / Step 224** | Weighted Merging | Combines colors proportionally to pixel frequency: $(c_i n_i + c_j n_j)/(n_i + n_j)$. |
| **Claims 6, 13 / Step 228** | Post-Processing | `post_process_refine=True` averages original source image pixels in each cluster. |

---

## 3. Usage Examples

### Python API

```python
import cv2
from fast_color_clustering import FastColorClustering, fast_color_clustering

# Load an image using OpenCV
img = cv2.imread("input_image.png")

# Method 1: Scikit-learn style Estimator API
model = FastColorClustering(n_colors=16, bits=4, distance="rgb")
clustered_img = model.fit_transform(img)

# Obtain cluster palette and label map
palette = model.palette_          # (16, 3) uint8 RGB/BGR colors
labels = model.predict(img)       # (H, W) int32 cluster assignments

# Ultra-fast transform of subsequent video frames / images
for frame in video_stream:
    processed_frame = model.transform(frame)  # >500 FPS throughput

# Method 2: One-shot functional API
clustered_img, palette, label_map = fast_color_clustering(img, n_colors=16, bits=4)
```

### Command Line Interface

```bash
# Basic run with 16 colors
python3 fast_color_clustering.py input_image.png -o output_16colors.png -k 16

# Run with CIELAB perceptual color distance and benchmark against cv2.kmeans
python3 fast_color_clustering.py input_image.png -o output.png -k 16 --distance lab --compare-kmeans --refine
```

---

## 4. Benchmark & Performance Results

Testing across standard test images (resolution up to 600x400) on Apple Silicon / macOS:

| Image | Resolution | Initial Reduced Colors | US11763422 Time (PSNR) | OpenCV `cv2.kmeans` Time (PSNR) | Speedup Factor |
|---|---|---|---|---|---|
| **Astronaut** | 512x512 | 768 | **32.2 ms** (24.39 dB) | 185.4 ms (27.59 dB) | **5.7x faster** |
| **Coffee** | 600x400 | 471 | **17.4 ms** (26.55 dB) | 125.2 ms (29.64 dB) | **7.2x faster** |
| **Chelsea** | 451x300 | 240 | **6.5 ms** (27.38 dB) | 67.5 ms (30.82 dB) | **10.4x faster** |

- **Inference Speed with Precomputed LUT**: **1.85 ms / frame** (~540 FPS), making it ideal for real-time video stream preprocessing.
- **Verification Tests**: Full mathematical checks against brute-force reference implementations and patent claim constraints are automated in [`demo_fast_color_clustering.py`](demo_fast_color_clustering.py:1).
