# ME5405 Machine Vision – Computer Project (2026)

MATLAB implementation of image processing and machine learning classification tasks for the ME5405 Machine Vision course project. All core image processing algorithms are implemented from scratch without Image Processing Toolbox functions. The classification section (`main_classifier.m`) additionally uses `fitcecoc` and `predict` from the Statistics and Machine Learning Toolbox for the SVM classifier.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [How to Run](#2-how-to-run)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Algorithm Details](#4-algorithm-details)
5. [Results & Outputs](#5-results--outputs)

---

## 1. Project Structure

```
ME5405-Project-Group-17/
├── data/
│   ├── chromo.txt                  % Image 1: 64×64 chromosome image (32 gray levels)
│   ├── charact1.txt                % Image 2: 64×64 character image — "1 2 3 / A B C"
│   └── p_dataset_26/               % Training dataset: 6 classes, ~1016 samples each
│       ├── Sample1/                %   Character '1' samples (26×26 uint8 .mat files)
│       ├── Sample2/                %   Character '2' samples (26×26 uint8 .mat files)
│       ├── Sample3/                %   Character '3' samples (26×26 uint8 .mat files)
│       ├── SampleA/                %   Character 'A' samples (26×26 uint8 .mat files)
│       ├── SampleB/                %   Character 'B' samples (26×26 uint8 .mat files)
│       └── SampleC/                %   Character 'C' samples (26×26 uint8 .mat files)
├── docs/
│   ├── Prj_Computer - Examples     % Reference examples provided with the project brief
│   └── Prj_Computer - ME5405 2026  % Official project brief (ME5405, 2026)
├── results/                        % Auto-generated output figures saved as .png
├── src/
│   ├── lib/
│   │   ├── arrange_chars.m         % Arranges CCL-labeled components in a user-defined sequence
│   │   ├── boundary_detect.m       % Two-pass row-column boundary/outline detection
│   │   ├── ccl_segment.m           % Two-pass connected-component labeling (Hoshen–Kopelman)
│   │   ├── load_txt2img.m          % Loads and decodes a coded .txt image into a numeric matrix
│   │   ├── otsu_threshold.m        % Automatic global thresholding via Otsu's method
│   │   ├── rotate_image.m          % Image rotation via backward mapping + nearest-neighbor interpolation
│   │   └── zs_thinning.m           % Zhang-Suen thinning (skeletonization)
│   ├── main_image1.m               % Tasks 1.1–1.5: Chromosome image processing pipeline
│   ├── main_image2.m               % Tasks 2.1–2.7: Character image processing pipeline
│   └── main_classifier.m           % Tasks 2.8–2.9: Classifier training, tuning & evaluation
└── README.md
```

### Dataset Format

Each `.mat` file in `p_dataset_26/` contains a single variable `imageArray` — a 26×26 `uint8` matrix where pixel values range from 0 (black) to 255 (white). The classifier scripts binarize these samples before feature extraction to match the binary convention used in the processed query images.

---

## 2. How to Run

**Requirements:** MATLAB (any recent version) with the **Statistics and Machine Learning Toolbox** (required only for `main_classifier.m` — the SVM uses `fitcecoc` and `predict`).

Set your MATLAB working directory to the repository root `ME5405-Project-Group-17/`, then run the scripts in order:

```matlab
% Step 1 — Process the chromosome image (Image 1, Tasks 1.1–1.5)
run('src/main_image1.m')

% Step 2 — Process and segment the character image (Image 2, Tasks 2.1–2.7)
run('src/main_image2.m')

% Step 3 — Train classifiers and run classification experiments (Tasks 2.8–2.9)
run('src/main_classifier.m')
```

Each script is self-contained, prints progress to the MATLAB console, and saves all output figures as `.png` files to the `results/` folder. Scripts can be run independently as long as the working directory is set correctly.

> **Note:** `main_classifier.m` includes a fixed random seed (`rng(12)`) for reproducibility of the train/validation split and SOM weight initialisation.

---

## 3. Code Walkthrough

### `main_image1.m` — Chromosome Image (Tasks 1.1–1.5)

Processes `chromo.txt`, which encodes a 64×64 grayscale image of chromosomes as dark objects on a bright background (black foreground convention: foreground = 0).

| Task | Description | Key Call |
|------|-------------|----------|
| 1.1 | Load and display original image with intensity histogram | `load_txt2img(...)` |
| 1.2 | Binarize using Otsu's automatic thresholding | `otsu_threshold(img)` |
| 1.3 | Produce one-pixel-wide skeleton via Zhang-Suen thinning | `zs_thinning(T, 0)` |
| 1.4 | Detect and display object boundary outlines | `boundary_detect(T, 0)` |
| 1.5 | Segment and label individual chromosome objects | `ccl_segment(T, 0)` |

All figures are saved to `results/img1_*.png`.

---

### `main_image2.m` — Character Image (Tasks 2.1–2.7)

Processes `charact1.txt`, a 64×64 image containing characters "1 2 3" (top row) and "A B C" (bottom row) as bright objects on a black background (white foreground convention: foreground = 1).

| Task | Description | Key Call |
|------|-------------|----------|
| 2.1 | Load and display original image with intensity histogram | `load_txt2img(...)` |
| 2.2 | Binarize using Otsu's automatic thresholding | `otsu_threshold(img)` |
| 2.3 | Produce one-pixel-wide skeleton via Zhang-Suen thinning | `zs_thinning(T, 1)` |
| 2.4 | Detect and display character boundary outlines | `boundary_detect(T, 1)` |
| 2.5 | Segment and label the six individual characters | `ccl_segment(T, 1)` |
| 2.6 | Rearrange characters into sequence **AB123C** | `arrange_chars(T, label_matrix, [4 5 1 2 3 6], 0)` |
| 2.7 | Rotate the arranged sequence 30° clockwise about its center | `rotate_image(arranged, 30, 'cw', 1)` |

> **CCL label mapping** (established by raster-scan order, top-to-bottom left-to-right):
> Labels 1→'1', 2→'2', 3→'3', 4→'A', 5→'B', 6→'C'.
> The sequence `[4 5 1 2 3 6]` therefore produces **AB123C**.

All figures are saved to `results/img2_*.png`.

---

### `main_classifier.m` — Classification Pipeline (Tasks 2.8–2.9)

Trains and evaluates three classifiers on the `p_dataset_26` training dataset, then applies the best-performing classifier to the six characters extracted from Image 2.

The script is organised into nine sections:

| Section | Description |
|---------|-------------|
| 1 | Load and display the original character image |
| 2 | Binarize, segment, and extract the six character crops from Image 2 |
| 3 | Load the full `p_dataset_26` dataset; apply binarization, aspect-ratio-preserving resize, and 75/25 stratified train/test split |
| 4 | Hyperparameter sweep — effect of feature image size (10×10, 16×16, 20×20, 28×28) on KNN accuracy |
| 5 | KNN classifier — sweep k ∈ {1,3,5,7,9,11}, select best k, plot confusion matrix |
| 6 | SVM classifier — compare linear / Gaussian / polynomial kernels; sweep box constraint C; plot confusion matrix |
| 7 | SOM classifier — compare grid sizes (8×8, 10×10, 26×26, 32×32); retrain best grid for 250 epochs; plot confusion matrix |
| 8 | Compare all three classifiers and select the best-performing model |
| 9 | Apply KNN, SVM, and SOM to Image 2 characters; display final predictions per character |

**Preprocessing applied to training samples:**
- Load `imageArray` from `.mat` file as `double`
- Normalise to [0,1] then binarize at 0.5 threshold
- Correct polarity so character pixels = 1 (minority class)
- Pad to square canvas preserving aspect ratio
- Resize to target feature size (default 20×20) using nearest-neighbor interpolation
- Flatten to 1D feature vector

All figures are saved to `results/img3_*.png`.

---

## 4. Algorithm Details

### `load_txt2img.m` — Coded Image Loader

Reads a `.txt` file where each pixel is encoded as a single alphanumeric character:
- `'0'`–`'9'` → intensity levels 0–9
- `'A'`–`'V'` → intensity levels 10–31

Uses `fscanf` to read 64×64 characters in column-major order, builds an ASCII-indexed lookup table for O(1) character mapping, then transposes to restore row-major (image) orientation. Output is a 64×64 `double` matrix with intensity values in [0, 31].

---

### `otsu_threshold.m` — Otsu's Method

Finds the threshold k\* that maximises the **between-class variance**:

$$\sigma_B^2(k) = \frac{(\mu_G \cdot P_1(k) - m_k)^2}{P_1(k)(1 - P_1(k))}$$

where P₁(k) is the cumulative probability up to level k and mₖ is the cumulative mean. Division-by-zero at the extremes (P₁ = 0 or 1) is guarded explicitly. Output is a binary matrix T with dark pixels → 0 and bright pixels → 1.

**Reference:** Gonzalez & Woods, *Digital Image Processing* 4e, §10.3, pp. 747–751; [Wikipedia: Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method).

---

### `zs_thinning.m` — Zhang-Suen Thinning

Iteratively erodes a binary foreground to a **one-pixel-wide skeleton** while preserving connectivity and topology. Each iteration performs two sub-passes targeting south-east and north-west boundary pixels respectively. A pixel is marked for deletion when all four conditions hold:

1. It is a foreground pixel
2. It has between 2 and 6 foreground neighbors (B(P))
3. There is exactly one 1→0 transition in its clockwise 8-neighborhood (A(P) = 1)
4. The two complementary triple-product conditions (Pass 1 or Pass 2 variant)

Deletions within each sub-pass are applied simultaneously. Iteration continues until no pixels are removed. The function accepts a `foreground` parameter (0 or 1) and normalises internally, restoring the original convention on output.

**Reference:** [Rosetta Code: Zhang-Suen thinning algorithm](https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm).

---

### `boundary_detect.m` — Two-Pass Boundary Detection

Detects object outlines by scanning for gray-level band transitions in two passes:

- **Pass 1** (row scan): For each pixel, compare with its left neighbor `f(r, c-1)`. A transition → mark as edge (LE).
- **Pass 2** (column scan): For each pixel, compare with its upper neighbor `f(r-1, c)`. A transition → mark as edge (LE).
- **Combination** (Eq. 9): `g(r,c) = LE` if either `g₁` or `g₂` is LE, otherwise background (LB).

The result is a binary image where edge pixels carry the foreground value and all other pixels carry the background value, consistent with the input convention.

**Reference:** ME5405 Lecture Notes, Chapter 3 — Binary Image and Color Image Processing, pp. 29–33.

---

### `ccl_segment.m` — Connected-Component Labeling (Two-Pass / Hoshen–Kopelman)

Labels distinct foreground regions with unique integer identifiers using the classical two-pass algorithm:

- **Pass 1:** Raster scan (left→right, top→bottom). Each foreground pixel checks its NW, N, NE, W neighbors (8-connectivity). Assigns a new label if no labeled neighbors exist; otherwise assigns the minimum neighbor label and records equivalences via an inline Union-Find with path compression.
- **Between passes:** Flattens all equivalence chains to their roots; remaps roots to sequential integers 1, 2, 3 …
- **Pass 2:** Replaces every provisional label with its final sequential label.

Output is an integer matrix where 0 = background and each positive integer identifies one connected component. Background is displayed as `NaN` in visualisation to render white using `imagesc`.

**References:** ME5405 Lecture Notes, Chapter 3, pp. 41–46; [Wikipedia: Connected-component labeling](https://en.wikipedia.org/wiki/Connected-component_labeling#Two-pass); [Wikipedia: Hoshen–Kopelman algorithm](https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm).

---

### `arrange_chars.m` — Character Sequence Arranger

Extracts crops of selected CCL components from a binary image and pastes them onto a new canvas in a user-specified order. Key steps:

1. For each component ID in `component_sequence`, scan `label_matrix` to find tight bounding box extents (r\_min, r\_max, c\_min, c\_max).
2. Expand by `padding` pixels on all sides (clamped to image boundaries).
3. Canvas height = max padded crop height; canvas width = sum of padded crop widths + `spacing` × (n−1).
4. Each crop is centered vertically on the canvas and pasted at the computed column offset.

All spacing, padding, and the canvas itself are filled with the `background` color, ensuring visual separation between characters regardless of foreground/background convention.

---

### `rotate_image.m` — Image Rotation

Rotates a binary image about its center by an arbitrary angle using:

1. **Output canvas sizing:** The four corners of the input image are forward-rotated; the bounding box of the rotated corners determines the output canvas size, ensuring no content is clipped.
2. **Backward (inverse) mapping:** For each output pixel, the inverse rotation is applied to find the corresponding source location in the input image. This avoids holes that forward mapping produces.
3. **Nearest-neighbor interpolation:** Source coordinates are rounded to the nearest integer. This preserves exact binary (0/1) pixel values without introducing intermediate gray levels.
4. **Out-of-bounds handling:** Output pixels whose back-mapped source falls outside the input image are set to the `background` color.

Clockwise rotation is implemented by negating the angle (since the image y-axis points downward, negating compensates for the axis flip).

---

### Classifiers (`main_classifier.m`)

#### K-Nearest Neighbors (KNN)
Custom implementation using vectorised Euclidean distance. For each query vector, computes distances to all training samples, selects the k nearest, and assigns the majority-vote label. Hyperparameter sweep over k ∈ {1, 3, 5, 7, 9, 11}.

#### Support Vector Machine (SVM)
Uses MATLAB's `fitcecoc` (Statistics and Machine Learning Toolbox) with One-vs-One coding. Three kernels compared: linear, Gaussian (RBF), and polynomial (degree 2). Box constraint C swept over {0.01, 0.1, 1, 10, 100} for the best-performing kernel.

#### Self-Organising Map (SOM)
Custom implementation of Kohonen's SOM. Weights initialised by sampling random training vectors. Training uses exponential decay of learning rate (α) and neighbourhood radius (σ). After training, each neuron is labeled by majority vote over the training set. Classification assigns the label of the BMU (Best Matching Unit — neuron with minimum Euclidean distance to the query). Grid sizes compared: 8×8, 10×10, 26×26, 32×32.

---

## 5. Results & Outputs

All figures are saved to `results/` automatically when each script is run.

### Image 1 — Chromosomes (`main_image1.m`)

| Output File | Description |
|-------------|-------------|
| `img1_01_original.png` | Original 64×64 chromosome image displayed with intensity colorbar |
| `img1_02_histogram.png` | Intensity histogram — 32 gray levels, frequency per level |
| `img1_03_binarized.png` | Binary image after Otsu thresholding — chromosomes as black foreground |
| `img1_04_skeletonized.png` | One-pixel-wide skeleton of chromosomes via Zhang-Suen thinning |
| `img1_05_outline.png` | Boundary outline of each chromosome via two-pass boundary detection |
| `img1_06_segmenting.png` | Color-labeled segmentation map — each chromosome assigned a unique color |

### Image 2 — Characters (`main_image2.m`)

| Output File | Description |
|-------------|-------------|
| `img2_01_original.png` | Original 64×64 character image displayed with intensity colorbar |
| `img2_02_histogram.png` | Intensity histogram — 32 gray levels |
| `img2_03_binarized.png` | Binary image after Otsu thresholding — characters as white foreground |
| `img2_04_skeletonized.png` | One-pixel-wide skeleton of all six characters |
| `img2_05_outline.png` | Boundary outline of each character |
| `img2_06_segmenting.png` | Color-labeled segmentation — six components, one per character |
| `img2_07_arranged.png` | Characters rearranged into sequence **AB123C** on a single canvas |
| `img2_08_rotated.png` | AB123C sequence rotated 30° clockwise about its center (canvas auto-expanded) |

### Classification (`main_classifier.m`)

| Output File | Description |
|-------------|-------------|
| `img3_01_characters.png` | Six segmented character crops extracted from Image 2 |
| `img3_02_KNNfeatsize.png` | Bar chart — KNN accuracy vs feature image size (10×10 to 28×28) |
| `img3_03_KNNtuning.png` | Line chart — KNN accuracy vs k value (k = 1 to 11) |
| `img3_04_KNNconfusion.png` | Confusion matrix — KNN classifier on test set (best k) |
| `img3_05_SVMcompare.png` | Bar chart — SVM accuracy across linear / Gaussian / polynomial kernels |
| `img3_06_SVMconfusion.png` | Confusion matrix — SVM classifier on test set (best kernel) |
| `img3_07_SVMeffect.png` | Semi-log line chart — SVM accuracy vs box constraint C |
| `img3_08_SOMcompare.png` | Bar chart — SOM accuracy across four grid sizes |
| `img3_09_SOMconfusion.png` | Confusion matrix — SOM classifier on test set (best grid, 250 epochs) |
| `img3_10_ClassCompare.png` | Bar chart — final accuracy comparison: KNN vs SVM vs SOM |
| `img3_11_predicted.png` | Image 2 characters with KNN / SVM / SOM predictions and best-model label in subplot titles |

---

## Notes

- The `foreground` parameter accepted by `zs_thinning`, `boundary_detect`, `ccl_segment`, and `arrange_chars` allows all functions to handle both conventions (black-on-white and white-on-black) without code duplication. Internally each function normalises to black = foreground, processes, then restores the original convention on output.
- `rotate_image` automatically expands the output canvas to prevent clipping — for a 64×64 image rotated 30°, the output canvas will be approximately 87×87 pixels.
- The random seed `rng(12)` at the top of `main_classifier.m` ensures the 75/25 split and SOM initialisation are reproducible across runs.
- All `.png` outputs are saved with `saveas(gcf, ...)` — ensure the `results/` directory exists before running, or add `mkdir('results')` at the top of each script if needed.