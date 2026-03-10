# ME5405 Machine Vision – Computer Project (2026)

MATLAB implementation of image processing and classification tasks for the ME5405 course project. All core algorithms are implemented from scratch — no Image Processing Toolbox functions are used, except `fitcecoc`/`predict` from the Statistics and Machine Learning Toolbox for SVM.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How to Run](#how-to-run)
3. [Code Walkthrough](#code-walkthrough)
4. [Algorithm Details](#algorithm-details)
5. [Results & Outputs](#results--outputs)

---

## Project Structure

```
ME5405_MATLAB_Code_v2/
├── main_image1.m          % Tasks 1.1–1.5: Chromosome image processing
├── main_image2.m          % Tasks 2.1–2.7: Character image processing
├── main_classifier.m      % Tasks 2.8–2.9: SVM training and experiments
│
├── read_txt_image.m       % Helper: decode alphanumeric .txt image
├── otsu_threshold.m       % Helper: Otsu's optimal thresholding
├── my_thin.m              % Helper: Zhang-Suen skeletonization
├── my_outline.m           % Helper: 4-connected boundary extraction
├── my_bwlabel.m           % Helper: BFS connected component labeling
├── my_rotate.m            % Helper: inverse-mapping image rotation
├── my_imresize.m          % Helper: bilinear/nearest-neighbor resize
│
├── chromo.txt             % Image 1: 64x64 chromosome image
├── charact1.txt           % Image 2: 64x64 character image (1,2,3,A,B,C)
├── p_dataset_26/          % Training dataset: 6 classes, ~1016 samples each
│   ├── Sample1/           % Digit '1' samples (26x26 .mat files)
│   ├── Sample2/
│   ├── Sample3/
│   ├── SampleA/
│   ├── SampleB/
│   └── SampleC/
│
└── char_info.mat          % Auto-generated: segmented chars from Image 2
```

---

## How to Run

> **Requirements:** MATLAB with the Statistics and Machine Learning Toolbox (for `fitcecoc`/`predict` in `main_classifier.m`).

Set your MATLAB working directory to `ME5405_MATLAB_Code_v2/`, then run the scripts **in order**:

```matlab
% Step 1 – Process the chromosome image (Image 1)
run('main_image1.m')

% Step 2 – Process and segment the character image (Image 2)
% This generates char_info.mat, which the classifier needs
run('main_image2.m')

% Step 3 – Train SVM and run classification experiments
run('main_classifier.m')
```

Each script is self-contained and saves all output figures as `.png` files in the working directory.

---

## Code Walkthrough

### `main_image1.m` — Chromosome Image (Tasks 1.1–1.5)

Processes `chromo.txt` (64×64, 32 gray levels). Chromosomes appear as **dark objects on a bright background**.

| Task | What it does | Key call |
|------|-------------|----------|
| 1.1 | Display original image | `read_txt_image`, `imshow` |
| 1.2 | Binarise using Otsu's method | `otsu_threshold(img, 'dark_fg')` |
| 1.3 | Skeletonise (one-pixel thin) | `my_thin(BW)` |
| 1.4 | Extract outlines | `my_outline(BW)` |
| 1.5 | Label connected components | `my_bwlabel(BW)`, `label2rgb` |

---

### `main_image2.m` — Character Image (Tasks 2.1–2.7)

Processes `charact1.txt` (64×64, 32 gray levels). Characters 1, 2, 3, A, B, C appear as **bright objects on a black background**.

| Task | What it does | Key call |
|------|-------------|----------|
| 2.1 | Display original image | `read_txt_image` |
| 2.2 | Binarise using Otsu's method | `otsu_threshold(img, 'bright_fg')` |
| 2.3 | Skeletonise | `my_thin(BW)` |
| 2.4 | Extract outlines | `my_outline(BW)` |
| 2.5 | Segment and identify characters | `my_bwlabel`, centroid-based row/column sorting |
| 2.6 | Rearrange characters as **AB123C** | Crop-and-place with 3px spacing |
| 2.7 | Rotate AB123C by 30° | `my_rotate(arranged, 30)` |

Character identification in Task 2.5 works by splitting detected components into top row (digits 1, 2, 3) and bottom row (letters A, B, C) using the midpoint of centroid row values, then sorting each row left-to-right by centroid column.

The script saves `char_info.mat` containing the cropped binary patch and ground-truth label for each of the 6 characters — this is consumed by `main_classifier.m`.

---

### `main_classifier.m` — SVM Classification (Tasks 2.8–2.9)

Trains a multi-class SVM on the `p_dataset_26` dataset and evaluates it on held-out test data and on the Image 2 characters.

**Pipeline:**

```
p_dataset_26/ (6 classes x ~1016 samples)
    │
    ▼
Polarity inversion (255 - img)   ← training images are white-bg/dark-fg;
    │                               we invert to match Image 2 convention
    ▼
Flatten to 676-d feature vector (26x26 pixels)
    │
    ▼
75/25 train/test split (rng seed 42)
    │
    ▼
fitcecoc with RBF-kernel SVM (one-vs-one multi-class)
    │
    ├──► Evaluate on test set (accuracy + confusion matrix)
    │
    └──► Classify Image 2 characters (center-pad crop to 26x26)
```

**Task 2.9 experiments** systematically vary:
1. Kernel type: linear vs RBF vs polynomial
2. Box constraint C: 0.1, 1, 10, 100
3. RBF kernel scale σ: 1, 5, 10, 50, 100
4. Image 2 preprocessing method: resize, center-pad, binarize+resize, resize-then-pad

---

## Algorithm Details

### Otsu's Thresholding (`otsu_threshold.m`)

Finds the threshold `T` that maximises the **between-class variance**:

```
σ²_B(T) = w₀(T) · w₁(T) · (μ₀(T) − μ₁(T))²
```

where `w₀`, `w₁` are class weights (probabilities) and `μ₀`, `μ₁` are class means, computed from the normalised histogram. All gray levels are scanned exhaustively.

- `'dark_fg'` mode: `BW = img < T` (chromosomes)
- `'bright_fg'` mode: `BW = img > T` (characters)

---

### Zhang-Suen Thinning (`my_thin.m`)

Iterative two-pass algorithm that removes border pixels while preserving topology. Each pixel is removed if it satisfies all of:

- **2 ≤ B(p) ≤ 6** — it has between 2 and 6 non-zero 8-neighbours
- **A(p) = 1** — exactly one 0→1 transition around the 8-neighbourhood ring
- **Pass 1:** P2·P4·P6 = 0 AND P4·P6·P8 = 0
- **Pass 2:** P2·P4·P8 = 0 AND P2·P6·P8 = 0

Where P2=N, P4=E, P6=S, P8=W in the standard Zhang-Suen labelling. Repeats until no more pixels change.

---

### Outline Extraction (`my_outline.m`)

A foreground pixel is classified as a boundary pixel if **any** of its 4-connected neighbours (N, S, E, W) is either a background pixel or out-of-bounds. Simple and efficient — single pass over the image.

---

### Connected Component Labeling (`my_bwlabel.m`)

BFS flood-fill with **8-connectivity**. Scans the image raster-order; when an unlabelled foreground pixel is found, it seeds a new BFS that propagates to all 8-connected foreground neighbours, assigning them the same label.

---

### Image Rotation (`my_rotate.m`)

Uses **inverse mapping** to avoid holes:

1. Forward-rotate the 4 image corners to compute the bounding box of the output canvas.
2. For each output pixel `(r, c)`, apply inverse rotation (by `-θ`) to find the corresponding source coordinate.
3. Apply **nearest-neighbour interpolation** — round source coordinates and copy the pixel value.

Positive angles rotate counter-clockwise. The output canvas is automatically expanded to contain the full rotated image.

---

### Image Resize (`my_imresize.m`)

Maps each output pixel `(r, c)` back to source coordinates using:

```
src_r = (r - 0.5) × scale_r + 0.5
src_c = (c - 0.5) × scale_c + 0.5
```

Supports **nearest-neighbour** and **bilinear interpolation**. The 0.5-offset convention centres pixel sampling correctly.

---

### SVM Classification (`main_classifier.m`)

- **Feature representation:** raw flattened pixel values (676-d vector from 26×26 image)
- **Multi-class strategy:** one-vs-one via `fitcecoc`
- **Kernel:** RBF with automatic scale selection (`'KernelScale', 'auto'`) by default
- **Standardisation:** enabled (`'Standardize', true`) — zero-mean, unit-variance per feature
- **Polarity:** training images (white background) are inverted before feature extraction to align with Image 2 (black background)

---

## Results & Outputs

### Image 1 — Chromosomes

| Output file | Description |
|-------------|-------------|
| `img1_01_original.png` | Raw 64×64 grayscale image (32 levels) |
| `img1_02_binary.png` | Otsu binary + manual threshold comparison |
| `img1_03_skeleton.png` | Zhang-Suen skeleton overlay |
| `img1_04_outline.png` | 1-pixel boundary of each chromosome |
| `img1_05_labeled.png` | Color-coded connected components |

### Image 2 — Characters

| Output file | Description |
|-------------|-------------|
| `img2_01_original.png` | Raw 64×64 grayscale image |
| `img2_02_binary.png` | Otsu binary (bright-foreground mode) |
| `img2_03_skeleton.png` | Zhang-Suen skeleton |
| `img2_04_outline.png` | Character outlines |
| `img2_05_segmented.png` | Color-coded labeled characters |
| `img2_05b_individual_chars.png` | Each of the 6 characters cropped individually |
| `img2_06_AB123C.png` | Characters rearranged in order A B 1 2 3 C |
| `img2_07_rotated30.png` | AB123C arrangement rotated 30° CCW |
| `img2_08_classification_results.png` | Per-character classification result tiles |
| `img2_08_confusion_matrix.png` | 6×6 confusion matrix on held-out test set |
| `img2_09_experiments.png` | 4-panel hyperparameter experiment plots |

### Classifier Performance

The baseline model uses RBF-kernel SVM with `C=1` and automatic kernel scale, trained on 75% of the dataset. Experiment plots in `img2_09_experiments.png` show accuracy across kernel types, box constraint values, kernel scales, and preprocessing methods.

---

## Notes for Team

- All custom functions are in the same directory — no `addpath` needed if you run from `ME5405_MATLAB_Code_v2/`.
- `main_classifier.m` requires `char_info.mat` to exist — always run `main_image2.m` first.
- The `rng(42)` seed in `main_classifier.m` ensures reproducible train/test splits.
- `p_dataset_26` uses 26×26 `.mat` files; each file contains a variable named `imageArray`.
