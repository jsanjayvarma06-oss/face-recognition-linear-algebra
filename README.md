# Face Recognition Using Linear Algebra Pipeline

### PES University — UE24MA241B: Linear Algebra and Its Applications

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [File-by-File Explanation](#file-by-file-explanation)
   - [main.py](#1-mainpy--the-entry-point)
   - [preprocessing.py](#2-preprocessingpy--face-preparation)
   - [data_collection.py](#3-data_collectionpy--collecting-training-faces)
   - [la_pipeline.py](#4-la_pipelinepy--the-linear-algebra-engine)
   - [recognizer.py](#5-recognizerpy--live-face-recognition)
   - [utils.py](#6-utilspy--display-helpers)
5. [How All Files Connect](#how-all-files-connect)
6. [The Full Linear Algebra Pipeline — Step by Step](#the-full-linear-algebra-pipeline)
7. [Key Equations](#key-equations)
8. [Viva Preparation](#viva-preparation)

---

## What This Project Does

This project builds a **live face recognition system** using only Linear Algebra — no deep learning, no neural networks. A webcam captures faces, and the system uses the **Eigenfaces method** to:

1. Represent every face as a mathematical vector
2. Learn the most important "face patterns" (eigenfaces) from training data
3. Identify a new face by comparing it to those learned patterns

The entire pipeline follows the 9-step workflow from the UE24MA241B Mini Project guidelines.

---

## Project Structure

```
face_recognition_project/
│
├── main.py               ← Start here. Runs the menu-driven program.
├── preprocessing.py      ← Cleans and standardises every face image
├── data_collection.py    ← Uses webcam to collect training photos
├── la_pipeline.py        ← The full 9-step Linear Algebra engine
├── recognizer.py         ← Live recognition using the trained pipeline
├── utils.py              ← Console print formatting helpers
├── requirements.txt      ← Python packages needed
│
└── faces_db/             ← Created automatically when you collect faces
    ├── Alice/
    │   ├── frame_0000.npy
    │   └── frame_0001.npy ...
    └── Bob/
        └── frame_0000.npy ...
```

---

## How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Start the program
python main.py
```

The program shows a menu:

```
  1. Collect training faces (camera)    ← Do this first, for each person
  2. Train the Linear Algebra Pipeline  ← Do this after collecting
  3. Recognize a face (camera)          ← Live recognition
  4. Show eigenfaces & explained variance
  5. Exit
```

**Recommended workflow:**
- Run Option 1 for Person A (collect 50 photos)
- Run Option 1 for Person B (collect 50 photos)
- Run Option 2 to train
- Run Option 3 to recognize live

---

## File-by-File Explanation

---

### 1. `main.py` — The Entry Point

**What it does:** This is the only file you directly run. It creates the menu and connects all the other modules together.

**How it works:**

When you run `python main.py`, it does three things at startup:

1. Creates a `LinearAlgebraPipeline` object (from `la_pipeline.py`) — this object will eventually hold all the trained eigenfaces, mean face, and projections.
2. Creates a `FaceRecognizer` object (from `recognizer.py`) and passes the pipeline to it so recognition and training share the same object.
3. Enters a loop that keeps showing the menu until you choose Exit.

**What each menu option triggers:**

| Option | Calls | What happens |
|--------|-------|--------------|
| 1 | `collect_faces(label)` in `data_collection.py` | Opens camera, saves face photos |
| 2 | `pipeline.train()` in `la_pipeline.py` | Runs all 9 LA steps on saved photos |
| 3 | `recognizer.recognize_from_camera()` | Opens camera, predicts identity live |
| 4 | `pipeline.visualize()` | Shows eigenfaces and variance chart |

The `pipeline` object is shared between `main.py` and `recognizer.py`. This means when you train in Option 2, the trained eigenfaces are immediately available in the recognizer for Option 3 without saving to disk.

---

### 2. `preprocessing.py` — Face Preparation

**What it does:** Cleans and standardises every face image before it is used — both when saving training photos AND when recognising live. Consistency between training and inference is the single most important factor for accuracy.

**Why this file exists:** If you train on bright-lit faces but test on shadowy faces, the math will fail because the pixel values look completely different even though it is the same person. This file removes those differences so the eigenspace comparison is fair.

**The four steps applied to every face:**

**Step A — Eye Alignment**

Detects the two eyes inside the face crop using OpenCV's `haarcascade_eye.xml`. It then:
- Takes the centre coordinates of both eyes
- Computes the angle between them using `arctan2(dy, dx)`
- Computes how much to scale the face so the eye distance always matches a target
- Builds a rotation-and-scale matrix using `cv2.getRotationMatrix2D`
- Applies `cv2.warpAffine` so both eyes always land at fixed positions (left eye at 35%, right eye at 65% horizontally, both at 40% vertically)

This removes head tilt completely. If fewer than 2 eyes are found, it falls back to a plain resize.

```
Before:  tilted face  → *  *         After:  straight face  →    *     *
                                              (eyes always at same coords)
```

**Step B — Resize**

Every aligned face is resized to exactly 100×100 pixels regardless of how close or far the person was from the camera. This makes d = 10,000 for every sample.

**Step C — CLAHE (Contrast Limited Adaptive Histogram Equalisation)**

A smarter version of histogram equalisation. It divides the image into an 8×8 grid and equalises the contrast separately in each small region, so a face lit from one side is treated similarly to a uniformly lit face. `clipLimit=2.0` prevents noise from being amplified.

**Step D — Z-score Normalisation**

```
normalised_pixel = (pixel_value - image_mean) / image_std_deviation
```

This makes every face vector have zero mean and unit variance regardless of how bright or dark the original image was. It is applied per image, not globally.

A face is rejected (returns `None`) if the standard deviation is near zero, which means it is a blank or uniform patch rather than a real face.

**Step E — Flatten**

The 100×100 pixel grid is unrolled into a single row vector of 10,000 numbers. This is one row in the data matrix **A**.

**Public functions used by other files:**

- `detect_faces(grey)` — called by `data_collection.py` and `recognizer.py` to find face bounding boxes
- `preprocess(grey, face_box)` — called by both modules to turn a face crop into a clean vector

---

### 3. `data_collection.py` — Collecting Training Faces

**What it does:** Opens the webcam, detects your face in each frame, runs it through the preprocessing pipeline, and saves it as a `.npy` file on disk.

**How it works step by step:**

1. Creates the folder `faces_db/<label>/` if it does not exist
2. Opens the webcam with `cv2.VideoCapture(0)`
3. For each frame, converts to greyscale and calls `detect_faces()` from `preprocessing.py`
4. For the first detected face, calls `preprocess()` to get a clean 10,000-dimensional vector
5. Saves that vector as `faces_db/<label>/frame_XXXX.npy` using `numpy.save()`
6. Shows a green rectangle, sample counter, and a tip on screen
7. Stops when `n_samples = 50` photos have been saved or you press Q

**Why `.npy` files?**
NumPy's binary format is extremely fast to save and load. Each file stores exactly one flat float32 array — one row of the data matrix A.

**Why 50 samples?**
More samples give the eigenspace more variation to learn from — different lighting angles, slight head movements, different expressions. This makes the trained eigenfaces generalise better to new poses during recognition.

**The on-screen tips** ("Move slightly left", "Tilt head a little", "Change expression") directly prompt you to provide variation during collection. Each tip cycles every time a new photo is saved.

**The progress bar** shown at the bottom of the frame fills from left to right as photos are collected.

---

### 4. `la_pipeline.py` — The Linear Algebra Engine

**What it does:** This is the mathematical core of the entire project. It loads all saved `.npy` training files, builds the data matrix, and executes all 9 pipeline steps. After training, it stores the eigenfaces, projections, and labels that the recognizer uses.

**How it works — all 9 steps in detail:**

**Step 1 — Matrix Representation**

Loads every `.npy` file from `faces_db/`. Each file is a 10,000-dimensional vector. Stacks them all row-by-row into the data matrix **A** of shape (n × d) where n = total training photos and d = 10,000.

```
A  =  [ face_1 vector ]   ← 10,000 numbers (row 0)
      [ face_2 vector ]   ← 10,000 numbers (row 1)
      [ face_3 vector ]   ← 10,000 numbers (row 2)
            ...
      shape: (n, 10000)
```

Any `.npy` file whose vector length is not 10,000 is skipped automatically — this prevents old v1 files (which were 64×64 = 4,096) from corrupting the matrix.

**Step 2 — Matrix Simplification (Mean-Centering)**

Computes the mean face — the average pixel value across all n training photos at each of the 10,000 positions:

```
μ = (1/n) × sum of all rows of A          shape: (10000,)
```

Subtracts μ from every row to get the centred matrix **Ã = A − μ**. This removes the "average face" shared by everyone and leaves only the differences — the features that distinguish one person from another. This is analogous to row reduction in that it normalises the matrix to a standard form.

**Step 3 — Structure of the Space**

- Computes `rank(Ã)` using `numpy.linalg.matrix_rank` with a small tolerance
- Derives `nullity = d − rank` using the Rank-Nullity Theorem
- Builds the surrogate covariance **L = ÃÃᵀ/n** of shape (n × n)

The reason for using L rather than the true covariance C = ÃᵀÃ/n is computational: C would be 10,000 × 10,000 (100 million entries) while L is only n × n (typically 100–200 entries). L has exactly the same non-zero eigenvalues as C, so we lose no mathematical information.

**Step 4 — Remove Redundancy**

The rank tells us the true dimensionality of the face space. Any training sample that can be expressed as a linear combination of others is redundant. The pipeline reports how many independent face directions exist and how many samples are linearly dependent on others.

**Step 5 — Orthogonalization**

Applies `numpy.linalg.eigh` (eigendecomposition for symmetric matrices) to L. The Spectral Theorem guarantees that eigenvectors of a symmetric matrix are orthogonal — this is equivalent to running Gram-Schmidt and is verified explicitly. Eigenvalues are sorted from largest to smallest.

The true eigenfaces (eigenvectors of C) are recovered from the eigenvectors of L using:

```
v_i = Ãᵀ × u_i         (then normalised to unit length)
```

where u_i is the i-th eigenvector of L. The top k=50 eigenfaces are kept.

**Step 6 — Projection**

Projects every centred training face onto the eigenface subspace:

```
raw_projection = Ã × Eᵀ              shape: (n, 50)
projection = raw / ‖raw‖              L2-normalised to unit vectors
```

Each face is now a point in ℝ⁵⁰ instead of ℝ¹⁰⁰⁰⁰. These 50 coordinates are stored in `self.projections` and used during recognition to compare new faces against training faces.

**Step 7 — Least Squares**

Demonstrates the least squares formula on the first training image as a proof of concept:

```
x̂ = (EᵀE)⁻¹Eᵀb
```

Since E has orthonormal columns, EᵀE ≈ I, so x̂ ≈ Eᵀb. The reconstruction **b̂ = Ex̂** is compared to the original b and the reconstruction error ‖b − b̂‖ is printed. During inference, the recognizer uses k-NN in eigenspace which is mathematically the minimum-residual (least squares) match.

**Step 8 — Pattern Discovery**

Eigenvalues are ranked by size. A large eigenvalue means that eigenface captures a dominant mode of variation across all training faces — for example, the difference between two people's face shapes. A small eigenvalue means that direction captures mainly noise. The percentage of total variance explained by each eigenface is computed and printed.

**Step 9 — System Simplification (Diagonalisation)**

The covariance matrix is expressed in its diagonalised form:

```
C ≈ E × Λ × Eᵀ
```

where E is the 50×10,000 eigenface matrix and Λ is the 50×50 diagonal matrix of eigenvalues. Retaining only 50 dimensions compresses each face from 10,000 numbers to 50 — a 200× reduction.

**The `knn_predict()` method — used during live recognition:**

1. Computes cosine similarity between the query projection and every training projection: `sims = self.projections @ coords`
2. Finds the top 5 most similar training samples
3. Each of those 5 neighbours casts a weighted vote for its label (weight = similarity score)
4. The label with the highest total weighted vote wins
5. A ratio test checks if the two best identities have very similar scores — if so, confidence is penalised because the system is uncertain
6. Returns the winning label and a confidence value between 0 and 1

**`visualize()` method — Option 4 in the menu:**

Shows two matplotlib plots:
- A grid of the top-20 eigenfaces rendered as 100×100 images (scaled to visible range)
- A bar + line chart of individual and cumulative explained variance per eigenface

---

### 5. `recognizer.py` — Live Face Recognition

**What it does:** Opens the webcam and identifies all faces in the frame in real time using the trained pipeline.

**How it works — what happens every frame:**

```
1. cap.read()              → grab one frame from webcam
2. cvtColor(BGR→GREY)      → convert to greyscale
3. detect_faces()          → find bounding boxes of all faces
4. _predict_unique()       → project + assign identities uniquely
5. _smooth()               → stabilise with temporal voting
6. Draw boxes and labels   → display on screen
```

**`_predict_unique()` in detail — preventing two faces sharing a label:**

This method processes ALL detected faces together rather than one at a time, which is what prevents the duplicate-label bug from v1.

1. For each detected face, calls `preprocess()` then `pipeline.project()` to get its eigenspace coordinates
2. Computes the normalised mean projection for each known identity
3. Builds a cosine distance matrix D where D[i][j] = how far face i is from identity j's average in eigenspace
4. Greedy assignment: finds the minimum entry in D, assigns that face→identity pair, then crosses out that row and column so neither can be reused
5. For the assigned identity, calls `pipeline.knn_predict()` to get a confidence score
6. If the k-NN winner disagrees with the uniquely assigned identity, confidence is reduced by 30% as a penalty
7. If final confidence is below `MIN_CONFIDENCE = 0.40`, shows "Unknown" instead of a wrong name

**`_smooth()` in detail — stopping label flickering:**

Maintains a separate `deque` of length 7 for each face slot (slot 0 = leftmost face, slot 1 = next, etc.). Every frame's prediction is appended to the deque. The displayed result is:
- Label: majority vote across the last 7 frames
- Confidence: mean confidence only from frames that agree with the majority label

This means a single blurry frame or bad detection cannot flip the displayed name. The label only changes when several consecutive frames agree on a new identity.

**What you see on screen:**

| Visual element | What it means |
|----------------|---------------|
| Green box | Recognised identity above MIN_CONFIDENCE |
| Red box | "Unknown" — below threshold |
| `Name  72%` above box | Identity label and confidence percentage |
| Filled bar below box | Confidence bar — wider = more confident |
| `w1:0.32 w2:-0.14 w3:0.07` | First 3 eigenspace coordinates of this face |

---

### 6. `utils.py` — Display Helpers

**What it does:** Provides three simple console-formatting functions used by `la_pipeline.py` and `main.py` to keep terminal output readable.

- `print_banner()` — prints the PES University / project title header when the menu loads
- `print_section(title)` — prints a horizontal line with a section title
- `print_step(n, title)` — prints a numbered step header during training, e.g. `┌── STEP 3: Structure of the Space`

No mathematics here — purely cosmetic output formatting.

---

## How All Files Connect

```
                         ┌─────────────┐
                         │   main.py   │  ← you run this
                         └──────┬──────┘
          ┌────────────────────┼───────────────────┐
          ▼                    ▼                   ▼
data_collection.py      la_pipeline.py       recognizer.py
          │                    ▲                   │
          │                    │ (shared object)   │
          └──────────────────── pipeline ──────────┘
          │                                        │
          └──────────── preprocessing.py ──────────┘
                        (used by BOTH for
                         consistent face prep)
          
la_pipeline.py ──uses──► utils.py
                          (for console output)
```

**Data flow through the system:**

```
Webcam frame
     ↓
preprocessing.py    → aligned, equalised, normalised 10,000-dim vector
     ↓
data_collection.py  → saved as faces_db/<name>/frame_XXXX.npy
     ↓
la_pipeline.py      → loads all .npy files
                    → builds data matrix A (n × 10,000)
                    → runs steps 1-9
                    → stores: mean_face, eigenfaces, projections, labels
     ↓
recognizer.py       → uses pipeline.project() and pipeline.knn_predict()
                    → unique assignment + temporal smoothing
                    → draws result on camera frame
```

---

## The Full Linear Algebra Pipeline

```
REAL-WORLD DATA  (camera frames via OpenCV)
        ↓
[Step 1]  Matrix Representation
          Each 100×100 face → 10,000-dim vector → stacked into A (n × 10,000)

        ↓
[Step 2]  Matrix Simplification
          Ã = A − μ  (subtract mean face from every row)

        ↓
[Step 3]  Structure of the Space
          rank(Ã), nullity, surrogate covariance L = ÃÃᵀ/n

        ↓
[Step 4]  Remove Redundancy
          rank = independent face directions; n−rank = redundant samples

        ↓
[Step 5]  Orthogonalization
          eigh(L) → eigenvectors → recover eigenfaces vᵢ = Ãᵀuᵢ (normalised)

        ↓
[Step 6]  Projection
          Ω = Ã × Eᵀ  then normalise → each face is now a point in ℝ⁵⁰

        ↓
[Step 7]  Prediction / Least Squares
          x̂ = (EᵀE)⁻¹Eᵀb; k-NN with cosine similarity for recognition

        ↓
[Step 8]  Pattern Discovery
          Eigenvalues ranked; variance explained per eigenface computed

        ↓
[Step 9]  System Simplification
          C = EΛEᵀ; retain top-50; compression 10,000 → 50 (200×)

        ↓
FINAL OUTPUT  (live identity + confidence % on webcam feed)
```

---

## Key Equations

| Equation | Meaning |
|----------|---------|
| **A ∈ ℝⁿˣᵈ** | Data matrix — n faces, each d = 10,000 pixels |
| **μ = (1/n) Σ aᵢ** | Mean face — pixel-wise average across all training images |
| **Ã = A − 1μᵀ** | Centred matrix — mean face removed from every row |
| **L = ÃÃᵀ/n** | Surrogate covariance (n×n) — same eigenvalues as true C |
| **L uᵢ = λᵢ uᵢ** | Eigendecomposition of surrogate covariance |
| **vᵢ = Ãᵀ uᵢ / ‖Ãᵀ uᵢ‖** | Recover and normalise true eigenfaces |
| **Ω = Eᵀ(b − μ)** | Project a new face b onto eigenface subspace |
| **x̂ = (EᵀE)⁻¹Eᵀb** | Least squares reconstruction of face from eigenspace |
| **C = EΛEᵀ** | Spectral decomposition — diagonalisation of covariance |
| **sim = Ωᵢ · Ωⱼ** | Cosine similarity (dot product of unit vectors) for matching |

---

## Viva Preparation

Answer every question using: **What concept → Why needed → What result**

| Concept | What was used | Why it was needed | What result it produced |
|---------|--------------|-------------------|------------------------|
| Data Matrix A | Stacked face vectors into A (n×10,000) | Need a matrix to apply linear algebra to images | Single object representing all training faces |
| Mean-centering | Ã = A − μ | Remove average brightness; focus on differences between people | Centred matrix where individual features are visible |
| Rank & Nullity | numpy.linalg.matrix_rank | Know true dimensionality of face space | rank = independent directions; nullity = redundant dimensions |
| Surrogate Covariance | L = ÃÃᵀ/n instead of ÃᵀÃ/n | True covariance is 10,000×10,000 — impossible to compute | Compact n×n matrix with the same eigenvalues |
| Eigendecomposition | eigh(L) | Extract the principal face patterns from data | Orthogonal eigenvectors sorted by how much face variation they capture |
| Spectral Theorem | Eigenvectors of symmetric matrix are orthogonal | Guarantees no redundancy in eigenface basis | Orthogonal basis without needing to run Gram-Schmidt manually |
| Projection | Ω = Eᵀ(b−μ) | Compress face from 10,000 dims to 50 for fast comparison | Each face = 50 coordinates in eigenspace |
| Least Squares | x̂ = (EᵀE)⁻¹Eᵀb | System is inconsistent due to missing data; need best approximate solution | Minimum reconstruction error; nearest neighbour in eigenspace = LS match |
| Eigenvalues | λ₁ ≥ λ₂ ≥ ... ≥ 0 | Large λ = dominant face pattern; small λ = noise | Variance explained chart; chose top 50 to keep |
| Diagonalisation | C = EΛEᵀ | Simplify the system to a compressed orthogonal coordinate space | 200× compression with minimal information loss |
| Cosine Similarity | dot product of unit vectors | Better than Euclidean in high-dimensional spaces; scale-invariant | More accurate matching across lighting conditions |
| k-NN Voting | Top-5 neighbours by cosine similarity | Single nearest neighbour sensitive to one bad training sample | Robust majority vote reduces misclassification rate |
| Temporal Smoothing | Majority vote over last 7 frames | Individual video frames are noisy due to motion or detection error | Stable label that does not flicker during live recognition |
