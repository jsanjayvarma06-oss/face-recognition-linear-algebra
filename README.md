# 🎓 Face Recognition Using Linear Algebra Pipeline

**PES University — UE24MA241B: Linear Algebra and Its Applications**
**Mini Project**

---

## 📌 Project Overview

This project demonstrates a **real-time face recognition system** built entirely from **first principles of linear algebra** — no pre-trained neural networks or black-box ML libraries. Every step of the pipeline directly maps to a core linear algebra concept taught in the course.

The system uses your webcam to:
1. **Collect** labelled face samples
2. **Train** an Eigenface model using SVD / PCA (via covariance eigendecomposition)
3. **Recognize** faces live from the camera in real time

---

## 🔢 The 9-Step Linear Algebra Pipeline

The pipeline mirrors the course workflow exactly:

| Step | Concept | What it does |
|------|---------|--------------|
| **1** | Matrix Representation | Load face images → data matrix **A** ∈ ℝⁿˣᵈ |
| **2** | Matrix Simplification | Mean-centre **A** (subtract mean face) |
| **3** | Structure of the Space | Compute surrogate covariance **L**, rank & nullity |
| **4** | Remove Redundancy | Identify linearly independent directions |
| **5** | Orthogonalization | Spectral Theorem → compute orthonormal eigenfaces |
| **6** | Projection | Project each face onto the eigenface subspace |
| **7** | Least Squares Prediction | k-NN in eigenspace (cosine similarity) |
| **8** | Pattern Discovery | Eigenvalues → explained variance per eigenface |
| **9** | System Simplification | Diagonalization **C = EΛEᵀ**, retain top-k components |

---

## 📂 File Structure

```
files/
│
├── main.py              # Entry point — interactive menu
├── la_pipeline.py       # Core linear algebra pipeline (Steps 1–9)
├── recognizer.py        # Live inference with temporal smoothing & unique assignment
├── preprocessing.py     # Face alignment, CLAHE, Z-score normalisation
├── data_collection.py   # Webcam face data collector (Step 1)
├── utils.py             # Banner / section printing helpers
├── requirements.txt     # Python dependencies
│
└── faces_db/            # Auto-created; stores training samples
    └── <person_name>/
        ├── frame_0000.npy
        ├── frame_0001.npy
        └── ...
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.10+**
- A working **webcam**

### Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:
```
numpy>=1.24
opencv-python>=4.8
matplotlib>=3.7
```

---

## 🚀 How to Run

```bash
python main.py
```

You will see an interactive menu:

```
  1. Collect training faces (camera)
  2. Train the Linear Algebra Pipeline
  3. Recognize a face (camera)
  4. Show eigenfaces & explained variance
  5. Exit
```

### Recommended Workflow

**Step 1 — Collect face data for each person**
- Choose option `1`
- Enter a name/ID (e.g., `Alice`)
- The webcam opens and automatically saves **30 frames** per person
- **Tips shown on screen**: vary your distance, tilt, and expression for better generalisation
- Repeat for every person you want the system to recognise

**Step 2 — Train the pipeline**
- Choose option `2`
- The system runs all 9 linear algebra steps and prints detailed diagnostics
- Training is fast (seconds on a normal laptop)

**Step 3 — Live face recognition**
- Choose option `3`
- The webcam opens with real-time bounding boxes, identity labels, and confidence bars
- Press **Q** to stop

**Step 4 — Visualise eigenfaces**
- Choose option `4`
- Displays the top-20 eigenfaces and an explained variance chart

---

## 🧠 Technical Deep-Dive

### Preprocessing (`preprocessing.py`)

Consistent preprocessing is the **single biggest accuracy lever**. Every face — at both training and inference time — goes through:

1. **Eye-based alignment** — detects both eyes with Haar cascades, rotates and scales the face so eyes always land at fixed pixel positions. This removes pose variation.
2. **Tight crop & resize** — fixed output size of **100 × 100** pixels.
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) — removes lighting and shadow variation.
4. **Z-score normalisation** — zero mean, unit variance per image.

### Eigenface Training (`la_pipeline.py`)

- Builds the surrogate covariance matrix **L = (1/n) AₒAₒᵀ** ∈ ℝⁿˣⁿ (feasible because n « d).
- Uses `np.linalg.eigh` for stable symmetric eigendecomposition.
- Recovers the true d-dimensional eigenvectors via **v_i = Aₒᵀ u_i** (the "kernel trick" for PCA).
- Retains the top **k = 50** eigenfaces (configurable).
- All training projections are **L2-normalised**, enabling cosine similarity at recognition time.

### Recognition (`recognizer.py`)

Three accuracy improvements over a naive approach:

| Technique | Purpose |
|-----------|---------|
| **k-NN cosine similarity** (k=5) | More robust than single nearest-neighbour Euclidean distance |
| **Confidence ratio test** | Suppresses "Unknown" ambiguity when top-2 identities are too close |
| **Temporal smoothing** | Majority vote over the last 7 frames prevents flickering labels |
| **Unique assignment** | Greedy optimal matching — no two faces share the same identity label |

Confidence threshold: labels below **40%** are displayed as `Unknown`.

---

## 🎯 Tips for Best Accuracy

| Tip | Why it helps |
|-----|-------------|
| Collect ≥ 30 samples per person | More data → better eigenspace coverage |
| Vary head angle and distance during collection | Improves robustness to pose |
| Use consistent, neutral lighting | Reduces CLAHE workload |
| Collect data in the same environment you test in | Domain shift hurts accuracy |
| Re-train after adding new people | The pipeline must see all classes at once |

---

## 📊 Key Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `n_components` (k) | `la_pipeline.py` | `50` | Number of eigenfaces; higher = richer representation |
| `n_neighbours` (knn) | `la_pipeline.py` | `5` | k in k-NN vote |
| `IMG_SIZE` | `preprocessing.py` | `(100, 100)` | Face crop resolution |
| `MIN_CONFIDENCE` | `recognizer.py` | `0.40` | Threshold below which label shows as Unknown |
| `WINDOW` | `recognizer.py` | `7` | Temporal smoothing window (frames) |
| `n_samples` | `data_collection.py` | `30` | Samples collected per person |

---

## 🔗 Linear Algebra Concepts Demonstrated

- **Vector spaces & subspaces** — face images as vectors in ℝ¹⁰⁰⁰⁰
- **Matrix rank & nullity** — Rank-Nullity Theorem verified at training time
- **Covariance matrices** — surrogate L for efficient PCA
- **Spectral Theorem** — symmetric matrix → orthogonal eigenvectors
- **Gram–Schmidt orthogonalisation** — guaranteed orthonormal eigenface basis
- **Projection onto a subspace** — Step 6 (inference and training identical)
- **Least Squares** — `(AᵀA)⁻¹Aᵀb` demonstrated in Step 7
- **Diagonalization** — C = EΛEᵀ shown in Step 9
- **Cosine similarity** — inner product in normalised eigenspace

---

## 📝 License

This project is submitted as academic coursework for **UE24MA241B** at **PES University**. All code is original work by the submitting student(s).
