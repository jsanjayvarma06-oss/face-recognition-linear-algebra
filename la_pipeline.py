"""
la_pipeline.py
──────────────
THE CORE LINEAR ALGEBRA PIPELINE
Following the PES University UE24MA241B workflow diagram exactly.

Accuracy improvements over v1
──────────────────────────────
• IMG_SIZE 64→100  : more pixel information per face
• n_components 20→50: richer eigenspace
• Cosine similarity  : better than Euclidean in high-dimensional spaces
• k-NN with majority vote (k=5): more robust than single nearest neighbour
• Confidence ratio test: best_dist / second_best_dist threshold
• Per-identity mean projection: smoother class centres

Steps implemented
─────────────────
1  Matrix Representation     — load images -> data matrix  A in R^(n x d)
2  Matrix Simplification     — mean-centre A
3  Structure of the Space    — covariance matrix, rank, nullity
4  Remove Redundancy         — linear independence check
5  Orthogonalization         — eigenvectors orthogonal (Spectral Theorem)
6  Projection                — project face onto eigenface subspace
7  Least Squares Prediction  — x_hat = (A^T A)^-1 A^T b; k-NN in subspace
8  Pattern Discovery         — eigenvalues / eigenfaces
9  System Simplification     — diagonalization, top-k retention
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from preprocessing import IMG_SIZE, IMG_DIM
from utils import print_step


class LinearAlgebraPipeline:

    def __init__(self, n_components: int = 50, n_neighbours: int = 5):
        self.k           = n_components
        self.knn         = n_neighbours
        self.is_trained  = False

        self.mean_face   = None
        self.eigenfaces  = None
        self.eigenvalues = None
        self.projections = None
        self.labels      = []
        self.rank        = 0
        self.nullity     = 0

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def train(self, data_dir: str = "faces_db") -> None:
        # STEP 1
        print_step(1, "Matrix Representation")
        A, labels = self._load_data(data_dir)
        if A is None:
            print("  [ERROR] No training data found. Collect faces first (option 1).")
            return
        n, d = A.shape
        print(f"  Data matrix A : {n} samples x {d} features")
        print(f"  Labels        : {sorted(set(labels))}")

        # STEP 2
        print_step(2, "Matrix Simplification - Mean-Centering")
        mean_face = np.mean(A, axis=0)
        A_centred = A - mean_face
        print(f"  Mean face shape : {mean_face.shape}  |  norm={np.linalg.norm(mean_face):.3f}")

        # STEP 3
        print_step(3, "Structure of the Space - Rank, Nullity, Covariance")
        rank    = int(np.linalg.matrix_rank(A_centred, tol=1e-3))
        nullity = d - rank
        print(f"  rank(A_c) = {rank}  |  nullity = {nullity}  |  rank+nullity = {rank+nullity} = d")
        L = (1.0 / n) * (A_centred @ A_centred.T)
        print(f"  Surrogate covariance L : {L.shape}")

        # STEP 4
        print_step(4, "Remove Redundancy - Linear Independence")
        print(f"  Independent directions : {rank}  |  Redundant samples : {n - rank}")

        # STEP 5
        print_step(5, "Orthogonalization - Spectral Theorem")
        eigenvalues_L, eigenvectors_L = np.linalg.eigh(L)
        idx            = np.argsort(eigenvalues_L)[::-1]
        eigenvalues_L  = eigenvalues_L[idx]
        eigenvectors_L = eigenvectors_L[:, idx]

        k_use = min(self.k, rank, n - 1)
        eigenfaces = np.zeros((k_use, d), dtype=np.float32)
        eigenvals  = np.zeros(k_use,      dtype=np.float32)

        for i in range(k_use):
            u_i  = eigenvectors_L[:, i]
            v_i  = A_centred.T @ u_i
            norm = np.linalg.norm(v_i)
            if norm > 1e-10:
                v_i /= norm
            eigenfaces[i] = v_i.astype(np.float32)
            eigenvals[i]  = max(float(eigenvalues_L[i]), 0.0)

        self._verify_orthogonality(eigenfaces[:5])
        print(f"  {k_use} eigenfaces computed")

        # STEP 6
        print_step(6, "Projection - Project Training Faces onto Eigenface Subspace")
        raw_proj = A_centred @ eigenfaces.T           # (n, k)
        norms    = np.linalg.norm(raw_proj, axis=1, keepdims=True)
        norms    = np.where(norms < 1e-10, 1.0, norms)
        projections = raw_proj / norms                # L2-normalised -> cosine sim
        print(f"  Projection matrix : {projections.shape}  (L2-normalised)")

        # STEP 7
        print_step(7, "Prediction / Approximation - Least Squares")
        E      = eigenfaces.T
        b      = A_centred[0]
        x_hat  = np.linalg.lstsq(E.T @ E, E.T @ b, rcond=None)[0]
        ls_err = np.linalg.norm(b - E @ x_hat)
        print(f"  x_hat shape : {x_hat.shape}  |  reconstruction error = {ls_err:.4f}")

        # STEP 8
        print_step(8, "Pattern Discovery - Eigenvalues & Explained Variance")
        total_var  = float(np.sum(eigenvals)) or 1.0
        explained  = eigenvals / total_var * 100
        cumulative = np.cumsum(explained)
        for i in range(min(5, k_use)):
            print(f"    lambda_{i+1:02d} = {eigenvals[i]:10.3f} | "
                  f"{explained[i]:5.2f}% | cum {cumulative[i]:5.2f}%")
        print(f"  Total explained ({k_use} eigenfaces): {cumulative[-1]:.2f}%")

        # STEP 9
        print_step(9, "System Simplification - Diagonalisation  C = E*Lambda*E^T")
        Lambda = np.diag(eigenvals)
        print(f"  Lambda shape : {Lambda.shape}  | compression {d}->{k_use} ({d/k_use:.1f}x)")

        self.mean_face   = mean_face
        self.eigenfaces  = eigenfaces
        self.eigenvalues = eigenvals
        self.projections = projections
        self.labels      = labels
        self.rank        = rank
        self.nullity     = nullity
        self.k           = k_use
        self.is_trained  = True
        print("\n  Training complete.\n")

    def project(self, face_vec: np.ndarray) -> np.ndarray:
        """Step 6 inference: returns L2-normalised eigenspace coords."""
        centred = face_vec.astype(np.float32) - self.mean_face
        raw     = centred @ self.eigenfaces.T
        norm    = np.linalg.norm(raw)
        return raw / norm if norm > 1e-10 else raw

    def knn_predict(self, coords: np.ndarray) -> Tuple[str, float]:
        """
        Step 7 inference: k-NN with cosine similarity + ratio test.
        Returns (label, confidence 0-1).
        """
        sims  = self.projections @ coords        # cosine similarity (unit vectors)
        order = np.argsort(sims)[::-1]
        top_k = order[:self.knn]

        # Weighted majority vote
        votes = {}
        for idx in top_k:
            lbl = self.labels[idx]
            votes[lbl] = votes.get(lbl, 0.0) + float(sims[idx])

        best_label   = max(votes, key=votes.get)
        total_weight = sum(max(v, 0) for v in votes.values()) or 1.0
        confidence   = max(votes[best_label], 0.0) / total_weight

        # Ratio test between top two identities
        id_best = {}
        for idx in order:
            lbl = self.labels[idx]
            if lbl not in id_best:
                id_best[lbl] = float(sims[idx])
            if len(id_best) == len(set(self.labels)):
                break
        id_sorted = sorted(id_best.values(), reverse=True)
        if len(id_sorted) >= 2:
            ratio = (1 - id_sorted[0]) / (1 - id_sorted[1] + 1e-6)
            if ratio > 0.92:
                confidence *= (1 - ratio)

        return best_label, float(confidence)

    def visualize(self) -> None:
        if not self.is_trained:
            print("  Not trained yet.")
            return
        n_show = min(self.k, 20)
        cols, rows = 5, (n_show + 4) // 5
        fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows))
        fig.suptitle(f"Top-{n_show} Eigenfaces", fontsize=13, fontweight="bold")
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                ef = self.eigenfaces[i].reshape(IMG_SIZE)
                ef_n = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
                ax.imshow(ef_n, cmap="bone")
                ax.set_title(f"EF-{i+1}\nlambda={self.eigenvalues[i]:.1f}", fontsize=7)
            ax.axis("off")
        plt.tight_layout()

        total_var  = float(np.sum(self.eigenvalues)) or 1.0
        explained  = self.eigenvalues / total_var * 100
        cumulative = np.cumsum(explained)
        fig2, ax2  = plt.subplots(figsize=(7, 4))
        ax2.bar(range(1, self.k+1), explained,  color="steelblue", alpha=0.7, label="Individual %")
        ax2.plot(range(1, self.k+1), cumulative, color="crimson", marker="o", markersize=3, label="Cumulative %")
        ax2.axhline(95, linestyle="--", color="grey", alpha=0.6, label="95% threshold")
        ax2.set_xlabel("Eigenface index"); ax2.set_ylabel("Variance explained (%)")
        ax2.set_title("Step 8 - Eigenvalue Analysis"); ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    @staticmethod
    def _load_data(data_dir: str) -> Tuple[Optional[np.ndarray], List[str]]:
        if not os.path.isdir(data_dir):
            return None, []
        rows, labels = [], []
        for person in sorted(os.listdir(data_dir)):
            person_dir = os.path.join(data_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for fname in sorted(os.listdir(person_dir)):
                if not fname.endswith(".npy"):
                    continue
                vec = np.load(os.path.join(person_dir, fname))
                if vec.shape[0] != IMG_DIM:
                    continue
                rows.append(vec)
                labels.append(person)
        if not rows:
            return None, []
        return np.array(rows, dtype=np.float32), labels

    @staticmethod
    def _verify_orthogonality(vectors: np.ndarray) -> None:
        n = len(vectors)
        max_off = max(
            abs(float(np.dot(vectors[i], vectors[j])))
            for i in range(n) for j in range(i+1, n)
        ) if n > 1 else 0.0
        ok = "orthogonal" if max_off < 1e-3 else "not perfectly orthogonal"
        print(f"  Orthogonality check (first {n}): max|vi.vj| = {max_off:.2e}  ({ok})")
