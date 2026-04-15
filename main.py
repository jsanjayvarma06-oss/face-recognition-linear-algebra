"""
==============================================================================
PES UNIVERSITY — UE24MA241B: Linear Algebra and Its Applications
Mini Project: Face Recognition Using Linear Algebra Pipeline
==============================================================================
Pipeline:
  Real-World Data (Camera Frames)
      ↓ Matrix Representation       (Step 1)
      ↓ Matrix Simplification       (Step 2 — mean-centering)
      ↓ Structure of the Space      (Step 3 — covariance, rank)
      ↓ Remove Redundancy           (Step 4 — linear independence)
      ↓ Orthogonalization           (Step 5 — Gram–Schmidt / eigenvectors)
      ↓ Projection                  (Step 6 — project onto eigenface subspace)
      ↓ Prediction via Least Squares(Step 7 — nearest neighbour in subspace)
      ↓ Pattern Discovery           (Step 8 — eigenvalues / eigenfaces)
      ↓ System Simplification       (Step 9 — diagonalization, top-k)
      ↓ Final Output                (Face Recognition / Identity)
==============================================================================
"""

import sys
import os

# ── make sure project root is on PYTHONPATH ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_collection  import collect_faces
from la_pipeline      import LinearAlgebraPipeline
from recognizer       import FaceRecognizer
from utils            import print_banner, print_section


def menu() -> str:
    print_banner()
    print("  1. Collect training faces (camera)")
    print("  2. Train the Linear Algebra Pipeline")
    print("  3. Recognize a face (camera)")
    print("  4. Show eigenfaces & explained variance")
    print("  5. Exit")
    return input("\n  Choose option [1-5]: ").strip()


def main() -> None:
    pipeline   = LinearAlgebraPipeline()
    recognizer = FaceRecognizer(pipeline)

    while True:
        choice = menu()

        if choice == "1":
            print_section("STEP 1 — Collect Training Faces")
            label = input("  Enter person name/ID: ").strip()
            if label:
                collect_faces(label, n_samples=30)
            else:
                print("  [!] No label entered — skipping.")

        elif choice == "2":
            print_section("TRAINING — Linear Algebra Pipeline")
            pipeline.train(data_dir="faces_db")

        elif choice == "3":
            if not pipeline.is_trained:
                print("\n  [!] Pipeline not trained yet. Run option 2 first.")
            else:
                print_section("STEP 6 & 7 — Projection + Recognition")
                recognizer.recognize_from_camera()

        elif choice == "4":
            if not pipeline.is_trained:
                print("\n  [!] Pipeline not trained yet. Run option 2 first.")
            else:
                print_section("STEP 8 & 9 — Eigenfaces + System Simplification")
                pipeline.visualize()

        elif choice == "5":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Please enter 1–5.")


if __name__ == "__main__":
    main()
