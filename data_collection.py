"""
data_collection.py
──────────────────
STEP 1 — Real-World Data → Matrix Representation

Uses OpenCV webcam + the shared preprocessing pipeline so that training
images are stored in the SAME normalised form used at inference time.
Consistent preprocessing is essential for accurate eigenface matching.

Collects n_samples per person; stores each as a flat float32 .npy vector
of dimension d = IMG_DIM (100×100 = 10 000).
"""

import os
import cv2
import numpy as np
from preprocessing import detect_faces, preprocess, IMG_SIZE


def collect_faces(label: str, n_samples: int = 50, data_dir: str = "faces_db") -> None:
    """
    Open webcam, detect + preprocess faces, save n_samples for `label`.
    Saved path: faces_db/<label>/frame_XXXX.npy

    Tips shown on screen:
      • Move closer / farther
      • Tilt head slightly left / right
      • Change expression
    → variation helps eigenface generalisation
    """
    save_dir = os.path.join(data_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    cap      = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Cannot open camera. Check permissions / device index.")
        return

    count = 0
    print(f"\n  [INFO] Collecting {n_samples} samples for '{label}'.")
    print("  Tips: vary distance, slight head tilts, expressions.")
    print("  Press  Q  to quit early.\n")

    tips = [
        "Look straight at camera",
        "Move slightly left",
        "Move slightly right",
        "Tilt head a little",
        "Change expression",
        "Move closer",
        "Move farther",
    ]

    while count < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(grey)

        for (x, y, w, h) in faces:
            vec = preprocess(grey, face_box=(x, y, w, h))
            if vec is None:
                continue

            idx  = existing + count
            path = os.path.join(save_dir, f"frame_{idx:04d}.npy")
            np.save(path, vec)
            count += 1

            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}  [{count}/{n_samples}]",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show alignment tip based on progress
            tip = tips[count % len(tips)]
            cv2.putText(frame, f"Tip: {tip}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
            break  # one face per frame

        # Progress bar
        if n_samples > 0:
            bar_w    = int(frame.shape[1] * count / n_samples)
            cv2.rectangle(frame, (0, frame.shape[0]-5),
                          (bar_w, frame.shape[0]), (0, 200, 100), -1)

        cv2.putText(frame, "Collecting — press Q to stop",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  [DONE] Saved {count} samples → '{save_dir}'\n")
