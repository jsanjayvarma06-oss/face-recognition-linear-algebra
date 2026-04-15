"""
preprocessing.py
────────────────
Face preprocessing pipeline applied consistently at both training AND inference.
Consistent preprocessing is the single biggest accuracy lever.

Steps applied to every face crop:
  1. Align      — detect eyes, rotate + scale so eyes are always horizontal
                  at fixed pixel positions  (removes pose variation)
  2. Crop       — tight crop around face region after alignment
  3. Resize     — fixed IMG_SIZE
  4. CLAHE      — Contrast Limited Adaptive Histogram Equalisation
                  (removes lighting / shadow variation)
  5. Normalise  — zero mean, unit variance per image (Z-score)
"""

import cv2
import numpy as np

IMG_SIZE = (100, 100)       # larger crop → more detail for eigenfaces
IMG_DIM  = IMG_SIZE[0] * IMG_SIZE[1]

# Haar cascades
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_EYE_CASCADE  = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Target eye positions as fraction of output image size
_EYE_LEFT_TARGET  = (0.35, 0.40)   # (x_frac, y_frac)
_EYE_RIGHT_TARGET = (0.65, 0.40)


# ─────────────────────────────────────────────────────────────────────────────
def detect_faces(grey: np.ndarray):
    """Return list of (x,y,w,h) bounding boxes detected in grey image."""
    return _FACE_CASCADE.detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))


def preprocess(grey: np.ndarray, face_box=None) -> np.ndarray | None:
    """
    Full preprocessing pipeline for a single face.

    grey      : full grey frame  OR  already-cropped face region
    face_box  : (x,y,w,h) if grey is the full frame; None if already cropped

    Returns   : flat float32 vector of length IMG_DIM, or None if processing fails
    """
    if face_box is not None:
        x, y, w, h = face_box
        face = grey[y:y+h, x:x+w]
    else:
        face = grey

    # ── 1. Attempt eye-based alignment ───────────────────────────────────────
    aligned = _align_face(face)
    if aligned is None:
        # Fall back: just resize without alignment
        aligned = cv2.resize(face, IMG_SIZE)

    # ── 2. CLAHE equalisation ─────────────────────────────────────────────────
    equalised = _CLAHE.apply(aligned)

    # ── 3. Z-score normalisation per image ───────────────────────────────────
    img_f = equalised.astype(np.float32)
    mu, sigma = img_f.mean(), img_f.std()
    if sigma < 1e-6:
        return None                         # blank / uniform patch — skip
    normalised = (img_f - mu) / sigma

    return normalised.flatten()


# ─────────────────────────────────────────────────────────────────────────────
def _align_face(face: np.ndarray) -> np.ndarray | None:
    """
    Detect eyes inside the face crop; compute rotation + scale to align them
    to fixed target positions; return the warped, resized face.

    Returns None if fewer than 2 eyes are found (caller falls back to resize).
    """
    h, w   = face.shape[:2]
    eyes   = _EYE_CASCADE.detectMultiScale(face, scaleFactor=1.1, minNeighbors=10)

    if len(eyes) < 2:
        return None

    # Take the two largest eye detections
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    # Eye centres
    centres = [(ex + ew // 2, ey + eh // 2) for (ex, ey, ew, eh) in eyes]
    # Sort left→right
    centres = sorted(centres, key=lambda c: c[0])
    lx, ly  = centres[0]
    rx, ry  = centres[1]

    # Angle between eyes
    dy    = ry - ly
    dx    = rx - lx
    angle = np.degrees(np.arctan2(dy, dx))

    # Desired eye distance in output image
    out_w, out_h = IMG_SIZE
    desired_dist = (_EYE_RIGHT_TARGET[0] - _EYE_LEFT_TARGET[0]) * out_w
    current_dist = np.sqrt(dx**2 + dy**2) + 1e-6
    scale        = desired_dist / current_dist

    # Centre of eyes in current image
    eye_cx = (lx + rx) / 2.0
    eye_cy = (ly + ry) / 2.0

    # Rotation + scale matrix around eye centre
    M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, scale)

    # Shift so the eye midpoint lands at target position in output
    target_cx = out_w * (_EYE_LEFT_TARGET[0] + _EYE_RIGHT_TARGET[0]) / 2.0
    target_cy = out_h * (_EYE_LEFT_TARGET[1] + _EYE_RIGHT_TARGET[1]) / 2.0
    M[0, 2] += target_cx - eye_cx
    M[1, 2] += target_cy - eye_cy

    warped = cv2.warpAffine(face, M, IMG_SIZE,
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped
