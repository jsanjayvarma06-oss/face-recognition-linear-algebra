"""
recognizer.py  — improved v2
─────────────────────────────
Accuracy improvements:
• Shared preprocessing (CLAHE + alignment) — SAME as training
• k-NN cosine similarity instead of Euclidean nearest neighbour
• Temporal smoothing: majority vote over last WINDOW frames
• Confidence threshold: labels hidden below MIN_CONFIDENCE
• Unique per-frame assignment: no two faces share the same label
"""

import cv2
import numpy as np
from collections import deque, Counter
from preprocessing import detect_faces, preprocess
from la_pipeline   import LinearAlgebraPipeline

MIN_CONFIDENCE = 0.40
WINDOW         = 7


class FaceRecognizer:

    def __init__(self, pipeline: LinearAlgebraPipeline):
        self.pipeline   = pipeline
        self._history   = {}          # slot -> deque of (label, conf)
        self._last_boxes= {}          # slot -> (cx, cy)
        self._next_slot = 0

    def recognize_from_camera(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  [ERROR] Cannot open camera.")
            return
        print("  Live recognition active — press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(grey)

            if len(faces) > 0:
                raw = self._predict_unique(grey, faces)
                slots = self._track_faces(faces)
                
                for slot, (x, y, w, h), (label, conf, coords) in zip(slots, faces, raw):
                    label, conf = self._smooth(slot, label, conf)
                    color = (0, 200, 0) if label != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label}  {int(conf*100)}%",
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    bar = int((w-4) * conf)
                    cv2.rectangle(frame, (x+2, y+h+4), (x+2+bar, y+h+12), color, -1)
                    cv2.rectangle(frame, (x+2, y+h+4), (x+w-2,   y+h+12), (100,100,100), 1)
                    cs = " ".join([f"w{i+1}:{v:.2f}" for i,v in enumerate(coords[:3])])
                    cv2.putText(frame, cs, (x, y+h+24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,50), 1)
            else:
                self._last_boxes = {}
                self._history.clear()

            cv2.putText(frame, "Eigenface Recognition  |  Q = quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
            cv2.imshow("Face Recognition (Linear Algebra Pipeline)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _track_faces(self, current_faces):
        current_slots = []
        new_boxes = {}
        cents = [(x + w/2, y + h/2) for (x, y, w, h) in current_faces]
        used_slots = set()
        
        for cx, cy in cents:
            best_slot = None
            best_dist = 100000
            for slot, (lcx, lcy) in self._last_boxes.items():
                if slot in used_slots: continue
                dist = (cx - lcx)**2 + (cy - lcy)**2
                if dist < best_dist and dist < 15000:
                    best_dist = dist
                    best_slot = slot
            
            if best_slot is None:
                best_slot = self._next_slot
                self._next_slot += 1
                
            current_slots.append(best_slot)
            used_slots.add(best_slot)
            new_boxes[best_slot] = (cx, cy)
            
        self._last_boxes = new_boxes
        for slot in list(self._history):
            if slot not in self._last_boxes:
                del self._history[slot]
                
        return current_slots

    # ── Private ───────────────────────────────────────────────────────────────

    def _predict_unique(self, grey, faces):
        p             = self.pipeline
        identity_list = sorted(set(p.labels))

        # Project all faces
        all_coords = []
        for (x, y, w, h) in faces:
            vec = preprocess(grey, face_box=(x, y, w, h))
            all_coords.append(p.project(vec) if vec is not None else None)

        # Cosine distance matrix built from k-NN votes
        nf, ni = len(faces), len(identity_list)
        D = np.full((nf, ni), np.inf)

        for i, coords in enumerate(all_coords):
            if coords is None: continue

            sims  = p.projections @ coords
            order = np.argsort(sims)[::-1]
            top_k = order[:p.knn]

            votes = {ident: 0.0 for ident in identity_list}
            for idx in top_k:
                lbl = p.labels[idx]
                votes[lbl] += float(sims[idx])

            total_weight = sum(max(v, 0) for v in votes.values()) or 1.0

            # Ratio test logic matching la_pipeline.knn_predict
            id_best = {}
            for idx in order:
                lbl = p.labels[idx]
                if lbl not in id_best:
                    id_best[lbl] = float(sims[idx])
                if len(id_best) == len(identity_list):
                    break

            id_sorted = sorted(id_best.values(), reverse=True)
            ratio_penalty = 1.0
            if len(id_sorted) >= 2:
                ratio = (1 - id_sorted[0]) / (1 - id_sorted[1] + 1e-6)
                if ratio > 0.92:
                    ratio_penalty = (1 - ratio)

            for j, ident in enumerate(identity_list):
                base_conf = max(votes[ident], 0.0) / total_weight
                conf = base_conf * ratio_penalty
                D[i, j] = 1.0 - conf

        # Greedy unique assignment
        labels  = ["Unknown"] * nf
        confs   = [0.0]       * nf
        coords_ = [np.zeros(p.k)] * nf
        used_f, used_i = set(), set()

        for _ in range(min(nf, ni)):
            Dm = D.copy()
            Dm[list(used_f), :] = np.inf
            Dm[:, list(used_i)] = np.inf
            if np.isinf(Dm).all(): break
            
            fi, ji = np.unravel_index(np.argmin(Dm), Dm.shape)
            if all_coords[fi] is not None:
                conf = 1.0 - Dm[fi, ji]
                labels[fi]  = identity_list[ji] if conf >= MIN_CONFIDENCE else "Unknown"
                confs[fi]   = conf
                coords_[fi] = all_coords[fi]
            
            used_f.add(fi); used_i.add(ji)

        return [(labels[i], confs[i], coords_[i]) for i in range(nf)]

    def _smooth(self, slot, label, conf):
        if slot not in self._history:
            self._history[slot] = deque(maxlen=WINDOW)
        self._history[slot].append((label, conf))
        history = self._history[slot]
        labels  = [h[0] for h in history]
        confs   = [h[1] for h in history]
        voted   = Counter(labels).most_common(1)[0][0]
        wc      = [c for l,c in zip(labels,confs) if l == voted]
        return voted, float(np.mean(wc)) if wc else 0.0
