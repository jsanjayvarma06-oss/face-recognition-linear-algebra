import os
import numpy as np
from la_pipeline import LinearAlgebraPipeline
from recognizer import FaceRecognizer

p = LinearAlgebraPipeline(n_components=50, n_neighbours=5)
p.train("faces_db")

correct = 0
total = 0
for ident in os.listdir("faces_db"):
    files = os.listdir(os.path.join("faces_db", ident))
    for f in files:
        vec = np.load(os.path.join("faces_db", ident, f))
        coords = p.project(vec)
        pred, conf = p.knn_predict(coords)
        if pred == ident:
            correct += 1
        total += 1

print(f"Accuracy with knn_predict: {correct}/{total} ({correct/total:.2%})")
