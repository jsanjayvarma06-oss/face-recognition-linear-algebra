import os
import numpy as np
from la_pipeline import LinearAlgebraPipeline
from recognizer import FaceRecognizer

p = LinearAlgebraPipeline(n_components=50, n_neighbours=5)
p.train("faces_db")

identities = sorted(set(p.labels))
id_mean = {}
for ident in identities:
    idxs = [i for i,l in enumerate(p.labels) if l == ident]
    m    = np.mean(p.projections[idxs], axis=0)
    n    = np.linalg.norm(m)
    id_mean[ident] = m / n if n > 1e-10 else m

centroid_correct = 0
knn_correct = 0
agreements = 0
total = 0

for ident in os.listdir("faces_db"):
    files = os.listdir(os.path.join("faces_db", ident))
    for f in files:
        vec = np.load(os.path.join("faces_db", ident, f))
        coords = p.project(vec)
        
        # knn predict
        lbl_knn, _ = p.knn_predict(coords)
        
        # centroid predict
        dists = [1.0 - float(np.dot(coords, id_mean[i])) for i in identities]
        lbl_cent = identities[np.argmin(dists)]
        
        if lbl_knn == ident: knn_correct += 1
        if lbl_cent == ident: centroid_correct += 1
        if lbl_knn == lbl_cent: agreements += 1
        total += 1

print(f"KNN Correct: {knn_correct}/{total}")
print(f"Centroid Correct: {centroid_correct}/{total}")
print(f"Agreements: {agreements}/{total}")
