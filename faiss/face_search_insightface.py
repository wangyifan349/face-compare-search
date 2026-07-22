"""
Directory layout:

face_vector_search_examples/
├── faces/
│   ├── alice.jpg
│   ├── bob.jpg
│   └── charlie.jpg
├── query.jpg
└── face_search_insightface.py

Place reference face images inside the faces directory.
The filename without its extension is displayed as the person name.
Place the face to search for in query.jpg.
Run this script from the face_vector_search_examples directory:

python face_search_insightface.py
"""

from pathlib import Path

import cv2
import faiss
import numpy as np
from insightface.app import FaceAnalysis

database_directory = Path("faces")
query_image_path = Path("query.jpg")
supported_extensions = {".jpg", ".jpeg", ".png"}
detection_size = (640, 640)

if not database_directory.exists():
    raise FileNotFoundError(f"Database directory not found: {database_directory}")

if not query_image_path.exists():
    raise FileNotFoundError(f"Query image not found: {query_image_path}")

# Load the InsightFace buffalo_l model package with CPU inference.
face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_analyzer.prepare(ctx_id=0, det_size=detection_size)

face_names = []
face_vectors = []
database_image_paths = sorted(
    image_path
    for image_path in database_directory.iterdir()
    if image_path.is_file() and image_path.suffix.lower() in supported_extensions
)

# Encode the largest face found in each database image.
for image_path in database_image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipped {image_path.name}: unable to read image")
        continue

    try:
        detected_faces = face_analyzer.get(image)
    except Exception as error:
        print(f"Skipped {image_path.name}: {error}")
        continue

    if not detected_faces:
        print(f"Skipped {image_path.name}: no face detected")
        continue

    largest_face = max(
        detected_faces,
        key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
    )
    face_names.append(image_path.stem)
    face_vectors.append(np.asarray(largest_face.embedding, dtype="float32"))
    print(f"Loaded: {image_path.name}")

if not face_vectors:
    raise RuntimeError("No valid face vectors were loaded from the faces directory.")

# Convert the database vectors to a contiguous float32 matrix and normalize them.
database_vectors = np.ascontiguousarray(np.vstack(face_vectors), dtype="float32")
faiss.normalize_L2(database_vectors)

# IndexFlatIP performs exact inner-product search.
# For L2-normalized vectors, inner product equals cosine similarity.
vector_dimension = database_vectors.shape[1]
index = faiss.IndexFlatIP(vector_dimension)
index.add(database_vectors)

# Detect and encode the largest face in query.jpg.
query_image = cv2.imread(str(query_image_path))

if query_image is None:
    raise ValueError(f"Unable to read query image: {query_image_path}")

query_faces = face_analyzer.get(query_image)

if not query_faces:
    raise RuntimeError(f"No face detected in query image: {query_image_path}")

largest_query_face = max(
    query_faces,
    key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
)
query_vector = np.ascontiguousarray([largest_query_face.embedding], dtype="float32")
faiss.normalize_L2(query_vector)

# Search every indexed vector and sort the final results from highest to lowest score.
similarity_scores, vector_indices = index.search(query_vector, index.ntotal)
search_results = [
    (face_names[int(vector_index)], float(similarity_score))
    for similarity_score, vector_index in zip(similarity_scores[0], vector_indices[0])
    if vector_index >= 0
]
search_results.sort(key=lambda result: result[1], reverse=True)

print("\n" + "=" * 70)
print("INSIGHTFACE VECTOR SEARCH RESULTS")
print("=" * 70)
print(f"{'Rank':<8}{'Name':<32}{'Cosine similarity':>20}")
print("-" * 70)

for rank, (face_name, similarity_score) in enumerate(search_results, start=1):
    print(f"{rank:<8}{face_name:<32}{similarity_score:>20.6f}")

print("-" * 70)
print(f"Best match: {search_results[0][0]}")
print(f"Best cosine similarity: {search_results[0][1]:.6f}")
print(f"Indexed vectors: {index.ntotal}")
print(f"Vector dimension: {vector_dimension}")
