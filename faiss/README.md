# 🔎 Face Vector Search Examples

This small project demonstrates face encoding and exact vector search with FAISS. It contains two face-search scripts with nearly identical search logic and one standalone FAISS example that does not use any face-recognition library.

The two face-search scripts use a simple linear teaching structure. The code follows the execution order directly instead of hiding the main steps behind multiple helper functions.

## 📚 Official Project Links

- [`face_recognition`](https://github.com/ageitgey/face_recognition): Python face-recognition library built on dlib.
- [`face_recognition` API documentation](https://face-recognition.readthedocs.io/en/latest/face_recognition.html): Documentation for `face_locations()`, `face_encodings()`, `model="cnn"`, and `model="large"`.
- [InsightFace](https://github.com/deepinsight/insightface): Face detection, alignment, recognition, and analysis toolkit.
- [InsightFace Python package](https://github.com/deepinsight/insightface/tree/master/python-package): Documentation for the `FaceAnalysis` Python interface.
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo): Information about InsightFace model packages.
- [FAISS](https://github.com/facebookresearch/faiss): High-performance vector similarity search and clustering library.
- [FAISS metric and distance documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances): Official explanation of L2 distance, inner product, and cosine similarity with normalized vectors.

## 📁 Project Structure

```text
face_vector_search_examples/
├── faces/
│   ├── alice.jpg
│   ├── bob.jpg
│   └── charlie.jpg
├── query.jpg
├── face_search_face_recognition.py
├── face_search_insightface.py
├── faiss_vector_search_example.py
└── README.md
```

Place reference face images inside the `faces/` directory. The filename without its extension is used as the displayed person name.

Example:

```text
faces/alice.jpg   -> displayed name: alice
faces/bob.jpg     -> displayed name: bob
```

Place the face that you want to search for in:

```text
query.jpg
```

Run all commands from the `face_vector_search_examples` directory.

## 🧠 Shared Search Process

Both face-search scripts use the same FAISS search process:

1. Read all supported images from the `faces/` directory.
2. Detect faces in each image.
3. Select the largest face when an image contains multiple faces.
4. Generate one embedding vector for each database image.
5. Convert all vectors to contiguous `float32` NumPy arrays.
6. Apply L2 normalization with `faiss.normalize_L2()`.
7. Build an exact FAISS index with `faiss.IndexFlatIP`.
8. Encode and normalize the face in `query.jpg`.
9. Search every indexed vector.
10. Print all results in descending cosine-similarity order.

`IndexFlatIP` is an exact search index. It does not use IVF, PQ, HNSW, or another approximate-search method. Every query is compared with every vector stored in the index.

## 📷 Version 1: face_recognition

Script:

```text
face_search_face_recognition.py
```

This version uses:

- `face_recognition.face_locations(..., model="cnn")` for CNN-based face detection.
- `face_recognition.face_encodings(..., model="large")` for the 68-point facial landmark model.
- The dlib face-recognition network for 128-dimensional face embeddings.
- `faiss.IndexFlatIP` for exact cosine-similarity search.

### Install

```bash
pip install face-recognition faiss-cpu numpy
```

The `face_recognition` package depends on dlib. On some systems, installing dlib may require CMake, a C++ compiler, and Python development tools.

### Run

```bash
python face_search_face_recognition.py
```

### Important Model Detail

`model="cnn"` selects the CNN face detector.

`model="large"` selects the 68-point facial landmark predictor used before encoding. It does not replace the 128-dimensional dlib face-embedding network with a different large embedding model.

The final face vector is still 128-dimensional.

The CNN detector can be slow when it runs only on a CPU.

## 🧬 Version 2: InsightFace

Script:

```text
face_search_insightface.py
```

This version uses:

- `FaceAnalysis(name="buffalo_l")` to load the InsightFace `buffalo_l` model package.
- `CPUExecutionProvider` for ONNX Runtime CPU inference.
- `face.embedding` to obtain the face-recognition vector.
- `faiss.IndexFlatIP` for exact cosine-similarity search.

### Install

```bash
pip install insightface onnxruntime faiss-cpu opencv-python numpy
```

For an NVIDIA GPU environment, the ONNX Runtime package selection may be different. The included script is configured for CPU inference.

### Run

```bash
python face_search_insightface.py
```

InsightFace may download the `buffalo_l` model files automatically during the first run. Model files are commonly stored under:

```text
~/.insightface/models/
```

### License Notice

The InsightFace source code is open source, but pretrained model licensing can be more restrictive. Official pretrained models are commonly provided for non-commercial research use. Confirm the model license before using the project commercially.

## 🧮 Standalone FAISS Example

Script:

```text
faiss_vector_search_example.py
```

This script uses only NumPy and `faiss-cpu`. It does not use a face encoder.

It demonstrates:

- Adding database vectors with `index.add()`.
- Searching vectors with `index.search()`.
- Exact squared L2-distance search with `IndexFlatL2`.
- Exact cosine-similarity search using normalized vectors and `IndexFlatIP`.
- Mapping FAISS vector indices back to readable names.
- Printing Top-K search results.

### Install

```bash
pip install faiss-cpu numpy
```

### Run

```bash
python faiss_vector_search_example.py
```

## 📊 Understanding the Search Scores

### Cosine Similarity

The two face-search scripts use:

```python
faiss.normalize_L2(database_vectors)
faiss.normalize_L2(query_vector)
index = faiss.IndexFlatIP(vector_dimension)
```

`IndexFlatIP` calculates inner products.

After both database vectors and query vectors are L2-normalized, the inner product is equal to cosine similarity.

A larger score means the vectors point in a more similar direction.

General interpretation:

```text
Higher score  -> more similar
Lower score   -> less similar
```

The score is not a probability.

A value such as `0.85` does not mean that the system is “85% certain” that the faces belong to the same person.

### L2 Distance

`IndexFlatL2` returns squared L2 distance.

It calculates:

```text
(x1 - y1)² + (x2 - y2)² + ... + (xn - yn)²
```

FAISS does not apply the final square root.

General interpretation:

```text
Smaller distance -> more similar
Larger distance  -> less similar
```

For L2-normalized vectors, squared L2 distance and cosine similarity satisfy:

```text
squared_L2 = 2 - 2 × cosine_similarity
```

Therefore, normalized L2 search and normalized cosine search usually produce the same nearest-neighbor ordering, although the returned score values have different meanings.

## 🔍 Why the Results Are Sorted Again

FAISS normally returns nearest-neighbor results in metric order:

- L2 distance: smallest first.
- Inner product: largest first.

The face-search scripts still create a result list and sort it explicitly in descending cosine-similarity order. This makes the final output logic easy to read and guarantees that the displayed list is ordered from the highest similarity score to the lowest.

## ⚠️ Important Notes

- Face similarity is not identity probability.
- Do not copy a fixed threshold from another project without testing it.
- Camera quality, pose, lighting, age, blur, occlusion, and image compression can change similarity scores.
- Build a validation dataset containing both same-person and different-person image pairs before selecting a threshold.
- Each reference image should preferably contain one clear and reasonably frontal face.
- When an image contains multiple faces, the scripts encode only the largest detected face.
- The `face_recognition` and InsightFace embeddings must not be mixed in one FAISS index.
- A 128-dimensional dlib embedding and an InsightFace embedding belong to different vector spaces and cannot be compared directly.
- These examples do not include liveness detection, anti-spoofing, encryption, database persistence, access control, or production API design.
- Face embeddings are sensitive biometric data. Real systems should use encryption, strict access controls, limited retention, audit logging, and applicable privacy-compliance procedures.

## ✅ Recommended Learning Order

1. Run `faiss_vector_search_example.py` first.
2. Study how `IndexFlatL2` returns squared L2 distance.
3. Study how normalization plus `IndexFlatIP` produces cosine similarity.
4. Run `face_search_face_recognition.py` and inspect the 128-dimensional search workflow.
5. Run `face_search_insightface.py` and compare its scores with the first encoder.
6. Test multiple images of the same person and different people.
7. Record the score distributions before choosing any recognition threshold.
