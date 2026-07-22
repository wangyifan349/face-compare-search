"""
Interactive Face Search

This script performs a 1:N face similarity search using InsightFace.

At startup, enter the directory containing the reference face images.
The script scans that directory and all its subdirectories with os.walk(),
extracts the face embeddings, and stores them in memory.
After the face database is loaded, you can repeatedly enter query image
paths. Each query face is compared with all reference faces, and the
results are displayed from highest to lowest cosine similarity.
Enter "exit" or "quit" to close the program.
Install dependencies:
    pip install opencv-python numpy insightface onnxruntime
"""
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
# Image formats included when scanning the reference directory
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Initialize the InsightFace model
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
# ctx_id=-1 selects CPU inference
face_analyzer.prepare(ctx_id=-1)
def get_face_embedding(image_path):
    """Return the normalized embedding of the first detected face."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    detected_faces = face_analyzer.get(image)
    if not detected_faces:
        raise ValueError(f"No face detected: {image_path}")
    return detected_faces[0].normed_embedding
# Ask for the reference directory once
while True:
    reference_directory = input(
        "Enter the reference image directory: "
    ).strip().strip('"')
    if os.path.isdir(reference_directory):
        break
    print("Invalid directory. Please try again.")
# Scan the directory and build the face database
face_database = []
print("\nBuilding face database...")
for current_directory, _, filenames in os.walk(reference_directory):
    for filename in filenames:
        extension = os.path.splitext(filename)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        image_path = os.path.join(current_directory, filename)
        try:
            embedding = get_face_embedding(image_path)
            face_database.append((image_path, embedding))
            print(f"Loaded: {image_path}")
        except ValueError as error:
            print(f"Skipped: {error}")
if not face_database:
    print("\nNo valid face images were found.")
    raise SystemExit(1)
print(f"\nFace database ready: {len(face_database)} image(s)")
print('Enter "exit" or "quit" to stop the program.')
# Continuously accept query images
while True:
    query_image_path = input(
        "\nEnter a query image path: "
    ).strip().strip('"')
    if query_image_path.lower() in {"exit", "quit"}:
        print("Program closed.")
        break
    if not os.path.isfile(query_image_path):
        print("Invalid image path.")
        continue
    try:
        query_embedding = get_face_embedding(query_image_path)
    except ValueError as error:
        print(f"Error: {error}")
        continue
    search_results = []
    # Compare the query face with every face in the database
    for reference_image_path, reference_embedding in face_database:
        # Both embeddings are normalized, so their dot product is
        # equivalent to cosine similarity
        similarity = float(
            np.dot(query_embedding, reference_embedding)
        )
        search_results.append(
            (reference_image_path, similarity)
        )
    # Sort from the highest similarity to the lowest
    search_results.sort(
        key=lambda result: result[1],
        reverse=True,
    )
    print("\nSearch results:")
    for rank, (image_path, similarity) in enumerate(
        search_results,
        start=1,
    ):
        print(
            f"{rank:03d}. "
            f"{similarity:.4f} | "
            f"{image_path}"
        )
