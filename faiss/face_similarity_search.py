"""
1:N Face Similarity Comparison
This script compares one query face image with all images in a target
directory and its subdirectories. Results are sorted by similarity
in descending order.
Install dependencies:
    pip install opencv-python numpy insightface onnxruntime
"""
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Load the InsightFace face recognition model
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
# Use CPU for model inference
face_analyzer.prepare(ctx_id=-1)

def get_embedding(image_path):
    """Extract the normalized face embedding from an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    faces = face_analyzer.get(image)
    if not faces:
        raise ValueError(f"No face detected: {image_path}")
    # Use the first detected face
    return faces[0].normed_embedding

def collect_image_paths(directory_path):
    """Recursively collect supported image files using os.walk."""
    image_paths = []
    for root_directory, _, file_names in os.walk(directory_path):
        for file_name in file_names:
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension in SUPPORTED_EXTENSIONS:
                image_paths.append(
                    os.path.join(root_directory, file_name)
                )
    return image_paths


def compare_with_directory(query_image_path, target_directory):
    """Compare one face image with all images in a directory."""
    query_embedding = get_embedding(query_image_path)
    comparison_results = []
    # Recursively collect target images
    target_image_paths = collect_image_paths(target_directory)
    for target_image_path in target_image_paths:
        try:
            target_embedding = get_embedding(target_image_path)
            # The embeddings are normalized, so the dot product equals
            # cosine similarity
            similarity = float(
                np.dot(query_embedding, target_embedding)
            )
            comparison_results.append(
                (target_image_path, similarity)
            )
        except ValueError as error:
            print(f"Skipped: {error}")
    # Sort results from highest to lowest similarity
    comparison_results.sort(
        key=lambda result: result[1],
        reverse=True,
    )
    return comparison_results

#-----------------------------------
# Compare one query image with all images in the target directory
results = compare_with_directory(
    query_image_path="path/to/query.jpg",
    target_directory="path/to/faces",
)
#----------------------------------
# Print ranked results
for rank, (image_path, similarity) in enumerate(results, start=1):
    print(f"{rank:03d}. {image_path}: {similarity:.4f}")
