"""
Face Similarity Comparison Script
This script uses InsightFace's buffalo_l model to extract normalized face
embeddings from two images and calculate their cosine similarity.
Install dependencies:
    pip install opencv-python numpy insightface onnxruntime
For NVIDIA GPU acceleration, replace onnxruntime with onnxruntime-gpu:
    pip uninstall -y onnxruntime
    pip install onnxruntime-gpu
"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load the face recognition model using the CPU
face_analyzer = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
)
face_analyzer.prepare(ctx_id=-1)


def get_embedding(image_path):
    """Extract the normalized face embedding from an image."""
    image = cv2.imread(image_path)
    face = face_analyzer.get(image)[0]
    return face.normed_embedding

def get_similarity(image_path_1, image_path_2):
    """Compute the cosine similarity between two face images."""
    embedding_1 = get_embedding(image_path_1)
    embedding_2 = get_embedding(image_path_2)
    return float(np.dot(embedding_1, embedding_2))

similarity = get_similarity(
    "path/to/face1.jpg",
    "path/to/face2.jpg",
)

print(f"Similarity: {similarity:.4f}")
