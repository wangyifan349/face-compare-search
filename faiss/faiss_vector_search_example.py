import faiss
import numpy as np

# Four simple two-dimensional vectors stored in the database.
vector_names = ["point_a", "point_b", "point_c", "point_d"]
database_vectors = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [8.0, 8.0],
        [9.0, 9.0],
    ],
    dtype="float32",
)

# The query vector is close to point_c and point_d.
query_vector = np.array([[8.5, 8.5]], dtype="float32")
top_k = 4

# IndexFlatL2 performs an exact search with squared L2 distance.
# A smaller distance means that two vectors are closer.
l2_index = faiss.IndexFlatL2(database_vectors.shape[1])
l2_index.add(database_vectors)
l2_distances, l2_indices = l2_index.search(query_vector, top_k)

print("=" * 60)
print("EXACT L2 SEARCH: SMALLER DISTANCE MEANS MORE SIMILAR")
print("=" * 60)

for rank, (distance, vector_index) in enumerate(zip(l2_distances[0], l2_indices[0]), start=1):
    name = vector_names[int(vector_index)]
    vector = database_vectors[int(vector_index)]
    print(f"{rank}. {name:<10} vector={vector} squared_l2={distance:.4f}")

# Copy the vectors because faiss.normalize_L2 changes the arrays in place.
normalized_database = database_vectors.copy()
normalized_query = query_vector.copy()

# After L2 normalization, inner product is equal to cosine similarity.
# A larger similarity score means that two vectors point in a closer direction.
faiss.normalize_L2(normalized_database)
faiss.normalize_L2(normalized_query)

cosine_index = faiss.IndexFlatIP(normalized_database.shape[1])
cosine_index.add(normalized_database)
cosine_scores, cosine_indices = cosine_index.search(normalized_query, top_k)

print("\n" + "=" * 60)
print("EXACT COSINE SEARCH: LARGER SCORE MEANS MORE SIMILAR")
print("=" * 60)

for rank, (score, vector_index) in enumerate(zip(cosine_scores[0], cosine_indices[0]), start=1):
    name = vector_names[int(vector_index)]
    vector = database_vectors[int(vector_index)]
    print(f"{rank}. {name:<10} vector={vector} cosine_similarity={score:.4f}")

print("\nThe index returns vector positions, so each position is mapped back to vector_names.")
