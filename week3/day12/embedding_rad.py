import os
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# It has total featured of 384 dimensions.
model = SentenceTransformer("all-MiniLM-L6-v2")
text = "Machine learning is fun."


# embedding = model.encode(text)

# print(embedding.shape)
# print(embedding)


t1 = "There are 24 paid leaves"
t2 = "there are 24 holidays allocated"

v1 = model.encode(t1)
v2 = model.encode(t2)
print(cosine_similarity(v1, v2))
