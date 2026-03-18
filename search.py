import numpy as np
from embedder import get_embedding

# Load documents
with open("data/docs.txt", "r") as f:
    documents = f.readlines()

# Create embeddings
doc_embeddings = [get_embedding(doc) for doc in documents]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=2):
    query_embedding = get_embedding(query)

    scores = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((documents[i], score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[:top_k]

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = search(query)

    print("\nTop Results:")
    for res in results:
        print(res[0])