import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Load API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
with open("data/docs.txt", "r") as f:
    documents = [doc.strip() for doc in f.readlines() if doc.strip()]

# Create embeddings
doc_embeddings = [model.encode(doc) for doc in documents]

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search function
def search(query, top_k=2):
    query_embedding = model.encode(query)

    scores = []
    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((documents[i], score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]

# Generate answer using Groq (FINAL)
def generate_answer(query, context):
    context_text = "\n".join(context)

    prompt = f"""
You are an AI assistant.

Answer ONLY using the context below.

Context:
{context_text}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ FINAL WORKING MODEL
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("⚠️ Groq API error:", e)
        return f"Fallback answer:\n{context_text}"

# Chat loop
if __name__ == "__main__":
    while True:
        query = input("\nAsk something (or type exit): ")

        if query.lower() == "exit":
            break

        context = search(query)
        answer = generate_answer(query, context)

        print("\n🤖 Answer:")
        print(answer)