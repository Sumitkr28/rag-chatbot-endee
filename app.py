import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

# 🔐 Load API key from Streamlit secrets
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
with open("data/docs.txt", "r") as f:
    documents = [doc.strip() for doc in f.readlines() if doc.strip()]

doc_embeddings = [model.encode(doc) for doc in documents]

# Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query, top_k=2):
    query_embedding = model.encode(query)
    scores = []

    for i, doc_embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((documents[i], score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]

def generate_answer(query, context):
    context_text = "\n".join(context)

    prompt = f"""
Answer ONLY using the context below.

Context:
{context_text}

Question:
{query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}\n\nFallback:\n{context_text}"

# UI
st.title("🤖 RAG Chatbot")
query = st.text_input("Ask your question:")

if query:
    context = search(query)
    answer = generate_answer(query, context)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Context")
    for doc in context:
        st.write("- " + doc)