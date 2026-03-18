print("🔥 Script started")

try:
    from sentence_transformers import SentenceTransformer
    from endee import Endee

    print("⏳ Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    print("Initializing DB...")
    db = Endee("my_db")

    print("Loading documents...")
    with open("data/docs.txt", "r") as f:
        documents = [doc.strip() for doc in f.readlines() if doc.strip()]

    print("Documents loaded:", documents)

    print("Creating embeddings...")
    embeddings = [model.encode(doc) for doc in documents]

    print("Preparing data for DB...")
    data = []
    for i, emb in enumerate(embeddings):
        print(f"Preparing doc {i}")
        data.append({
            "id": str(i),
            "vector": emb.tolist(),
            "metadata": {"text": documents[i]}
        })

    print("Storing in DB...")
    
    # Try all possible methods safely
    if hasattr(db, "store"):
        db.store(data)
    elif hasattr(db, "upsert"):
        for item in data:
            db.upsert(**item)
    elif hasattr(db, "insert"):
        for item in data:
            db.insert(**item)
    else:
        print("❌ No valid method found in Endee. Available methods:")
        print(dir(db))
        raise Exception("Endee method not found")

    print("✅ Data stored in Endee DB successfully!")

except Exception as e:
    print("❌ ERROR:", e)