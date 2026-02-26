import sys
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Backend root path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.chunker import chunk_text

print("QUERY START 🚀")

# Load local embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load docs
with open("data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Create chunks
chunks = []
for doc in docs:
    text = doc.get("content", "")
    chunks.extend(chunk_text(text))

print("TOTAL CHUNKS:", len(chunks))

# Create embeddings
chunk_embeddings = model.encode(chunks)

# Ask question
query = input("Ask Question: ")

query_embedding = model.encode([query])[0]

# Cosine similarity
scores = np.dot(chunk_embeddings, query_embedding)

best_index = np.argmax(scores)

print("\n🔥 BEST MATCHED CONTEXT:\n")
print(chunks[best_index])