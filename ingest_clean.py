import sys
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Backend root path add
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.chunker import chunk_text

load_dotenv()

print("INGEST START HO GAYA 🚀")

# Load docs.json
with open("data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print("TOTAL DOCS:", len(docs))

chunks = []

# Create chunks
for doc in docs:
    text = doc.get("content", "")
    print("DOC TEXT:", text)
    chunks.extend(chunk_text(text))

print("TOTAL CHUNKS:", len(chunks))

# FREE LOCAL EMBEDDINGS
print("LOCAL EMBEDDINGS START (FREE MODE)")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

for i in range(len(embeddings)):
    print(f"Local Embedding {i+1}/{len(embeddings)} DONE")