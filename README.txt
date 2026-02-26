RAG Assistant Assignment

Steps to run:

1. Create virtual env
   python -m venv venv

2. Activate
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Run ingest
   python scripts/ingest_clean.py

5. Run query
   python scripts/query.py

This project uses FREE local embeddings
(sentence-transformers all-MiniLM-L6-v2).