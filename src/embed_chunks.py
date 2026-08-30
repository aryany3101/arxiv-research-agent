import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def load_chunks():
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_and_store():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("arxiv_papers")

    texts = [c["text"] for c in chunks]
    ids = [f"{c['arxiv_id']}_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {"arxiv_id": c["arxiv_id"], "title": c["title"], "url": c["url"]}
        for c in chunks
    ]

    print("Embedding chunks (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    embed_and_store()