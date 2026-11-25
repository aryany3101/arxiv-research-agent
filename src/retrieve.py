import json
import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = Path(os.environ.get("ARXIV_DATA_DIR", DEFAULT_DATA_DIR))
CHROMA_DIR = DATA_DIR / "chroma_db"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection("arxiv_papers")

# --- Load chunks once and build a BM25 index over them ---
def load_chunks():
    chunks = []
    with open(CHUNKS_PATH) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

all_chunks = load_chunks()
tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Lookup dict: chunk_id -> full chunk record (needed for text + title in rerank/printing)
chunks_by_id = {
    f"{c['arxiv_id']}_{c['chunk_index']}": c for c in all_chunks
}


def semantic_search(query, k=10):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    ids = results["ids"][0]
    return ids


def keyword_search(query, k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    ids = [f"{all_chunks[i]['arxiv_id']}_{all_chunks[i]['chunk_index']}" for i in top_indices]
    return ids


def reciprocal_rank_fusion(sem_ids, kw_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(sem_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(kw_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def rerank(query, fused_ids, all_chunks_by_id, top_k=5):
    # Build (query, chunk_text) pairs for every candidate
    pairs = [(query, all_chunks_by_id[doc_id]["text"]) for doc_id, _ in fused_ids]
    scores = reranker.predict(pairs)

    # Attach scores and sort by them, descending
    reranked = sorted(
        zip([doc_id for doc_id, _ in fused_ids], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return reranked[:top_k]


if __name__ == "__main__":
    query = "how do language models represent moral concepts"

    print("=== Semantic search results ===")
    sem_ids = semantic_search(query)
    print(sem_ids)

    print("\n=== Keyword (BM25) search results ===")
    kw_ids = keyword_search(query)
    print(kw_ids)

    print("\n=== Fused (hybrid) results ===")
    fused_results = reciprocal_rank_fusion(sem_ids, kw_ids)
    for doc_id, score in fused_results[:10]:
        print(f"{doc_id}  (score: {score:.4f})")

    print("\n=== Reranked results ===")
    reranked_results = rerank(query, fused_results, chunks_by_id)
    for doc_id, score in reranked_results:
        title = chunks_by_id[doc_id]["title"]
        print(f"{doc_id}  (score: {score:.4f})  — {title}")