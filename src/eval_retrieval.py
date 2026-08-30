import json
import sys
from pathlib import Path

sys.path.append(".")
from retrieve import semantic_search, keyword_search, reciprocal_rank_fusion, rerank, chunks_by_id

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_questions.json"


def retrieve_top_papers(query, top_k=5):
    """Run the full hybrid + rerank pipeline and return the set of
    arxiv_ids present in the top_k results (not chunk ids)."""
    sem_ids = semantic_search(query, k=10)
    kw_ids = keyword_search(query, k=10)
    fused = reciprocal_rank_fusion(sem_ids, kw_ids)
    top_chunks = rerank(query, fused, chunks_by_id, top_k=top_k)

    arxiv_ids = []
    for doc_id, score in top_chunks:
        arxiv_id = chunks_by_id[doc_id]["arxiv_id"]
        if arxiv_id not in arxiv_ids:
            arxiv_ids.append(arxiv_id)
    return arxiv_ids

def retrieve_top_papers_naive(query, top_k=5):
    """Baseline: semantic search only, no BM25, no reranking."""
    sem_ids = semantic_search(query, k=top_k)
    arxiv_ids = []
    for doc_id in sem_ids:
        arxiv_id = chunks_by_id[doc_id]["arxiv_id"]
        if arxiv_id not in arxiv_ids:
            arxiv_ids.append(arxiv_id)
    return arxiv_ids


def evaluate():
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    total = len(eval_set)
    hybrid_hits = 0
    naive_hits = 0

    for item in eval_set:
        question = item["question"]
        expected = set(item["expected_arxiv_ids"])

        hybrid_retrieved = set(retrieve_top_papers(question, top_k=5))
        naive_retrieved = set(retrieve_top_papers_naive(question, top_k=5))

        hybrid_hit = len(expected & hybrid_retrieved) > 0
        naive_hit = len(expected & naive_retrieved) > 0

        if hybrid_hit:
            hybrid_hits += 1
        if naive_hit:
            naive_hits += 1

        if hybrid_hit != naive_hit:
            print(f"[DIFFERS] {question}")
            print(f"    naive: {naive_retrieved} ({'hit' if naive_hit else 'miss'})")
            print(f"    hybrid: {hybrid_retrieved} ({'hit' if hybrid_hit else 'miss'})\n")

    print(f"\n=== Summary ===")
    print(f"Naive (semantic-only) hit rate:  {naive_hits}/{total} ({naive_hits/total*100:.1f}%)")
    print(f"Hybrid+rerank hit rate:          {hybrid_hits}/{total} ({hybrid_hits/total*100:.1f}%)")


if __name__ == "__main__":
    evaluate()