import argparse
import json
from pathlib import Path

from retrieve import (
    chunks_by_id,
    keyword_search,
    reciprocal_rank_fusion,
    rerank,
    semantic_search,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_questions.json"
METHODS = ("Semantic only", "Hybrid fusion", "Hybrid + rerank")


def paper_ids(chunk_ids):
    return [chunks_by_id[doc_id]["arxiv_id"] for doc_id in chunk_ids]


def evaluate(eval_path):
    questions = json.loads(eval_path.read_text())
    if isinstance(questions, dict):
        questions = questions["questions"]
    if not questions:
        raise ValueError(f"No evaluation questions in {eval_path}")

    counts = {method: {1: 0, 5: 0, "full": 0} for method in METHODS}
    for item in questions:
        question = item["question"]
        expected = set(item["expected_arxiv_ids"])
        semantic = semantic_search(question, k=10)
        fused = reciprocal_rank_fusion(semantic, keyword_search(question, k=10))
        reranked = rerank(question, fused, chunks_by_id, top_k=5)
        rankings = {
            "Semantic only": paper_ids(semantic[:5]),
            "Hybrid fusion": paper_ids(doc_id for doc_id, _ in fused[:5]),
            "Hybrid + rerank": paper_ids(doc_id for doc_id, _ in reranked),
        }

        hits = {
            method: {
                1: bool(expected.intersection(ids[:1])),
                5: bool(expected.intersection(ids[:5])),
                "full": expected.issubset(ids[:5]),
            }
            for method, ids in rankings.items()
        }
        for method, scores in hits.items():
            for k, hit in scores.items():
                counts[method][k] += hit

        if not all(scores[1] and scores["full"] for scores in hits.values()):
            print(f"\n{question}\n  Expected: {', '.join(sorted(expected))}")
            for method, ids in rankings.items():
                print(f"  {method}: {', '.join(ids)}")

    total = len(questions)
    print(f"\n{'Method':<18} {'Hit@1':>8} {'Hit@5':>8} {'All@5':>8}")
    for method, scores in counts.items():
        fractions = [f"{scores[metric]}/{total}" for metric in (1, 5, "full")]
        print(f"{method:<18}" + "".join(f" {score:>8}" for score in fractions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare three retrieval configurations.")
    parser.add_argument("--questions", type=Path, default=EVAL_PATH)
    evaluate(parser.parse_args().questions)
