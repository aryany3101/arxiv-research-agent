import argparse
import json
from collections import Counter
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


def top_paper_ids(chunk_ids, max_chunks_per_paper=None):
    paper_counts = Counter()
    selected = []
    for doc_id in chunk_ids:
        paper_id = chunks_by_id[doc_id]["arxiv_id"]
        if max_chunks_per_paper is not None and paper_counts[paper_id] >= max_chunks_per_paper:
            continue
        selected.append(paper_id)
        paper_counts[paper_id] += 1
        if len(selected) == 5:
            break
    return selected


def evaluate(eval_path, candidate_limit=10, max_chunks_per_paper=None, show_rankings=False):
    questions = json.loads(eval_path.read_text())
    if isinstance(questions, dict):
        questions = questions["questions"]
    if not questions:
        raise ValueError(f"No evaluation questions in {eval_path}")

    counts = {method: {1: 0, 5: 0, "full": 0} for method in METHODS}
    for item in questions:
        question = item["question"]
        expected = set(item["expected_arxiv_ids"])
        semantic = semantic_search(question, k=candidate_limit)
        fused = reciprocal_rank_fusion(semantic, keyword_search(question, k=candidate_limit))
        rerank_limit = len(fused) if max_chunks_per_paper is not None else 5
        reranked = rerank(question, fused, chunks_by_id, top_k=rerank_limit)
        rankings = {
            "Semantic only": top_paper_ids(semantic, max_chunks_per_paper),
            "Hybrid fusion": top_paper_ids((doc_id for doc_id, _ in fused), max_chunks_per_paper),
            "Hybrid + rerank": top_paper_ids((doc_id for doc_id, _ in reranked), max_chunks_per_paper),
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

        if show_rankings or not all(scores[1] and scores["full"] for scores in hits.values()):
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
    parser.add_argument("--candidate-limit", type=int, default=10, help="Chunks per search method before fusion")
    parser.add_argument("--max-chunks-per-paper", type=int, help="Limit results from each paper after ranking")
    parser.add_argument("--show-rankings", action="store_true", help="Print ranked paper IDs for every question")
    args = parser.parse_args()
    if args.candidate_limit < 5:
        parser.error("--candidate-limit must be at least 5")
    if args.max_chunks_per_paper is not None and args.max_chunks_per_paper < 1:
        parser.error("--max-chunks-per-paper must be positive")
    evaluate(args.questions, args.candidate_limit, args.max_chunks_per_paper, args.show_rankings)
