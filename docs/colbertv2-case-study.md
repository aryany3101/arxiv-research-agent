# Why the agent missed ColBERTv2

The question asked which successor to late-interaction passage search uses residual compression for document token embeddings. The expected paper was [ColBERTv2](https://arxiv.org/abs/2112.10752v2). Semantic search, hybrid fusion, and hybrid fusion with reranking each returned five chunks from [the original ColBERT paper](https://arxiv.org/abs/2004.12832v2).

## The experiment

I kept the eight-paper corpus, 12 questions, retrieval models, and five-chunk output limit fixed. I applied a cap of one chunk per paper after each method produced a ranking. The capped runs used either 10 or 30 candidates from each search method before fusion. Reranking scored the full fused candidate list before applying the cap.

| Configuration | Papers returned by hybrid + rerank, in order | ColBERTv2 in output |
| --- | --- | --- |
| Original, 10 candidates per search | ColBERT occupied all five slots | No |
| One chunk per paper, 10 candidates per search | ColBERT, RAG, DPR, Active RAG | No |
| One chunk per paper, 30 candidates per search | ColBERT, RAG, DPR, Self-RAG, Active RAG | No |

With 10 candidates, neither semantic search nor BM25 surfaced a ColBERTv2 chunk. At 30, BM25 brought in one at rank 21. It reached rank 31 after fusion and rank 40 after reranking. The original index contained 33 chunks from ColBERTv2. The failure started with candidate recall and persisted in ranking after the relevant paper entered the pool.

The cap improved full-source coverage for some other questions. **All@5** counts questions for which every expected paper appeared in the first five chunks.

| Configuration | Semantic All@5 | Hybrid fusion All@5 | Hybrid + rerank All@5 |
| --- | ---: | ---: | ---: |
| Original, 10 candidates | 9/12 | 9/12 | 10/12 |
| One chunk per paper, 10 candidates | 9/12 | 10/12 | 10/12 |
| One chunk per paper, 30 candidates | 10/12 | 10/12 | 10/12 |

My next experiment would give each paper's title and abstract their own retrieval entry before ranking its body chunks. With one query and a small hand-written suite, I'd treat that as a hypothesis to test on other closely related papers.

## Reproduce it

Use the setup in the [project README](../README.md#focused-rag-benchmark), then run this from the repository root. It uses local embedding and ranking models. No Gemini key is needed.

```sh
export ARXIV_DATA_DIR="$(mktemp -d)"
python src/prepare_rag_benchmark.py "$ARXIV_DATA_DIR"
python src/embed_chunks.py

python src/eval_retrieval.py --questions data/rag_benchmark.json --show-rankings
python src/eval_retrieval.py --questions data/rag_benchmark.json --show-rankings \
  --max-chunks-per-paper 1
python src/eval_retrieval.py --questions data/rag_benchmark.json --show-rankings \
  --max-chunks-per-paper 1 --candidate-limit 30
```

To trace where ColBERTv2 first enters the 30-candidate run:

```sh
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

from retrieve import chunks_by_id, keyword_search, reciprocal_rank_fusion, rerank, semantic_search

question = json.loads(Path("data/rag_benchmark.json").read_text())["questions"][3]["question"]
semantic = semantic_search(question, k=30)
keyword = keyword_search(question, k=30)
fused = reciprocal_rank_fusion(semantic, keyword)
reranked = rerank(question, fused, chunks_by_id, top_k=len(fused))

for label, ids in [
    ("Semantic", semantic),
    ("BM25", keyword),
    ("Fusion", [chunk_id for chunk_id, _ in fused]),
    ("Rerank", [chunk_id for chunk_id, _ in reranked]),
]:
    rank = next(
        (i for i, chunk_id in enumerate(ids, 1)
         if chunks_by_id[chunk_id]["arxiv_id"] == "2112.10752v2"),
        None,
    )
    print(label, rank)
PY
unset ARXIV_DATA_DIR
```
