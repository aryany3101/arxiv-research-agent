# arXiv Research Agent

A multi-step RAG agent that answers research questions by retrieving and
synthesizing information across a corpus of arXiv papers — with hybrid
search, cross-encoder reranking, citation-backed answers, and a
self-critique loop that catches unsupported claims and gaps in coverage
before returning an answer.

Unlike a basic RAG chatbot, this agent:
- Breaks complex questions into focused sub-questions before retrieving
- Combines semantic (vector) search with keyword (BM25) search, then
  reranks candidates with a cross-encoder
- Critiques its own answer against the retrieved sources, and can loop
  back to fetch more evidence if the critique finds gaps
- Is explicitly instructed to admit when the source material doesn't
  fully cover part of a question, rather than filling gaps with guesses

## Demo

**Question:** How do video world models generalize across different robot embodiments?

**Excerpt from a saved run:**

> By scaling data across heterogeneous human and robotic agents, the models learn generalizable physical priors.
>
> To accommodate disparately-sized operational spaces across platforms, absolute end-effector actions are normalized into uniform, finite bounds [...] using platform-unique normalization factors.

Both statements cite [CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators](https://arxiv.org/abs/2608.27406v1). The [saved run](data/faithfulness_results.json) contains the complete answer and the agent’s critique.

## Architecture

```
                    ┌─────────────────┐
   User Question →  │ Query            │
                    │ Decomposition    │  (LLM breaks question into
                    └────────┬─────────┘   2-4 focused sub-questions)
                             │
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Retrieval │
                    │  Semantic (Chroma)│
                    │  + BM25 (keyword) │
                    │  → RRF fusion      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Cross-Encoder    │  (re-scores top candidates,
                    │ Reranking        │   filters low-relevance chunks)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer Synthesis │  (LLM writes cited answer from
                    │                  │   retrieved chunks only)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
              ┌────▶│ Self-Critique    │  (checks faithfulness +
              │     │                  │   coverage against sources)
              │     └────────┬─────────┘
              │              │
              │     needs_revision?
              │              │ yes
              └──────────────┘
                             │ pass
                             ▼
                       Final Answer
                    (with citations)
```

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Data source | arXiv API | Free, no auth, real messy PDFs |
| PDF extraction | PyMuPDF | Fast, clean text extraction |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Free, no API cost, runs on CPU |
| Vector store | Chroma | Simple, embedded, no separate server |
| Keyword search | BM25 (`rank_bm25`) | Catches exact term/acronym matches semantic search misses |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Jointly encodes query+doc for more accurate relevance scoring |
| Agent LLM | Gemini API | Decomposition, synthesis, critique |
| Interface | Streamlit | Fast to build, good for demoing agent transparency |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key-here"
```

## Usage

```bash
cd src

# 1. Fetch papers from arXiv
python3 fetch_arxiv.py

# 2. Extract text and chunk it
python3 extract_text.py

# 3. Embed chunks into the vector store
python3 embed_chunks.py

# 4. Run the agent directly
python3 agent.py

# 5. Or launch the web interface
python3 -m streamlit run app.py
```

## Evaluation

Evaluation was run on a hand-written set of 12 question/answer pairs
covering the paper corpus, with a mix of straightforward single-paper
questions and questions targeting specific named systems/acronyms.

### Retrieval Evaluation

The retrieval benchmark compares semantic-only search, BM25 + semantic fusion, and fusion with a cross-encoder reranker. Hit@1 and Hit@5 check whether any expected paper appears in the first one or five chunks. All@5 requires every expected paper in the first five chunks.

With the checked-in corpus of 15 papers, all three methods found the expected paper at rank one for all 12 original questions.

```bash
python src/embed_chunks.py
python src/eval_retrieval.py
```

#### Focused RAG benchmark

The [benchmark manifest](data/rag_benchmark.json) pins eight closely related arXiv papers and twelve questions, including four that require two sources. This benchmark downloads the papers to build its own index from scratch. The embedding and reranking models download on first use. Neither retrieval benchmark calls Gemini.

```bash
export ARXIV_DATA_DIR="$(mktemp -d)"
python src/prepare_rag_benchmark.py "$ARXIV_DATA_DIR"
python src/embed_chunks.py
python src/eval_retrieval.py --questions data/rag_benchmark.json
unset ARXIV_DATA_DIR
```

| Method | Hit@1 | Hit@5 | All@5 |
| --- | ---: | ---: | ---: |
| Semantic only | 10/12 | 10/12 | 9/12 |
| Hybrid fusion | 11/12 | 11/12 | 9/12 |
| Hybrid + rerank | 10/12 | 11/12 | 10/12 |

Reranking brought [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983v1) into the top five alongside [Self-RAG](https://arxiv.org/abs/2310.11511v1) for the question comparing their retrieval triggers. It also dropped a relevant first-place result to third for the ColBERT comparison. All three methods missed [ColBERTv2](https://arxiv.org/abs/2112.10752v2) when asked which successor uses residual compression. Their top five were all chunks from the original ColBERT paper for that question.

### Faithfulness Evaluation

Ran the full agent (decompose → retrieve → synthesize → critique) on 5
test questions, using the built-in self-critique step to judge whether
every claim in the answer was supported by retrieved source chunks.

| Metric | Result |
|---|---|
| Faithfulness pass rate | 5/5 (100%) |
| Revision loops triggered | 0/5 |

**Manual spot-check:** one answer's citation of specific model names
(DeepSeek-V3.2, Qwen3-Max, GLM-4.7, Kimi-k2) looked suspiciously precise
for a synthesized answer, so it was manually verified against the raw
chunk text using `grep`. All four names were confirmed present in the
cited source chunks — the agent was accurately reporting content, not
inventing it. This kind of manual verification is a useful complement to
automated faithfulness scoring, since an LLM-as-judge critique step can
in principle share blind spots with the LLM that wrote the answer.

## Known Limitations

- **Corpus size:** both retrieval suites use small corpora and hand-written questions. Their results describe these fixtures and need a larger, independently labeled corpus to generalize.
- **Self-critique blind spots:** the critique step uses the same class of
  model as the answer generator; a manual spot-check caught nothing wrong
  in this case, but the critique step alone shouldn't be treated as a
  fully independent verification method.
- **Free-tier API limits:** development was constrained by Gemini's free
  tier (20 requests/day), which shaped how much of the evaluation set
  could be run in a single session.

## Roadmap / Next Steps

- [ ] Scale the benchmark to 100–300 papers and broaden the question set
- [ ] Run full 12-question faithfulness eval (currently 5/12 run due to
      API quota constraints)
- [ ] Dockerize and deploy to HuggingFace Spaces / Railway
- [ ] Add citation links directly to arxiv.org/abs/{id} in the UI

## Project Structure

```
arxiv-research-agent/
├── data/
│   ├── raw_pdfs/            # downloaded papers
│   ├── processed/
│   │   └── chunks.jsonl     # chunked, metadata-tagged text
│   ├── metadata.json        # paper titles, authors, abstracts
│   ├── chroma_db/           # persisted vector store
│   ├── eval_questions.json  # hand-written eval set
│   └── faithfulness_results.json
├── src/
│   ├── fetch_arxiv.py       # Step 1: download papers
│   ├── extract_text.py      # Step 2: extract + chunk text
│   ├── embed_chunks.py      # Step 3: embed + store in Chroma
│   ├── retrieve.py          # Step 4: hybrid search + reranking
│   ├── agent.py             # Step 5: decompose/synthesize/critique loop
│   ├── eval_retrieval.py    # Step 6a: retrieval precision eval
│   ├── eval_faithfulness.py # Step 6b: answer faithfulness eval
│   └── app.py                # Step 7: Streamlit interface
├── requirements.txt
└── README.md
```