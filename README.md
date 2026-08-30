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

**Question:** *"What are LLMs and evals?"*

The agent correctly identified and cited two different source papers for
different parts of the compound question, and pulled specific model names
mentioned in those papers' related-work sections — verified against the
raw source chunks with `grep` to confirm no hallucination (see
[Evaluation](#evaluation) below).

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

| Method | Hit Rate (paper found in top-5) |
|---|---|
| Naive (semantic search only) | 12/12 (100%) |
| Hybrid (BM25 + semantic + rerank) | 12/12 (100%) |

**Honest finding:** at the current corpus size (15 papers on largely
non-overlapping topics), there isn't enough topical overlap between
papers to create real retrieval ambiguity — even naive semantic search
finds the right paper every time. This means the current eval set doesn't
yet demonstrate hybrid search's advantage, which typically shows up when
multiple similar-but-distinct papers compete for relevance. **Next step:**
scale the corpus to 100-300 papers within the same narrow subfields, which
should surface real cases where hybrid search and reranking outperform
naive retrieval — documented here as a known limitation rather than
glossed over.

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

- **Corpus size:** at 15 papers, the retrieval evaluation can't yet
  distinguish hybrid search from naive semantic search — see above.
- **Self-critique blind spots:** the critique step uses the same class of
  model as the answer generator; a manual spot-check caught nothing wrong
  in this case, but the critique step alone shouldn't be treated as a
  fully independent verification method.
- **Free-tier API limits:** development was constrained by Gemini's free
  tier (20 requests/day), which shaped how much of the evaluation set
  could be run in a single session.

## Roadmap / Next Steps

- [ ] Scale corpus to 100-300 papers to properly stress-test hybrid
      retrieval's advantage over naive search
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