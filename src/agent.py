import os
from google import genai

import sys
sys.path.append(".")
from retrieve import semantic_search, keyword_search, reciprocal_rank_fusion, rerank, chunks_by_id

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def decompose_query(question):
    prompt = f"""You are a research assistant. Break the following question into 2-4 focused sub-questions that, together, would help fully answer it. If the question is already simple and doesn't need breaking down, just return the original question as the only item.

Question: {question}

Respond with ONLY a numbered list, one sub-question per line. No preamble, no explanation."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    # Parse the numbered list into a clean Python list of strings
    lines = response.text.strip().split("\n")
    sub_questions = []
    for line in lines:
        line = line.strip()
        if line:
            # strip leading numbers like "1. " or "1)"
            cleaned = line.lstrip("0123456789.) ").strip()
            if cleaned:
                sub_questions.append(cleaned)
    return sub_questions

def retrieve_for_subquestions(sub_questions, top_k=3, min_score=0.0):
    all_retrieved = []
    for sub_q in sub_questions:
        sem_ids = semantic_search(sub_q, k=10)
        kw_ids = keyword_search(sub_q, k=10)
        fused = reciprocal_rank_fusion(sem_ids, kw_ids)
        top_chunks = rerank(sub_q, fused, chunks_by_id, top_k=top_k)
        # Filter out chunks the reranker actively considers irrelevant
        filtered_chunks = [(doc_id, score) for doc_id, score in top_chunks if score >= min_score]
        all_retrieved.append({"sub_question": sub_q, "chunks": filtered_chunks})
    return all_retrieved

def synthesize_answer(original_question, retrieved):
    # Build a context block from all retrieved chunks across all sub-questions
    context_parts = []
    for item in retrieved:
        for doc_id, score in item["chunks"]:
            chunk = chunks_by_id[doc_id]
            context_parts.append(
                f"[Source: {chunk['arxiv_id']}] {chunk['text']}"
            )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant answering questions using ONLY the provided source excerpts. 

Question: {original_question}

Source excerpts:
{context}

Instructions:
- Answer the question using only information from the excerpts above.
- After each claim, cite the source in brackets like [2608.27402v1].
- If the excerpts don't fully cover some part of the question, say so explicitly rather than guessing.
- Do not use any outside knowledge beyond what's in the excerpts.

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text

def critique_answer(original_question, answer, retrieved):
    context_parts = []
    for item in retrieved:
        for doc_id, score in item["chunks"]:
            chunk = chunks_by_id[doc_id]
            context_parts.append(f"[Source: {chunk['arxiv_id']}] {chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""You are a strict fact-checker reviewing an AI-generated answer against its source material.

Original question: {original_question}

Source excerpts:
{context}

Generated answer:
{answer}

Review the answer and respond in this exact format:
FAITHFUL: yes or no
UNSUPPORTED_CLAIMS: list any claims not backed by the excerpts, or "none"
MISSING: does the answer fail to address any part of the original question? If so, what?
VERDICT: pass or needs_revision"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text

def parse_verdict(critique_text):
    for line in critique_text.split("\n"):
        if line.strip().upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().lower()
            return verdict
    return "pass"  # default to pass if parsing fails, rather than looping forever

def generate_followup_query(original_question, answer, critique_text):
    prompt = f"""The following answer to a question was reviewed and found to need revision.

Original question: {original_question}

Answer given:
{answer}

Critique:
{critique_text}

Based on the critique, write ONE focused search query that would help retrieve the missing or better information needed to fix this answer. Respond with ONLY the query text, nothing else."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text.strip()

def run_agent(question, max_loops=2):
    sub_qs = decompose_query(question)
    retrieved = retrieve_for_subquestions(sub_qs)
    answer = synthesize_answer(question, retrieved)
    critique = critique_answer(question, answer, retrieved)
    verdict = parse_verdict(critique)

    loop_count = 0
    while verdict == "needs_revision" and loop_count < max_loops:
        print(f"\n[Revision needed — running follow-up retrieval, attempt {loop_count + 1}]")
        followup_query = generate_followup_query(question, answer, critique)
        print(f"Follow-up query: {followup_query}")

        # Retrieve more chunks targeting the gap, and add them to what we have
        extra_retrieved = retrieve_for_subquestions([followup_query])
        retrieved = retrieved + extra_retrieved

        # Re-synthesize and re-critique with the expanded context
        answer = synthesize_answer(question, retrieved)
        critique = critique_answer(question, answer, retrieved)
        verdict = parse_verdict(critique)
        loop_count += 1

    return {
        "question": question,
        "sub_questions": sub_qs,
        "retrieved": retrieved,
        "answer": answer,
        "critique": critique,
        "verdict": verdict,
        "loops_used": loop_count,
    }


if __name__ == "__main__":
    question = "How do language models represent moral concepts, and what makes this different from how they represent other types of knowledge?"

    result = run_agent(question)

    print(f"Question: {result['question']}\n")
    print(f"Final Answer:\n{result['answer']}\n")
    print(f"Final Verdict: {result['verdict']} (used {result['loops_used']} revision loop(s))")