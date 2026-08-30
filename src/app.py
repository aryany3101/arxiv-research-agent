import streamlit as st
import sys
import json
from pathlib import Path

sys.path.append(".")
from agent import run_agent

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "answer_cache.json"

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)

st.set_page_config(page_title="arXiv Research Agent", layout="wide")

st.title("arXiv Research Agent")
st.write("Ask a question and get a cited answer synthesized from research papers.")

cache = load_cache()

with st.sidebar:
    st.subheader("Previously Asked")
    if cache:
        for past_question in cache.keys():
            if st.button(past_question, key=f"history_{past_question}"):
                question = past_question
    else:
        st.write("No questions asked yet.")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    if question in cache:
        st.info("✓ Loaded from cache — no API calls used")
        result = cache[question]
    else:
        with st.spinner("Thinking... (decomposing, retrieving, synthesizing, checking)"):
            result = run_agent(question)
        cache[question] = result
        save_cache(cache)
        st.success("New answer generated and cached for next time")

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Verdict:** {result['verdict']}")
    with col2:
        st.write(f"**Revision loops used:** {result['loops_used']}")

    with st.expander("Show sub-questions and retrieved sources"):
        for item in result["retrieved"]:
            st.write(f"**Sub-question:** {item['sub_question']}")
            for doc_id, score in item["chunks"]:
                st.write(f"- `{doc_id}` (relevance: {score:.2f})")

    with st.expander("Show self-critique"):
        st.text(result["critique"])