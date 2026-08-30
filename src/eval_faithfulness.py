import json
import sys
from pathlib import Path

sys.path.append(".")
from agent import run_agent

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_PATH = DATA_DIR / "eval_questions.json"
RESULTS_PATH = DATA_DIR / "faithfulness_results.json"


def run_faithfulness_eval(num_questions=None):
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    if num_questions:
        eval_set = eval_set[:num_questions]

    results = []
    pass_count = 0

    for item in eval_set:
        question = item["question"]
        print(f"\nRunning: {question}")

        result = run_agent(question)

        verdict = result["verdict"]
        is_pass = verdict == "pass"
        if is_pass:
            pass_count += 1

        results.append({
            "question": question,
            "verdict": verdict,
            "loops_used": result["loops_used"],
            "answer": result["answer"],
            "critique": result["critique"],
        })

        status = "✓ PASS" if is_pass else "✗ NEEDS_REVISION"
        print(f"{status} (loops used: {result['loops_used']})")

    total = len(eval_set)
    print(f"\n=== Faithfulness Summary ===")
    print(f"Passed: {pass_count}/{total} ({pass_count/total*100:.1f}%)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    run_faithfulness_eval(num_questions=5)  # start small since each question = multiple LLM calls