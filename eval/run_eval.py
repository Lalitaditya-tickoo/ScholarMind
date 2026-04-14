#!/usr/bin/env python3
"""
ScholarMind Evaluation Runner
Usage:
    python run_eval.py --dataset test_questions.json --api http://localhost:8000
    python run_eval.py --dataset test_questions.json --api http://localhost:8000 --output results.json --markdown RESULTS.md
"""

import argparse
import json
import sys
import time
import httpx


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["samples"]


def run_evaluation(api_url: str, samples: list[dict]) -> dict:
    payload = {
        "samples": [
            {
                "question": s["question"],
                "ground_truth": s["ground_truth"],
                "paper_ids": s.get("paper_ids", []),
            }
            for s in samples
        ]
    }

    print(f"\n{'='*60}")
    print(f" ScholarMind Evaluation - {len(samples)} samples")
    print(f"{'='*60}\n")
    print(f"Sending to {api_url}/evaluate ...")
    print("(This may take a few minutes - each sample runs 4 LLM judge calls)\n")

    start = time.time()
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(f"{api_url}/evaluate", json=payload)
        resp.raise_for_status()
        result = resp.json()
    elapsed = time.time() - start
    return result, elapsed


def print_report(result: dict, elapsed: float):
    print(f"\n{'='*60}")
    print(f" EVALUATION RESULTS")
    print(f"{'='*60}\n")
    print(f"  Samples evaluated:     {result['num_samples']}")
    print(f"  Total eval time:       {elapsed:.1f}s\n")
    print(f"  Faithfulness:          {result['avg_faithfulness']:.4f}")
    print(f"  Answer Relevance:      {result['avg_answer_relevance']:.4f}")
    print(f"  Context Precision:     {result['avg_context_precision']:.4f}")
    print(f"  Context Recall:        {result['avg_context_recall']:.4f}")
    print(f"  Avg Latency:           {result['avg_latency_ms']:.0f}ms")

    print(f"\n{'='*60}")
    print(f" PER-SAMPLE BREAKDOWN")
    print(f"{'='*60}\n")
    for i, r in enumerate(result["results"]):
        q_short = r["question"][:55] + "..." if len(r["question"]) > 55 else r["question"]
        print(f"  Q{i+1}: {q_short}")
        print(f"      Faith={r['faithfulness']:.2f}  Rel={r['answer_relevance']:.2f}  "
              f"Prec={r['context_precision']:.2f}  Rec={r['context_recall']:.2f}  "
              f"({r['latency_ms']:.0f}ms)")
        print()


def generate_markdown_report(result: dict, elapsed: float) -> str:
    lines = [
        "## Evaluation Results\n",
        f"Evaluated on **{result['num_samples']} test questions** using LLM-as-judge across 4 RAGAS-inspired metrics.\n",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Faithfulness | **{result['avg_faithfulness']:.4f}** |",
        f"| Answer Relevance | **{result['avg_answer_relevance']:.4f}** |",
        f"| Context Precision | **{result['avg_context_precision']:.4f}** |",
        f"| Context Recall | **{result['avg_context_recall']:.4f}** |",
        f"| Avg Latency | **{result['avg_latency_ms']:.0f}ms** |\n",
        "### Per-Question Scores\n",
        "| # | Question | Faith. | Relev. | Prec. | Recall | Latency |",
        "|---|----------|--------|--------|-------|--------|---------|",
    ]
    for i, r in enumerate(result["results"]):
        q_short = r["question"][:45] + "..." if len(r["question"]) > 45 else r["question"]
        lines.append(
            f"| {i+1} | {q_short} | {r['faithfulness']:.2f} | {r['answer_relevance']:.2f} | "
            f"{r['context_precision']:.2f} | {r['context_recall']:.2f} | {r['latency_ms']:.0f}ms |"
        )
    lines.append(f"\n*Evaluation completed in {elapsed:.1f}s using claude-sonnet as judge.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ScholarMind Evaluation Runner")
    parser.add_argument("--dataset", required=True, help="Path to test_questions.json")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default=None, help="Save JSON results")
    parser.add_argument("--markdown", default=None, help="Save Markdown report")
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    try:
        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{args.api}/health")
            health.raise_for_status()
            h = health.json()
            print(f"API healthy - {h['papers_indexed']} papers, {h['vector_store_documents']} vectors")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.api}/health - {e}")
        sys.exit(1)

    if h["papers_indexed"] == 0:
        print("\nWARNING: No papers uploaded! Upload papers first.")
        sys.exit(1)

    result, elapsed = run_evaluation(args.api, samples)
    print_report(result, elapsed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.markdown:
        md = generate_markdown_report(result, elapsed)
        with open(args.markdown, "w") as f:
            f.write(md)
        print(f"Markdown report saved to {args.markdown}")


if __name__ == "__main__":
    main()
