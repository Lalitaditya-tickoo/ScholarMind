"""
RAGAS-inspired evaluation pipeline for ScholarMind.
Measures: Faithfulness, Answer Relevance, Context Precision, Context Recall.
No external RAGAS dependency — custom implementation for full control.
"""

import json
import time
import logging
import httpx
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EVAL_MODEL = os.getenv("EVAL_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


@dataclass
class EvalSample:
    """Single evaluation sample."""
    question: str
    ground_truth: str
    paper_ids: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""
    question: str
    generated_answer: str
    ground_truth: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    latency_ms: float
    num_sources: int


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    num_samples: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_context_recall: float
    avg_latency_ms: float
    results: list[EvalResult]

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_answer_relevance": round(self.avg_answer_relevance, 4),
            "avg_context_precision": round(self.avg_context_precision, 4),
            "avg_context_recall": round(self.avg_context_recall, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "results": [asdict(r) for r in self.results],
        }


class ScholarMindEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-as-judge approach.

    Metrics:
    - Faithfulness: Are claims in the answer grounded in retrieved context?
    - Answer Relevance: Does the answer actually address the question?
    - Context Precision: Are the retrieved chunks relevant to the question?
    - Context Recall: Does the retrieved context cover key info from ground truth?
    """

    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = EVAL_MODEL

    async def _llm_judge(self, system: str, prompt: str) -> str:
        """Call Claude as an evaluation judge."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{ANTHROPIC_BASE_URL}/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("content", [])
        return "".join(b["text"] for b in content if b.get("type") == "text")

    def _extract_score(self, text: str) -> float:
        """Extract a 0-1 score from LLM judge response."""
        patterns = [
            r'[Ss]core[:\s]+([01]\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*1',
            r'(\d+)%',
            r'\b(0\.\d+|1\.0|1)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                if val > 1:
                    val = val / 100.0
                return min(max(val, 0.0), 1.0)
        logger.warning(f"Could not extract score from: {text[:100]}")
        return 0.5

    async def eval_faithfulness(self, answer: str, context_chunks: list[str]) -> float:
        context = "\\n\\n---\\n\\n".join(context_chunks)
        system = """You are an evaluation judge. Your task is to assess whether
an AI-generated answer is faithfully grounded in the provided source context.
Rate ONLY on whether claims are supported by the sources — not on quality or completeness."""

        prompt = f"""Given the following source context and generated answer, evaluate faithfulness.

**Source Context:**
{context}

**Generated Answer:**
{answer}

Instructions:
1. Identify each factual claim in the answer.
2. Check if each claim is supported by the source context.
3. Calculate: (number of supported claims) / (total claims).

Respond with ONLY a JSON object:
{{"supported_claims": <int>, "total_claims": <int>, "score": <float 0-1>, "unsupported": ["list of unsupported claims if any"]}}"""

        response = await self._llm_judge(system, prompt)
        try:
            cleaned = response.strip().strip("`").replace("json\\n", "").strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            return self._extract_score(response)

    async def eval_answer_relevance(self, question: str, answer: str) -> float:
        system = """You are an evaluation judge. Assess whether an answer
is relevant to and addresses the given question. Focus on topical relevance,
not factual correctness."""

        prompt = f"""**Question:** {question}

**Answer:** {answer}

Rate how relevant the answer is to the question on a scale of 0 to 1:
- 1.0: Directly and completely addresses the question
- 0.7-0.9: Mostly addresses the question with minor tangents
- 0.4-0.6: Partially addresses the question
- 0.1-0.3: Barely related to the question
- 0.0: Completely irrelevant

Respond with ONLY: Score: <float>"""

        response = await self._llm_judge(system, prompt)
        return self._extract_score(response)

    async def eval_context_precision(self, question: str, context_chunks: list[str]) -> float:
        system = """You are an evaluation judge. For each retrieved text chunk,
determine if it is relevant to answering the given question."""

        chunk_list = "\\n".join(
            f"[Chunk {i+1}]: {chunk[:500]}" for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""**Question:** {question}

**Retrieved Chunks:**
{chunk_list}

For each chunk, mark it as RELEVANT or IRRELEVANT to answering the question.
Then calculate precision = relevant_chunks / total_chunks.

Respond with ONLY a JSON object:
{{"relevant_chunks": <int>, "total_chunks": {len(context_chunks)}, "score": <float 0-1>}}"""

        response = await self._llm_judge(system, prompt)
        try:
            cleaned = response.strip().strip("`").replace("json\\n", "").strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            return self._extract_score(response)

    async def eval_context_recall(self, ground_truth: str, context_chunks: list[str]) -> float:
        context = "\\n\\n---\\n\\n".join(context_chunks)
        system = """You are an evaluation judge. Assess whether the retrieved
context contains enough information to support a known ground-truth answer."""

        prompt = f"""**Ground Truth Answer:** {ground_truth}

**Retrieved Context:**
{context}

Identify the key claims/facts in the ground truth answer.
Check how many of these are covered by the retrieved context.
Calculate: (covered facts) / (total facts in ground truth).

Respond with ONLY a JSON object:
{{"covered_facts": <int>, "total_facts": <int>, "score": <float 0-1>}}"""

        response = await self._llm_judge(system, prompt)
        try:
            cleaned = response.strip().strip("`").replace("json\\n", "").strip()
            data = json.loads(cleaned)
            return float(data.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            return self._extract_score(response)

    async def evaluate_sample(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
        context_chunks: list[str],
        latency_ms: float,
    ) -> EvalResult:
        faithfulness = await self.eval_faithfulness(generated_answer, context_chunks)
        relevance = await self.eval_answer_relevance(question, generated_answer)
        precision = await self.eval_context_precision(question, context_chunks)
        recall = await self.eval_context_recall(ground_truth, context_chunks)

        return EvalResult(
            question=question,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(relevance, 4),
            context_precision=round(precision, 4),
            context_recall=round(recall, 4),
            latency_ms=round(latency_ms, 1),
            num_sources=len(context_chunks),
        )

    def aggregate_results(self, results: list[EvalResult]) -> EvalReport:
        n = len(results)
        if n == 0:
            return EvalReport(
                num_samples=0,
                avg_faithfulness=0, avg_answer_relevance=0,
                avg_context_precision=0, avg_context_recall=0,
                avg_latency_ms=0, results=[],
            )

        return EvalReport(
            num_samples=n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n,
            avg_answer_relevance=sum(r.answer_relevance for r in results) / n,
            avg_context_precision=sum(r.context_precision for r in results) / n,
            avg_context_recall=sum(r.context_recall for r in results) / n,
            avg_latency_ms=sum(r.latency_ms for r in results) / n,
            results=results,
        )
