"""
Cross-paper synthesis: query across multiple papers, find agreements,
contradictions, and research gaps.
"""

import os
import logging
import httpx
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


@dataclass
class PaperPosition:
    paper_id: str
    paper_title: str
    position: str
    evidence: str
    page_numbers: list[int] = field(default_factory=list)


@dataclass
class SynthesisResult:
    query: str
    consensus: str
    contradictions: list[dict]
    gaps: list[str]
    positions: list[PaperPosition]
    paper_count: int
    synthesis_summary: str


class CrossPaperSynthesizer:
    """
    Analyzes a question across multiple papers to find:
    - Consensus: where papers agree
    - Contradictions: where papers disagree
    - Gaps: aspects of the question no paper covers
    """

    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = CLAUDE_MODEL

    async def _call_claude(self, system: str, user_msg: str, max_tokens: int = 3000) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{ANTHROPIC_BASE_URL}/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )

    async def synthesize(
        self,
        question: str,
        paper_chunks: dict[str, list[dict]],
    ) -> SynthesisResult:
        paper_sections = []
        for paper_id, chunks in paper_chunks.items():
            if not chunks:
                continue
            title = chunks[0].get("paper_title", paper_id)
            text_parts = []
            for c in chunks:
                pg = c.get("page_number", "?")
                text_parts.append(f"  [p.{pg}] {c['text']}")
            combined = "\n".join(text_parts)
            paper_sections.append(
                f"=== PAPER: {title} (ID: {paper_id}) ===\n{combined}"
            )

        all_context = "\n\n".join(paper_sections)

        system = """You are ScholarMind's cross-paper synthesis engine. Your job is to
analyze how multiple research papers relate to each other on a given topic.

You must identify:
1. CONSENSUS - findings that multiple papers agree on
2. CONTRADICTIONS - where papers disagree
3. GAPS - aspects no paper addresses
4. PER-PAPER POSITIONS - each paper's stance with evidence

Be precise. Cite specific papers by title. Only report what is in the provided text."""

        user_msg = f"""Analyze the following research question across multiple papers:

**Research Question:** {question}

**Papers and their relevant excerpts:**
{all_context}

Respond with ONLY a JSON object in this exact format:
{{
  "consensus": "What the papers collectively agree on",
  "contradictions": [
    {{
      "topic": "The specific point of disagreement",
      "paper_a": "Paper title A",
      "position_a": "What paper A says",
      "paper_b": "Paper title B",
      "position_b": "What paper B says"
    }}
  ],
  "gaps": ["Research gap 1", "Gap 2"],
  "positions": [
    {{
      "paper_id": "id",
      "paper_title": "Title",
      "position": "This paper's stance",
      "evidence": "Key finding from this paper",
      "page_numbers": [1, 3]
    }}
  ],
  "synthesis_summary": "A 2-3 sentence narrative connecting findings across papers"
}}"""

        response = await self._call_claude(system, user_msg)

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse synthesis JSON: {response[:300]}")
            return SynthesisResult(
                query=question,
                consensus="Failed to parse synthesis.",
                contradictions=[], gaps=[], positions=[],
                paper_count=len(paper_chunks),
                synthesis_summary=response[:500],
            )

        positions = []
        for p in data.get("positions", []):
            positions.append(PaperPosition(
                paper_id=p.get("paper_id", ""),
                paper_title=p.get("paper_title", ""),
                position=p.get("position", ""),
                evidence=p.get("evidence", ""),
                page_numbers=p.get("page_numbers", []),
            ))

        return SynthesisResult(
            query=question,
            consensus=data.get("consensus", ""),
            contradictions=data.get("contradictions", []),
            gaps=data.get("gaps", []),
            positions=positions,
            paper_count=len(paper_chunks),
            synthesis_summary=data.get("synthesis_summary", ""),
        )
