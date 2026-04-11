"""
Claude Client for ScholarMind
Routes through AICredits.in (OpenAI-compatible endpoint) for UPI billing.
"""

import logging
from openai import OpenAI
from typing import Optional

from rag_engine import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ScholarMind, an expert AI research assistant that answers questions about scientific papers.

You will be given:
1. A user's question
2. Retrieved text chunks from relevant papers (with page numbers and paper names)

Your job:
- Answer the question accurately using ONLY the provided context
- Cite your sources using [Paper: X, Page: Y] format after each claim
- If the context doesn't contain enough information, say so honestly
- Use clear, academic language but keep it accessible
- Structure your answer with clear paragraphs for readability
- When comparing across papers, organize by theme, not by paper

IMPORTANT:
- Never make up information not in the provided context
- Always cite which paper and page number your answer comes from
- Be concise but thorough
"""


class ClaudeClient:
    """Claude API client via AICredits.in gateway."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.aicredits.in/v1"
        )
        self.model = model
        logger.info(f"Claude client initialized via AICredits with model: {model}")

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        include_figures: bool = False,
    ) -> str:
        """Generate a cited answer using retrieved context."""

        # ── Build context from retrieved chunks ──
        context_parts = []

        for i, rc in enumerate(retrieved_chunks):
            chunk = rc.chunk
            context_parts.append(
                f"--- Chunk {i+1} ---\n"
                f"Paper: {chunk.paper_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Section: {chunk.section}\n"
                f"Relevance Score: {rc.score:.3f}\n"
                f"Content:\n{chunk.text}\n"
            )

        context_text = "\n".join(context_parts)

        user_message = (
            f"## Retrieved Context:\n\n{context_text}\n\n"
            f"## Question:\n{question}\n\n"
            f"Please answer the question using the retrieved context above. "
            f"Cite sources using [Paper: filename, Page: number] format."
        )

        # ── Call Claude via AICredits (OpenAI-compatible format) ──
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response.choices[0].message.content
            logger.info(
                f"Generated answer: {len(answer)} chars, "
                f"tokens used: {response.usage.prompt_tokens}+{response.usage.completion_tokens}"
            )
            return answer

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def summarize_paper(self, paper_chunks: list[RetrievedChunk]) -> str:
        """Generate a concise summary of a paper from its chunks."""
        context = "\n\n".join([rc.chunk.text for rc in paper_chunks[:10]])

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research paper summarizer. Provide a concise, structured summary.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize this scientific paper based on the following excerpts:\n\n"
                        f"{context}\n\n"
                        f"Structure: Objective, Methods, Key Findings, Significance (2-3 sentences each)"
                    ),
                },
            ],
        )
        return response.choices[0].message.content
