"""Pydantic models for ScholarMind API."""

from pydantic import BaseModel
from typing import Optional


class PaperMetadata(BaseModel):
    paper_id: str
    filename: str
    title: Optional[str] = None
    num_pages: int
    num_chunks: int
    num_figures: int
    status: str = "processing"


class QueryRequest(BaseModel):
    question: str
    paper_ids: Optional[list[str]] = None  # None = search all papers
    top_k: int = 8


class Citation(BaseModel):
    paper_id: str
    paper_name: str
    page_number: int
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    figures_used: list[str]  # base64 encoded figures if any


class PaperListResponse(BaseModel):
    papers: list[PaperMetadata]
