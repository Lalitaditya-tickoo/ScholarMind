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


# ===============================================================
# Cross-Paper Synthesis Models
# ===============================================================

class CrossPaperQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    paper_ids: list[str] = Field(default_factory=list, description="Paper IDs to compare (empty = all)")
    top_k_per_paper: int = Field(3, ge=1, le=10, description="Chunks to retrieve per paper")


class PaperPositionResponse(BaseModel):
    paper_id: str
    paper_title: str
    position: str
    evidence: str
    page_numbers: list[int]


class ContradictionResponse(BaseModel):
    topic: str
    paper_a: str
    position_a: str
    paper_b: str
    position_b: str


class CrossPaperResponse(BaseModel):
    query: str
    consensus: str
    contradictions: list[ContradictionResponse]
    gaps: list[str]
    positions: list[PaperPositionResponse]
    paper_count: int
    synthesis_summary: str


# ===============================================================
# Table Extraction Models
# ===============================================================

class TableResponse(BaseModel):
    page_number: int
    table_index: int
    caption: str
    headers: list[str]
    rows: list[list[str]]
    num_rows: int
    num_cols: int
    markdown: str


class PaperTablesResponse(BaseModel):
    paper_id: str
    title: str
    tables: list[TableResponse]
    total_tables: int


# ===============================================================
# Evaluation Models
# ===============================================================

class EvalSampleInput(BaseModel):
    question: str
    ground_truth: str
    paper_ids: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    samples: list[EvalSampleInput] = Field(..., min_length=1, max_length=50)


class EvalResultResponse(BaseModel):
    question: str
    generated_answer: str
    ground_truth: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    latency_ms: float
    num_sources: int


class EvalReportResponse(BaseModel):
    num_samples: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_context_recall: float
    avg_latency_ms: float
    results: list[EvalResultResponse]
