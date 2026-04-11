"""
ScholarMind — FastAPI Backend
Multi-Modal RAG Pipeline for Scientific Literature
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from models import (
    PaperMetadata, QueryRequest, QueryResponse,
    Citation, PaperListResponse,
)
from pdf_processor import PDFProcessor
from rag_engine import RAGEngine
from claude_client import ClaudeClient

# ── Config ──
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Global state ──
pdf_processor: PDFProcessor = None
rag_engine: RAGEngine = None
claude: ClaudeClient = None
papers_db: dict[str, PaperMetadata] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global pdf_processor, rag_engine, claude

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set in .env file!")
        raise RuntimeError("Set ANTHROPIC_API_KEY in your .env file")

    pdf_processor = PDFProcessor()
    rag_engine = RAGEngine()
    claude = ClaudeClient(api_key=api_key)

    logger.info("ScholarMind backend ready!")
    yield
    logger.info("Shutting down ScholarMind...")


app = FastAPI(
    title="ScholarMind API",
    description="Multi-Modal RAG Pipeline for Scientific Literature",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow React frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────


@app.get("/health")
async def health_check():
    stats = rag_engine.get_stats() if rag_engine else {}
    return {
        "status": "healthy",
        "papers_loaded": len(papers_db),
        **stats,
    }


@app.post("/papers/upload", response_model=PaperMetadata)
async def upload_paper(file: UploadFile = File(...)):
    """Upload and process a scientific paper (PDF)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    paper_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{paper_id}_{file.filename}"

    # Save uploaded file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Step 1: Extract content from PDF
        processed = pdf_processor.process_pdf(str(save_path), paper_id)

        # Step 2: Ingest into vector store
        num_chunks = rag_engine.ingest_paper(processed, paper_id)

        # Step 3: Store metadata
        metadata = PaperMetadata(
            paper_id=paper_id,
            filename=file.filename,
            title=processed.title,
            num_pages=processed.total_pages,
            num_chunks=num_chunks,
            num_figures=processed.total_figures,
            status="ready",
        )
        papers_db[paper_id] = metadata

        logger.info(f"Paper uploaded: {file.filename} → {num_chunks} chunks, {processed.total_figures} figures")
        return metadata

    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        # Cleanup
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")


@app.get("/papers", response_model=PaperListResponse)
async def list_papers():
    """List all uploaded papers."""
    return PaperListResponse(papers=list(papers_db.values()))


@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper and its chunks from the system."""
    if paper_id not in papers_db:
        raise HTTPException(404, "Paper not found")

    rag_engine.delete_paper(paper_id)
    del papers_db[paper_id]

    # Remove uploaded file
    for f in UPLOAD_DIR.glob(f"{paper_id}_*"):
        f.unlink()

    return {"message": f"Paper {paper_id} deleted"}


@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Ask a question about uploaded papers."""
    if not papers_db:
        raise HTTPException(400, "No papers uploaded yet. Upload at least one paper first.")

    # Step 1: Retrieve relevant chunks
    retrieved = rag_engine.retrieve(
        query=request.question,
        top_k=request.top_k,
        paper_ids=request.paper_ids,
    )

    if not retrieved:
        return QueryResponse(
            answer="I couldn't find any relevant information in the uploaded papers for your question.",
            citations=[],
            figures_used=[],
        )

    # Step 2: Generate answer with Claude
    answer = claude.generate_answer(
        question=request.question,
        retrieved_chunks=retrieved,
        include_figures=True,
    )

    # Step 3: Build citations
    citations = [
        Citation(
            paper_id=rc.chunk.paper_id,
            paper_name=rc.chunk.paper_name,
            page_number=rc.chunk.page_number,
            chunk_text=rc.chunk.text[:200] + "..." if len(rc.chunk.text) > 200 else rc.chunk.text,
            relevance_score=round(rc.score, 4),
        )
        for rc in retrieved
    ]

    # Step 4: Collect figure base64 strings
    figures_used = []
    for rc in retrieved:
        for fig in rc.figures:
            if fig.image_base64 not in figures_used:
                figures_used.append(fig.image_base64)

    return QueryResponse(
        answer=answer,
        citations=citations,
        figures_used=figures_used[:4],  # Max 4 figures in response
    )


@app.post("/papers/{paper_id}/summarize")
async def summarize_paper(paper_id: str):
    """Generate an AI summary of a specific paper."""
    if paper_id not in papers_db:
        raise HTTPException(404, "Paper not found")

    # Get all chunks for this paper
    chunks = rag_engine.retrieve(
        query="main findings methodology results conclusions",
        top_k=10,
        paper_ids=[paper_id],
    )

    summary = claude.summarize_paper(chunks)
    return {"paper_id": paper_id, "summary": summary}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
