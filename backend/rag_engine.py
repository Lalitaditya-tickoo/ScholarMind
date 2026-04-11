"""
RAG Engine for ScholarMind
Handles: section-aware chunking, embedding, vector storage, and retrieval.
Uses ChromaDB (free, local) + sentence-transformers (free, local).
"""

import re
import uuid
import logging
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Optional

from pdf_processor import ProcessedPaper, ExtractedFigure

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    text: str
    paper_id: str
    paper_name: str
    page_number: int
    section: str = ""
    has_table: bool = False
    figure_ids: list[str] = None

    def __post_init__(self):
        if self.figure_ids is None:
            self.figure_ids = []


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval with relevance score."""
    chunk: Chunk
    score: float
    figures: list[ExtractedFigure] = None

    def __post_init__(self):
        if self.figures is None:
            self.figures = []


# ── Section header patterns for academic papers ──
SECTION_PATTERNS = [
    r"^(?:\d+\.?\s+)?(Abstract|Introduction|Background|Related Work|Methodology|"
    r"Methods|Experiments?|Results?|Discussion|Conclusion|References|Acknowledgements?"
    r"|Appendix|Supplementary|Materials?\s+and\s+Methods?|Data\s+Collection|"
    r"Experimental\s+Setup|Evaluation|Analysis|Future\s+Work|Limitations?)"
    r"\s*$",
]


class RAGEngine:
    """Core RAG pipeline: chunk → embed → store → retrieve."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ── Load embedding model (FREE, runs locally) ──
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # ── Initialize ChromaDB (FREE, local vector store) ──
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="scholar_mind",
            metadata={"hnsw:space": "cosine"},
        )

        # ── In-memory figure store ──
        self.figures: dict[str, ExtractedFigure] = {}

        logger.info("RAG Engine initialized successfully")

    def ingest_paper(self, paper: ProcessedPaper, paper_id: str) -> int:
        """Ingest a processed paper into the vector store. Returns chunk count."""
        # Step 1: Section-aware chunking
        chunks = self._chunk_paper(paper, paper_id)

        if not chunks:
            logger.warning(f"No chunks created for {paper.filename}")
            return 0

        # Step 2: Store figures
        for page in paper.pages:
            for fig in page.figures:
                self.figures[fig.figure_id] = fig

        # Step 3: Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Step 4: Store in ChromaDB
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "paper_id": c.paper_id,
                    "paper_name": c.paper_name,
                    "page_number": c.page_number,
                    "section": c.section,
                    "has_table": c.has_table,
                    "figure_ids": ",".join(c.figure_ids),
                }
                for c in chunks
            ],
        )

        logger.info(f"Ingested {len(chunks)} chunks from {paper.filename}")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        paper_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()

        # Build filter
        where_filter = None
        if paper_ids:
            if len(paper_ids) == 1:
                where_filter = {"paper_id": paper_ids[0]}
            else:
                where_filter = {"paper_id": {"$in": paper_ids}}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Build RetrievedChunk objects
        retrieved = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1 - (distance / 2)

            chunk = Chunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                paper_id=meta["paper_id"],
                paper_name=meta["paper_name"],
                page_number=meta["page_number"],
                section=meta.get("section", ""),
                has_table=meta.get("has_table", False),
                figure_ids=meta.get("figure_ids", "").split(",") if meta.get("figure_ids") else [],
            )

            # Attach figures
            figures = []
            for fig_id in chunk.figure_ids:
                if fig_id and fig_id in self.figures:
                    figures.append(self.figures[fig_id])

            retrieved.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                figures=figures,
            ))

        # Sort by relevance (highest score first)
        retrieved.sort(key=lambda x: x.score, reverse=True)
        return retrieved

    def delete_paper(self, paper_id: str):
        """Remove all chunks for a paper from the vector store."""
        self.collection.delete(where={"paper_id": paper_id})
        # Remove figures
        to_remove = [fid for fid, fig in self.figures.items() if paper_id in fid]
        for fid in to_remove:
            del self.figures[fid]
        logger.info(f"Deleted paper {paper_id} from vector store")

    def get_stats(self) -> dict:
        """Get stats about the vector store."""
        return {
            "total_chunks": self.collection.count(),
            "total_figures": len(self.figures),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": self.embedding_dim,
        }

    # ──────────────────────────────────────────────
    #  CHUNKING LOGIC
    # ──────────────────────────────────────────────

    def _chunk_paper(self, paper: ProcessedPaper, paper_id: str) -> list[Chunk]:
        """Section-aware chunking with overlap."""
        chunks = []
        current_section = "Unknown"

        for page in paper.pages:
            text = page.text
            if not text.strip():
                continue

            # Detect section headers
            current_section = self._detect_section(text) or current_section

            # Get figure IDs for this page
            page_figure_ids = [f.figure_id for f in page.figures]

            # Check for tables
            has_table = len(page.tables) > 0

            # Split into chunks with overlap
            page_chunks = self._split_text(
                text=text,
                paper_id=paper_id,
                paper_name=paper.filename,
                page_number=page.page_number,
                section=current_section,
                has_table=has_table,
                figure_ids=page_figure_ids,
            )
            chunks.extend(page_chunks)

        return chunks

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect if text starts with a section header."""
        lines = text.strip().split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            for pattern in SECTION_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip().title()
        return None

    def _split_text(
        self,
        text: str,
        paper_id: str,
        paper_name: str,
        page_number: int,
        section: str,
        has_table: bool,
        figure_ids: list[str],
    ) -> list[Chunk]:
        """Split text into chunks by words with overlap."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if len(chunk_text.strip()) < 20:
                start = end - self.chunk_overlap
                continue

            chunks.append(Chunk(
                chunk_id=f"{paper_id}_{page_number}_{len(chunks)}_{uuid.uuid4().hex[:6]}",
                text=chunk_text,
                paper_id=paper_id,
                paper_name=paper_name,
                page_number=page_number,
                section=section,
                has_table=has_table,
                figure_ids=figure_ids if start == 0 else [],  # attach figs to first chunk only
            ))

            # Move forward with overlap
            start = end - self.chunk_overlap
            if start <= 0 and end >= len(words):
                break  # Prevent infinite loop on short text

        return chunks
