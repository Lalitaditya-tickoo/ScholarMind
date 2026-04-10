"""
PDF Processor for ScholarMind
Extracts text, figures, tables, and equations from scientific papers.
Handles multi-column layouts, OCR fallback, and figure captioning.
"""

import fitz  # PyMuPDF
import base64
import io
import os
import re
import logging
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFigure:
    """A figure extracted from a PDF."""
    page_number: int
    image_base64: str
    caption: str = ""
    bbox: tuple = ()
    figure_id: str = ""


@dataclass
class ExtractedPage:
    """Extracted content from a single page."""
    page_number: int
    text: str
    figures: list[ExtractedFigure] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    is_ocr: bool = False


@dataclass
class ProcessedPaper:
    """Complete processed paper."""
    filename: str
    title: str
    pages: list[ExtractedPage] = field(default_factory=list)
    total_figures: int = 0
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """Extract structured content from scientific PDFs."""

    FIGURE_MIN_SIZE = 15000    # min pixels to consider as figure (not icon)
    FIGURE_MIN_DIM = 100       # min width/height in pixels
    TABLE_PATTERNS = [
        r"Table\s+\d+",
        r"TABLE\s+\d+",
    ]
    FIGURE_CAPTION_PATTERNS = [
        r"(Fig(?:ure)?\.?\s*\d+[.:]\s*.+?)(?:\n|$)",
        r"(Figure\s+\d+[.:]\s*.+?)(?:\n|$)",
    ]

    def __init__(self, storage_dir: str = "./paper_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.storage_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str, paper_id: str) -> ProcessedPaper:
        """Main entry point: process a PDF and extract all content."""
        logger.info(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        paper = ProcessedPaper(
            filename=os.path.basename(pdf_path),
            title=self._extract_title(doc),
            total_pages=len(doc),
        )

        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted = self._process_page(page, page_num + 1, paper_id)
            paper.pages.append(extracted)
            paper.total_figures += len(extracted.figures)

        doc.close()
        logger.info(
            f"Extracted {paper.total_pages} pages, "
            f"{paper.total_figures} figures from {paper.filename}"
        )
        return paper

    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract paper title from metadata or first page."""
        # Try metadata first
        meta = doc.metadata
        if meta.get("title") and len(meta["title"].strip()) > 5:
            return meta["title"].strip()

        # Fallback: largest font text on first page
        if len(doc) == 0:
            return "Untitled Paper"

        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        max_size = 0
        title_text = ""

        for block in blocks:
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > max_size and len(span["text"].strip()) > 3:
                        max_size = span["size"]
                        title_text = span["text"].strip()

        return title_text or "Untitled Paper"

    def _process_page(
        self, page: fitz.Page, page_number: int, paper_id: str
    ) -> ExtractedPage:
        """Process a single page: extract text, figures, tables."""
        # ---- Text extraction ----
        text = page.get_text("text")

        # OCR fallback if text is too short (likely scanned page)
        is_ocr = False
        if len(text.strip()) < 50:
            try:
                import pytesseract
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                is_ocr = True
                logger.info(f"OCR fallback used for page {page_number}")
            except Exception as e:
                logger.warning(f"OCR failed for page {page_number}: {e}")

        # ---- Clean text ----
        text = self._clean_text(text)

        # ---- Figure extraction ----
        figures = self._extract_figures(page, page_number, paper_id, text)

        # ---- Table detection ----
        tables = self._detect_tables(text)

        return ExtractedPage(
            page_number=page_number,
            text=text,
            figures=figures,
            tables=tables,
            is_ocr=is_ocr,
        )

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: fix hyphenation, spacing, encoding."""
        # Fix hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Fix excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix spacing issues
        text = re.sub(r"[ \t]+", " ", text)
        # Remove page numbers (standalone numbers on a line)
        text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    def _extract_figures(
        self, page: fitz.Page, page_number: int, paper_id: str, page_text: str
    ) -> list[ExtractedFigure]:
        """Extract figures/images from a page."""
        figures = []
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))

                # Filter out tiny images (icons, bullets, etc.)
                w, h = img.size
                if w < self.FIGURE_MIN_DIM or h < self.FIGURE_MIN_DIM:
                    continue
                if w * h < self.FIGURE_MIN_SIZE:
                    continue

                # Convert to base64 (PNG format for consistency)
                buffer = io.BytesIO()
                img_rgb = img.convert("RGB")
                img_rgb.save(buffer, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                # Try to find caption
                caption = self._find_figure_caption(page_text, len(figures) + 1)

                figure_id = f"{paper_id}_p{page_number}_fig{img_idx}"

                figures.append(ExtractedFigure(
                    page_number=page_number,
                    image_base64=img_base64,
                    caption=caption,
                    figure_id=figure_id,
                ))

            except Exception as e:
                logger.warning(
                    f"Failed to extract image {img_idx} from page {page_number}: {e}"
                )

        return figures

    def _find_figure_caption(self, text: str, figure_num: int) -> str:
        """Try to find a figure caption in the page text."""
        for pattern in self.FIGURE_CAPTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the caption closest to the figure number
                for match in matches:
                    if str(figure_num) in match:
                        return match.strip()
                # If no exact match, return the last one found
                return matches[-1].strip() if matches else ""
        return ""

    def _detect_tables(self, text: str) -> list[str]:
        """Detect table content in the text."""
        tables = []
        for pattern in self.TABLE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract table context (surrounding text)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 500)
                table_context = text[start:end].strip()
                tables.append(table_context)
        return tables
