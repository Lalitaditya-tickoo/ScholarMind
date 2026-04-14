"""
Table extraction from PDFs using PyMuPDF layout analysis.
"""

import fitz
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str
    caption: str = ""
    num_rows: int = 0
    num_cols: int = 0

    def __post_init__(self):
        self.num_rows = len(self.rows)
        self.num_cols = len(self.headers) if self.headers else (
            len(self.rows[0]) if self.rows else 0
        )

    def to_markdown(self) -> str:
        if not self.headers and not self.rows:
            return self.raw_text
        lines = []
        if self.caption:
            lines.append(f"**{self.caption}**\n")
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        for row in self.rows:
            padded = row + [""] * (len(self.headers) - len(row)) if self.headers else row
            lines.append("| " + " | ".join(padded) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "table_index": self.table_index,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "markdown": self.to_markdown(),
        }


class TableExtractor:
    def __init__(self):
        self.min_columns = 2
        self.min_rows = 2

    def extract_tables(self, pdf_path: str) -> list[ExtractedTable]:
        doc = fitz.open(pdf_path)
        all_tables = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_tables = self._extract_from_page(page, page_num + 1)
            all_tables.extend(page_tables)
        doc.close()
        logger.info(f"Extracted {len(all_tables)} tables from {pdf_path}")
        return all_tables

    def _extract_from_page(self, page, page_number: int) -> list[ExtractedTable]:
        tables = []
        blocks = page.get_text("dict")["blocks"]
        table_candidates = self._find_table_blocks(blocks, page_number)
        tables.extend(table_candidates)
        raw_text = page.get_text("text")
        line_tables = self._detect_from_lines(raw_text, page_number, start_idx=len(tables))
        tables.extend(line_tables)
        return tables

    def _find_table_blocks(self, blocks: list, page_number: int) -> list[ExtractedTable]:
        tables = []
        text_blocks = [b for b in blocks if b.get("type") == 0]
        for block in text_blocks:
            lines_data = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans if s["text"].strip())
                if text:
                    lines_data.append({"text": text, "y": line["bbox"][1], "spans": spans})
            if len(lines_data) >= self.min_rows + 1:
                tab_lines = sum(1 for l in lines_data if "\t" in l["text"] or "  " in l["text"])
                pipe_lines = sum(1 for l in lines_data if "|" in l["text"])
                if tab_lines > len(lines_data) * 0.5 or pipe_lines > len(lines_data) * 0.5:
                    table = self._parse_table_block(lines_data, page_number, len(tables))
                    if table:
                        tables.append(table)
        return tables

    def _detect_from_lines(self, text: str, page_number: int, start_idx: int = 0) -> list[ExtractedTable]:
        tables = []
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            table_lines = []
            caption = ""
            if self._is_table_line(lines[i]):
                if i > 0 and re.match(r'^(Table|Tab\.?)\s*\d', lines[i-1], re.IGNORECASE):
                    caption = lines[i-1].strip()
                while i < len(lines) and self._is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                if len(table_lines) >= self.min_rows + 1:
                    table = self._parse_text_table(table_lines, page_number, start_idx + len(tables), caption)
                    if table:
                        tables.append(table)
                continue
            i += 1
        return tables

    def _is_table_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        if stripped.count("\t") >= 1:
            return True
        if stripped.count("|") >= 2:
            return True
        space_groups = re.findall(r'\s{2,}', stripped)
        if len(space_groups) >= 1 and len(stripped) > 10:
            parts = re.split(r'\s{2,}', stripped)
            non_empty = [p for p in parts if p.strip()]
            if len(non_empty) >= self.min_columns:
                return True
        return False

    def _parse_table_block(self, lines_data, page_number, table_index):
        rows = []
        for line_d in lines_data:
            text = line_d["text"]
            cells = re.split(r'\t|\s{2,}', text)
            cells = [c.strip() for c in cells if c.strip()]
            if cells:
                rows.append(cells)
        if len(rows) < self.min_rows:
            return None
        headers = rows[0]
        data_rows = rows[1:]
        raw = "\n".join(l["text"] for l in lines_data)
        return ExtractedTable(page_number=page_number, table_index=table_index,
                              headers=headers, rows=data_rows, raw_text=raw)

    def _parse_text_table(self, lines, page_number, table_index, caption=""):
        rows = []
        for line in lines:
            stripped = line.strip()
            if "|" in stripped:
                cells = [c.strip() for c in stripped.split("|") if c.strip()]
            elif "\t" in stripped:
                cells = [c.strip() for c in stripped.split("\t") if c.strip()]
            else:
                cells = [c.strip() for c in re.split(r'\s{2,}', stripped) if c.strip()]
            if cells and all(re.match(r'^[-=:]+$', c) for c in cells):
                continue
            if cells:
                rows.append(cells)
        if len(rows) < self.min_rows:
            return None
        headers = rows[0]
        data_rows = rows[1:]
        raw = "\n".join(lines)
        return ExtractedTable(page_number=page_number, table_index=table_index,
                              headers=headers, rows=data_rows, raw_text=raw, caption=caption)
