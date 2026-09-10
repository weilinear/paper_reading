#!/usr/bin/env python3
"""
pdf_reader.py - Poppler-based scientific paper PDF reader for AI agents and researchers.
Specialized in layout-aware extraction of two-column and multi-column scientific documents.

Features:
- Accurate column layout detection (1-column, 2-column, mixed banner/table layouts).
- Preserves correct human reading order across multi-column text.
- Filters out margin watermarks (e.g. arXiv sidebar), running headers, and footers.
- Smart de-hyphenation across line and column breaks.
- Robust section heading hierarchy extraction (#, ##, ###).
- Identifies figure and table captions (> **Figure 1.** ...).
- Supports outline/TOC extraction, section-specific reading, keyword search.
- High-fidelity visual rendering of pages/figures via pdftoppm for visual inspection.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys

# Ensure poppler fontconfig cache doesn't warn on read-only system dirs
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache")
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0


@dataclass
class Word:
    bbox: BBox
    text: str


@dataclass
class Line:
    bbox: BBox
    words: List[Word] = field(default_factory=list)
    text: str = ""

    @property
    def height(self) -> float:
        return self.bbox.height


@dataclass
class Block:
    bbox: BBox
    lines: List[Line] = field(default_factory=list)
    text: str = ""
    block_type: str = "paragraph"  # 'title', 'frontmatter', 'heading', 'paragraph', 'caption', 'table', 'footnote', 'header', 'footer'
    column_idx: int = 0  # 0: spanning/full-width, 1: col1, 2: col2
    heading_level: int = 0
    caption_label: str = ""


@dataclass
class Page:
    page_num: int
    width: float
    height: float
    is_two_column: bool
    blocks: List[Block] = field(default_factory=list)
    header: Optional[str] = None
    footer: Optional[str] = None
    margin_notes: List[str] = field(default_factory=list)


@dataclass
class Section:
    title: str
    level: int
    page_num: int
    content: str = ""


# ==============================================================================
# Helper Functions
# ==============================================================================

def check_poppler_installed() -> None:
    """Check if required poppler utilities are available in PATH."""
    required = ["pdftotext", "pdfinfo", "pdftoppm"]
    missing = [tool for tool in required if not shutil.which(tool)]
    if missing:
        sys.stderr.write(
            f"Error: Required poppler utilities not found in PATH: {', '.join(missing)}\n"
            f"Please install poppler (e.g., 'brew install poppler' on macOS or 'apt install poppler-utils' on Linux).\n"
        )
        sys.exit(1)


def get_pdf_info(pdf_path: str | Path) -> Dict[str, Any]:
    """Extract PDF metadata using poppler's pdfinfo."""
    try:
        res = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        info = {}
        for line in res.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
        return info
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"pdfinfo failed: {e.stderr}\n")
        return {}


def dehyphenate_lines(line1: str, line2: str) -> str:
    """
    Intelligently join two lines, de-hyphenating words split across line breaks.
    Preserves legitimate hyphens in compound words or numbers.
    """
    line1 = line1.rstrip()
    line2 = line2.lstrip()

    if not line1:
        return line2
    if not line2:
        return line1

    m = re.search(r"([A-Za-z0-9]+)[-–—]\s*$", line1)
    if not m:
        return f"{line1} {line2}"

    w1 = m.group(1)
    m2 = re.match(r"^([A-Za-z0-9]+)(.*)$", line2)
    if not m2:
        return f"{line1} {line2}"

    w2 = m2.group(1)
    rest2 = m2.group(2)
    prefix = line1[: m.start()]

    # If w1 ends with digits (e.g. 18-layer, 56-layer), keep hyphen
    if re.search(r"\d+$", w1):
        return f"{prefix}{w1}-{w2}{rest2}"

    # If w2 is capitalized (e.g. non-English, multi-GPU), keep hyphen
    if w2[0].isupper():
        return f"{prefix}{w1}-{w2}{rest2}"

    # Known prefixes that commonly stay hyphenated
    common_hyphen_prefixes = {
        "multi", "non", "self", "cross", "pre", "post", "quasi", "semi", "ultra", "all",
    }
    if w1.lower() in common_hyphen_prefixes and len(w2) > 3:
        return f"{prefix}{w1}-{w2}{rest2}"

    # Default: word was split across line break by TeX/renderer, merge without hyphen
    return f"{prefix}{w1}{w2}{rest2}"


def is_heading_text(text: str) -> Tuple[bool, int]:
    """
    Classify whether a line of text is a bona fide section heading and return its level.
    Avoids classifying figure labels, axis legends, footnotes, or affiliations as headings.
    """
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 3 or len(cleaned) > 85:
        return False, 0

    # Reject text containing typical non-heading characters/words
    if any(cleaned.lower().startswith(p) for p in ["figure", "fig.", "table", "algorithm", "arxiv:"]):
        return False, 0
    if re.search(r"(%|\bflops\b|\bstd\b|\bacc\b|\bval\b|\btest error\b|\btraining error\b)", cleaned, re.I):
        return False, 0

    # Headings in papers are never entirely lowercase
    if cleaned.islower():
        return False, 0

    # Numbered heading: '1. Introduction', '1 Introduction', '3.1. Residual Learning', 'A. Object Detection'
    m_num = re.match(r"^((?:\d+(?:\.\d+)*|[A-Z])\.?)\s+([A-Z][A-Za-z0-9\s,\-:–—]+)$", cleaned)
    if m_num:
        title_body = m_num.group(2).strip()
        words = title_body.split()
        # Single-word numbered items like '2. Log' or '3. Update' are diagram steps, not sections
        if len(words) == 1 and words[0] not in {
            "Introduction", "Background", "Overview", "Method", "Methods", "Methodology",
            "Approach", "Architecture", "Implementation", "Setup", "Experiments", "Experimentation",
            "Results", "Evaluation", "Analysis", "Discussion", "Conclusion", "Conclusions",
            "References", "Abstract", "Ablations", "Baselines", "Preliminaries", "Formulation",
            "Training", "Optimization", "Theory", "Proofs", "Limitations",
        }:
            return False, 0

        num = m_num.group(1).rstrip(".")
        if "." in num:
            level = num.count(".") + 1
        elif num.isdigit():
            level = 1
        else:  # Appendix 'A'
            level = 2
        return True, min(3, level)

    # Standard unnumbered section titles (Title Case or UPPERCASE)
    m_std = re.match(
        r"^(Abstract|Introduction|Related Work|Background|Methods?|Methodology|Approach|Architecture|Experiments?|Results?|Discussion|Conclusions?|References|Acknowledgements?|Appendix(?:\s+[A-Z0-9]+)?)$",
        cleaned.title(),
    )
    if m_std:
        return True, 1

    return False, 0


# ==============================================================================
# PDF Layout Parser
# ==============================================================================

class PDFLayoutParser:
    """Extracts and parses poppler pdftotext bbox-layout XML."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.info = get_pdf_info(self.pdf_path)
        self.num_pages = int(self.info.get("Pages", 1))

    def extract_page_xml(self, page_num: int) -> str:
        """Call pdftotext -bbox-layout for a single page."""
        cmd = [
            "pdftotext",
            "-bbox-layout",
            "-f", str(page_num),
            "-l", str(page_num),
            str(self.pdf_path),
            "-",
        ]
        res = subprocess.run(cmd, capture_output=True, check=True)
        raw_xml = res.stdout.decode("utf-8", errors="replace")
        # Sanitize control characters that break XML parsing
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_xml)

    def parse_page(self, page_num: int) -> Page:
        """Parse XML into structured Page object with layout analysis."""
        xml_str = self.extract_page_xml(page_num)

        # Extract page dimensions
        m_page = re.search(r'<page width="([0-9.]+)" height="([0-9.]+)">', xml_str)
        if not m_page:
            return Page(page_num=page_num, width=612.0, height=792.0, is_two_column=False)

        pw = float(m_page.group(1))
        ph = float(m_page.group(2))

        # Extract blocks
        raw_blocks = re.findall(
            r'<block xMin="([0-9.]+)" yMin="([0-9.]+)" xMax="([0-9.]+)" yMax="([0-9.]+)">'
            r"(.*?)</block>",
            xml_str,
            re.DOTALL,
        )

        blocks: List[Block] = []
        margin_notes: List[str] = []
        header_text: Optional[str] = None
        footer_text: Optional[str] = None

        for xmin_s, ymin_s, xmax_s, ymax_s, content in raw_blocks:
            xmin, ymin, xmax, ymax = float(xmin_s), float(ymin_s), float(xmax_s), float(ymax_s)
            bbox = BBox(xmin, ymin, xmax, ymax)

            # Parse lines and words
            lines: List[Line] = []
            for line_xml in re.findall(r"<line [^>]*>(.*?)</line>", content, re.DOTALL):
                words: List[Word] = []
                for w_m in re.finditer(
                    r'<word xMin="([0-9.]+)" yMin="([0-9.]+)" xMax="([0-9.]+)" yMax="([0-9.]+)">'
                    r"(.*?)</word>",
                    line_xml,
                ):
                    w_text = html.unescape(w_m.group(5)).strip()
                    if w_text:
                        words.append(
                            Word(
                                bbox=BBox(
                                    float(w_m.group(1)),
                                    float(w_m.group(2)),
                                    float(w_m.group(3)),
                                    float(w_m.group(4)),
                                ),
                                text=w_text,
                            )
                        )
                if words:
                    line_text = " ".join(str(w.text) for w in words)
                    line_bbox = BBox(
                        min(w.bbox.xmin for w in words),
                        min(w.bbox.ymin for w in words),
                        max(w.bbox.xmax for w in words),
                        max(w.bbox.ymax for w in words),
                    )
                    lines.append(Line(bbox=line_bbox, words=words, text=line_text))

            if not lines:
                continue

            block_text = "\n".join(l.text for l in lines)
            block = Block(bbox=bbox, lines=lines, text=block_text)

            # 1. Filter margin watermarks (e.g. arXiv sidebar)
            if (bbox.xmax < pw * 0.08 and bbox.height > ph * 0.15) or (
                bbox.xmin > pw * 0.92 and bbox.height > ph * 0.15
            ):
                margin_notes.append(block_text.replace("\n", " "))
                continue

            # 2. Running headers (skip on page 1 where title is at the top)
            if page_num > 1 and bbox.ymax < ph * 0.085 and bbox.height < 30 and len(lines) <= 2:
                header_text = block_text.replace("\n", " ")
                block.block_type = "header"
                continue

            # 3. Running footers (page numbers, copyright)
            if bbox.ymin > ph * 0.915 and bbox.height < 30 and len(lines) <= 2:
                footer_text = block_text.replace("\n", " ")
                block.block_type = "footer"
                continue

            blocks.append(block)

        # Classify block types and determine layout
        is_two_column, ordered_blocks = self._layout_and_order_blocks(pw, ph, blocks, page_num)

        return Page(
            page_num=page_num,
            width=pw,
            height=ph,
            is_two_column=is_two_column,
            blocks=ordered_blocks,
            header=header_text,
            footer=footer_text,
            margin_notes=margin_notes,
        )

    def _layout_and_order_blocks(
        self, pw: float, ph: float, blocks: List[Block], page_num: int
    ) -> Tuple[bool, List[Block]]:
        """
        Determine column structure and return blocks in correct human reading order.
        Handles:
        - 1-column pages
        - 2-column pages
        - Mixed pages: Title/Author frontmatter banner + 2 columns
        - Full-width wide figures/tables spanning across columns
        """
        if not blocks:
            return False, []

        # Find median body font size
        word_heights = [
            w.bbox.height
            for b in blocks
            for l in b.lines
            for w in l.words
            if len(str(w.text)) > 2 and w.bbox.width > 10
        ]
        median_font_h = (
            sorted(word_heights)[len(word_heights) // 2] if word_heights else 9.0
        )

        # Classify block semantics
        for b in blocks:
            raw_stripped = b.text.strip()
            first_line = b.lines[0].text.strip() if b.lines else ""

            # Caption check
            cap_m = re.match(r"^(Figure|Fig\.|Table|Algorithm)\s+(\d+[\w\.\-]*)[.:\s]", first_line, re.I)
            if cap_m:
                b.block_type = "caption"
                # Strip trailing dots in label to prevent double dot in markdown
                label_prefix = cap_m.group(1).capitalize()
                if label_prefix.startswith("Fig."):
                    label_prefix = "Figure"
                b.caption_label = f"{label_prefix} {cap_m.group(2).rstrip('.')}"
                continue

            # Section heading check
            is_heading, lvl = is_heading_text(raw_stripped)
            if is_heading:
                b.block_type = "heading"
                b.heading_level = lvl
                continue

            # Footnote check (bottom 15% of page, starts with footnote mark)
            if b.bbox.ymin > ph * 0.85 and re.match(r"^(\d+|[*†‡])\s+", first_line):
                b.block_type = "footnote"
                continue

        # Column assignment based on horizontal layout
        c_left = 0.44 * pw
        c_right = 0.56 * pw

        col1_blocks = []
        col2_blocks = []
        span_blocks = []

        for b in blocks:
            # A block spans if it is significantly wider than half the page
            # or crosses both the left and right center boundaries
            is_span = (b.bbox.width > 0.58 * pw) or (b.bbox.xmin < c_left and b.bbox.xmax > c_right)
            if is_span:
                b.column_idx = 0
                span_blocks.append(b)
            elif b.bbox.center_x < pw / 2:
                b.column_idx = 1
                col1_blocks.append(b)
            else:
                b.column_idx = 2
                col2_blocks.append(b)

        # Decide if page is two-column
        c1_chars = sum(len(b.text) for b in col1_blocks)
        c2_chars = sum(len(b.text) for b in col2_blocks)
        is_two_column = (c1_chars > 200 and c2_chars > 200) or (
            len(col1_blocks) >= 2 and len(col2_blocks) >= 2 and (c1_chars + c2_chars > 350)
        )

        if not is_two_column:
            # Single-column page: sort top to bottom
            sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.ymin, b.bbox.xmin))
            # Tag title and frontmatter on page 1
            if page_num == 1 and sorted_blocks:
                body_start_candidates = [
                    b.bbox.ymin
                    for b in sorted_blocks
                    if (b.block_type == "heading" and re.match(r"^(Abstract|1\.?\s+Intro)", b.text.strip(), re.I))
                    or len(b.lines) >= 6
                ]
                body_start_y = min(body_start_candidates) if body_start_candidates else ph * 0.28
                sorted_blocks[0].block_type = "title"
                for b in sorted_blocks[1:]:
                    if b.bbox.ymax <= body_start_y:
                        b.block_type = "frontmatter"
            return False, sorted_blocks

        # ----------------------------------------------------------------------
        # Two-Column Page Ordering via Vertical Band Slicing
        # ----------------------------------------------------------------------

        frontmatter_blocks = []
        if page_num == 1:
            # Find the start of the two-column body (e.g. Abstract or Introduction or first long paragraph)
            body_start_candidates = [
                b.bbox.ymin
                for b in blocks
                if (b.block_type == "heading" and re.match(r"^(Abstract|1\.?\s+Intro)", b.text.strip(), re.I))
                or (len(b.lines) >= 4 and b.column_idx == 1)
            ]
            body_start_y = min(body_start_candidates) if body_start_candidates else ph * 0.28

            # Frontmatter includes title, authors, affiliations, emails above body_start_y
            for b in list(blocks):
                if b.bbox.ymax <= body_start_y and b.bbox.ymin < body_start_y - 10:
                    # Don't grab column 2 figure blocks if figure starts early
                    if b.bbox.center_x < pw * 0.85:
                        frontmatter_blocks.append(b)

        # Remove frontmatter blocks from col1/col2/span for band computation
        fm_ids = set(id(b) for b in frontmatter_blocks)
        remaining_blocks = [b for b in blocks if id(b) not in fm_ids]

        # Group remaining spanning blocks into span bands
        active_span_blocks = [b for b in remaining_blocks if b in span_blocks]
        span_bands: List[Dict[str, Any]] = []

        for sb in sorted(active_span_blocks, key=lambda b: b.bbox.ymin):
            if not span_bands:
                span_bands.append({"ymin": sb.bbox.ymin, "ymax": sb.bbox.ymax, "blocks": [sb]})
            else:
                last = span_bands[-1]
                # If overlapping or very close (< 12 pt), merge into single band
                if sb.bbox.ymin <= last["ymax"] + 12.0:
                    last["ymax"] = max(last["ymax"], sb.bbox.ymax)
                    last["blocks"].append(sb)
                else:
                    span_bands.append({"ymin": sb.bbox.ymin, "ymax": sb.bbox.ymax, "blocks": [sb]})

        ordered: List[Block] = []

        # 1. Frontmatter at top of page 1
        if frontmatter_blocks:
            # Sort frontmatter by Y row (bucket by 12 pt), then X
            fm_sorted = sorted(
                frontmatter_blocks, key=lambda b: (round(b.bbox.ymin / 12) * 12, b.bbox.xmin)
            )
            # Mark the very first large text as title, and rest as frontmatter
            if fm_sorted:
                fm_sorted[0].block_type = "title"
                for b in fm_sorted[1:]:
                    if b.block_type == "heading":
                        b.block_type = "frontmatter"
            ordered.extend(fm_sorted)
            cur_y = max(b.bbox.ymax for b in frontmatter_blocks)
        else:
            cur_y = 0.0

        all_band_blocks = set(id(b) for band in span_bands for b in band["blocks"])
        two_col_blocks = [b for b in remaining_blocks if id(b) not in all_band_blocks]

        # 2. Process vertical bands
        for band in span_bands:
            b_ymin = band["ymin"]
            b_ymax = band["ymax"]

            # Two-column slice before this span band
            slice_blocks = [b for b in two_col_blocks if cur_y - 2 <= b.bbox.center_y < b_ymin]
            c1 = sorted([b for b in slice_blocks if b.column_idx == 1], key=lambda b: b.bbox.ymin)
            c2 = sorted([b for b in slice_blocks if b.column_idx == 2], key=lambda b: b.bbox.ymin)

            # Left column first, then Right column!
            ordered.extend(c1)
            ordered.extend(c2)

            # The span band itself
            band_blocks_sorted = sorted(
                band["blocks"], key=lambda b: (round(b.bbox.ymin / 10) * 10, b.bbox.xmin)
            )
            ordered.extend(band_blocks_sorted)
            cur_y = b_ymax

        # 3. Two-column slice after the last span band
        bottom_blocks = [b for b in two_col_blocks if b.bbox.center_y >= cur_y - 2]
        c1 = sorted([b for b in bottom_blocks if b.column_idx == 1], key=lambda b: b.bbox.ymin)
        c2 = sorted([b for b in bottom_blocks if b.column_idx == 2], key=lambda b: b.bbox.ymin)
        ordered.extend(c1)
        ordered.extend(c2)

        # 4. Catch any missed blocks
        seen = set(id(b) for b in ordered)
        missed = [b for b in blocks if id(b) not in seen and b.block_type not in ("header", "footer")]
        if missed:
            ordered.extend(sorted(missed, key=lambda b: (b.bbox.ymin, b.bbox.xmin)))

        return True, ordered


# ==============================================================================
# Markdown and High-Level Document Formatter
# ==============================================================================

class DocumentFormatter:
    """Formats parsed Page and Block structures into readable Markdown or JSON."""

    @staticmethod
    def format_block(block: Block, dehyphen: bool = True) -> str:
        """Format a single block into Markdown according to its semantic type."""
        lines = block.lines
        if not lines:
            return ""

        # Join lines with dehyphenation
        joined_text = ""
        for line in lines:
            if not joined_text:
                joined_text = line.text
            else:
                if dehyphen:
                    joined_text = dehyphenate_lines(joined_text, line.text)
                else:
                    joined_text = f"{joined_text} {line.text}"

        # Clean spaces
        joined_text = re.sub(r"\s+", " ", joined_text).strip()

        # Format by block type
        if block.block_type == "title":
            return f"# {joined_text}"

        if block.block_type == "heading":
            prefix = "#" * max(1, min(4, block.heading_level))
            return f"{prefix} {joined_text}"

        if block.block_type == "caption":
            label = block.caption_label
            if label and joined_text.lower().startswith(label.lower()):
                rest = joined_text[len(label):].lstrip(".: ")
                return f"> **{label}.** {rest}"
            return f"> **{joined_text}**"

        if block.block_type == "footnote":
            return f"_{joined_text}_"

        # Check for run-in bold heading at start of paragraph (e.g. "Shortcut Connections. Practices and...")
        runin_m = re.match(r"^([A-Z][A-Za-z0-9\s,\-]{2,40}\.)\s+(.+)$", joined_text)
        if runin_m and len(runin_m.group(1).split()) <= 4:
            lead = runin_m.group(1)
            body = runin_m.group(2)
            return f"**{lead}** {body}"

        return joined_text

    @classmethod
    def page_to_markdown(cls, page: Page, include_page_header: bool = True, dehyphen: bool = True) -> str:
        """Render a single page to Markdown."""
        parts = []
        if include_page_header:
            parts.append(f"<!-- Page {page.page_num} ({'2-column' if page.is_two_column else '1-column'}) -->")
            parts.append(f"## Page {page.page_num}\n")

        for block in page.blocks:
            md_block = cls.format_block(block, dehyphen=dehyphen)
            if md_block:
                parts.append(md_block)

        return "\n\n".join(parts)

    @classmethod
    def document_to_markdown(cls, pages: List[Page], continuous: bool = True, dehyphen: bool = True) -> str:
        """Render a full document to Markdown with continuous paragraph joining."""
        if not pages:
            return ""

        output_chunks = []
        for page in pages:
            page_md = cls.page_to_markdown(page, include_page_header=not continuous, dehyphen=dehyphen)
            output_chunks.append(page_md)

        full_text = "\n\n".join(output_chunks)

        if continuous:
            # Join paragraphs that were split across page boundaries
            def merge_cross_page(m: re.Match) -> str:
                t1 = m.group(1)
                t2 = m.group(2)
                if t1.endswith("-"):
                    return dehyphenate_lines(t1, t2)
                if re.search(r"[A-Za-z0-9]$", t1) and t2 and t2[0].islower():
                    return f"{t1} {t2}"
                return f"{t1}\n\n{t2}"

            full_text = re.sub(
                r"([^\n\.\?\!:\>#]+)\n\n<!-- Page \d+ -->\n\n([a-z][^\n]+)",
                merge_cross_page,
                full_text,
            )

        return full_text.strip()


# ==============================================================================
# PDFReader High-Level API
# ==============================================================================

class PDFReader:
    """Main interface for inspecting, reading, and searching scientific papers."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.parser = PDFLayoutParser(self.pdf_path)
        self.num_pages = self.parser.num_pages
        self._cached_pages: Dict[int, Page] = {}

    def get_info(self) -> Dict[str, Any]:
        """Return PDF metadata and layout summary."""
        info = dict(self.parser.info)
        info["Pages"] = self.num_pages
        # Sample first 5 pages to describe document layout
        layouts = []
        sample_limit = min(5, self.num_pages)
        for p in range(1, sample_limit + 1):
            page = self.get_page(p)
            layouts.append("2-column" if page.is_two_column else "1-column")
        info["detected_layouts"] = layouts
        info["primary_layout"] = "2-column" if layouts.count("2-column") > len(layouts) / 2 else "1-column"
        return info

    def get_page(self, page_num: int) -> Page:
        """Get or parse a specific page."""
        if page_num < 1 or page_num > self.num_pages:
            raise IndexError(f"Page number {page_num} out of range (1..{self.num_pages})")
        if page_num not in self._cached_pages:
            self._cached_pages[page_num] = self.parser.parse_page(page_num)
        return self._cached_pages[page_num]

    def get_pages(self, page_range: Optional[List[int]] = None) -> List[Page]:
        """Get a list of parsed pages."""
        if page_range is None:
            page_nums = list(range(1, self.num_pages + 1))
        else:
            page_nums = [p for p in page_range if 1 <= p <= self.num_pages]
        return [self.get_page(p) for p in page_nums]

    def read(
        self,
        pages: Optional[List[int]] = None,
        continuous: bool = True,
        dehyphen: bool = True,
    ) -> str:
        """Read document or page range as formatted Markdown."""
        parsed_pages = self.get_pages(pages)
        return DocumentFormatter.document_to_markdown(
            parsed_pages, continuous=continuous, dehyphen=dehyphen
        )

    def get_outline(self) -> List[Section]:
        """Extract table of contents / outline sections from the document."""
        sections: List[Section] = []
        for p in range(1, self.num_pages + 1):
            page = self.get_page(p)
            for b in page.blocks:
                if b.block_type in ("title", "heading"):
                    text = b.text.replace("\n", " ").strip()
                    level = 1 if b.block_type == "title" else b.heading_level
                    sections.append(Section(title=text, level=level, page_num=p))
        return sections

    def get_section(self, section_query: str) -> str:
        """Extract a specific section by title or number."""
        query_norm = section_query.strip().lower()
        outline = self.get_outline()

        # Find matching section
        target_idx = -1
        for i, sec in enumerate(outline):
            title_norm = sec.title.lower()
            if query_norm in title_norm or re.match(rf"^{re.escape(query_norm)}[\.\s]", title_norm):
                target_idx = i
                break

        if target_idx == -1:
            available = [f"- {s.title} (Page {s.page_num})" for s in outline]
            return f"Section '{section_query}' not found.\nAvailable sections:\n" + "\n".join(available)

        target_sec = outline[target_idx]
        # Next section of equal or higher hierarchy (same or smaller level number)
        next_sec = next(
            (s for s in outline[target_idx + 1 :] if s.level <= target_sec.level),
            None,
        )

        start_page = target_sec.page_num
        end_page = next_sec.page_num if next_sec else self.num_pages

        content_blocks = []
        started = False

        for p in range(start_page, end_page + 1):
            page = self.get_page(p)
            for b in page.blocks:
                b_text = b.text.replace("\n", " ").strip()
                if not started:
                    if target_sec.title.lower() in b_text.lower():
                        started = True
                        content_blocks.append(DocumentFormatter.format_block(b))
                else:
                    if next_sec and next_sec.title.lower() in b_text.lower():
                        # Reached next section
                        return "\n\n".join(content_blocks)
                    content_blocks.append(DocumentFormatter.format_block(b))

        return "\n\n".join(content_blocks)

    def search(self, query: str, context_lines: int = 2) -> List[Dict[str, Any]]:
        """Search for keywords or regex across all pages, returning context snippets."""
        pattern = re.compile(re.escape(query), re.I)
        results = []

        for p in range(1, self.num_pages + 1):
            page = self.get_page(p)
            for b_idx, b in enumerate(page.blocks):
                lines = [l.text for l in b.lines]
                for l_idx, line in enumerate(lines):
                    if pattern.search(line):
                        c_start = max(0, l_idx - context_lines)
                        c_end = min(len(lines), l_idx + context_lines + 1)
                        snippet = " ".join(lines[c_start:c_end])
                        results.append({
                            "page": p,
                            "block_type": b.block_type,
                            "matching_line": line,
                            "snippet": snippet,
                        })
        return results

    def list_figures_and_tables(self) -> List[Dict[str, Any]]:
        """Extract all figures and tables with captions, labels, and bounding boxes."""
        items = []
        for p in range(1, self.num_pages + 1):
            page = self.get_page(p)
            for b in page.blocks:
                if b.block_type == "caption":
                    items.append({
                        "page": p,
                        "label": b.caption_label,
                        "caption": b.text.replace("\n", " "),
                        "bbox": asdict(b.bbox),
                    })
        return items

    def render_page(
        self,
        page_num: int,
        dpi: int = 150,
        output_path: Optional[str | Path] = None,
        crop: Optional[Tuple[int, int, int, int]] = None,
    ) -> Path:
        """
        Render a page or cropped area to a PNG image using pdftoppm.
        Allows the agent to visually inspect figures or formulas with read_image.
        """
        if output_path is None:
            output_path = Path(f"page_{page_num}.png")
        else:
            output_path = Path(output_path)

        prefix = output_path.with_suffix("")
        cmd = [
            "pdftoppm",
            "-png",
            "-r", str(dpi),
            "-f", str(page_num),
            "-l", str(page_num),
        ]
        if crop:
            x, y, w, h = crop
            cmd.extend(["-x", str(x), "-y", str(y), "-W", str(w), "-H", str(h)])

        cmd.extend([str(self.pdf_path), str(prefix)])

        subprocess.run(cmd, check=True)

        # pdftoppm appends page numbers like prefix-1.png or prefix-01.png
        possible = [
            output_path,
            Path(f"{prefix}-{page_num}.png"),
            Path(f"{prefix}-{page_num:02d}.png"),
            Path(f"{prefix}-{page_num:03d}.png"),
        ]
        for candidate in possible:
            if candidate.exists():
                if candidate != output_path:
                    candidate.rename(output_path)
                return output_path

        raise FileNotFoundError(f"pdftoppm output image not found for prefix {prefix}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_page_range(spec: str) -> List[int]:
    """Parse a page range string like '1-3', '2', '1,3,5' into a sorted list of ints."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.update(range(int(start), int(end) + 1))
        elif part:
            pages.add(int(part))
    return sorted(pages)


def main():
    check_poppler_installed()

    parser = argparse.ArgumentParser(
        description="Layout-aware Poppler PDF reader for scientific papers and multi-column documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read entire paper in clean Markdown
  python3 pdf_reader.py read paper.pdf

  # Read pages 1 to 3
  python3 pdf_reader.py read paper.pdf --pages 1-3

  # Extract Table of Contents / Outline
  python3 pdf_reader.py outline paper.pdf

  # Read a specific section
  python3 pdf_reader.py section paper.pdf --section "Related Work"

  # Search for a term with context
  python3 pdf_reader.py search paper.pdf --query "residual mapping"

  # List all figures and tables
  python3 pdf_reader.py figures paper.pdf

  # Render a page to PNG for visual inspection
  python3 pdf_reader.py render paper.pdf --page 2 --output page2.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    # Command: read
    p_read = subparsers.add_parser("read", help="Read document or page range in Markdown")
    p_read.add_argument("pdf", help="Path to PDF file")
    p_read.add_argument("-p", "--pages", help="Page range to read, e.g. '1-3' or '2' or '1,3,5'")
    p_read.add_argument("--no-continuous", action="store_true", help="Keep separate page banners instead of continuous text")
    p_read.add_argument("--no-dehyphen", action="store_true", help="Do not dehyphenate split words")
    p_read.add_argument("-o", "--output", help="Save markdown output to file instead of stdout")

    # Command: outline / toc
    p_outline = subparsers.add_parser("outline", aliases=["toc"], help="Extract document outline / sections")
    p_outline.add_argument("pdf", help="Path to PDF file")
    p_outline.add_argument("--json", action="store_true", help="Output as JSON")

    # Command: section
    p_section = subparsers.add_parser("section", help="Extract a specific section by name or number")
    p_section.add_argument("pdf", help="Path to PDF file")
    p_section.add_argument("-s", "--section", required=True, help="Section title or number, e.g. 'Introduction' or '3'")
    p_section.add_argument("-o", "--output", help="Save section markdown to file")

    # Command: search
    p_search = subparsers.add_parser("search", help="Search for keyword or regex in paper")
    p_search.add_argument("pdf", help="Path to PDF file")
    p_search.add_argument("-q", "--query", required=True, help="Text query to search")
    p_search.add_argument("-c", "--context", type=int, default=2, help="Context lines surrounding matches")

    # Command: figures
    p_figs = subparsers.add_parser("figures", help="List figures and tables with captions")
    p_figs.add_argument("pdf", help="Path to PDF file")
    p_figs.add_argument("--json", action="store_true", help="Output as JSON")

    # Command: info
    p_info = subparsers.add_parser("info", help="Display PDF metadata and layout analysis")
    p_info.add_argument("pdf", help="Path to PDF file")
    p_info.add_argument("--json", action="store_true", help="Output as JSON")

    # Command: render
    p_render = subparsers.add_parser("render", help="Render page or region to PNG")
    p_render.add_argument("pdf", help="Path to PDF file")
    p_render.add_argument("-p", "--page", type=int, default=1, help="Page number to render (default: 1)")
    p_render.add_argument("-r", "--dpi", type=int, default=150, help="Image resolution DPI (default: 150)")
    p_render.add_argument("-o", "--output", default="page.png", help="Output PNG file path (default: page.png)")
    p_render.add_argument("--crop", help="Crop box: x,y,w,h (in points)")

    # Command: json
    p_json = subparsers.add_parser("json", help="Dump entire document structure as JSON")
    p_json.add_argument("pdf", help="Path to PDF file")
    p_json.add_argument("-p", "--pages", help="Page range")
    p_json.add_argument("-o", "--output", help="Save JSON to file")

    args = parser.parse_args()
    reader = PDFReader(args.pdf)

    if args.command == "read":
        page_range = parse_page_range(args.pages) if args.pages else None
        md = reader.read(
            pages=page_range,
            continuous=not args.no_continuous,
            dehyphen=not args.no_dehyphen,
        )
        if args.output:
            Path(args.output).write_text(md, encoding="utf-8")
            print(f"Saved Markdown to {args.output}")
        else:
            print(md)

    elif args.command in ("outline", "toc"):
        outline = reader.get_outline()
        if args.json:
            print(json.dumps([asdict(s) for s in outline], indent=2))
        else:
            print(f"# Outline for {Path(args.pdf).name}\n")
            for sec in outline:
                indent = "  " * (sec.level - 1)
                print(f"{indent}- **{sec.title}** (Page {sec.page_num})")

    elif args.command == "section":
        text = reader.get_section(args.section)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Saved Section to {args.output}")
        else:
            print(text)

    elif args.command == "search":
        matches = reader.search(args.query, context_lines=args.context)
        print(f"Found {len(matches)} match(es) for '{args.query}':\n")
        for m in matches:
            print(f"--- Page {m['page']} [{m['block_type']}] ---")
            print(f"Snippet: ... {m['snippet']} ...\n")

    elif args.command == "figures":
        items = reader.list_figures_and_tables()
        if args.json:
            print(json.dumps(items, indent=2))
        else:
            print(f"# Figures & Tables in {Path(args.pdf).name}\n")
            for item in items:
                print(f"- **{item['label']}** (Page {item['page']}): {item['caption']}")

    elif args.command == "info":
        info = reader.get_info()
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"# Document Info: {Path(args.pdf).name}\n")
            for k, v in info.items():
                print(f"- **{k}**: {v}")

    elif args.command == "render":
        crop_tuple = None
        if args.crop:
            parts = [int(x.strip()) for x in args.crop.split(",")]
            if len(parts) == 4:
                crop_tuple = (parts[0], parts[1], parts[2], parts[3])
        out_path = reader.render_page(
            page_num=args.page,
            dpi=args.dpi,
            output_path=args.output,
            crop=crop_tuple,
        )
        print(f"Rendered Page {args.page} to image: {out_path}")

    elif args.command == "json":
        page_range = parse_page_range(args.pages) if args.pages else None
        pages = reader.get_pages(page_range)
        data = {
            "info": reader.get_info(),
            "outline": [asdict(s) for s in reader.get_outline()],
            "pages": [
                {
                    "page_num": p.page_num,
                    "width": p.width,
                    "height": p.height,
                    "is_two_column": p.is_two_column,
                    "header": p.header,
                    "footer": p.footer,
                    "margin_notes": p.margin_notes,
                    "blocks": [
                        {
                            "type": b.block_type,
                            "column": b.column_idx,
                            "heading_level": b.heading_level,
                            "caption_label": b.caption_label,
                            "bbox": asdict(b.bbox),
                            "text": b.text,
                        }
                        for b in p.blocks
                    ],
                }
                for p in pages
            ],
        }
        json_str = json.dumps(data, indent=2)
        if args.output:
            Path(args.output).write_text(json_str, encoding="utf-8")
            print(f"Saved JSON to {args.output}")
        else:
            print(json_str)


if __name__ == "__main__":
    main()
