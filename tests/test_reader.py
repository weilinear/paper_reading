#!/usr/bin/env python3
"""
Unit and integration tests for pdf_reader.py.
"""

import json
import os
import unittest
from pathlib import Path

from pdf_reader import (
    PDFReader,
    dehyphenate_lines,
    is_heading_text,
    parse_page_range,
)

RESNET_PDF = Path("test_resnet.pdf")
PROCEDURAL_PDF = Path("OriginalPapers/Procedural_Graphs_Self-Evolving_Execution_Structures_for_LLM_Agents.pdf")


class TestPDFReaderUnit(unittest.TestCase):
    def test_dehyphenation(self):
        # Case 1: Syllable break in word (should join without hyphen)
        self.assertEqual(
            dehyphenate_lines("a series of break-", "throughs for image classification"),
            "a series of breakthroughs for image classification",
        )
        self.assertEqual(
            dehyphenate_lines("still having lower complex-", "ity. An ensemble of these"),
            "still having lower complexity. An ensemble of these",
        )
        # Case 2: Numbered compound (should preserve hyphen)
        self.assertEqual(
            dehyphenate_lines("evaluate 18-layer and 34-", "layer residual nets"),
            "evaluate 18-layer and 34-layer residual nets",
        )
        # Case 3: Common prefix (should preserve hyphen)
        self.assertEqual(
            dehyphenate_lines("Many other non-", "trivial visual recognition tasks"),
            "Many other non-trivial visual recognition tasks",
        )

    def test_page_range_parsing(self):
        self.assertEqual(parse_page_range("1-3"), [1, 2, 3])
        self.assertEqual(parse_page_range("2"), [2])
        self.assertEqual(parse_page_range("1,3,5"), [1, 3, 5])
        self.assertEqual(parse_page_range("1-2, 5"), [1, 2, 5])

    def test_heading_classification(self):
        # Real headings
        is_h, lvl = is_heading_text("1. Introduction")
        self.assertTrue(is_h)
        self.assertEqual(lvl, 1)

        is_h, lvl = is_heading_text("3.2. Identity Mapping by Shortcuts")
        self.assertTrue(is_h)
        self.assertEqual(lvl, 2)

        is_h, lvl = is_heading_text("Abstract")
        self.assertTrue(is_h)
        self.assertEqual(lvl, 1)

        is_h, lvl = is_heading_text("References")
        self.assertTrue(is_h)
        self.assertEqual(lvl, 1)

        is_h, lvl = is_heading_text("A. Object Detection Baselines")
        self.assertTrue(is_h)
        self.assertEqual(lvl, 2)

        # Diagram / plot / table non-headings
        self.assertFalse(is_heading_text("training error (%)")[0])
        self.assertFalse(is_heading_text("34-layer")[0])
        self.assertFalse(is_heading_text("method")[0])
        self.assertFalse(is_heading_text("Figure 2. Residual learning")[0])
        self.assertFalse(is_heading_text("2. Log")[0])


class TestPDFReaderResNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESNET_PDF.exists():
            raise unittest.SkipTest(f"{RESNET_PDF} not found")
        cls.reader = PDFReader(RESNET_PDF)

    def test_layout_detection(self):
        info = self.reader.get_info()
        self.assertEqual(info["Pages"], 12)
        self.assertEqual(info["primary_layout"], "2-column")

    def test_page_2_reading_order(self):
        """
        Critical test: Verifies that on Page 2, Left Column text
        ('are comparably good...') precedes Right Column text
        ('ImageNet test set...' and '2. Related Work').
        """
        page2_md = self.reader.read(pages=[2])
        idx_left_col = page2_md.find("are comparably good or better than the constructed solution")
        idx_right_col = page2_md.find("ImageNet test set, and won the 1st place")
        idx_sec2 = page2_md.find("# 2. Related Work")

        self.assertNotEqual(idx_left_col, -1, "Left column text not found on page 2")
        self.assertNotEqual(idx_right_col, -1, "Right column text not found on page 2")
        self.assertNotEqual(idx_sec2, -1, "Section 2 heading not found on page 2")

        # Left column MUST be read BEFORE right column!
        self.assertLess(
            idx_left_col,
            idx_right_col,
            "Two-column reading order failed: Right column appeared before Left column!",
        )
        self.assertLess(
            idx_right_col,
            idx_sec2,
            "End of section 1 should appear before Section 2 heading!",
        )

    def test_outline_extraction(self):
        outline = self.reader.get_outline()
        titles = [s.title for s in outline]

        self.assertIn("1. Introduction", titles)
        self.assertIn("2. Related Work", titles)
        self.assertIn("3. Deep Residual Learning", titles)
        self.assertIn("3.1. Residual Learning", titles)
        self.assertIn("4. Experiments", titles)
        self.assertIn("References", titles)

        # Diagram labels should NOT be in outline
        self.assertNotIn("34-layer", titles)
        self.assertNotIn("relu", titles)
        self.assertNotIn("training error (%)", titles)

    def test_figures_and_tables(self):
        items = self.reader.list_figures_and_tables()
        labels = [item["label"] for item in items]

        self.assertIn("Figure 1", labels)
        self.assertIn("Figure 2", labels)
        self.assertIn("Table 1", labels)
        self.assertIn("Figure 4", labels)
        self.assertIn("Table 2", labels)

    def test_section_extraction(self):
        sec_text = self.reader.get_section("Related Work")
        self.assertIn("2. Related Work", sec_text)
        self.assertIn("Residual Representations.", sec_text)
        self.assertIn("Shortcut Connections.", sec_text)
        # Should not include Section 3
        self.assertNotIn("3. Deep Residual Learning", sec_text)

    def test_search(self):
        results = self.reader.search("residual mapping")
        self.assertTrue(len(results) >= 1)
        self.assertTrue(any(r["page"] in (1, 2, 3) for r in results))

    def test_render_page(self):
        out_img = Path("test_p1.png")
        try:
            rendered = self.reader.render_page(1, dpi=72, output_path=out_img)
            self.assertTrue(rendered.exists())
            self.assertTrue(rendered.stat().st_size > 1000)
        finally:
            if out_img.exists():
                out_img.unlink()


class TestPDFReaderProceduralGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROCEDURAL_PDF.exists():
            raise unittest.SkipTest(f"{PROCEDURAL_PDF} not found")
        cls.reader = PDFReader(PROCEDURAL_PDF)

    def test_layout_detection(self):
        info = self.reader.get_info()
        self.assertEqual(info["Pages"], 36)
        self.assertEqual(info["primary_layout"], "1-column")

    def test_outline(self):
        outline = self.reader.get_outline()
        titles = [s.title for s in outline]

        self.assertIn("1. Introduction", titles)
        self.assertIn("2. Related Work", titles)
        self.assertIn("3. The Procedural Graph Framework", titles)
        self.assertIn("4. Experimental Setup", titles)
        self.assertIn("5. Results", titles)
        self.assertIn("6. Conclusion", titles)
        self.assertIn("References", titles)


if __name__ == "__main__":
    unittest.main()
