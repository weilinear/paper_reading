# Scientific Paper Reading & Agent Research (`paper_reading`)

A unified suite for scientific paper reading, layout-aware multi-column parsing, and reproducible evaluations of agent architectures (Procedural Graphs, MemoryBank, ReAct, and multi-hop reasoning).

---

## Interactive Google Colab Demos

All demos are hosted on Google Colab and can be launched directly in your browser with a single click:

| Demo Notebook | Description | Launch in Colab |
| :--- | :--- | :---: |
| **01. Layout-Aware Paper Reader** | Two-column scientific paper reader, outline extraction & figure rendering (Poppler) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/01_paper_reader_demo.ipynb) |
| **02. Procedural Graphs on HotpotQA** | Multi-hop QA benchmark comparing Vanilla ReAct vs Procedural Graph (Mode 1 & Mode 5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/02_hotpotqa_procedural_graph.ipynb) |
| **03. MemoryBank & Ebbinghaus Decay** | Interactive simulation of Ebbinghaus forgetting curves, spacing effect & SiliconFriend | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weilinear/paper_reading/blob/main/notebooks/03_memory_bank_demo.ipynb) |

You can also manage and open Colab demos from your terminal using the included CLI:
```bash
./bin/colab list        # List all available demos and their Colab links
./bin/colab open 1      # Open Demo 1 directly in your browser
./bin/colab open 2      # Open Demo 2 in your browser
./bin/colab open 3      # Open Demo 3 in your browser
```

---

## 1. The Multi-Column Paper Problem & Solution

### Why standard PDF extraction breaks:
1. **Naive stream extraction (`pdftotext paper.pdf`)**: Poppler's default layout heuristic frequently jumps across columns when figures, tables, or margin notes exist. For example, text from the top of the right column is often emitted *before* the bottom of the left column.
2. **Physical layout extraction (`pdftotext -layout`)**: Side-by-side columns are rendered on the same horizontal line with whitespace gaps. When processed line-by-line by an LLM or script, fragments of column 1 and column 2 are merged into the same line, destroying paragraph coherence.
3. **TeX hyphenation**: Narrow columns frequently split words across line breaks (e.g., `degrada-` \n `tion`).
4. **Headers, footers, & watermarks**: arXiv margin stamps (e.g. `arXiv:1512.03385v1`) and running headers interrupt sentences crossing page boundaries.

### How `pdf_reader.py` solves it:
- **XML Bounding-Box Layout Parsing**: Uses `pdftotext -bbox-layout` to extract exact physical coordinates for every block, line, and word.
- **Margin & Noise Filtering**: Automatically detects and separates margin watermarks, running headers, and page number footers.
- **Vertical Band Slicing**:
  - Automatically identifies full-width spanning blocks (Title/Author frontmatter, wide tables, wide figures).
  - Partitions each page into horizontal bands.
  - Within two-column bands, processes **Column 1 (Left) strictly from top-to-bottom**, then **Column 2 (Right) strictly from top-to-bottom**.
- **Smart Dehyphenation**: Reconstitutes TeX-hyphenated words while preserving legitimate hyphens in numbers and compound words (`34-layer`, `non-trivial`, `end-to-end`).
- **Semantic Structure**: Automatically recognizes title, abstract, section headings (`#`, `##`, `###`), bold run-in headers, and figure/table captions (`> **Figure 1.** ...`).
- **Visual Inspection via `pdftoppm`**: Renders any page or cropped diagram to high-resolution PNG so multimodal LLMs can inspect complex diagrams, plots, and architectures using `read_image`.

---

## 2. Installation & Prerequisites

Requires `poppler` installed on your system:
```bash
# macOS (Homebrew)
brew install poppler

# Linux (Debian/Ubuntu)
apt-get install poppler-utils
```

Zero additional Python dependencies required (uses standard library `subprocess`, `xml`, `re`, `json`, `pathlib`).

---

## 3. CLI Usage

The tool is available as `./pdf_reader.py` or via convenience links `./bin/read-paper` and `./bin/pdf-read`.

### Read Entire Paper or Specific Pages
```bash
# Read pages 1 to 2 formatted as clean Markdown
./bin/read-paper read test_resnet.pdf --pages 1-2

# Read full paper and save to Markdown
./bin/read-paper read test_resnet.pdf -o resnet.md

# Read specific pages without continuous paragraph merging across page boundaries
./bin/read-paper read test_resnet.pdf --pages 1,3,5 --no-continuous
```

### Table of Contents / Outline
Extracts the document outline with section numbers, titles, hierarchical levels, and page numbers:
```bash
./bin/read-paper outline test_resnet.pdf
```
Output:
```markdown
# Outline for test_resnet.pdf

- **Deep Residual Learning for Image Recognition** (Page 1)
- **Abstract** (Page 1)
- **1. Introduction** (Page 1)
- **2. Related Work** (Page 2)
- **3. Deep Residual Learning** (Page 3)
  - **3.1. Residual Learning** (Page 3)
  - **3.2. Identity Mapping by Shortcuts** (Page 3)
  - **3.3. Network Architectures** (Page 3)
  - **3.4. Implementation** (Page 4)
- **4. Experiments** (Page 4)
  - **4.1. ImageNet Classification** (Page 4)
  - **4.2. CIFAR-10 and Analysis** (Page 7)
  - **4.3. Object Detection on PASCAL and MS COCO** (Page 8)
- **References** (Page 9)
  - **A. Object Detection Baselines** (Page 10)
  - **B. Object Detection Improvements** (Page 10)
  - **C. ImageNet Localization** (Page 12)
```

### Extract a Specific Section
Allows an agent or user to immediately read a single section without loading thousands of unnecessary tokens:
```bash
# Read Section 2 (Related Work)
./bin/read-paper section test_resnet.pdf --section "Related Work"

# Read Subsection 4.1 (ImageNet Classification)
./bin/read-paper section test_resnet.pdf --section "4.1"
```

### Keyword Search with Context
Search for phrases across the document, displaying page numbers, block types, and surrounding lines:
```bash
./bin/read-paper search test_resnet.pdf --query "bottleneck"
```

### List All Figures & Tables
Extracts all detected figure and table captions with page numbers:
```bash
./bin/read-paper figures test_resnet.pdf
```

### Render Pages / Figures to Image (for Multimodal Agents)
Render a page or crop area to a PNG image for visual inspection using `read_image`:
```bash
# Render Page 2 at 150 DPI
./bin/read-paper render test_resnet.pdf --page 2 --output resnet_p2.png

# Render with custom DPI
./bin/read-paper render test_resnet.pdf --page 5 --dpi 200 --output table1.png
```

### Document Metadata & Layout Info
```bash
./bin/read-paper info test_resnet.pdf
```

---

## 4. Python API Usage

```python
from pdf_reader import PDFReader

reader = PDFReader("test_resnet.pdf")

# 1. Metadata & detected column layout
info = reader.get_info()
print("Primary layout:", info["primary_layout"])  # "2-column" or "1-column"

# 2. Extract outline
outline = reader.get_outline()
for sec in outline:
    print(f"Page {sec.page_num}: {sec.title} (Level {sec.level})")

# 3. Read specific section
text = reader.get_section("3.1")
print(text)

# 4. Search
results = reader.search("residual function")
for r in results:
    print(f"Page {r['page']}: {r['snippet']}")

# 5. Render page to image
img_path = reader.render_page(page_num=2, dpi=150, output_path="page2.png")
```

---

## 5. Verification & Test Suite

Run the full automated test suite verifying layout detection, two-column reading order preservation, dehyphenation, and section extraction:

```bash
python3 -m unittest discover -s tests
```

---

## 6. Paper Reproduction: Procedural Graphs on HotpotQA

The repository includes a clean, modular evaluation framework under `benchmarks/` and `agents/` designed to be easily reused for subsequent research papers without over-engineering:

- `benchmarks/base.py`: Standard interfaces for `Task`, `Benchmark`, and metrics (`Exact Match`, `F1 score`).
- `benchmarks/hotpotqa/`: Complete multi-hop QA environment supporting `Search[entity]`, `Lookup[keyword]`, and `Finish[answer]`.
- `agents/react.py`: Standard ReAct solver loop with support for situational guidance injection.
- `agents/procedural_graph/`: Complete implementation of the Procedural Graph framework:
  - `graph.py`: The `(procedure, relation, procedure)` multigraph with `(condition, guidance, pitfalls)`.
  - `guidance.py`: Online $h$-hop local neighborhood extraction and situational guidance generation.
  - `refiner.py`: Offline feedback-driven self-evolution (`Add`, `Delete`, `Update`) with validation gating and rejection memory.

### Running HotpotQA Evaluation
```bash
# Run using convenience script
./bin/run-eval --split mini

# Or run directly
python3 evaluations/run_hotpotqa.py --split mini --mode all
```

Output:
```text
====================================================================
Method / Configuration             | Ans EM (%) | Ans F1 (%) | Avg Steps
--------------------------------------------------------------------
Vanilla ReAct                      | 60.00      | 60.00      | 2.60     
PG (Mode 1: Expert Prior)          | 100.00     | 100.00     | 3.00     
PG (Mode 5: Scratch + Evolution)   | 100.00     | 100.00     | 3.00     
====================================================================
```

---

## 7. Paper Reproduction: MemoryBank (Zhong et al., AAAI 2024)

An implementation of the **MemoryBank** long-term memory mechanism under `memory_bank/`:
- **Storage**: Dialogue log, hierarchical event summaries, and dynamic user personality portrait.
- **Ebbinghaus Decay**: Retention modeled as $R = e^{-t / S}$ with the spacing effect ($S \leftarrow S + 1, t \leftarrow 0$ upon recall).
- **Retrieval & Augmentation**: Semantic vector search combined with prompt assembly for AI companions (SiliconFriend).

### Run the Interactive MemoryBank Walkthrough
```bash
# Run interactive multi-day demo
./bin/demo-memory-bank

# Or directly via Python
python3 memory_bank/toy_demo.py
```
