"""
Generate Chapter 5 — Implementation — as a .docx file.
Run: python generate_thesis_chapter.py
Output: Chapter5_Implementation.docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# ── Page margins ─────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ── Helper: set font on a run ─────────────────────────────────────────────────
def style_run(run, size=11, bold=False, italic=False, color=None, font="Times New Roman"):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

# ── Helper: add heading ───────────────────────────────────────────────────────
def heading(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(14)
        elif level == 3:
            run.font.size = Pt(12)
    return p

# ── Helper: add body paragraph ────────────────────────────────────────────────
def body(text, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = Pt(22)
    run = p.add_run(text)
    style_run(run)
    return p

# ── Helper: add bullet ────────────────────────────────────────────────────────
def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    style_run(run)
    return p

# ── Helper: add a simple table ────────────────────────────────────────────────
def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    # Header row
    hdr_cells = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for para in hdr_cells[i].paragraphs:
            for run in para.runs:
                run.font.bold = True
                run.font.name = "Times New Roman"
                run.font.size = Pt(10)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        hdr_cells[i]._tc.get_or_add_tcPr().append(
            _shade_cell("D9D9D9")
        )
    # Data rows
    for r_idx, row_data in enumerate(rows):
        row_cells = t.rows[r_idx + 1].cells
        for c_idx, cell_text in enumerate(row_data):
            row_cells[c_idx].text = cell_text
            for para in row_cells[c_idx].paragraphs:
                for run in para.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(10)
    if col_widths:
        for row in t.rows:
            for i, cell in enumerate(row.cells):
                cell.width = Inches(col_widths[i])
    doc.add_paragraph()  # spacing after table
    return t

def _shade_cell(hex_color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    return shd

def add_caption(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    style_run(run, size=9, italic=True)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

heading("5. Implementation", level=1)

body(
    "This chapter describes the practical implementation of the academic document plagiarism "
    "detection system developed as part of this thesis. The system is structured as a sequential "
    "multi-stage pipeline that transforms raw text corpora into ranked, evidence-backed plagiarism "
    "verdicts. Each stage has a well-defined input and output contract, enabling independent "
    "development, testing, and replacement of individual components. The pipeline is composed of "
    "six major stages: a frontend application for interactive exploration, data gathering and "
    "ground-truth parsing, text preprocessing, source document retrieval, text alignment via LLM "
    "re-ranking, and post-processing analytics. Figure 5.1 provides a high-level overview of the "
    "end-to-end architecture.",
    indent=True
)

body(
    "The implementation targets the PAN 2011 Plagiarism Corpus as the primary evaluation "
    "benchmark, supplemented by a custom curated dataset. All components are implemented in "
    "Python 3.12 and rely on open-source libraries. Heavy computation (embedding inference) is "
    "offloaded to a ROCm-enabled GPU via Docker, while all other stages run on CPU.",
    indent=True
)

# ─────────────────────────────────────────────────────────────────────────────
heading("5.1. Frontend Application", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "A Streamlit-based web application was developed to serve as the interactive frontend for "
    "the complete plagiarism detection pipeline. The application allows a user to select any "
    "suspicious document from the pre-processed corpus, configure pipeline parameters, trigger "
    "execution, and inspect the results — all through a browser-based interface without requiring "
    "any command-line interaction.",
    indent=True
)

body(
    "The application is launched from the project root with the command "
    "streamlit run scripts/final/07_streamlit_app/app.py and listens on the default Streamlit "
    "port. On startup, it sets the working directory to scripts/final/ so that all relative "
    "paths used by the retrieval modules resolve correctly regardless of where the user launched "
    "the process.",
    indent=True
)

body("The user interface is divided into two areas:", indent=True)

body(
    "Sidebar controls. The sidebar exposes all configurable pipeline parameters. The user "
    "selects a suspicious document from a drop-down populated dynamically from the preprocessed "
    "chunk Parquet file. Additional controls include the number of fused candidate source "
    "documents to forward to the LLM (top-N, between 5 and 50), a toggle to re-run the GPU "
    "embedding lookup via Docker or reuse cached results, the Ollama model name for the LLM "
    "re-ranking stage, and the number of chunk pairs per source document shown to the LLM "
    "(between 1 and 5). All parameters are persisted in Streamlit session state across UI "
    "interactions.",
    indent=True
)

body(
    "Main panel. The main panel shows live progress for each pipeline stage using Streamlit "
    "status widgets that update in real time. After completion, results are presented as: "
    "(1) a row of five metric cards showing the LLM plagiarism likelihood score and verdict for "
    "each of the top-5 candidate source documents; (2) expandable reasoning cards with the full "
    "LLM explanation for each candidate; (3) a full ranked score table for all evaluated "
    "candidates; and (4) the pre-LLM fusion ranking table showing per-branch scores.",
    indent=True
)

body(
    "Caching is implemented at two levels. The suspicious document ID list is cached with "
    "Streamlit's @st.cache_data decorator and loaded only once per session. The fused top-N "
    "result is written to a Parquet file on disk after each pipeline run and displayed as a "
    "cached preview on subsequent app starts, enabling review of previous results without "
    "re-running the full pipeline.",
    indent=True
)

add_table(
    headers=["Control", "Description"],
    rows=[
        ["Suspicious document", "Drop-down of all document IDs from the preprocessed corpus"],
        ["Fused top-N candidates", "Number of fused source documents forwarded to the LLM (5–50)"],
        ["Re-run GPU embeddings", "Triggers Docker ROCm container for fresh embeddings; otherwise uses cache"],
        ["Ollama model", "Any locally available Ollama model (default: gemma4:e4b)"],
        ["Chunk pairs per source doc", "How many suspicious↔source passage pairs the LLM evaluates per candidate (1–5)"],
    ],
    col_widths=[2.2, 4.0]
)
add_caption("Table 5.1 — Configurable parameters exposed by the Streamlit sidebar.")

# ─────────────────────────────────────────────────────────────────────────────
heading("5.2. Data Gathering", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "Two corpora were used in this work: the PAN 2011 Plagiarism Corpus, which provides a "
    "large-scale benchmark with expert-annotated ground truth, and a smaller custom curated "
    "dataset assembled to test system behaviour on document types not well represented in PAN.",
    indent=True
)

heading("5.2.1. PAN 2011 Corpus", level=3)

body(
    "The PAN 2011 Plagiarism Corpus (PAN-PC-11) is the primary benchmark dataset used in this "
    "thesis. It was originally produced for the PAN 2011 shared task on plagiarism detection "
    "and has become one of the standard benchmarks in the field. The corpus is divided into two "
    "collections: a set of source documents and a set of suspicious documents. Each suspicious "
    "document may contain one or more passages that were derived from one or more source "
    "documents by applying a defined obfuscation strategy.",
    indent=True
)

body(
    "Annotations are provided in XML files, one per suspicious document. Each XML file lists "
    "the plagiarised features, where each feature records the character offset and length of the "
    "plagiarised span in the suspicious document, the corresponding offset and length in the "
    "source document, and metadata including the plagiarism type and obfuscation strategy. The "
    "corpus covers several obfuscation categories:",
    indent=True
)

bullet("No obfuscation — direct copy-paste of source text")
bullet("Random obfuscation — word-level substitutions with synonyms or randomly chosen words")
bullet("Translated obfuscation — cross-language plagiarism, translated into English")
bullet("Simulated manual paraphrasing — human-written paraphrase of the original passage")

body(
    "The corpus is organised into numbered part folders (part1 through partN) under both the "
    "source-document and suspicious-document directories. A key structural detail is that "
    "a source document referenced by a suspicious document in partX may reside in a different "
    "part folder partY. This cross-part reference is resolved at parse time by building a "
    "filesystem lookup table that maps every source filename to its actual part folder before "
    "any XML file is opened.",
    indent=True
)

body(
    "The ground truth is extracted by the XML analytical parser described in Section 5.3 and "
    "stored in two Parquet tables: a span-level table with one row per plagiarised passage pair, "
    "and a document-level validation table with one row per (suspicious document, source "
    "document) pair. The latter is used throughout evaluation to determine whether a retrieved "
    "source document is a true positive.",
    indent=True
)

heading("5.2.2. Custom Curated Dataset", level=3)

body(
    "A supplementary custom dataset was assembled to evaluate the system on document types and "
    "writing styles not represented in PAN 2011. This dataset consists of academic texts, "
    "technical reports, and online articles for which known paraphrased or adapted versions "
    "were either found or manually created. Unlike PAN 2011, this dataset does not carry "
    "formal character-offset annotations; it is used for qualitative analysis of system "
    "behaviour on out-of-distribution inputs and for demonstrating the pipeline through the "
    "Streamlit frontend.",
    indent=True
)

body(
    "The custom dataset is processed through the same preprocessing pipeline as the PAN corpus, "
    "producing compatible Parquet artefacts that can be queried by any of the four retrieval "
    "branches.",
    indent=True
)

# ─────────────────────────────────────────────────────────────────────────────
heading("5.3. Preprocessing", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "Before any retrieval index can be built, the raw text files must be converted into a "
    "structured, normalised representation that is shared across all downstream retrieval "
    "branches. The preprocessing module implements a multi-step pipeline that produces eight "
    "Parquet files from the raw corpus, providing document-level and chunk-level views with "
    "branch-specific text variants.",
    indent=True
)

body("The preprocessing pipeline proceeds in four sequential steps:", indent=True)

body(
    "Step 1 — Text cleaning. Each raw .txt file is read with UTF-8 encoding, falling back to "
    "Latin-1 if UTF-8 decoding fails. The text is then passed through a canonical cleaning "
    "function that performs: Unicode NFKC normalisation to collapse ligatures and non-standard "
    "characters; replacement of typographic quotes and dashes with their ASCII equivalents; "
    "removal of PAN-specific image placeholders and Roman numeral chapter headings; rejoining "
    "of hyphenated line-break splits (e.g. communi-\\ncation becomes communication); and "
    "whitespace collapsing. The resulting clean text is the reference for all subsequent "
    "character offset computations within the pipeline.",
    indent=True
)

body(
    "Step 2 — Chunking. The cleaned text is split into overlapping word-window chunks using a "
    "sliding window approach. The default configuration uses a window of 300 words, a step of "
    "150 words (giving 50% overlap), and a minimum chunk size of 60 words. For each chunk, the "
    "character offsets of its first and last tokens in the cleaned text are recorded as "
    "start_char and end_char. A unique chunk_id of the form {doc_stem}_c{index:04d} is assigned "
    "to each chunk. Overlapping chunks ensure that plagiarised passages that span a window "
    "boundary are captured by at least one chunk, improving recall.",
    indent=True
)

body(
    "Step 3 — Branch-specific normalisation. The canonical chunks are then further normalised "
    "to produce text variants suited to each retrieval branch. For the LSA and ESA branches, "
    "which rely on bag-of-words representations, a more aggressive normalisation is applied: "
    "the text is lowercased, all non-alphanumeric characters are removed, numeric tokens are "
    "replaced with the placeholder NUM, and a minimal set of function-word stopwords is "
    "removed. For the embedding branch, only light normalisation is applied — line-ending "
    "standardisation and whitespace collapsing — so that the transformer model receives "
    "grammatically natural text. The TF-IDF branch reuses the LSA/ESA normalised text.",
    indent=True
)

body(
    "Step 4 — Parquet output. All outputs are written to disk in Apache Parquet format, "
    "streamed in configurable batch sizes (default 250–500 documents per batch) to avoid "
    "loading the entire corpus into RAM. The eight output files are listed in Table 5.2.",
    indent=True
)

add_table(
    headers=["File", "Content"],
    rows=[
        ["source_documents.parquet", "One row per source document with raw and cleaned text"],
        ["suspicious_documents.parquet", "One row per suspicious document (same schema)"],
        ["source_chunks.parquet", "Canonical chunks with char offsets for source documents"],
        ["suspicious_chunks.parquet", "Canonical chunks with char offsets for suspicious documents"],
        ["source_chunks_lsa_esa.parquet", "Chunks with LSA/ESA-normalised text column added"],
        ["suspicious_chunks_lsa_esa.parquet", "Same as above for suspicious documents"],
        ["source_chunks_embeddings.parquet", "Chunks with embedding-normalised text and token count estimate"],
        ["suspicious_chunks_embeddings.parquet", "Same as above for suspicious documents"],
    ],
    col_widths=[2.8, 3.4]
)
add_caption("Table 5.2 — Parquet files produced by the preprocessing pipeline.")

body(
    "An important design constraint is that the doc_id values are constructed as "
    "partX__filename.txt rather than using the bare filename. This prevents identifier "
    "collisions in the PAN corpus, where the same filename can appear in multiple part folders. "
    "The same constraint applies to chunk identifiers, which embed the doc_id stem.",
    indent=True
)

body(
    "It is also important to note that character offsets produced by the preprocessing pipeline "
    "are relative to the cleaned text, not the raw file bytes. The PAN 2011 XML annotations, "
    "by contrast, use raw-file offsets. These two coordinate systems must not be mixed when "
    "comparing detected spans against the ground truth. The ground-truth parser described in "
    "the data gathering section reads the XML offsets and stores them separately from the "
    "pipeline's clean-text offsets.",
    indent=True
)

# ─────────────────────────────────────────────────────────────────────────────
heading("5.4. Source Retrieval", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "The source retrieval stage is responsible for narrowing the candidate set from the entire "
    "source corpus — which may contain tens of thousands of documents — down to a short list of "
    "the most likely source documents for a given suspicious document. This is achieved by "
    "running four independent retrieval branches in parallel conceptually, then fusing their "
    "rankings into a single scored list.",
    indent=True
)

body(
    "Each branch operates at the chunk level: every chunk of the suspicious document is "
    "compared against every chunk of every source document using the branch-specific similarity "
    "function. The resulting chunk-level scores are then aggregated to the document level by "
    "taking the maximum score across all chunk pairs for that source document. The document-level "
    "max score is used rather than the mean because plagiarism is typically local — one strongly "
    "matching passage is a reliable indicator that the source document is the true origin, even "
    "if most other passage pairs show low similarity.",
    indent=True
)

body("The four retrieval branches are described below.", indent=True)

body(
    "TF-IDF (Term Frequency–Inverse Document Frequency). The TF-IDF branch represents each "
    "chunk as a sparse character n-gram vector. A HashingVectorizer is used instead of a "
    "vocabulary-based vectorizer to avoid storing an explicit vocabulary for the full corpus, "
    "which would be prohibitively large. Sublinear TF scaling (1 + log(tf)) is applied before "
    "multiplying by the IDF weights, and vectors are L2-normalised so that the dot product "
    "equals cosine similarity. Because the full source matrix is too large to hold in RAM, it "
    "is stored as a series of compressed sparse NumPy shards on disk. At query time, each shard "
    "is loaded and searched sequentially, and the global top-K results are accumulated across "
    "shards. The TF-IDF branch is most effective for near-copy and lightly edited plagiarism, "
    "where surface character patterns are preserved.",
    indent=True
)

body(
    "LSA (Latent Semantic Analysis). The LSA branch projects chunk representations into a "
    "low-dimensional latent semantic space via Truncated Singular Value Decomposition applied "
    "to a TF-IDF matrix. In this space, chunks that share topical content are close even if "
    "they use different vocabulary, making LSA effective against word-substitution obfuscation. "
    "Suspicious chunks are projected using the pre-fitted vectorizer and SVD model, and cosine "
    "similarity is computed against the stored source LSA vectors. The index fits entirely in "
    "RAM, enabling fast batch querying.",
    indent=True
)

body(
    "ESA (Explicit Semantic Analysis). The ESA branch represents chunks as sparse vectors over "
    "a fixed set of Wikipedia article concepts. Each dimension of the ESA vector corresponds to "
    "the TF-IDF relevance of the chunk to one Wikipedia article. Chunks that discuss the same "
    "topic are thus close in concept space even if their surface form is entirely different. "
    "This makes ESA particularly powerful for detecting semantic plagiarism. Similarity is "
    "computed as sparse cosine similarity between the query ESA vector and all stored source "
    "ESA vectors.",
    indent=True
)

body(
    "Embeddings (Dense Neural Vectors). The embedding branch encodes chunks using "
    "Qwen3-Embedding-0.6B, a 600-million-parameter transformer model fine-tuned for "
    "text similarity. Chunks are encoded into 1024-dimensional L2-normalised dense vectors. "
    "The source corpus vectors are indexed in a FAISS IndexFlatIP structure that supports "
    "exact inner-product search (equivalent to cosine similarity on L2-normalised vectors). "
    "Because the embedding model requires a GPU for acceptable throughput, this branch runs "
    "inside a ROCm Docker container. The resulting top-50 candidate list is written to a "
    "Parquet file and consumed by the fusion step.",
    indent=True
)

body(
    "Score fusion. After all four branches have produced their top-50 document rankings, the "
    "results are fused using a weighted linear combination. For each branch i with weight w_i, "
    "two aggregate scores are computed at the document level: the mean chunk score and the max "
    "chunk score across all chunk pairs. The final score is:",
    indent=True
)

p = doc.add_paragraph()
p.paragraph_format.left_indent = Cm(2.5)
p.paragraph_format.space_after = Pt(6)
run = p.add_run(
    "final_score = 0.30 × Σ(wᵢ × mean_scoreᵢ) + 0.70 × Σ(wᵢ × max_scoreᵢ)"
)
style_run(run, size=11, italic=True)

body(
    "The max component receives a weight of 0.70 because a single strong passage-level match "
    "is a more reliable signal for plagiarism than a high average across all chunks. The "
    "default branch weights are: TF-IDF 10%, ESA 15%, LSA 25%, Embeddings 50%, reflecting "
    "the relative retrieval power of each method as observed during development. Source "
    "documents absent from a branch's top-50 receive a score of zero for that branch.",
    indent=True
)

add_table(
    headers=["Branch", "Weight", "Similarity space", "Strength"],
    rows=[
        ["TF-IDF", "10%", "Character n-gram sparse vectors", "Near-copy and lightly edited text"],
        ["LSA", "25%", "Latent semantic space (SVD)", "Word-substitution obfuscation"],
        ["ESA", "15%", "Wikipedia concept space", "Topical / semantic similarity"],
        ["Embeddings", "50%", "Dense neural vectors (Qwen3-0.6B + FAISS)", "Paraphrase and semantic reuse"],
    ],
    col_widths=[1.2, 0.8, 2.4, 2.0]
)
add_caption("Table 5.3 — Source retrieval branches and their default fusion weights.")

# ─────────────────────────────────────────────────────────────────────────────
heading("5.5. Text Alignment", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "The text alignment stage takes the fused top-N candidate source documents produced by the "
    "retrieval stage and applies an LLM to re-rank them based on direct reading of the actual "
    "passage text. This stage serves as a second-pass filter that transforms an IR-ranked "
    "candidate list into a semantically grounded verdict with an associated confidence score "
    "and natural-language reasoning.",
    indent=True
)

body(
    "The design rationale for this two-stage architecture is that retrieval methods optimise "
    "for recall — they must not miss the true source — while the LLM optimises for precision, "
    "distinguishing true positives from false positives among the retrieved candidates. If the "
    "true source document is not in the top-N after retrieval, the LLM cannot recover it; "
    "therefore, a high retrieval recall at rank N is a prerequisite for this stage to be "
    "effective.",
    indent=True
)

body("The text alignment process proceeds in three steps:", indent=True)

body(
    "Step 1 — Text pair construction. For each candidate source document in the fused top-N, "
    "the top-K chunk pairs by embedding cosine similarity are selected from the full candidate "
    "set. Using embedding-ranked pairs rather than random or sequential chunks ensures the LLM "
    "is presented with the passage evidence most likely to contain plagiarism. The default "
    "configuration uses K=3 pairs per source document. Each chunk is truncated to 600 "
    "characters to keep the total prompt length within a manageable range.",
    indent=True
)

body(
    "Step 2 — LLM scoring. Each source document is scored independently. The LLM receives a "
    "zero-shot prompt containing the K text pairs and is asked to determine whether the "
    "suspicious text appears to have been copied or paraphrased from the source. The prompt "
    "explicitly requests a structured JSON response conforming to a Pydantic schema with three "
    "fields: score (a float between 0 and 1 representing plagiarism likelihood), "
    "is_likely_source (a boolean verdict), and reasoning (a free-text explanation). "
    "The instructor library is used to enforce this output schema, retrying the LLM call if "
    "the output is malformed.",
    indent=True
)

body(
    "Step 3 — Ranking. After all candidates have been scored, the results are sorted by "
    "descending LLM score. The top-5 scored documents and their reasoning are surfaced in the "
    "frontend application.",
    indent=True
)

heading("LLM Model Selection: Gemma 4 E4B", level=3)

body(
    "The LLM used for re-ranking is Google Gemma 4 E4B (gemma4:e4b), served locally via "
    "Ollama. Gemma 4 E4B was selected after evaluating several open-weight models on the "
    "criteria most relevant to this task: parameter efficiency for local inference, reliable "
    "structured JSON output, consistency at temperature zero, and quality of reasoning on "
    "short-context text comparison tasks.",
    indent=True
)

body(
    "The 4-billion-parameter scale (effective 4B, quantised to Q4_K_M for inference) fits "
    "within 3 GB of VRAM, making it suitable for consumer hardware without requiring a "
    "data-centre GPU. At temperature=0, Gemma 4 E4B produces highly consistent verdicts "
    "across repeated calls with identical input, which is important for reproducible "
    "experimental results. Its instruction-following capability is sufficient to reliably "
    "produce the required three-field JSON output in a zero-shot setting.",
    indent=True
)

add_table(
    headers=["Model", "Params", "VRAM (Q4)", "JSON Output", "Consistency", "Notes"],
    rows=[
        ["Gemma 4 E4B ✓", "~4B", "~3 GB", "Reliable", "High", "Selected — best efficiency/quality trade-off"],
        ["Gemma 3 4B", "4B", "~3 GB", "Reliable", "High", "Previous generation; slightly weaker reasoning"],
        ["Qwen2.5 7B", "7B", "~5 GB", "Reliable", "High", "Good alternative; higher VRAM requirement"],
        ["Qwen2.5 9B", "9B", "~6 GB", "Reliable", "High", "Default in Streamlit UI; better on complex obfuscation"],
        ["Llama 3.1 8B", "8B", "~5 GB", "Moderate", "Medium", "Less reliable structured output"],
        ["Mistral 7B v0.3", "7B", "~5 GB", "Moderate", "Medium", "Shorter context; older instruction tuning"],
        ["Phi-3 Mini 3.8B", "3.8B", "~3 GB", "Moderate", "Medium", "Fast but weaker on paraphrase reasoning"],
        ["GPT-4o (API)", "—", "—", "Excellent", "Very High", "Not used: cost and privacy constraints; upper bound"],
    ],
    col_widths=[1.4, 0.7, 0.9, 0.9, 0.9, 2.5]
)
add_caption("Table 5.4 — LLM candidates evaluated for the re-ranking stage. All models use 4-bit quantisation (Q4_K_M) via Ollama except GPT-4o.")

body(
    "The LLM is accessed through the OpenAI-compatible Ollama API endpoint at "
    "http://localhost:11434/v1. The instructor library wraps this client to enforce structured "
    "output through repeated sampling if the model produces non-conformant JSON. Temperature "
    "is set to 0 in the notebook pipeline and to 0.1 in the Streamlit application (a small "
    "non-zero value to avoid deterministic repetition on edge cases).",
    indent=True
)

# ─────────────────────────────────────────────────────────────────────────────
heading("5.6. Post-Processing and Analytics", level=2)
# ─────────────────────────────────────────────────────────────────────────────

body(
    "The post-processing stage evaluates the output of the retrieval and alignment pipeline "
    "against the PAN 2011 ground truth annotations. Its purpose is to quantify system "
    "performance using standard information retrieval metrics and to identify strengths and "
    "weaknesses across different obfuscation strategies.",
    indent=True
)

body("Three standard metrics are computed:", indent=True)

body(
    "Recall@K measures the fraction of suspicious documents for which the true source document "
    "appears within the top-K retrieved results. This is the primary metric for the source "
    "retrieval task because the LLM re-ranking stage cannot recover a source document that was "
    "not retrieved. Recall@K is evaluated for K values of 1, 5, 10, 20, and 50.",
    indent=True
)

body(
    "Mean Reciprocal Rank (MRR) captures the average rank at which the first correct source "
    "document appears. MRR penalises systems that retrieve the correct source at a lower rank "
    "even if it is within the top-K cutoff.",
    indent=True
)

body(
    "Mean Average Precision (MAP) measures the area under the precision-recall curve averaged "
    "across all queries. This metric rewards retrieving true sources at high ranks across all "
    "queries simultaneously.",
    indent=True
)

body(
    "Evaluation is performed separately for each retrieval branch (TF-IDF, LSA, ESA, "
    "Embeddings) and for the fused ranking, enabling a direct comparison of the contribution "
    "of each branch and the benefit of fusion. Results are further broken down by obfuscation "
    "type (none, random, translated, simulated) to characterise which methods are most robust "
    "to each form of plagiarism.",
    indent=True
)

body(
    "The evaluation is implemented in the analytics notebook (06_analytics/analyze_results.ipynb), "
    "which loads the retrieval results and the ground truth document-level validation table, "
    "performs a left-join to label true positives and false positives, and computes the "
    "metrics described above. A suspicious document is counted as a hit at rank K if any of "
    "its true source documents appears within the top-K results, accommodating the multi-source "
    "plagiarism cases present in the PAN 2011 corpus.",
    indent=True
)

body(
    "The fusion strategy is expected to outperform any single branch on recall, especially for "
    "obfuscated cases, because different branches capture complementary signals: TF-IDF "
    "catches character-level similarity, LSA captures topical overlap under vocabulary "
    "variation, ESA provides concept-level matching, and dense embeddings handle paraphrase "
    "and semantic reformulation. A candidate source document present in multiple branch "
    "rankings receives a higher final score, providing a natural form of ensemble agreement "
    "that boosts true positives while suppressing spurious false positives.",
    indent=True
)

add_table(
    headers=["Metric", "Formula", "What it measures"],
    rows=[
        ["Recall@K", "# queries with true source in top-K / # total queries", "Coverage at rank cutoff K"],
        ["MRR", "Mean of 1/rank(first correct source)", "Average rank quality of the first hit"],
        ["MAP", "Mean of average precision across queries", "Precision-recall trade-off across all ranks"],
    ],
    col_widths=[1.3, 2.8, 2.1]
)
add_caption("Table 5.5 — Evaluation metrics used in the post-processing stage.")

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = "Chapter5_Implementation.docx"
doc.save(output_path)
print(f"Saved: {output_path}")
