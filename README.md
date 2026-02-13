# Academic Document Plagiarism Detection using LLMs and RAG

## Overview
This repository contains the work developed for a **master’s thesis** focused on **academic document plagiarism detection** using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** approaches, including **Graph-based RAG**.

The project investigates how modern AI techniques can be applied to detect plagiarism beyond surface-level text similarity, addressing semantic, structural, and idea-level reuse in academic writing.

---

## Research Motivation
Traditional plagiarism detection systems rely heavily on lexical and syntactic similarity, which limits their effectiveness against:
- Paraphrasing
- Structural reordering
- Semantically equivalent reformulations
- Idea-level plagiarism

Recent advances in **LLMs, embeddings, vector databases, and knowledge graphs** provide new opportunities to improve plagiarism detection by enabling deeper semantic understanding and contextual retrieval.

---

## Objectives
The main objectives of this thesis project are:
- Study and categorize **forms of academic plagiarism**, including:
  - Lexical plagiarism
  - Syntax-preserving plagiarism
  - Semantics-preserving plagiarism
  - Idea-level plagiarism
- Explore **LLM-based plagiarism detection** strategies
- Design and evaluate **RAG-based pipelines** for document comparison
- Investigate **Graph RAG** approaches to represent document structure, citations, and conceptual relationships
- Analyze strengths, limitations, and risks of LLM-based plagiarism detection


---
## Practical Steps

This project follows a 4-stage extrinsic plagiarism detection pipeline aligned with **PAN-PC-11** evaluation needs (offset-based segment detection). The core idea is: **classical semantic candidate search → bounded retrieval → LLM verification → span merging**.

### Stage A — Candidate Generation (ESA or LSA baseline)
**Goal:** For each suspicious text window, retrieve the most likely source candidates (**top-K**) from the full source corpus.

**Steps:**
1. **Chunk suspicious documents** into fixed windows (e.g., 150–250 tokens with overlap) or sentence blocks.
2. **Track character offsets** for every window: `char_start`, `char_end` (required for PAN scoring).
3. Compute a vector representation for each suspicious window:
   - **ESA**: concept vector per window
   - **LSA baseline**: TF-IDF → SVD projection per window
4. Search the indexed source corpus and return **top-K** candidate source documents/passages with similarity scores.
5. Persist Stage A results for traceability and debugging:
   - suspicious window id + offsets
   - candidate source ids (doc/passages) + scores
   - (optional) top terms/concepts used by ESA/LSA

**Where to use the Vector DB:**  
- Store **source passages** as vectors + metadata (`doc_id`, `passage_id`, `char_start`, `char_end`) to support fast top-K retrieval.

---

### Stage B — Filtered Retrieval inside Top-K (Bounded RAG)
**Goal:** Retrieve the best evidence passages **only inside Stage A’s top-K candidates** (avoid a second global search).

**Steps:**
1. Take the **top-K candidates** produced in Stage A.
2. Restrict retrieval to those candidates:
   - If Stage A returns **passages**, fetch them directly.
   - If Stage A returns **documents**, retrieve top-N passages **within those documents** (BM25 or vector similarity).
3. Build an **evidence pack** containing:
   - suspicious window text + offsets
   - top-N source passages (text + offsets + doc ids)

**Where to use the Vector DB:**  
- Apply metadata filtering (e.g., `doc_id IN topK`) so retrieval is constrained to Stage A candidates.

---

### Stage C — LLM Verification (Offsets + Evidence)
**Goal:** Use an LLM to confirm plagiarism and output **offset-aligned evidence**, grounded only in retrieved text.

**Steps:**
1. Send the suspicious window + evidence pack to the LLM with a strict output schema (JSON).
2. Require the LLM to output:
   - `verdict`: `CONFIRMED` or `NOT_CONFIRMED`
   - `best_source_doc_id`
   - suspicious span offsets: `start/end`
   - source span offsets: `start/end`
   - 1–3 short **evidence quotes** (must be exact substrings of provided text)
   - confidence score + reason label (e.g., near-copy, paraphrase)
3. Automatically validate the LLM output:
   - offsets must fall within provided passage bounds
   - evidence quotes must exist verbatim in the input context
   - reject ungrounded answers (no evidence → `NOT_CONFIRMED`)

**When to trigger the LLM:**  
- Only for suspicious windows above a similarity threshold OR within top-K candidates (keeps cost + noise down).

---

### Stage D — Merge Spans for PAN Scoring (Postprocessing)
**Goal:** Reduce fragmented detections and improve PAN-style scoring by merging compatible spans.

**Steps:**
1. Sort all `CONFIRMED` detections by suspicious offsets.
2. Merge adjacent/overlapping detections when:
   - they refer to the same `source_doc_id` (or same source cluster)
   - the gap between suspicious spans is below a threshold (e.g., 200–500 characters)
   - confidence remains above a minimum threshold after merging
3. Drop detections below a minimum length (tiny segments tend to inflate false positives and fragmentation).
4. Export final detections in a PAN-friendly format (offsets + source ids).

---

<br>

![Types of plagiarism and know techniques to uncover](images/forms_of_plagiarism.png)

---

## Methodological Focus
The project explores multiple complementary approaches:
- Text embeddings and similarity search
- Vector databases for scalable document retrieval
- RAG pipelines combining retrieval and generation
- Graph-based representations of documents (sections, citations, concepts)
- LLM reasoning over retrieved evidence

The emphasis is on **research, experimentation, and evaluation**, not on building a production-grade plagiarism detection system.

---

## Project Status
🚧 **Early research and prototyping phase**

Current focus:
- Environment and project setup
- Literature review and background concepts
- Initial experiments with embeddings and RAG workflows

---

## Tech Stack
- **Python 3.12**
- **pandas** – data processing and analysis
- **requests** – data acquisition and API interaction
- **Streamlit** – interactive experimentation and visualization
- **LLMs** – for semantic analysis and reasoning
- **RAG / Graph RAG** – retrieval and structured context modeling
- **Vector DB** – DB for vector storing (chroma db as it is open source)
- **uv** – dependency and environment management

---

## Environment Setup
This project uses **uv** for Python environment and dependency management.

```bash
# Install dependencies from lockfile
uv sync
```