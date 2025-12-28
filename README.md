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
# Create project virtual environment
uv venv

# Install dependencies from lockfile
uv sync
```