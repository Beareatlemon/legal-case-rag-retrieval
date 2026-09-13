# Legal Case Retrieval-Augmented Generation

A graduation-project prototype for Chinese legal case similarity retrieval and evidence-grounded question answering. The system retrieves comparable cases from the LeCaRDv2 benchmark and generates a concise explanation with citations to the retrieved evidence.

## Why this project

Legal case retrieval is not a generic chatbot task. A useful answer must first retrieve relevant cases reliably, keep the generated explanation grounded in evidence, and make the evidence visible to the user. This project explores that pipeline end to end, including retrieval evaluation rather than relying only on demo outputs.

## What I built

- A FastAPI backend for retrieval, benchmark evaluation, and retrieval-grounded answer generation.
- A Streamlit workbench for trying queries, inspecting retrieved cases, and running experiments.
- Structured legal-text chunking, FAISS vector indexes, BM25/hybrid retrieval, and dense-vector fusion.
- A fine-tuning and evaluation workflow for BGE-M3 on LeCaRDv2-style supervision.
- Case-level metrics including Recall@5, MRR@5, nDCG@5, and Hit Rate@5.

## Retrieval pipeline

```text
Case description
  -> retrieval-oriented query construction
  -> structured legal-section chunking
  -> BGE-M3 dense retrieval
  -> fine-tuned/base score fusion
  -> case-level ranking
  -> LLM explanation constrained to retrieved evidence and citations
```

## Results and interpretation

On the project's 159-query LeCaRDv2 evaluation set, the original BGE-M3 baseline achieved Recall@5 0.1409, MRR@5 0.8324, nDCG@5 0.6917, and Hit Rate@5 0.9245. The selected vector-fusion configuration reached Recall@5 0.1424, MRR@5 0.8344, nDCG@5 0.6972, and Hit Rate@5 0.9245.

The improvement is modest. I therefore present this as an experimental result, not as proof of production-grade legal advice. Fusion was retained because it preserved the base model's stable signal while incorporating the fine-tuned model's ranking information.

## Repository layout

```text
backend/app/       FastAPI application, retrieval, evaluation, and training helpers
frontend/          Streamlit workbench
scripts/           Index building, evaluation, fine-tuning, and verification entry points
.env.example       Safe local-configuration template
```

## Run locally

### 1. Set up Python and dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Obtain assets separately

This repository intentionally excludes datasets, models, vector indexes, generated answers, and credentials. Obtain LeCaRDv2 and any model checkpoints according to their respective licences, place them locally, then copy `.env.example` to `.env` and update the paths.

### 3. Start the API and interface

```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501` after the API is running.

## Reproduce an evaluation

After configuring local assets, run:

```bash
python scripts/evaluate_rag.py --dataset lecardv2 --max-queries 10000
```

The lightweight verification script checks that configured models and indexes exist and executes a small retrieval evaluation:

```bash
python scripts/verify_fyp_rag.py
```

## Limitations

- Retrieval metrics are automated and do not replace legal-expert review.
- Generation quality depends on the configured LLM endpoint and must not be treated as legal advice.
- Vector fusion loads two dense retrievers and is slower than a single-model path.
- Hardware constraints limited the scale of training and ablation experiments.

## Responsible use

This is an academic prototype for information-retrieval research. It is not a legal-advice service and should not be used to make legal decisions.

