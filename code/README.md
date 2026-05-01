# Offline Support Triage Agent

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r code/requirements.txt
```

## Run

```bash
python code/main.py --input support_issues/support_issues.csv --output support_issues/output.csv
```

## Evaluate On Sample Tickets

```bash
python code/evaluate.py
```

## Offline Notes

- The agent uses only `data/`.
- No hosted APIs are required.
- **Retrieval is Hybrid:** Combines `sentence-transformers` semantic search with TF-IDF and domain heuristics.
- **Reasoning is Gracefully Degrading:** Uses `llama-cpp-python` for drafting if available, gracefully falling back to deterministic heuristic extraction if the model is missing.
- **Safety is Deterministic:** Hardcoded Python regex guardrails run before and after generation to guarantee 0% action hallucinations.
- To enable the local LLM, provide a `.gguf` model and set the environment variable: `export LOCAL_LLM_PATH="./models/YourModel.gguf"`.

## Current Architecture

- `corpus.py`: builds a local markdown chunk index from `data/`, preserving heading boundaries.
- `retriever.py`: offline Hybrid Retrieval (TF-IDF + Embeddings) with reranking for metadata filtering and domain hints.
- `guardrails.py`: explicit "Guardrail Sandwich" for catching fraud, prompt injections, and hallucinated actions.
- `local_model.py`: the `LocalModelAdapter` combining deterministic routing/classification with LLM/heuristic response drafting.
- `pipeline.py`: orchestrates retrieval, reasoning, and guardrail enforcement into final outputs.
- `evaluate.py`: quick local evaluator for the labeled sample CSV
