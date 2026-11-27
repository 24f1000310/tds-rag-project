# TDS RAG v3 — Hybrid retrieval

This version keeps the original 50 passages and uses:
- FastEmbed `all-MiniLM-L6-v2` dense embeddings
- BM25 lexical retrieval
- Hybrid passage and sentence reranking
- Conservative abstention
- Explicit rejection of common prompt-exfiltration / override requests
- Lazy, locked model initialization for serverless concurrency

## Windows / VS Code setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_model.py
uvicorn main:app --reload
```

The setup script also creates `passage_vectors.npy`, `sentence_vectors.npy`, and `sentences.json` to reduce cold-start work. Keep these files and `model_cache/` in the project and commit them for Vercel deployment.
Do not commit `venv/`.

## Smoke test

```powershell
python eval_local.py
```

## Vercel

Push the project to GitHub and import it into Vercel.
`main.py` exports the FastAPI object as `app`.
Test `/`, `/health`, `/docs`, and `POST /query`.

## Important

The thresholds are initial heuristics, not calibrated against the hidden exam set.
The included evaluation script is a smoke test, not the official 20-question evaluator.
A regex guard is not a complete prompt-injection detector; retrieved passages are
treated as data and no retrieved instructions are executed.
