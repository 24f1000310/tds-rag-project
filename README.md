# TDS RAG API — Vercel deployment

This project uses FastAPI, FastEmbed ONNX, all-MiniLM-L6-v2, and NumPy cosine similarity.

## Prepare the model cache before deployment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_model.py
```

Keep `model_cache/` in the project and include it in the Git deployment.
Do not add `model_cache/` to `.gitignore`.

## Local run

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

## API

`POST /query`

```json
{"question":"What port does PostgreSQL use?","top_k":3}
```

## Evaluate

```powershell
python eval_local.py
```

Set `RAG_API_URL` to evaluate a deployed endpoint.

## Deploy

Push this folder to GitHub, import the repository in Vercel, and deploy.
Vercel detects the FastAPI app exported as `app` from `main.py`.
