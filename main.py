from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import RAGEngine

app = FastAPI(title="TDS RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
engine_lock = Lock()

def get_engine():
    global engine
    if engine is None:
        with engine_lock:
            if engine is None:
                engine = RAGEngine()
    return engine

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

@app.get("/")
def health():
    return {"status": "ok", "service": "TDS RAG API"}

@app.get("/health")
def health_check():
    # This lightweight route does not force-load the embedding model.
    return {"status": "ok", "model": "all-MiniLM-L6-v2"}

@app.post("/query")
def query_api(request: QueryRequest):
    return get_engine().query(request.question, request.top_k)
