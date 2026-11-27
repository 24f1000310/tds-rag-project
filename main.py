import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import RAGEngine

app = FastAPI(title="TDS RAG API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_engine_init_lock = threading.Lock()

def get_engine():
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                _engine = RAGEngine()
    return _engine

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

@app.get("/")
def root():
    return {"status": "ok", "service": "TDS RAG API", "version": "3.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_api(request: QueryRequest):
    return get_engine().query(request.question, request.top_k)
