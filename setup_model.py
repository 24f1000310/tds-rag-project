import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from passages import PASSAGES

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "model_cache"
CACHE.mkdir(exist_ok=True)

model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=str(CACHE),
    threads=1,
)

texts = [p["text"] for p in PASSAGES]
sentences = []
for passage in PASSAGES:
    sentences.extend(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", passage["text"])
        if part.strip()
    )

def embed_and_normalize(items):
    matrix = np.asarray(list(model.embed(items)), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)

passage_vectors = embed_and_normalize(texts)
sentence_vectors = embed_and_normalize(sentences)

np.save(ROOT / "passage_vectors.npy", passage_vectors)
np.save(ROOT / "sentence_vectors.npy", sentence_vectors)

# Save sentence order so runtime metadata stays aligned with cached vectors.
(ROOT / "sentences.json").write_text(
    __import__("json").dumps(sentences, ensure_ascii=False),
    encoding="utf-8",
)

print("Model downloaded and embeddings precomputed.")
print("Passage vectors:", passage_vectors.shape)
print("Sentence vectors:", sentence_vectors.shape)
print("Cache:", CACHE)
