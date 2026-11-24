from pathlib import Path
from fastembed import TextEmbedding

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "model_cache"
CACHE.mkdir(exist_ok=True)

model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=str(CACHE),
    threads=1,
)

# Force model files to be downloaded and verified locally.
vector = next(iter(model.embed(["model cache warmup"])))
print("Model downloaded and initialized.")
print("Embedding dimensions:", len(vector))
print("Cache directory:", CACHE)
