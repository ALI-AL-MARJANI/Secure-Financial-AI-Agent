FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 TQDM_DISABLE=1
WORKDIR /app

# CPU-only torch keeps the image small; the embedding model runs on CPU inside the container.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[api]"

COPY configs ./configs
COPY data/corpus ./data/corpus
COPY data/demo ./data/demo
# Pre-download the embedding model at build time so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8000
CMD ["uvicorn", "secfin.api:app", "--host", "0.0.0.0", "--port", "8000"]
