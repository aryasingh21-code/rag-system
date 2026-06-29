# -------------------------------------------------------
# Stage 1 — dependency builder
# -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first from the PyTorch index,
# then install everything else. This prevents pip from
# pulling the 2.5 GB CUDA build of torch.
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# -------------------------------------------------------
# Stage 2 — runtime image
# -------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY main.py .
COPY prompts/   prompts/
COPY templates/ templates/
COPY ingest.py .
COPY documents/ documents/

RUN mkdir -p /app/chroma_db

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
