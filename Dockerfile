# finagent_openai — FastAPI + OpenAI Agents SDK research orchestrator.
#
# Exposes POST /chat (Synapse web-compat) and POST /api/chat (native SSE) on :8000.

FROM python:3.12-slim

WORKDIR /app

# System deps:
#   git            — pip installing fin-kit / data-mcp from GitHub
#   build-essential — a few transitive wheels need a C compiler
#   curl           — docker-compose healthcheck fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a cached layer first.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Register the default ipykernel spec so jupyter_client.KernelManager can launch it.
RUN python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"

# Copy application code.
COPY . .

# Generated notebooks land here; mount a volume if you want them to survive restarts.
RUN mkdir -p outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
