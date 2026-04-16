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
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "from agents.mcp import MCPServerManager"  # fail fast if pin drifts

# Install fin-kit (lakshya-aga/fin-kit — the private mlfinlab fork) separately:
#   --ignore-requires-python: setup.cfg still declares python_requires <3.9,
#     which is a stale upstream pin; the relevant code paths work on 3.12.
#   --no-deps: install_requires pulls tensorflow>=2, networkx<2.6, dash,
#     decorator<5, POT, analytics-python, getmac — none of which the research
#     notebooks this agent generates actually use. Add specific runtime deps
#     to requirements.txt if a notebook hits ImportError (pandas, numpy,
#     scikit-learn, scipy, matplotlib, statsmodels are already installed).
# Nothing in this step touches the public PyPI `mlfinlab` — the package
# installed here IS mlfinlab, sourced entirely from your GitHub.
RUN pip install --no-cache-dir --ignore-requires-python --no-deps \
        git+https://github.com/lakshya-aga/fin-kit.git

# Register the ipykernel spec the agent actually asks for. agent_workflow.py
# hardcodes `kernel_name="finagent-python"` in KernelManager calls and writes
# that same name into every generated notebook's kernelspec metadata, so the
# spec that lives on disk must match exactly or validate_run fails with
# "KernelError: finagent-python not found" before a single cell executes.
RUN python -m ipykernel install --sys-prefix --name finagent-python --display-name "finagent-python"

# Copy application code.
COPY . .

# Generated notebooks land here; mount a volume if you want them to survive restarts.
RUN mkdir -p outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ >/dev/null || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
