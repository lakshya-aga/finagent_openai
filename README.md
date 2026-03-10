```bash
conda create -n finagentv2
```

```bash
conda activate finagentv2
```

```bash
git clone https://github.com/lakshya-aga/data-mcp
cd data-mcp
pip install -r requirements.txt
pip install -e .
```

```bash
cd ..
```

```bash
git clone https://github.com/lakshya-aga/fin-kit
cd fin-kit
pip install -r requirements.txt
pip install -e .
```

```bash
export OPENAI_API_KEY=sk-proj-...
```

```bash
pip install langgraph langchain-openai langchain-core nbformat jupyter_client
```

```bash
python run_finagent.py
```

## Architecture

The workflow now uses a LangGraph state machine with three sequential nodes:

1. `plan`: turns the user request into a compact execution plan.
2. `assemble`: creates the notebook and writes cells.
3. `validate`: runs the notebook, fixes failures, and returns the final status.

The notebook editing and validation operations are exposed as tools to the LangGraph agents in `agent_workflow.py`.
