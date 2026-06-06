"""Top-level workflow: classifies intent and drives the planner → orchestrator → validator pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Optional, TypedDict

import nbformat
from agents import RunConfig, Runner, TResponseInputItem, trace
from agents.mcp import MCPServerManager
from pydantic import BaseModel

from .agents import (
    analysis_orchestration_agent,
    analysis_planner,
    edit_orchestration_agent,
    orchestration_agent,
    planner,
    validatorandfixingagent,
)
from .agents.edit_planner import EDIT_PLANNER_INSTRUCTIONS
from .agents.question import answer_question
from .functions import extract_trace_markdown
from .functions.notebook_io import _get_current_path
from .hooks import StreamingHooks, build_notebook_outline, emit_phase
from .langchain_runner import run_tool_loop
from .langchain_tools import (
    LangChainMCPToolContext,
    notebook_build_tools,
    notebook_edit_tools,
    notebook_validation_tools,
)
from .lineage import extract_lineage_ast, extract_lineage_runtime
from .mcp_connections import mcp_servers

logging.basicConfig(
    filename="finagent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


_TRACE_SOURCE = "agent-builder"


def _new_trace_metadata() -> dict:
    """Build a fresh trace-metadata dict with a per-run workflow id."""
    return {
        "__trace_source__": _TRACE_SOURCE,
        "workflow_id": f"wf_{uuid.uuid4().hex}",
    }


def _stash_lineage_metadata(path: str, method: str, lineage: dict) -> None:
    """Persist a lineage graph onto nb.metadata['finagent_lineage'][method].

    Lets the viewer fetch lineage straight from the .ipynb file rather than
    re-running the extractor on every page load. Best-effort: failures are
    logged but don't break the workflow.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        bucket = nb.metadata.setdefault("finagent_lineage", {})
        bucket[method] = lineage
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
    except Exception:
        logging.exception("could not stash %s lineage on %s", method, path)


def _build_notebook_context(path: Optional[str]) -> str:
    """Render a compact, truncated view of a notebook for inline prompt injection.

    Per cell:
      - source capped at ~600 chars (head + tail if longer)
      - first text/plain or text/html output capped at ~400 chars
    Cells without source/output are still listed so the agent knows the index
    layout. Returns an empty string if `path` is missing or unreadable.
    """
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception:
        return ""

    lines: list[str] = []
    for i, cell in enumerate(nb.cells):
        ctype = cell.cell_type
        src = cell.source or ""
        if len(src) > 600:
            src = src[:400] + "\n…\n" + src[-180:]
        meta = cell.metadata.get("finagent") if hasattr(cell, "metadata") else None
        node = (meta or {}).get("node_id", "")
        head = f"[Cell {i} | {ctype}{f' | {node}' if node else ''}]"
        lines.append(f"{head}\n{src}")
        if ctype == "code":
            for out in cell.get("outputs") or []:
                otype = out.get("output_type")
                if otype == "stream":
                    txt = out.get("text") or ""
                    if isinstance(txt, list):
                        txt = "".join(txt)
                    if txt.strip():
                        snippet = txt if len(txt) <= 400 else txt[:380] + "…"
                        lines.append(f"  [stream] {snippet.strip()}")
                        break
                elif otype in ("execute_result", "display_data"):
                    data = out.get("data") or {}
                    plain = data.get("text/plain", "")
                    if isinstance(plain, list):
                        plain = "".join(plain)
                    if plain.strip():
                        snippet = plain if len(plain) <= 400 else plain[:380] + "…"
                        lines.append(f"  [output] {snippet.strip()}")
                        break
                elif otype == "error":
                    lines.append(
                        f"  [error] {out.get('ename', '')}: {out.get('evalue', '')}"
                    )
                    break
    return "\n\n".join(lines)


class WorkflowInput(BaseModel):
    input_as_text: str


async def _classify_intent(message: str, has_notebook: bool) -> str:
    """Return 'question', 'edit', 'explore', or 'new' based on the user message.

    The intent buckets correspond to four distinct routing paths in
    ``run_workflow``:

      * ``question`` — chat-only reply (no notebook write); answers
        "what is X" / "why does cell N do Y" without touching disk.
      * ``edit``     — modify an existing notebook in place.
      * ``explore``  — ad-hoc analysis or plot of real data
        (e.g. "plot the historical daily P/E of Micron",
        "show me the correlation matrix of these 5 stocks",
        "what does the yield curve look like over the last 10 years").
        Routes through ``analysis_planner`` + ``analysis_orchestrator``
        which DO NOT enforce the asset_weights/returns + backtest
        shape — the deliverable is the chart / dataframe itself.
      * ``new``      — brand-new TRADING STRATEGY notebook. Routes
        through the strategy planner/orchestrator which require the
        DAG to end in asset_weights + asset_returns and a backtest call.

    The "explore" vs "new" split matters because forcing a simple
    plotting request through the strategy pipeline produces a synthetic
    backtest on a one-row dataframe and answers nothing.
    """
    from .llm import ainvoke_text

    intent = (
        await ainvoke_text(
            "intent_classifier",
            system=(
                "Classify the user message as exactly one of:\n"
                "- question: asking for explanation, analysis, clarification, "
                "or general info (no chart, no dataframe — just an answer in chat)\n"
                "- edit: requesting a change, addition, or fix to an existing notebook\n"
                "- explore: requesting an ad-hoc analysis, plot, chart, or computation "
                "of REAL DATA (e.g. 'plot historical P/E of MU', "
                "'show correlation matrix', 'compare returns of A vs B'). "
                "The deliverable is the chart/dataframe, NOT a trading strategy.\n"
                "- new: requesting a brand-new TRADING STRATEGY notebook "
                "(the deliverable is asset weights + returns over time, suitable "
                "for backtesting).\n\n"
                "Reply with only the single word."
            ),
            user=f"Has existing notebook: {has_notebook}\nMessage: {message}",
        )
    ).strip().lower()
    if intent not in ("question", "edit", "explore", "new"):
        intent = "edit" if has_notebook else "explore"
    return intent


class _GraphWorkflowState(TypedDict, total=False):
    workflow_input: WorkflowInput
    existing_notebook_path: Optional[str]
    prior_history: Optional[list]
    progress_cb: object
    intent: str
    result: dict


async def _emit_status(progress_cb, msg: str):
    if progress_cb:
        await progress_cb({"type": "status", "message": msg})


async def _emit_outline(progress_cb, path: Optional[str]):
    if progress_cb and path:
        try:
            await progress_cb({"type": "event", "data": build_notebook_outline(path)})
        except Exception:
            logging.exception("emit_outline failed for %s", path)


async def _emit_lineage(progress_cb, path: Optional[str]) -> None:
    """Compute AST lineage now and runtime lineage in the background."""
    if not path:
        return
    try:
        ast_lin = extract_lineage_ast(path)
        _stash_lineage_metadata(path, "ast", ast_lin)
        if progress_cb:
            await progress_cb(
                {
                    "type": "event",
                    "data": {
                        "type": "notebook_lineage",
                        "method": "ast",
                        "path": str(path),
                        "node_count": len(ast_lin.get("nodes", [])),
                        "edge_count": len(ast_lin.get("edges", [])),
                    },
                }
            )
    except Exception:
        logging.exception("AST lineage failed for %s", path)

    async def _runtime_trace():
        try:
            rt_lin = await asyncio.to_thread(extract_lineage_runtime, path)
            _stash_lineage_metadata(path, "runtime", rt_lin)
            if progress_cb:
                await progress_cb(
                    {
                        "type": "event",
                        "data": {
                            "type": "notebook_lineage",
                            "method": "runtime",
                            "path": str(path),
                            "node_count": len(rt_lin.get("nodes", [])),
                            "edge_count": len(rt_lin.get("edges", [])),
                            "error": rt_lin.get("error"),
                        },
                    }
                )
        except Exception:
            logging.exception("runtime lineage failed for %s", path)

    asyncio.create_task(_runtime_trace())


async def _emit_agent_input(
    progress_cb,
    phase: str,
    agent_name: str,
    prompt: str,
) -> None:
    if not progress_cb:
        return
    text = prompt if len(prompt) <= 2000 else prompt[:2000] + "…"
    await progress_cb(
        {
            "type": "event",
            "data": {
                "type": "agent_input",
                "phase": phase,
                "agent": agent_name,
                "prompt": text,
            },
        }
    )


async def _name_next_notebook(progress_cb, user_request: str) -> None:
    await _emit_status(progress_cb, "Naming notebook...")
    try:
        from .agents.name_suggester import suggest_notebook_name
        from .functions.notebook_io import _path_for_named, set_next_notebook_name

        slug = await suggest_notebook_name(user_request)
        set_next_notebook_name(slug)
        preview_path = _path_for_named(slug)
        if progress_cb:
            await progress_cb(
                {
                    "type": "event",
                    "data": {
                        "type": "notebook_named",
                        "slug": slug,
                        "path": str(preview_path),
                        "filename": preview_path.name,
                    },
                }
            )
        logging.info("workflow: notebook named slug=%s preview=%s", slug, preview_path)
    except Exception:
        logging.exception("workflow: name suggester failed (non-fatal)")


async def _run_langchain_notebook_workflow(
    workflow_input: WorkflowInput,
    *,
    intent: str,
    existing_notebook_path: Optional[str] = None,
    progress_cb=None,
) -> dict:
    """Run notebook creation/edit/validation via LangChain tools only."""
    from .agents.analysis_orchestration import ANALYSIS_ORCHESTRATION_INSTRUCTIONS
    from .agents.analysis_planner import ANALYSIS_PLANNER_INSTRUCTIONS
    from .agents.edit_orchestration import EDIT_ORCHESTRATION_INSTRUCTIONS
    from .agents.edit_planner import EDIT_PLANNER_INSTRUCTIONS
    from .agents.orchestration import ORCHESTRATION_INSTRUCTIONS
    from .agents.planner import PLANNER_INSTRUCTIONS
    from .agents.validator import VALIDATOR_INSTRUCTIONS

    user_request = workflow_input.input_as_text

    if intent == "edit" and existing_notebook_path:
        with open(existing_notebook_path, "r", encoding="utf-8") as nb_f:
            nb = nbformat.read(nb_f, as_version=4)
        nb_summary = "\n".join(
            f"  [{i}] ({cell.cell_type}) {cell.source[:200].replace(chr(10), ' ')}"
            for i, cell in enumerate(nb.cells)
        )
        edit_planner_input = (
            f"EXISTING NOTEBOOK ({existing_notebook_path}):\n{nb_summary}\n\n"
            f"USER REQUEST: {user_request}"
        )
        await _emit_status(progress_cb, "Planning changes...")
        await emit_phase(progress_cb, "edit_plan", "start")
        await _emit_agent_input(
            progress_cb,
            "edit_plan",
            "EditPlanner",
            edit_planner_input,
        )
        from .llm import ainvoke_text

        diff_spec_raw = (
            await ainvoke_text(
                "chat_edit_planner",
                system=EDIT_PLANNER_INSTRUCTIONS,
                user=edit_planner_input,
            )
        ).strip()
        await emit_phase(progress_cb, "edit_plan", "end")

        try:
            diff_spec = json.loads(diff_spec_raw)
        except Exception:
            diff_spec = {"mode": "new"}

        if diff_spec.get("mode") != "new":
            await _emit_status(progress_cb, "Applying edits...")
            await emit_phase(progress_cb, "edit_apply", "start")
            edit_apply_prompt = (
                f"Apply this diff spec to the current notebook:\n{diff_spec_raw}"
            )
            await _emit_agent_input(
                progress_cb,
                "edit_apply",
                "EditOrchestrationAgent",
                edit_apply_prompt,
            )
            async with LangChainMCPToolContext() as mcp:
                edit_output = await run_tool_loop(
                    role="chat_edit_orchestrator",
                    system=EDIT_ORCHESTRATION_INSTRUCTIONS,
                    user=edit_apply_prompt,
                    tools=[*notebook_edit_tools(), *mcp.tools],
                    max_turns=40,
                    progress_cb=progress_cb,
                    phase="edit_apply",
                )
            await emit_phase(progress_cb, "edit_apply", "end")
            await _emit_outline(progress_cb, existing_notebook_path)

            await _emit_status(progress_cb, "Validating notebook...")
            await emit_phase(progress_cb, "edit_validate", "start")
            validate_prompt = (
                f"Validate the edited notebook at {existing_notebook_path}. "
                "Run it cell-by-cell, fix any errors, escalate if blocked."
            )
            await _emit_agent_input(
                progress_cb,
                "edit_validate",
                "ValidatorAndFixingAgent",
                validate_prompt,
            )
            async with LangChainMCPToolContext() as mcp:
                val_output = await run_tool_loop(
                    role="chat_validator",
                    system=VALIDATOR_INSTRUCTIONS,
                    user=validate_prompt,
                    tools=[*notebook_validation_tools(), *mcp.tools],
                    max_turns=24,
                    progress_cb=progress_cb,
                    phase="edit_validate",
                )
            await emit_phase(progress_cb, "edit_validate", "end")
            await _emit_outline(progress_cb, existing_notebook_path)
            await _emit_lineage(progress_cb, existing_notebook_path)
            return {
                "output_text": val_output or edit_output,
                "notebook_path": existing_notebook_path,
                "mode": "edit",
            }

        existing_notebook_path = None
        intent = "new"

    if intent == "explore":
        planner_instructions = ANALYSIS_PLANNER_INSTRUCTIONS
        orchestrator_instructions = ANALYSIS_ORCHESTRATION_INSTRUCTIONS
        planner_name = "AnalysisPlanner"
        orchestrator_name = "Analysis Orchestration Agent"
        mode_label = "explore"
    else:
        planner_instructions = PLANNER_INSTRUCTIONS
        orchestrator_instructions = ORCHESTRATION_INSTRUCTIONS
        planner_name = "Planner"
        orchestrator_name = "Orchestration Agent"
        mode_label = "new"

    await _name_next_notebook(progress_cb, user_request)

    await _emit_status(
        progress_cb,
        "Planning research..." if mode_label == "new" else "Planning analysis...",
    )
    await emit_phase(progress_cb, "plan", "start")
    plan_prompt = f"Research request: {user_request}"
    await _emit_agent_input(progress_cb, "plan", planner_name, plan_prompt)
    async with LangChainMCPToolContext() as mcp:
        planner_output = await run_tool_loop(
            role="chat_planner",
            system=planner_instructions,
            user=plan_prompt,
            tools=mcp.tools,
            max_turns=24,
            progress_cb=progress_cb,
            phase="plan",
        )
    await emit_phase(progress_cb, "plan", "end")

    await _emit_status(progress_cb, "Building notebook...")
    await emit_phase(progress_cb, "build", "start")
    build_prompt = f"User request: {user_request}\n\nDAG plan from planner:\n{planner_output}"
    await _emit_agent_input(progress_cb, "build", orchestrator_name, build_prompt)
    async with LangChainMCPToolContext() as mcp:
        build_output = await run_tool_loop(
            role="chat_orchestrator",
            system=orchestrator_instructions,
            user=build_prompt,
            tools=[*notebook_build_tools(), *mcp.tools],
            max_turns=80,
            progress_cb=progress_cb,
            phase="build",
        )
    await emit_phase(progress_cb, "build", "end")
    nb_path_after_build = str(_get_current_path())
    await _emit_outline(progress_cb, nb_path_after_build)

    await _emit_status(progress_cb, "Validating and fixing...")
    await emit_phase(progress_cb, "validate", "start")
    validate_prompt = (
        f"Validate the freshly built notebook at {nb_path_after_build}. "
        "Run it cell-by-cell, fix any errors, escalate if blocked."
    )
    await _emit_agent_input(
        progress_cb,
        "validate",
        "ValidatorAndFixingAgent",
        validate_prompt,
    )
    async with LangChainMCPToolContext() as mcp:
        validator_output = await run_tool_loop(
            role="chat_validator",
            system=VALIDATOR_INSTRUCTIONS,
            user=validate_prompt,
            tools=[*notebook_validation_tools(), *mcp.tools],
            max_turns=24,
            progress_cb=progress_cb,
            phase="validate",
        )
    await emit_phase(progress_cb, "validate", "end")

    final_path = str(_get_current_path())
    await _emit_outline(progress_cb, final_path)
    await _emit_lineage(progress_cb, final_path)

    return {
        "output_text": validator_output or build_output,
        "notebook_path": final_path,
        "mode": mode_label,
    }


async def _run_langgraph_workflow(
    workflow_input: WorkflowInput,
    existing_notebook_path: Optional[str] = None,
    prior_history: Optional[list] = None,
    progress_cb=None,
):
    """LangGraph entrypoint for the product workflow.

    The graph owns routing, question mode, and the notebook plan/build/validate
    path. Tool execution is LangChain-native; the OpenAI Agents SDK path remains
    available only through FINAGENT_WORKFLOW_RUNTIME=agents as an emergency
    rollback.
    """
    from langgraph.graph import END, START, StateGraph

    async def classify_node(state: _GraphWorkflowState) -> dict:
        wf = state["workflow_input"]
        intent = await _classify_intent(
            wf.input_as_text,
            bool(state.get("existing_notebook_path")),
        )
        return {"intent": intent}

    async def route_question_node(state: _GraphWorkflowState) -> dict:
        wf = state["workflow_input"]
        nb_path = state.get("existing_notebook_path")
        progress = state.get("progress_cb")
        if progress:
            await progress({"type": "status", "message": "Thinking..."})
        nb_context = _build_notebook_context(nb_path)
        answer = await answer_question(wf.input_as_text, notebook_context=nb_context)
        return {
            "result": {
                "output_text": answer,
                "notebook_path": nb_path,
                "mode": "question",
            }
        }

    async def notebook_tool_loop_node(state: _GraphWorkflowState) -> dict:
        result = await _run_langchain_notebook_workflow(
            state["workflow_input"],
            intent=state["intent"],
            existing_notebook_path=state.get("existing_notebook_path"),
            progress_cb=state.get("progress_cb"),
        )
        return {"result": result}

    def choose_branch(state: _GraphWorkflowState) -> str:
        return "question" if state.get("intent") == "question" else "notebook_tool_loop"

    graph = StateGraph(_GraphWorkflowState)
    graph.add_node("classify", classify_node)
    graph.add_node("question", route_question_node)
    graph.add_node("notebook_tool_loop", notebook_tool_loop_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        choose_branch,
        {"question": "question", "notebook_tool_loop": "notebook_tool_loop"},
    )
    graph.add_edge("question", END)
    graph.add_edge("notebook_tool_loop", END)
    compiled = graph.compile()
    final_state = await compiled.ainvoke(
        {
            "workflow_input": workflow_input,
            "existing_notebook_path": existing_notebook_path,
            "prior_history": prior_history,
            "progress_cb": progress_cb,
        }
    )
    return final_state["result"]


async def _run_agents_workflow(
    workflow_input: WorkflowInput,
    existing_notebook_path: Optional[str] = None,
    prior_history: Optional[list] = None,
    progress_cb=None,
    forced_intent: Optional[str] = None,
):
    async def _emit(msg: str):
        if progress_cb:
            await progress_cb({"type": "status", "message": msg})

    async def _emit_trace(result):
        if progress_cb:
            md = extract_trace_markdown(result)
            if md:
                await progress_cb({"type": "trace", "markdown": md})

    async def _emit_outline(path: Optional[str]):
        if progress_cb and path:
            try:
                await progress_cb(
                    {"type": "event", "data": build_notebook_outline(path)}
                )
            except Exception:
                logging.exception("emit_outline failed for %s", path)

    async def _emit_lineage(path: Optional[str]) -> None:
        """Compute AST lineage immediately and runtime lineage in the
        background; stash both in nb.metadata so the viewer can fetch them
        on demand. Runtime tracing is best-effort — it can fail if the
        notebook needs missing deps or hangs, so we don't await it.
        """
        if not path:
            return
        try:
            ast_lin = extract_lineage_ast(path)
            _stash_lineage_metadata(path, "ast", ast_lin)
            if progress_cb:
                await progress_cb(
                    {
                        "type": "event",
                        "data": {
                            "type": "notebook_lineage",
                            "method": "ast",
                            "path": str(path),
                            "node_count": len(ast_lin.get("nodes", [])),
                            "edge_count": len(ast_lin.get("edges", [])),
                        },
                    }
                )
        except Exception:
            logging.exception("AST lineage failed for %s", path)

        # Runtime tracer runs in a subprocess and may take a while; fire-and-
        # forget so the UI gets the AST view immediately and the runtime view
        # whenever it lands.
        async def _runtime_trace():
            try:
                rt_lin = await asyncio.to_thread(extract_lineage_runtime, path)
                _stash_lineage_metadata(path, "runtime", rt_lin)
                if progress_cb:
                    await progress_cb(
                        {
                            "type": "event",
                            "data": {
                                "type": "notebook_lineage",
                                "method": "runtime",
                                "path": str(path),
                                "node_count": len(rt_lin.get("nodes", [])),
                                "edge_count": len(rt_lin.get("edges", [])),
                                "error": rt_lin.get("error"),
                            },
                        }
                    )
            except Exception:
                logging.exception("runtime lineage failed for %s", path)

        asyncio.create_task(_runtime_trace())

    def _hooks(phase: str) -> StreamingHooks:
        return StreamingHooks(progress_cb, phase)

    async def _emit_agent_input(phase: str, agent_name: str, prompt: str) -> None:
        """Surface the prompt sent to each agent so users see the full CoT chain.

        Truncates the prompt to ~2000 chars on the wire — a long DAG / notebook
        dump would otherwise flood the chat. The full prompt remains in the
        OpenAI Agents trace.
        """
        if not progress_cb:
            return
        text = prompt if len(prompt) <= 2000 else prompt[:2000] + "…"
        await progress_cb(
            {
                "type": "event",
                "data": {
                    "type": "agent_input",
                    "phase": phase,
                    "agent": agent_name,
                    "prompt": text,
                },
            }
        )

    # One workflow_id per request — trace_metadata gets a fresh value each run
    # so OpenAI traces filter cleanly per-conversation.
    trace_metadata = _new_trace_metadata()
    logging.info(
        f"run_workflow mode={'edit' if existing_notebook_path else 'new'} "
        f"workflow_id={trace_metadata['workflow_id']}"
    )

    with trace("Finagent"):
        workflow = workflow_input.model_dump()
        conversation_history: list[TResponseInputItem] = list(prior_history or [])

        # ── CLASSIFY INTENT ──────────────────────────────────────────────────
        intent = forced_intent or await _classify_intent(
            workflow["input_as_text"], bool(existing_notebook_path)
        )
        logging.info(f"run_workflow intent={intent}")

        # ── QUESTION MODE ────────────────────────────────────────────────────
        if intent == "question":
            await _emit("Thinking...")
            # Inject a compact notebook context inline so the agent has
            # immediate ground truth and almost never needs to call
            # read_notebook. Without this, agents tended to call the tool,
            # receive a large dump (cells + outputs), and stall before
            # producing a final answer. read_notebook stays available for
            # cases the agent really needs the full source/output of one
            # specific cell.
            nb_context = _build_notebook_context(existing_notebook_path)
            if nb_context:
                q_input = (
                    f"NOTEBOOK CONTEXT (truncated — call `read_notebook` only "
                    f"if you need the *full* source or all outputs of a specific cell):\n"
                    f"{nb_context}\n\n"
                    f"QUESTION: {workflow['input_as_text']}"
                )
            else:
                q_input = workflow["input_as_text"]

            await emit_phase(progress_cb, "question", "start")
            await _emit_agent_input("question", "QuestionAgent", q_input)
            q_answer = await answer_question(
                workflow["input_as_text"],
                notebook_context=nb_context,
            )
            await emit_phase(progress_cb, "question", "end")
            return {
                "output_text": q_answer,
                "notebook_path": existing_notebook_path,
                "mode": "question",
            }

        # ── EDIT MODE ────────────────────────────────────────────────────────
        if intent == "edit" and existing_notebook_path:
            with open(existing_notebook_path, "r", encoding="utf-8") as _nb_f:
                nb = nbformat.read(_nb_f, as_version=4)
            nb_summary_lines = []
            for i, cell in enumerate(nb.cells):
                snippet = cell.source[:200].replace("\n", " ")
                nb_summary_lines.append(f"  [{i}] ({cell.cell_type}) {snippet}")
            nb_summary = "\n".join(nb_summary_lines)

            await _emit("Planning changes...")
            edit_planner_input = (
                f"EXISTING NOTEBOOK ({existing_notebook_path}):\n{nb_summary}\n\n"
                f"USER REQUEST: {workflow['input_as_text']}"
            )
            await emit_phase(progress_cb, "edit_plan", "start")
            await _emit_agent_input("edit_plan", "EditPlanner", edit_planner_input)
            from .llm import ainvoke_text

            diff_spec_raw = (
                await ainvoke_text(
                    "chat_edit_planner",
                    system=EDIT_PLANNER_INSTRUCTIONS,
                    user=edit_planner_input,
                )
            ).strip()
            await emit_phase(progress_cb, "edit_plan", "end")

            try:
                diff_spec = json.loads(diff_spec_raw)
            except Exception:
                diff_spec = {"mode": "new", "rationale": "Could not parse diff spec"}

            if diff_spec.get("mode") == "new":
                existing_notebook_path = None  # fall through to new-notebook flow
            else:
                await _emit("Applying edits...")
                await emit_phase(progress_cb, "edit_apply", "start")
                edit_apply_prompt = (
                    f"Apply this diff spec to the current notebook:\n{diff_spec_raw}"
                )
                await _emit_agent_input(
                    "edit_apply", edit_orchestration_agent.name, edit_apply_prompt
                )
                async with MCPServerManager(mcp_servers()) as _edit_mcp_mgr:
                    _edit_orch = edit_orchestration_agent.clone(
                        mcp_servers=_edit_mcp_mgr.active_servers
                    )
                    edit_orch_result_temp = await Runner.run(
                        _edit_orch,
                        input=[
                            *conversation_history,
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": edit_apply_prompt,
                                    }
                                ],
                            },
                        ],
                        run_config=RunConfig(),
                        hooks=_hooks("edit_apply"),
                        max_turns=40,
                    )
                await emit_phase(progress_cb, "edit_apply", "end")
                await _emit_trace(edit_orch_result_temp)
                conversation_history.extend(
                    [item.to_input_item() for item in edit_orch_result_temp.new_items]
                )
                # Notebook structure has changed; emit a fresh outline so the
                # frontend's Outline panel matches the new cell layout.
                await _emit_outline(existing_notebook_path)

                await _emit("Validating notebook...")
                await emit_phase(progress_cb, "edit_validate", "start")
                await _emit_agent_input(
                    "edit_validate",
                    validatorandfixingagent.name,
                    f"Validate the edited notebook at {existing_notebook_path}.",
                )
                async with MCPServerManager(mcp_servers()) as _val_mcp_mgr:
                    _val = validatorandfixingagent.clone(
                        mcp_servers=_val_mcp_mgr.active_servers
                    )
                    val_result_temp = await Runner.run(
                        _val,
                        input=conversation_history,
                        run_config=RunConfig(),
                        hooks=_hooks("edit_validate"),
                        max_turns=20,
                    )
                await emit_phase(progress_cb, "edit_validate", "end")
                await _emit_trace(val_result_temp)
                # Validator may have replaced cells (and rationales). Refresh outline + lineage.
                await _emit_outline(existing_notebook_path)
                await _emit_lineage(existing_notebook_path)
                return {
                    "output_text": val_result_temp.final_output_as(str),
                    "notebook_path": existing_notebook_path,
                    "mode": "edit",
                }

        # ── NEW NOTEBOOK MODE (strategy build OR ad-hoc explore) ─────────────
        #
        # Both `new` and `explore` use the same plan → build → validate flow.
        # The ONLY difference is the planner + orchestrator pair: `new` uses
        # the strategy pair (asset_weights / asset_returns / backtest), `explore`
        # uses the analysis pair (free-form chart / dataframe).
        if intent == "explore":
            _intent_planner = analysis_planner
            _intent_orchestrator = analysis_orchestration_agent
            mode_label = "explore"
        else:
            _intent_planner = planner
            _intent_orchestrator = orchestration_agent
            mode_label = "new"

        conversation_history.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
            }
        )

        # Phase 1 of the signal-dashboard pipeline: give every chat-driven
        # notebook a meaningful filename instead of the opaque
        # ``notebook_N.ipynb`` counter. A tiny LLM call turns the user's
        # request into a kebab-case slug; the notebook_io._path_for_named
        # helper appends ``__YYYYMMDD-HHMMSS`` for uniqueness. The chosen
        # name is emitted as a ``notebook_named`` event so the chat UI
        # can echo it as the build progresses.
        await _emit("Naming notebook...")
        try:
            from .agents.name_suggester import suggest_notebook_name
            from .functions.notebook_io import _path_for_named, set_next_notebook_name

            slug = await suggest_notebook_name(workflow["input_as_text"])
            set_next_notebook_name(slug)
            preview_path = _path_for_named(slug)
            if progress_cb:
                await progress_cb(
                    {
                        "type": "event",
                        "data": {
                            "type": "notebook_named",
                            "slug": slug,
                            "path": str(preview_path),
                            "filename": preview_path.name,
                        },
                    }
                )
            logging.info(
                "workflow: notebook named slug=%s preview=%s", slug, preview_path.name
            )
        except Exception:
            # Naming is best-effort — never block the build on a name-
            # suggester failure. Falls back to the legacy notebook_N
            # counter inside _get_latest_path.
            logging.exception("workflow: name suggester failed (non-fatal)")

        await _emit(
            "Planning research..." if mode_label == "new" else "Planning analysis..."
        )
        await emit_phase(progress_cb, "plan", "start")
        plan_prompt = f"Research request: {workflow['input_as_text']}"
        await _emit_agent_input("plan", _intent_planner.name, plan_prompt)
        async with MCPServerManager(mcp_servers()) as _planner_mcp_mgr:
            _planner = _intent_planner.clone(
                mcp_servers=_planner_mcp_mgr.active_servers
            )
            planner_result_temp = await Runner.run(
                _planner,
                input=[
                    *conversation_history,
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": plan_prompt}],
                    },
                ],
                run_config=RunConfig(trace_metadata=trace_metadata),
                hooks=_hooks("plan"),
            )
        await emit_phase(progress_cb, "plan", "end")
        await _emit_trace(planner_result_temp)
        conversation_history.extend(
            [item.to_input_item() for item in planner_result_temp.new_items]
        )
        planner_result = {"output_text": planner_result_temp.final_output_as(str)}

        await _emit("Building notebook...")
        await emit_phase(progress_cb, "build", "start")
        # Previously this was an accidental tuple-repr — `f"Question: {a, b}"`
        # serialises a Python tuple, so the orchestrator received the planner
        # output stringified as a dict literal. Pass the user request and the
        # DAG explicitly so the model sees clean, labelled inputs.
        build_prompt = (
            f"User request: {workflow['input_as_text']}\n\n"
            f"DAG plan from planner:\n{planner_result['output_text']}"
        )
        await _emit_agent_input("build", _intent_orchestrator.name, build_prompt)
        async with MCPServerManager(mcp_servers()) as _orch_mcp_mgr:
            _orchestration_agent = _intent_orchestrator.clone(
                mcp_servers=_orch_mcp_mgr.active_servers
            )
            try:
                orchestration_agent_result_temp = await Runner.run(
                    _orchestration_agent,
                    input=[
                        *conversation_history,
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": build_prompt}],
                        },
                    ],
                    run_config=RunConfig(trace_metadata=trace_metadata),
                    hooks=_hooks("build"),
                    # Bumped from 40 → 80. The build phase often needs ~25-30
                    # tool calls (search_code, list_data_sources, write_cell ×N)
                    # to compose a single notebook, and 40 was too tight on
                    # complex requests — orchestrator hit the cap and returned
                    # silently, leaving the chat UI showing only the plan.
                    max_turns=80,
                )
            except Exception as exc:
                # Surface the failure explicitly so the chat UI doesn't go
                # silent after the plan. Without this catch, a build-phase
                # exception bubbled to run_workflow's outer except and the
                # frontend rendered the plan + an unfriendly server error
                # — unhelpful in a customer demo. Now the error reaches the
                # user as a structured event AND we re-raise so the outer
                # error handler still records the failure.
                logging.exception("orchestrator build phase failed")
                await _emit(
                    f"⚠ Build phase failed: {type(exc).__name__}: {str(exc)[:200]}"
                )
                raise
        await emit_phase(progress_cb, "build", "end")
        await _emit_trace(orchestration_agent_result_temp)
        conversation_history.extend(
            [item.to_input_item() for item in orchestration_agent_result_temp.new_items]
        )
        # Surface the notebook structure right after build so the frontend can
        # render the Outline panel before the (potentially long) validate phase.
        nb_path_after_build = str(_get_current_path())
        await _emit_outline(nb_path_after_build)

        await _emit("Validating and fixing...")
        await emit_phase(progress_cb, "validate", "start")
        await _emit_agent_input(
            "validate",
            validatorandfixingagent.name,
            f"Validate the freshly built notebook at {nb_path_after_build}. "
            "Run it cell-by-cell, fix any errors, escalate if blocked.",
        )
        async with MCPServerManager(mcp_servers()) as _val_mcp_mgr:
            _validatorandfixingagent = validatorandfixingagent.clone(
                mcp_servers=_val_mcp_mgr.active_servers
            )
            validatorandfixingagent_result_temp = await Runner.run(
                _validatorandfixingagent,
                input=conversation_history,
                run_config=RunConfig(trace_metadata=trace_metadata),
                hooks=_hooks("validate"),
                max_turns=20,
            )
        await emit_phase(progress_cb, "validate", "end")
        await _emit_trace(validatorandfixingagent_result_temp)
        conversation_history.extend(
            [
                item.to_input_item()
                for item in validatorandfixingagent_result_temp.new_items
            ]
        )
        final_path = str(_get_current_path())
        # Validator may have rewritten cells; emit a final outline reflecting that.
        await _emit_outline(final_path)
        # Lineage graph: AST runs synchronously, runtime tracer fires in
        # the background and lands a few seconds later via its own event.
        await _emit_lineage(final_path)

        return {
            "output_text": validatorandfixingagent_result_temp.final_output_as(str),
            "notebook_path": final_path,
            "mode": mode_label,
        }


async def run_workflow(
    workflow_input: WorkflowInput,
    existing_notebook_path: Optional[str] = None,
    prior_history: Optional[list] = None,
    progress_cb=None,
):
    runtime = os.environ.get("FINAGENT_WORKFLOW_RUNTIME", "langgraph").strip().lower()
    if runtime in {"langgraph", "graph"}:
        return await _run_langgraph_workflow(
            workflow_input,
            existing_notebook_path=existing_notebook_path,
            prior_history=prior_history,
            progress_cb=progress_cb,
        )
    if runtime in {"agents", "legacy", "openai_agents"}:
        return await _run_agents_workflow(
            workflow_input,
            existing_notebook_path=existing_notebook_path,
            prior_history=prior_history,
            progress_cb=progress_cb,
        )
    raise ValueError(
        "FINAGENT_WORKFLOW_RUNTIME must be 'langgraph' or 'agents', "
        f"got {runtime!r}"
    )
