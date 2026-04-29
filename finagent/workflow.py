"""Top-level workflow: classifies intent and drives the planner → orchestrator → validator pipeline."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

import nbformat
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import RunConfig, Runner, TResponseInputItem, trace
from agents.mcp import MCPServerManager

from .agents import (
    edit_orchestration_agent,
    edit_planner,
    orchestration_agent,
    planner,
    question_agent,
    validatorandfixingagent,
)
from .functions import extract_trace_markdown
from .functions.notebook_io import _get_current_path
from .hooks import StreamingHooks, build_notebook_outline, emit_phase
from .mcp_connections import make_data_mcp, make_fruit_thrower


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


class WorkflowInput(BaseModel):
    input_as_text: str


async def _classify_intent(message: str, has_notebook: bool) -> str:
    """Return 'question', 'edit', or 'new' based on the user message."""
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user message as exactly one of:\n"
                    "- question: asking for explanation, analysis, clarification, or general info\n"
                    "- edit: requesting a change, addition, or fix to an existing notebook\n"
                    "- new: requesting a brand-new research notebook or strategy\n\n"
                    "Reply with only the single word."
                ),
            },
            {
                "role": "user",
                "content": f"Has existing notebook: {has_notebook}\nMessage: {message}",
            },
        ],
        max_tokens=5,
        temperature=0,
    )
    intent = resp.choices[0].message.content.strip().lower()
    if intent not in ("question", "edit", "new"):
        intent = "edit" if has_notebook else "new"
    return intent


async def run_workflow(
    workflow_input: WorkflowInput,
    existing_notebook_path: Optional[str] = None,
    prior_history: Optional[list] = None,
    progress_cb=None,
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
                await progress_cb({"type": "event", "data": build_notebook_outline(path)})
            except Exception:
                logging.exception("emit_outline failed for %s", path)

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
        await progress_cb({"type": "event", "data": {
            "type": "agent_input",
            "phase": phase,
            "agent": agent_name,
            "prompt": text,
        }})

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
        intent = await _classify_intent(workflow["input_as_text"], bool(existing_notebook_path))
        logging.info(f"run_workflow intent={intent}")

        # ── QUESTION MODE ────────────────────────────────────────────────────
        if intent == "question":
            await _emit("Thinking...")
            # The question agent now has the read_notebook tool, so we don't
            # dump the full notebook source into the prompt anymore. Just tell
            # it where to look — it pulls cells (and outputs!) on demand.
            if existing_notebook_path:
                q_input = (
                    f"There is an existing notebook at {existing_notebook_path}. "
                    f"If the question requires inspecting cells, code, or outputs, "
                    f"call `read_notebook` to fetch them.\n\n"
                    f"Question: {workflow['input_as_text']}"
                )
            else:
                q_input = workflow["input_as_text"]

            await emit_phase(progress_cb, "question", "start")
            await _emit_agent_input("question", question_agent.name, q_input)
            q_result = await Runner.run(
                question_agent,
                input=[
                    *conversation_history,
                    {"role": "user", "content": [{"type": "input_text", "text": q_input}]},
                ],
                run_config=RunConfig(),
                hooks=_hooks("question"),
            )
            await emit_phase(progress_cb, "question", "end")
            await _emit_trace(q_result)
            return {
                "output_text": q_result.final_output_as(str),
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
            await _emit_agent_input("edit_plan", edit_planner.name, edit_planner_input)
            edit_planner_result_temp = await Runner.run(
                edit_planner,
                input=[{"role": "user", "content": [{"type": "input_text", "text": edit_planner_input}]}],
                run_config=RunConfig(),
                hooks=_hooks("edit_plan"),
            )
            await emit_phase(progress_cb, "edit_plan", "end")
            await _emit_trace(edit_planner_result_temp)
            diff_spec_raw = edit_planner_result_temp.final_output_as(str).strip()

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
                await _emit_agent_input("edit_apply", edit_orchestration_agent.name, edit_apply_prompt)
                async with MCPServerManager([make_fruit_thrower(), make_data_mcp()]) as _edit_mcp_mgr:
                    _edit_orch = edit_orchestration_agent.clone(mcp_servers=_edit_mcp_mgr.active_servers)
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
                conversation_history.extend([item.to_input_item() for item in edit_orch_result_temp.new_items])
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
                async with MCPServerManager([make_fruit_thrower(), make_data_mcp()]) as _val_mcp_mgr:
                    _val = validatorandfixingagent.clone(mcp_servers=_val_mcp_mgr.active_servers)
                    val_result_temp = await Runner.run(
                        _val,
                        input=conversation_history,
                        run_config=RunConfig(),
                        hooks=_hooks("edit_validate"),
                        max_turns=20,
                    )
                await emit_phase(progress_cb, "edit_validate", "end")
                await _emit_trace(val_result_temp)
                # Validator may have replaced cells (and rationales). Refresh outline.
                await _emit_outline(existing_notebook_path)
                return {
                    "output_text": val_result_temp.final_output_as(str),
                    "notebook_path": existing_notebook_path,
                    "mode": "edit",
                }

        # ── NEW NOTEBOOK MODE ────────────────────────────────────────────────
        conversation_history.append({
            "role": "user",
            "content": [{"type": "input_text", "text": workflow["input_as_text"]}],
        })

        await _emit("Planning research...")
        await emit_phase(progress_cb, "plan", "start")
        plan_prompt = f"Research request: {workflow['input_as_text']}"
        await _emit_agent_input("plan", planner.name, plan_prompt)
        async with MCPServerManager([make_fruit_thrower(), make_data_mcp()]) as _planner_mcp_mgr:
            _planner = planner.clone(mcp_servers=_planner_mcp_mgr.active_servers)
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
        conversation_history.extend([item.to_input_item() for item in planner_result_temp.new_items])
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
        await _emit_agent_input("build", orchestration_agent.name, build_prompt)
        async with MCPServerManager([make_fruit_thrower(), make_data_mcp()]) as _orch_mcp_mgr:
            _orchestration_agent = orchestration_agent.clone(mcp_servers=_orch_mcp_mgr.active_servers)
            orchestration_agent_result_temp = await Runner.run(
                _orchestration_agent,
                input=[
                    *conversation_history,
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": build_prompt}
                        ],
                    },
                ],
                run_config=RunConfig(trace_metadata=trace_metadata),
                hooks=_hooks("build"),
                max_turns=40,
            )
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
        async with MCPServerManager([make_fruit_thrower(), make_data_mcp()]) as _val_mcp_mgr:
            _validatorandfixingagent = validatorandfixingagent.clone(mcp_servers=_val_mcp_mgr.active_servers)
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
            [item.to_input_item() for item in validatorandfixingagent_result_temp.new_items]
        )
        final_path = str(_get_current_path())
        # Validator may have rewritten cells; emit a final outline reflecting that.
        await _emit_outline(final_path)

        return {
            "output_text": validatorandfixingagent_result_temp.final_output_as(str),
            "notebook_path": final_path,
            "mode": "new",
        }
