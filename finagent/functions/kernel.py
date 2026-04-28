"""Jupyter kernel execution helpers."""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Dict, Optional

from jupyter_client import KernelManager


def _serialize_output(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    msg_type = msg.get("msg_type")
    content = msg.get("content", {})

    if msg_type == "stream":
        return {
            "output_type": "stream",
            "name": content.get("name"),
            "text": content.get("text", ""),
        }

    if msg_type in {"display_data", "execute_result"}:
        return {
            "output_type": msg_type,
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
            "execution_count": content.get("execution_count"),
        }

    if msg_type == "error":
        return {
            "output_type": "error",
            "ename": content.get("ename"),
            "evalue": content.get("evalue"),
            "traceback": content.get("traceback", []),
        }

    return None


def _run_code_in_kernel(code: str, timeout: int = 120) -> Dict[str, Any]:
    """Execute code in a temporary Jupyter kernel and collect outputs."""
    logging.info("_run_code_in_kernel using kernel_name=finagent-python")
    km = KernelManager(kernel_name="finagent-python")
    km.start_kernel()

    try:
        kc = km.client()
        kc.start_channels()
        kc.wait_for_ready(timeout=timeout)

        msg_id = kc.execute(code)
        outputs = []
        execute_reply = None
        deadline = time.time() + timeout

        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                msg = kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                break

            parent = msg.get("parent_header", {})
            if parent.get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            if msg_type == "status" and msg.get("content", {}).get("execution_state") == "idle":
                break

            out = _serialize_output(msg)
            if out is not None:
                outputs.append(out)

        shell_deadline = time.time() + 5
        while time.time() < shell_deadline:
            try:
                reply = kc.get_shell_msg(timeout=0.5)
            except queue.Empty:
                continue
            if reply.get("parent_header", {}).get("msg_id") == msg_id:
                execute_reply = reply
                break

        status = None
        if execute_reply:
            status = execute_reply.get("content", {}).get("status")

        error = next((o for o in outputs if o["output_type"] == "error"), None)

        return {
            "success": error is None and status != "error",
            "status": status or ("error" if error else "ok"),
            "outputs": outputs,
            "error": error,
        }

    finally:
        try:
            kc.stop_channels()
        except Exception:
            pass
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass
