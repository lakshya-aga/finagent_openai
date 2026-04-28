"""Question agent: answers questions about an existing notebook without modifying it."""

from __future__ import annotations

from agents import Agent, ModelSettings


QUESTION_INSTRUCTIONS = """You are a helpful quantitative research assistant with deep knowledge of
financial mathematics, statistics, and Python-based research workflows.

You may be given the content of an existing research notebook as context.
Answer the user's question clearly and concisely. You can:
- Explain methodology, code, or results
- Suggest improvements or flag issues
- Answer general quant finance questions

Do NOT modify the notebook. Do NOT call any tools. Just answer.
"""


question_agent = Agent(
    name="QuestionAgent",
    instructions=QUESTION_INSTRUCTIONS,
    model="gpt-5",
    model_settings=ModelSettings(store=True),
)
