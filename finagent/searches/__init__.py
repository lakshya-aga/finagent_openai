"""Search artifact + policies + executor.

A Search is a budgeted sweep over recipe parameters. The executor turns
``policy.propose(history)`` events into recipe variants, runs them via
the existing recipe runner, and tracks the best result on the search
record. Random + grid policies ship today; LLM-driven creativity is the
follow-up phase.
"""

from .executor import execute_search
from .policy import GridPolicy, Policy, RandomPolicy, make_policy
from .types import (
    Budget,
    ChoiceDimension,
    Dimension,
    FloatDimension,
    IntDimension,
    Objective,
    PolicyKind,
    SearchHistoryEntry,
    SearchSubmission,
)

__all__ = [
    "Budget",
    "ChoiceDimension",
    "Dimension",
    "FloatDimension",
    "GridPolicy",
    "IntDimension",
    "Objective",
    "Policy",
    "PolicyKind",
    "RandomPolicy",
    "SearchHistoryEntry",
    "SearchSubmission",
    "execute_search",
    "make_policy",
]
