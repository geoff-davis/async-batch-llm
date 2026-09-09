"""The effective request and runtime state for one accepted logical item."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from ..base import LLMWorkItem, RetryState, TContext, TInput, TOutput, WorkItemResult
from .execution_state import ItemRuntimeState, runtime_state


@dataclass
class PreparedLogicalItem(Generic[TInput, TOutput, TContext]):
    """Carry preprocessing decisions unchanged through replay and retries."""

    original_item: LLMWorkItem[TInput, TOutput, TContext]
    effective_item: LLMWorkItem[TInput, TOutput, TContext]
    deadline: float | None
    started: float
    retry_state: RetryState = field(default_factory=RetryState)
    terminal_result: WorkItemResult[TOutput, TContext] | None = None

    @property
    def runtime_state(self) -> ItemRuntimeState:
        return runtime_state(self.retry_state)
