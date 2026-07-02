"""Middleware system for batch processing pipeline."""

from abc import ABC, abstractmethod
from typing import Generic

from ..base import LLMWorkItem, TContext, TInput, TOutput, WorkItemResult


class Middleware(ABC, Generic[TInput, TOutput, TContext]):
    """Abstract base class for middleware that intercepts the processing pipeline.

    Ordering and failure semantics (onion model):

    - ``before_process`` runs in **registration order**;
      ``after_process`` runs in **reverse** registration order, so the
      first-registered middleware wraps outermost.
    - ``on_error`` runs in registration order and the **first non-None**
      result wins (remaining middlewares are not consulted).
    - A middleware that **raises** is logged and skipped — the item
      continues through the rest of the chain (fail-open). Don't rely on
      an exception in ``before_process`` to block an item; return ``None``
      instead, which skips it explicitly.
    """

    @abstractmethod
    async def before_process(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """
        Called before processing each item, in registration order.

        Args:
            work_item: The work item about to be processed

        Returns:
            Modified work item, or None to skip processing this item
            (the item is recorded as failed with "Skipped by middleware").
        """
        pass

    @abstractmethod
    async def after_process(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """
        Called after processing each item, in reverse registration order
        (onion-style: the first-registered middleware sees the result last).

        Args:
            result: The result of processing

        Returns:
            Modified result (can add metadata, modify output, etc.)
        """
        pass

    @abstractmethod
    async def on_error(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        error: Exception,
    ) -> WorkItemResult[TOutput, TContext] | None:
        """
        Called when an error occurs during processing, in registration
        order; the first middleware returning non-None handles the error.

        Args:
            work_item: The work item that failed
            error: The exception that was raised

        Returns:
            Custom result to use instead of error, or None to use default error handling
        """
        pass


class BaseMiddleware(Middleware[TInput, TOutput, TContext]):
    """Base middleware with no-op implementations."""

    async def before_process(
        self, work_item: LLMWorkItem[TInput, TOutput, TContext]
    ) -> LLMWorkItem[TInput, TOutput, TContext] | None:
        """Default: pass through unchanged."""
        return work_item

    async def after_process(
        self, result: WorkItemResult[TOutput, TContext]
    ) -> WorkItemResult[TOutput, TContext]:
        """Default: pass through unchanged."""
        return result

    async def on_error(
        self,
        work_item: LLMWorkItem[TInput, TOutput, TContext],
        error: Exception,
    ) -> WorkItemResult[TOutput, TContext] | None:
        """Default: use default error handling."""
        return None
