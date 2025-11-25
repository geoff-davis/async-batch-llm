"""Batch LLM processing utilities for handling bulk LLM requests.

This module provides a flexible framework for processing multiple LLM requests
efficiently using a strategy pattern for provider-agnostic LLM integration.

Key features:
- Strategy pattern for any LLM provider (OpenAI, Anthropic, Google, LangChain, custom)
- Built-in strategies: PydanticAIStrategy, GeminiStrategy, GeminiCachedStrategy
- Provider-agnostic error classification
- Pluggable rate limit strategies
- Middleware pipeline for extensibility
- Observer pattern for monitoring
- Configuration-based setup

Example:
    >>> from batch_llm import (
    ...     ParallelBatchProcessor,
    ...     ProcessorConfig,
    ...     LLMWorkItem,
    ...     PydanticAIStrategy,
    ... )
    >>> from pydantic_ai import Agent
    >>>
    >>> agent = Agent("openai:gpt-4o-mini", result_type=MyOutput)
    >>> strategy = PydanticAIStrategy(agent=agent)
    >>> config = ProcessorConfig(max_workers=5, timeout_per_item=60.0)
    >>>
    >>> async with ParallelBatchProcessor(config=config) as processor:
    ...     await processor.add_work(LLMWorkItem(
    ...         item_id="item_1",
    ...         strategy=strategy,
    ...         prompt="Process this",
    ...     ))
    ...     result = await processor.process_all()

Type Aliases:
    For convenience, type aliases are provided to reduce verbosity:

    - ``SimpleBatchProcessor[T]``: Processor with string input, output type T, no context
      Equivalent to ``ParallelBatchProcessor[str, T, None]``

    - ``SimpleWorkItem[T]``: Work item with string input, output type T, no context
      Equivalent to ``LLMWorkItem[str, T, None]``

    - ``SimpleResult[T]``: Result with output type T, no context
      Equivalent to ``WorkItemResult[T, None]``

    Example using type aliases:
        >>> from batch_llm import SimpleBatchProcessor, SimpleWorkItem
        >>>
        >>> async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...     await processor.add_work(SimpleWorkItem[MyOutput](
        ...         item_id="item_1",
        ...         strategy=strategy,
        ...         prompt="Process this",
        ...     ))
"""

from typing import TypeVar

# Core classes
from .base import (
    BatchProcessor,
    BatchResult,
    LLMWorkItem,
    PostProcessorFunc,
    ProcessingStats,
    ProgressCallbackFunc,
    RetryState,
    TokenUsage,
    WorkItemResult,
)

# Classifiers
from .classifiers import GeminiErrorClassifier

# Configuration
from .core import ProcessorConfig, RateLimitConfig, RetryConfig

# LLM call strategies
from .llm_strategies import (
    GeminiCachedStrategy,
    GeminiResponse,
    GeminiStrategy,
    LLMCallStrategy,
    PydanticAIStrategy,
)

# Middleware
from .middleware import BaseMiddleware, Middleware

# Observers
from .observers import BaseObserver, MetricsObserver, ProcessingEvent, ProcessorObserver

# Main processor
from .parallel import ParallelBatchProcessor

# Error classification and rate limit strategies
from .strategies import (
    DefaultErrorClassifier,
    ErrorClassifier,
    ErrorInfo,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    FrameworkTimeoutError,
    RateLimitStrategy,
    TokenTrackingError,
)

# Type variable for output type in simplified aliases
_T = TypeVar("_T")

# Type aliases for common use cases
# These reduce verbosity for the most common pattern: string input, typed output, no context
SimpleBatchProcessor = ParallelBatchProcessor[str, _T, None]
"""Type alias for ParallelBatchProcessor[str, T, None].

Use when you have string prompts, a typed output, and no context.

Example:
    async with SimpleBatchProcessor[MyOutput](config=config) as processor:
        ...
"""

SimpleWorkItem = LLMWorkItem[str, _T, None]
"""Type alias for LLMWorkItem[str, T, None].

Use when creating work items with string prompts, typed output, and no context.

Example:
    item = SimpleWorkItem[MyOutput](item_id="1", strategy=strategy, prompt="Hello")
"""

SimpleResult = WorkItemResult[_T, None]
"""Type alias for WorkItemResult[T, None].

Use when working with results that have no context.

Example:
    result: SimpleResult[MyOutput] = results[0]
"""

__all__ = [
    # Core
    "BatchProcessor",
    "BatchResult",
    "LLMWorkItem",
    "PostProcessorFunc",
    "ProcessingStats",
    "ProgressCallbackFunc",
    "RetryState",
    "TokenUsage",
    "WorkItemResult",
    # Configuration
    "ProcessorConfig",
    "RateLimitConfig",
    "RetryConfig",
    # LLM Strategies
    "GeminiCachedStrategy",
    "GeminiResponse",
    "GeminiStrategy",
    "LLMCallStrategy",
    "PydanticAIStrategy",
    # Error Classification Strategies
    "ErrorClassifier",
    "ErrorInfo",
    "DefaultErrorClassifier",
    "FrameworkTimeoutError",
    "TokenTrackingError",
    "RateLimitStrategy",
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    # Middleware
    "Middleware",
    "BaseMiddleware",
    # Observers
    "ProcessorObserver",
    "BaseObserver",
    "MetricsObserver",
    "ProcessingEvent",
    # Classifiers
    "GeminiErrorClassifier",
    # Processor
    "ParallelBatchProcessor",
    # Type aliases (convenience)
    "SimpleBatchProcessor",
    "SimpleWorkItem",
    "SimpleResult",
]

# Version is read from package metadata (single source of truth in pyproject.toml)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("batch-llm")
except PackageNotFoundError:
    # Package not installed (e.g., running from source in development)
    __version__ = "0.0.0+dev"
