"""Mock agents for testing."""

import asyncio
import random
from collections.abc import Callable
from typing import Any, Generic, TypeVar

TOutput = TypeVar("TOutput")


class MockUsage:
    """Mock usage information mirroring pydantic-ai's v1 ``RunUsage`` names.

    Stores the modern ``input_tokens``/``output_tokens``/``cache_read_tokens``
    fields; ``request_tokens``/``response_tokens`` remain as read-only aliases
    for code written against pydantic-ai's legacy (pre-v1) usage API. The
    constructor keeps the legacy keyword names for backward compatibility.
    """

    def __init__(
        self,
        request_tokens: int = 100,
        response_tokens: int = 50,
        cache_read_tokens: int = 0,
        total_tokens: int | None = None,
    ):
        """Initialize mock usage. ``total_tokens`` defaults to input + output."""
        self.input_tokens = request_tokens
        self.output_tokens = response_tokens
        self.cache_read_tokens = cache_read_tokens
        self.total_tokens = (
            total_tokens if total_tokens is not None else request_tokens + response_tokens
        )

    @property
    def request_tokens(self) -> int:
        """Legacy alias for ``input_tokens``."""
        return self.input_tokens

    @property
    def response_tokens(self) -> int:
        """Legacy alias for ``output_tokens``."""
        return self.output_tokens


class MockResult(Generic[TOutput]):
    """Mock agent result."""

    def __init__(self, output: TOutput, usage_info: MockUsage | None = None):
        """
        Initialize mock result.

        Args:
            output: The output data
            usage_info: Token usage information
        """
        self.output = output
        self._usage = usage_info or MockUsage()

    def usage(self) -> MockUsage:
        """Get usage information."""
        return self._usage

    def all_messages(self) -> list[Any]:
        """Get all messages (empty for mock)."""
        return []


class MockAgent(Generic[TOutput]):
    """Mock agent for testing."""

    def __init__(
        self,
        response_factory: Callable[[str], TOutput] | None = None,
        latency: float = 0.1,
        failure_rate: float = 0.0,
        rate_limit_on_call: int | None = None,
        timeout_on_call: int | None = None,
        tokens_per_call: dict[str, int] | None = None,
        rng: random.Random | None = None,
    ):
        """
        Initialize mock agent.

        Args:
            response_factory: Function to generate responses from prompts
            latency: Simulated latency in seconds
            failure_rate: Probability of random failures (0.0 to 1.0)
            rate_limit_on_call: Call number to simulate rate limit (1-indexed, only triggers once)
            timeout_on_call: Call number to simulate timeout (1-indexed, only triggers once)
            tokens_per_call: Token usage to report per call (default: 10 input, 20 output,
                30 total). ``cached_input_tokens`` and ``total_tokens`` are honored too.
            rng: Random source for ``failure_rate``. Pass a seeded
                ``random.Random(42)`` for reproducible failure sequences;
                defaults to a fresh unseeded instance.
        """
        self.response_factory: Callable[[str], TOutput] = response_factory or self._default_response  # type: ignore[assignment,unused-ignore]
        self.latency = latency
        self.failure_rate = failure_rate
        self.rate_limit_on_call = rate_limit_on_call
        self.timeout_on_call = timeout_on_call
        self.tokens_per_call = tokens_per_call or {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        }
        self.call_count = 0
        self._rng = rng if rng is not None else random.Random()
        self._rate_limit_triggered = False
        self._timeout_triggered = False

    def _default_response(self, prompt: str) -> Any:
        """Default response generator."""
        return {"response": f"Mock response to: {prompt[:50]}"}

    async def run(self, prompt: str, **kwargs) -> MockResult[TOutput]:
        """
        Simulate agent.run().

        Args:
            prompt: The prompt to process
            **kwargs: Additional arguments (ignored)

        Returns:
            MockResult with generated output

        Raises:
            Exception: For simulated failures
        """
        self.call_count += 1

        # Simulate latency
        await asyncio.sleep(self.latency)

        # Simulate rate limit (only once)
        if (
            self.rate_limit_on_call is not None
            and self.call_count == self.rate_limit_on_call
            and not self._rate_limit_triggered
        ):
            self._rate_limit_triggered = True

            # The message alone matches the classifiers' rate-limit patterns
            # ("429", "resource_exhausted") — no class-name spoofing needed.
            class MockRateLimitError(Exception):
                """Mock rate limit error shaped like a Gemini 429."""

            raise MockRateLimitError("429 RESOURCE_EXHAUSTED")

        # Simulate timeout (only once)
        if (
            self.timeout_on_call is not None
            and self.call_count == self.timeout_on_call
            and not self._timeout_triggered
        ):
            self._timeout_triggered = True
            await asyncio.sleep(999)  # Will trigger timeout in processor

        # Simulate random failures
        if self._rng.random() < self.failure_rate:
            raise Exception(f"Random failure on call {self.call_count}")

        # Generate response
        output = self.response_factory(prompt)

        # Create usage info from tokens_per_call
        usage_info = MockUsage(
            request_tokens=self.tokens_per_call.get("input_tokens", 10),
            response_tokens=self.tokens_per_call.get("output_tokens", 20),
            cache_read_tokens=self.tokens_per_call.get("cached_input_tokens", 0),
            total_tokens=self.tokens_per_call.get("total_tokens"),
        )

        return MockResult(output, usage_info=usage_info)
