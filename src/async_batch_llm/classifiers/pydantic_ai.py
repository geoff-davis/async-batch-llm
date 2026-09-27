"""Classification of PydanticAI's provider-neutral exceptions."""

from ..strategies.errors import DefaultErrorClassifier, ErrorInfo
from .openai import OpenAIErrorClassifier


class PydanticAIErrorClassifier(OpenAIErrorClassifier):
    """Use HTTP status policy for model failures and stop on usage limits."""

    def classify(self, exception: Exception) -> ErrorInfo:
        info = self._framework_prelude(exception)
        if info is not None:
            return info
        try:
            from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
        except ImportError:
            return DefaultErrorClassifier().classify(exception)
        if isinstance(exception, UsageLimitExceeded):
            return ErrorInfo(False, False, False, "usage_limit_exceeded")
        if isinstance(exception, ModelHTTPError):
            return self._classify_status_error(exception)
        return DefaultErrorClassifier().classify(exception)
