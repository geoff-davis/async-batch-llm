"""Tests for the string-based strategy factory (issue #95).

No live API calls — every test constructs strategies with an explicit
dummy api_key (or asserts on the error path).
"""

import pytest

import async_batch_llm.models as models_module
from async_batch_llm import (
    DeepSeekModel,
    DeepSeekStrategy,
    GeminiModel,
    GeminiStrategy,
    OpenAIModel,
    OpenAIStrategy,
    OpenRouterModel,
    OpenRouterStrategy,
    llm,
)


class TestProviderPrefixes:
    def test_openai(self):
        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        assert isinstance(strategy, OpenAIStrategy)
        assert isinstance(strategy.model, OpenAIModel)
        assert strategy.model._model == "gpt-4o-mini"

    def test_openrouter(self):
        strategy = llm("openrouter:anthropic/claude-haiku-4-5", api_key="sk-or-test")
        assert isinstance(strategy, OpenRouterStrategy)
        assert isinstance(strategy.model, OpenRouterModel)
        assert strategy.model._model == "anthropic/claude-haiku-4-5"

    def test_deepseek(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test")
        assert isinstance(strategy, DeepSeekStrategy)
        assert isinstance(strategy.model, DeepSeekModel)
        assert strategy.model._model == "deepseek-v4-flash"

    def test_gemini(self):
        strategy = llm("gemini:gemini-2.5-flash", api_key="test-key")
        assert isinstance(strategy, GeminiStrategy)
        assert isinstance(strategy.model, GeminiModel)
        assert strategy.model._model == "gemini-2.5-flash"

    def test_model_id_may_contain_colons(self):
        strategy = llm("openrouter:meta-llama/llama-3.1-8b-instruct:free", api_key="sk-or-test")
        assert strategy.model._model == "meta-llama/llama-3.1-8b-instruct:free"

    def test_provider_prefix_is_case_insensitive(self):
        strategy = llm("OpenAI:gpt-4o-mini", api_key="sk-test")
        assert isinstance(strategy, OpenAIStrategy)

    def test_openai_family_models_own_their_client(self):
        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        assert strategy.model._owns_client is True


class TestKwargForwarding:
    def test_max_connections_reaches_capacity_advertisement(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test", max_connections=7)
        assert strategy.model.max_concurrency == 7
        assert strategy.max_concurrency == 7

    def test_deepseek_thinking_toggle(self):
        strategy = llm("deepseek:deepseek-v4-flash", api_key="sk-test", thinking=False)
        assert strategy.model._default_extra_body == {"thinking": {"type": "disabled"}}

    def test_system_instruction_forwarded(self):
        strategy = llm(
            "gemini:gemini-2.5-flash", api_key="test-key", system_instruction="Be terse."
        )
        assert strategy.model._default_system_instruction == "Be terse."

    def test_strategy_kwargs_forwarded(self):
        def parser(response):
            return len(response.text)

        strategy = llm(
            "openai:gpt-4o-mini",
            api_key="sk-test",
            response_parser=parser,
            temperature=None,
            generation_config={"max_tokens": 5},
        )
        assert strategy.response_parser is parser
        assert strategy.temperature is None
        assert strategy.generation_config == {"max_tokens": 5}

    def test_default_parser_returns_text(self):
        from async_batch_llm import LLMResponse

        strategy = llm("openai:gpt-4o-mini", api_key="sk-test")
        response = LLMResponse(text="hello", input_tokens=1, output_tokens=1, total_tokens=2)
        assert strategy.response_parser(response) == "hello"


class TestErrorMessages:
    def test_unknown_prefix_lists_valid_prefixes(self):
        with pytest.raises(ValueError) as exc_info:
            llm("anthropic:claude-sonnet-4-5")
        message = str(exc_info.value)
        assert "anthropic" in message
        for prefix in ("gemini", "openai", "openrouter", "deepseek"):
            assert f"'{prefix}:'" in message
        assert "async-batch-llm[openai]" in message

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="provider:model"):
            llm("gpt-4o-mini")

    def test_empty_model_id(self):
        with pytest.raises(ValueError, match="provider:model"):
            llm("openai:")

    def test_missing_openai_dependency_names_extra(self, monkeypatch):
        monkeypatch.setattr(models_module, "AsyncOpenAI", None)
        with pytest.raises(ImportError, match=r"async-batch-llm\[openai\]"):
            llm("openai:gpt-4o-mini", api_key="sk-test")

    def test_missing_gemini_dependency_names_extra(self, monkeypatch):
        monkeypatch.setattr(models_module, "genai", None)
        with pytest.raises(ImportError, match=r"async-batch-llm\[gemini\]"):
            llm("gemini:gemini-2.5-flash", api_key="test-key")

    def test_gemini_no_api_key_names_env_vars(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            llm("gemini:gemini-2.5-flash")

    def test_openrouter_no_api_key_names_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            llm("openrouter:anthropic/claude-haiku-4-5")


class TestEnvKeyResolution:
    def test_gemini_reads_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        strategy = llm("gemini:gemini-2.5-flash")
        assert isinstance(strategy, GeminiStrategy)

    def test_gemini_falls_back_to_gemini_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        strategy = llm("gemini:gemini-2.5-flash")
        assert isinstance(strategy, GeminiStrategy)

    def test_openrouter_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        strategy = llm("openrouter:anthropic/claude-haiku-4-5")
        assert isinstance(strategy, OpenRouterStrategy)


class TestErrorClassifierSelection:
    """The factory returns provider-named strategies, so the streaming API's
    classifier auto-selection keeps working."""

    @pytest.mark.parametrize(
        ("spec", "classifier_name"),
        [
            ("gemini:gemini-2.5-flash", "GeminiErrorClassifier"),
            ("openai:gpt-4o-mini", "OpenAIErrorClassifier"),
            ("openrouter:anthropic/claude-haiku-4-5", "OpenRouterErrorClassifier"),
            ("deepseek:deepseek-v4-flash", "OpenAIErrorClassifier"),
        ],
    )
    def test_recommended_classifier(self, spec, classifier_name):
        strategy = llm(spec, api_key="test-key")
        assert type(strategy.recommended_error_classifier()).__name__ == classifier_name
