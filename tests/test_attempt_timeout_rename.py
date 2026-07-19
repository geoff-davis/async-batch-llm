"""Tests for the timeout_per_item -> attempt_timeout rename (issue #98)."""

import dataclasses
import warnings

import pytest

from async_batch_llm import ProcessorConfig


class TestNewName:
    def test_attempt_timeout_default(self):
        config = ProcessorConfig()
        assert config.attempt_timeout == 120.0

    def test_attempt_timeout_explicit(self):
        config = ProcessorConfig(attempt_timeout=45.0)
        assert config.attempt_timeout == 45.0

    def test_new_name_emits_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            config = ProcessorConfig(attempt_timeout=45.0)
        assert config.attempt_timeout == 45.0

    def test_attempt_timeout_validated(self):
        with pytest.raises(ValueError, match="attempt_timeout must be > 0"):
            ProcessorConfig(attempt_timeout=0)


class TestDeprecatedAlias:
    def test_alias_still_works_with_warning(self):
        with pytest.warns(DeprecationWarning, match="timeout_per_item.*deprecated"):
            config = ProcessorConfig(timeout_per_item=45.0)
        assert config.attempt_timeout == 45.0

    def test_alias_works_positionally(self):
        # timeout_per_item was historically the second positional field.
        with pytest.warns(DeprecationWarning):
            config = ProcessorConfig(5, 45.0)
        assert config.max_workers == 5
        assert config.attempt_timeout == 45.0

    def test_warning_emitted_once_per_construction(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ProcessorConfig(timeout_per_item=45.0)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1

    def test_both_names_raise(self):
        with pytest.raises(ValueError, match="attempt_timeout only"):
            ProcessorConfig(timeout_per_item=45.0, attempt_timeout=60.0)

    def test_both_names_raise_even_when_equal(self):
        with pytest.raises(ValueError, match="attempt_timeout only"):
            ProcessorConfig(timeout_per_item=45.0, attempt_timeout=45.0)

    def test_alias_validated(self):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="attempt_timeout"):
            ProcessorConfig(timeout_per_item=-1)


class TestAttributeCompat:
    def test_reading_old_attribute_warns_and_returns_value(self):
        config = ProcessorConfig(attempt_timeout=45.0)
        with pytest.warns(DeprecationWarning, match="timeout_per_item is deprecated"):
            assert config.timeout_per_item == 45.0

    def test_writing_old_attribute_warns_and_sets_new_field(self):
        config = ProcessorConfig()
        with pytest.warns(DeprecationWarning, match="timeout_per_item is deprecated"):
            config.timeout_per_item = 33.0
        assert config.attempt_timeout == 33.0

    def test_dataclasses_replace_round_trips_silently(self):
        config = ProcessorConfig(attempt_timeout=45.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            clone = dataclasses.replace(config)
            override = dataclasses.replace(config, attempt_timeout=60.0)
        assert clone.attempt_timeout == 45.0
        assert override.attempt_timeout == 60.0

    def test_repr_shows_new_field_only(self):
        text = repr(ProcessorConfig(attempt_timeout=45.0))
        assert "attempt_timeout=45.0" in text
        # The deprecated alias is repr=False (total_timeout_per_item on the
        # nested GuardrailConfig is a different, non-deprecated knob).
        assert ", timeout_per_item=" not in text


class TestLegacyProcessorParameter:
    def test_processor_kwarg_still_forwards(self):
        from async_batch_llm import ParallelBatchProcessor

        with pytest.warns(DeprecationWarning, match="attempt_timeout"):
            processor = ParallelBatchProcessor(timeout_per_item=45.0)
        assert processor.config.attempt_timeout == 45.0

    def test_processor_kwarg_overrides_config(self):
        from async_batch_llm import ParallelBatchProcessor

        config = ProcessorConfig(attempt_timeout=30.0)
        with pytest.warns(DeprecationWarning):
            processor = ParallelBatchProcessor(timeout_per_item=45.0, config=config)
        assert processor.config.attempt_timeout == 45.0
        # The caller's config object is not mutated.
        assert config.attempt_timeout == 30.0
