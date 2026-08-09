from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError

import semantic_search_skill.embedding as embedding
from semantic_search_skill.embedding import EmbeddingClient, _normalize_input, _retry_delay, estimate_embedding_input_chars


def test_retry_delay_uses_retry_after_header() -> None:
    exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "2.5"}))

    assert _retry_delay(exc, attempt=4) == 2.5


def test_retry_delay_falls_back_to_exponential_backoff() -> None:
    assert _retry_delay(Exception("boom"), attempt=2) == 2.0


def test_normalize_input_replaces_newlines_and_clamps_length() -> None:
    assert _normalize_input("a\nb\n" + "c" * 20, max_chars=5) == "a b c"


def test_estimate_embedding_input_chars_uses_normalized_inputs() -> None:
    assert estimate_embedding_input_chars(["a\nb", "c" * 10], max_chars=4) == 7


def test_fallback_not_configured_reraises(monkeypatch) -> None:
    client = EmbeddingClient()
    primary_error = RuntimeError("primary failed")
    monkeypatch.setattr(client, "_embed_with", Mock(side_effect=primary_error))

    with pytest.raises(RuntimeError) as excinfo:
        client.embed_batch(["text"])
    assert excinfo.value is primary_error
    assert client._use_fallback is False


def test_fallback_triggers_on_primary_failure(monkeypatch) -> None:
    client = EmbeddingClient(fallback_base_url="http://localhost:1234/v1", fallback_model="fallback-model")
    embed_with = Mock(side_effect=[RuntimeError("primary failed"), [[1.0]]])
    monkeypatch.setattr(client, "_embed_with", embed_with)

    assert client.embed_batch(["text"]) == [[1.0]]
    assert client._use_fallback is True
    assert embed_with.call_count == 2
    primary_call, fallback_call = embed_with.call_args_list
    assert primary_call.args[0] is client.client
    assert primary_call.args[1] == client.model
    assert fallback_call.args[0] is client._fallback_client
    assert fallback_call.args[1] == "fallback-model"


def test_fallback_is_sticky(monkeypatch) -> None:
    client = EmbeddingClient(fallback_base_url="http://localhost:1234/v1", fallback_model="fallback-model")
    embed_with = Mock(side_effect=[RuntimeError("primary failed"), [[1.0]], [[2.0]]])
    monkeypatch.setattr(client, "_embed_with", embed_with)

    assert client.embed_batch(["first"]) == [[1.0]]
    assert client.embed_batch(["second"]) == [[2.0]]
    assert embed_with.call_count == 3
    assert embed_with.call_args_list[0].args[0] is client.client
    assert all(call.args[0] is client._fallback_client for call in embed_with.call_args_list[1:])


def test_fallback_uses_fallback_model_id(monkeypatch) -> None:
    client = EmbeddingClient(
        model="primary-model",
        fallback_base_url="http://localhost:1234/v1",
        fallback_model="fallback-model",
    )
    primary = Mock()
    primary.embeddings.create.side_effect = RuntimeError("primary failed")
    fallback = Mock()
    fallback.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])
    client.client = primary
    client._fallback_client = fallback

    assert client.embed_batch(["text"]) == [[1.0]]
    primary.embeddings.create.assert_called_once_with(input=["text"], model="primary-model")
    fallback.embeddings.create.assert_called_once_with(input=["text"], model="fallback-model")


def test_both_fail_raises_fallback_error(monkeypatch) -> None:
    client = EmbeddingClient(fallback_base_url="http://localhost:1234/v1", fallback_model="fallback-model")
    fallback_error = RuntimeError("fallback failed")
    embed_with = Mock(side_effect=[RuntimeError("primary failed"), fallback_error])
    monkeypatch.setattr(client, "_embed_with", embed_with)

    with pytest.raises(RuntimeError) as excinfo:
        client.embed_batch(["text"])
    assert excinfo.value is fallback_error
    assert client._use_fallback is True
    assert embed_with.call_args_list[1].args[0] is client._fallback_client


def test_primary_retries_before_fallback(monkeypatch) -> None:
    request = httpx.Request("POST", "https://example.test/embeddings")
    primary_error = APIConnectionError(message="primary unavailable", request=request)
    client = EmbeddingClient(fallback_base_url="http://localhost:1234/v1", fallback_model="fallback-model")
    primary = Mock()
    primary.embeddings.create.side_effect = [primary_error, primary_error, RuntimeError("primary failed")]
    fallback = Mock()
    fallback.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])
    client.client = primary
    client._fallback_client = fallback
    monkeypatch.setattr(embedding.time, "sleep", Mock())

    assert client.embed_batch(["text"]) == [[1.0]]
    assert primary.embeddings.create.call_count == 3
    assert fallback.embeddings.create.call_count == 1
