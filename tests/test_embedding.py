from types import SimpleNamespace

from semantic_search_skill.embedding import _normalize_input, _retry_delay


def test_retry_delay_uses_retry_after_header() -> None:
    exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "2.5"}))

    assert _retry_delay(exc, attempt=4) == 2.5


def test_retry_delay_falls_back_to_exponential_backoff() -> None:
    assert _retry_delay(Exception("boom"), attempt=2) == 2.0


def test_normalize_input_replaces_newlines_and_clamps_length() -> None:
    assert _normalize_input("a\nb\n" + "c" * 20, max_chars=5) == "a b c"
