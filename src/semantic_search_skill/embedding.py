from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Iterable

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_PROVIDER = "openai"
DEFAULT_MAX_INPUT_CHARS = 3000


class EmbeddingClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 8,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.max_retries = max_retries
        self.max_input_chars = max_input_chars

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        normalized = [_normalize_input(text, self.max_input_chars) for text in texts]
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(input=normalized, model=self.model)
                return [item.embedding for item in response.data]
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                if attempt >= self.max_retries:
                    raise
                time.sleep(_retry_delay(exc, attempt))
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt >= self.max_retries:
                    raise
                time.sleep(_retry_delay(exc, attempt))
        raise RuntimeError("unreachable embedding retry state")

    def embed_batches_parallel(self, texts: list[str], batch_size: int, workers: int) -> tuple[list[list[float]], int]:
        batches = list(_batched(texts, batch_size))
        if not batches:
            return [], 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            results = list(executor.map(self.embed_batch, batches))
        embeddings: list[list[float]] = []
        for batch_result in results:
            embeddings.extend(batch_result)
        return embeddings, len(batches)


def _batched(values: Iterable[str], size: int) -> Iterable[list[str]]:
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(0.5 * (2**attempt), 30.0)


def _normalize_input(text: str, max_chars: int) -> str:
    normalized = text.replace("\n", " ").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars]
