from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Iterable

from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_PROVIDER = "openai"


class EmbeddingClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        normalized = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=normalized, model=self.model)
        return [item.embedding for item in response.data]

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
