from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    source_file: str
    text: str
    header: str = ""
    position: tuple[int, int] = (0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["position"] = list(self.position)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        position = data.get("position", (0, 0))
        return cls(
            id=str(data["id"]),
            source_file=str(data["source_file"]),
            text=str(data["text"]),
            header=str(data.get("header", "")),
            position=(int(position[0]), int(position[1])),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": float(self.score), "chunk": self.chunk.to_dict()}
