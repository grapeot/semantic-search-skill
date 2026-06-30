from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk


class MarkdownChunker:
    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size

    def _split_frontmatter(self, content: str) -> tuple[dict[str, str], str, int]:
        if not content.startswith("---\n"):
            return {}, content.strip(), 1
        match = re.search(r"^---\s*$", content[4:], flags=re.MULTILINE)
        if not match:
            return {}, content.strip(), 1
        end = match.end() + 4
        frontmatter = content[4 : match.start() + 4]
        metadata: dict[str, str] = {}
        for line in frontmatter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"')
        return metadata, content[end:].strip(), content[:end].count("\n") + 1

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        metadata, body, line_offset = self._split_frontmatter(content)
        lines = body.splitlines()
        chunks: list[Chunk] = []
        current_header = ""
        current_lines: list[str] = []
        start_line = line_offset

        def flush(end_line: int) -> None:
            nonlocal current_lines
            if not current_lines:
                return
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{file_path}:{index}",
                    source_file=file_path,
                    text="\n".join(current_lines).strip(),
                    header=current_header,
                    position=(start_line, end_line),
                    metadata=metadata,
                )
            )
            current_lines = []

        for offset, line in enumerate(lines, start=line_offset):
            if line.startswith("#") and current_lines:
                flush(offset - 1)
                current_header = line
                current_lines = [line]
                start_line = offset
                continue
            if line.startswith("#"):
                current_header = line
                start_line = offset
            current_lines.append(line)
            if len("\n".join(current_lines)) >= self.max_chunk_size:
                flush(offset)
                current_lines = [current_header] if current_header else []
                start_line = offset
        flush(line_offset + len(lines) - 1)
        return [chunk for chunk in chunks if chunk.text]


def read_text_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")
