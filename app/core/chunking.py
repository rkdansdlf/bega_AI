"""Semantic-first text chunking utilities for ingestion and re-embedding."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, List, Optional

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 650
    max_chars: int = 900
    min_chars: int = 180
    overlap_chars: int = 80


@dataclass(frozen=True)
class _Segment:
    text: str
    kind: str


def resolve_chunking_config(
    *,
    settings: Optional[Any] = None,
    target_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> ChunkingConfig:
    config = ChunkingConfig(
        target_chars=int(
            target_chars
            if target_chars is not None
            else getattr(
                settings, "rag_chunk_target_chars", ChunkingConfig.target_chars
            )
        ),
        max_chars=int(
            max_chars
            if max_chars is not None
            else getattr(settings, "rag_chunk_max_chars", ChunkingConfig.max_chars)
        ),
        min_chars=int(
            min_chars
            if min_chars is not None
            else getattr(settings, "rag_chunk_min_chars", ChunkingConfig.min_chars)
        ),
        overlap_chars=int(
            overlap_chars
            if overlap_chars is not None
            else getattr(
                settings,
                "rag_chunk_overlap_chars",
                ChunkingConfig.overlap_chars,
            )
        ),
    )
    if config.target_chars < 1:
        raise ValueError("Chunk target_chars must be >= 1")
    if config.max_chars < 1:
        raise ValueError("Chunk max_chars must be >= 1")
    if config.min_chars < 1:
        raise ValueError("Chunk min_chars must be >= 1")
    if config.overlap_chars < 0:
        raise ValueError("Chunk overlap_chars must be >= 0")
    if config.target_chars > config.max_chars:
        raise ValueError("Chunk target_chars must be <= max_chars")
    if config.min_chars > config.max_chars:
        raise ValueError("Chunk min_chars must be <= max_chars")
    if config.overlap_chars >= config.max_chars:
        raise ValueError("Chunk overlap_chars must be < max_chars")
    return config


def legacy_window_chunks(
    text: str,
    *,
    max_chars: int = 800,
    overlap_chars: int = 100,
) -> List[str]:
    if not text:
        return []
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(start + max_chars, length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        next_start = end - overlap_chars
        start = next_start if next_start > start else end
    return chunks


def _is_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line))


def _is_list_line(line: str) -> bool:
    return bool(_LIST_RE.match(line))


def _looks_like_table_start(lines: List[str], index: int) -> bool:
    current = lines[index].strip()
    if "|" not in current:
        return False
    if index + 1 >= len(lines):
        return False
    separator = lines[index + 1].strip()
    if "|" not in separator:
        return False
    cells = [cell.strip() for cell in separator.strip("|").split("|")]
    if not cells:
        return False
    return all(cell and "-" in cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and "|" in stripped)


def _segment_text(text: str) -> List[_Segment]:
    stripped_text = text.strip()
    if not stripped_text:
        return []

    lines = stripped_text.splitlines()
    segments: List[_Segment] = []
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue

        if _is_heading(line):
            segments.append(_Segment(line.strip(), "heading"))
            index += 1
            continue

        if _looks_like_table_start(lines, index):
            block = [lines[index].rstrip()]
            index += 1
            while (
                index < total_lines
                and lines[index].strip()
                and _is_table_line(lines[index])
            ):
                block.append(lines[index].rstrip())
                index += 1
            segments.append(_Segment("\n".join(block).strip(), "table"))
            continue

        if _is_list_line(line):
            block = [line.rstrip()]
            index += 1
            while index < total_lines:
                candidate = lines[index]
                if not candidate.strip():
                    break
                if _is_heading(candidate) or _looks_like_table_start(lines, index):
                    break
                if _is_list_line(candidate) or candidate.startswith("  "):
                    block.append(candidate.rstrip())
                    index += 1
                    continue
                break
            segments.append(_Segment("\n".join(block).strip(), "list"))
            continue

        block = [line.rstrip()]
        index += 1
        while index < total_lines:
            candidate = lines[index]
            if not candidate.strip():
                break
            if (
                _is_heading(candidate)
                or _is_list_line(candidate)
                or _looks_like_table_start(lines, index)
            ):
                break
            block.append(candidate.rstrip())
            index += 1
        segments.append(_Segment("\n".join(block).strip(), "paragraph"))

    return [segment for segment in segments if segment.text]


def _merge_small_tail(parts: List[str], *, max_chars: int, joiner: str) -> List[str]:
    if len(parts) < 2:
        return parts
    last = parts[-1]
    prev = parts[-2]
    if len(prev) + len(joiner) + len(last) > max_chars:
        return parts
    return parts[:-2] + [f"{prev}{joiner}{last}"]


def _merge_small_tail_with_kinds(
    parts: List[str],
    kinds: List[str],
    *,
    max_chars: int,
    joiner: str,
) -> List[str]:
    if len(parts) < 2 or len(parts) != len(kinds):
        return parts
    if kinds[-1] in {"table", "list"} or kinds[-2] in {"table", "list"}:
        return parts
    return _merge_small_tail(parts, max_chars=max_chars, joiner=joiner)


def _tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0:
        return ""
    suffix = text[-overlap_chars:].strip()
    if not suffix:
        return ""
    first_space = suffix.find(" ")
    if first_space > 0 and suffix[0].isalnum():
        trimmed = suffix[first_space + 1 :].strip()
        if trimmed:
            return trimmed
    return suffix


def _apply_sentence_overlap(
    parts: List[str],
    *,
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    if len(parts) <= 1 or overlap_chars <= 0:
        return parts

    overlapped: List[str] = []
    for index, part in enumerate(parts):
        if index == 0:
            overlapped.append(part)
            continue
        prefix = _tail_overlap(parts[index - 1], overlap_chars)
        if not prefix:
            overlapped.append(part)
            continue

        candidate = f"{prefix}\n{part}"
        if len(candidate) <= max_chars:
            overlapped.append(candidate)
            continue

        available_prefix = max_chars - len(part) - 1
        if available_prefix <= 0:
            overlapped.append(part)
            continue
        short_prefix = _tail_overlap(parts[index - 1], available_prefix)
        if short_prefix:
            overlapped.append(f"{short_prefix}\n{part}")
        else:
            overlapped.append(part)
    return overlapped


def _hard_split(text: str, *, max_chars: int, overlap_chars: int) -> List[str]:
    return legacy_window_chunks(
        text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


def _split_segment(segment: _Segment, config: ChunkingConfig) -> List[_Segment]:
    if len(segment.text) <= config.max_chars:
        return [segment]

    sentences = [
        piece.strip()
        for piece in _SENTENCE_SPLIT_RE.split(segment.text)
        if piece and piece.strip()
    ]
    if len(sentences) <= 1:
        return [
            _Segment(piece, "hard_split")
            for piece in _hard_split(
                segment.text,
                max_chars=config.max_chars,
                overlap_chars=config.overlap_chars,
            )
        ]

    parts: List[str] = []
    current = sentences[0]
    for sentence in sentences[1:]:
        candidate = f"{current} {sentence}"
        if len(candidate) <= config.target_chars:
            current = candidate
            continue
        if len(current) < config.min_chars and len(candidate) <= config.max_chars:
            current = candidate
            continue
        parts.append(current)
        current = sentence
    parts.append(current)
    parts = _merge_small_tail(parts, max_chars=config.max_chars, joiner=" ")

    normalized_parts: List[str] = []
    for part in parts:
        if len(part) <= config.max_chars:
            normalized_parts.append(part)
            continue
        normalized_parts.extend(
            _hard_split(
                part,
                max_chars=config.max_chars,
                overlap_chars=config.overlap_chars,
            )
        )

    overlapped_parts = _apply_sentence_overlap(
        normalized_parts,
        max_chars=config.max_chars,
        overlap_chars=config.overlap_chars,
    )
    return [_Segment(piece, "sentence_split") for piece in overlapped_parts if piece]


def smart_chunks(
    text: str,
    max_chars: Optional[int] = None,
    overlap: Optional[int] = None,
    *,
    settings: Optional[Any] = None,
    target_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[str]:
    config = resolve_chunking_config(
        settings=settings,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_chars=overlap_chars if overlap_chars is not None else overlap,
    )
    if not text:
        return []

    normalized_text = text.strip()
    if not normalized_text:
        return []
    if len(normalized_text) <= config.max_chars:
        return [normalized_text]

    expanded_segments: List[_Segment] = []
    for segment in _segment_text(normalized_text):
        expanded_segments.extend(_split_segment(segment, config))

    if not expanded_segments:
        return []

    chunks: List[str] = []
    chunk_kinds: List[str] = []
    current = expanded_segments[0].text
    current_kind = expanded_segments[0].kind
    for segment in expanded_segments[1:]:
        structural_boundary = current_kind in {"table", "list"} or segment.kind in {
            "table",
            "list",
        }
        if structural_boundary:
            chunks.append(current.strip())
            chunk_kinds.append(current_kind)
            current = segment.text
            current_kind = segment.kind
            continue
        joiner = "\n\n"
        candidate = f"{current}{joiner}{segment.text}"
        if len(candidate) <= config.target_chars:
            current = candidate
            current_kind = current_kind if current_kind == segment.kind else "mixed"
            continue
        if len(current) < config.min_chars and len(candidate) <= config.max_chars:
            current = candidate
            current_kind = current_kind if current_kind == segment.kind else "mixed"
            continue
        if len(candidate) <= config.max_chars and segment.kind == "heading":
            current = candidate
            current_kind = current_kind if current_kind == segment.kind else "mixed"
            continue
        chunks.append(current.strip())
        chunk_kinds.append(current_kind)
        current = segment.text
        current_kind = segment.kind

    if current.strip():
        chunks.append(current.strip())
        chunk_kinds.append(current_kind)

    chunks = _merge_small_tail_with_kinds(
        chunks,
        chunk_kinds,
        max_chars=config.max_chars,
        joiner="\n\n",
    )
    return [chunk for chunk in chunks if chunk]
