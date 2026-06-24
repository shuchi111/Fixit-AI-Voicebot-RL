"""Parse raw call transcript files into structured Call objects."""

from __future__ import annotations

import re
import statistics
from datetime import datetime
from pathlib import Path

from src.models import Call, ParseSummary, Speaker, Turn

HEADER_RE = re.compile(r"^Call SID\s*:\s*(\S+)\s*$", re.MULTILINE)
TURN_RE = re.compile(
    r"^\[([^\]]+)\]\s+(AI ASSISTANT|CUSTOMER)\s*:\s*(.*)$",
    re.MULTILINE,
)
FOOTER_RE = re.compile(
    r"^END OF TRANSCRIPT\s+\((\d+)\s+messages?\)\s*$",
    re.MULTILINE,
)

DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S UTC"


class TranscriptParseError(ValueError):
    """Raised when a transcript file cannot be parsed."""


def _parse_timestamp(raw: str, fmt: str) -> datetime:
    try:
        return datetime.strptime(raw.strip(), fmt)
    except ValueError as exc:
        raise TranscriptParseError(f"Invalid timestamp '{raw}' (expected format {fmt})") from exc


def _normalise_speaker(raw: str) -> Speaker:
    cleaned = re.sub(r"\s+", " ", raw.strip())
    try:
        return Speaker(cleaned)
    except ValueError as exc:
        raise TranscriptParseError(f"Unknown speaker '{raw}'") from exc


def parse_transcript_text(
    text: str,
    *,
    source_path: Path | None = None,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    validate_message_count: bool = True,
) -> Call:
    """Parse transcript content into a Call object."""
    header = HEADER_RE.search(text)
    if not header:
        raise TranscriptParseError("Missing Call SID header")

    call_sid = header.group(1)
    footer = FOOTER_RE.search(text)
    declared_count: int | None = None
    if footer:
        declared_count = int(footer.group(1))

    turns: list[Turn] = []
    for index, match in enumerate(TURN_RE.finditer(text)):
        timestamp_raw = match.group(1)
        speaker = _normalise_speaker(match.group(2))
        utterance = match.group(3).strip()
        timestamp = _parse_timestamp(timestamp_raw, timestamp_format)
        turns.append(
            Turn(
                index=index,
                speaker=speaker,
                text=utterance,
                timestamp=timestamp,
                timestamp_raw=timestamp_raw,
            )
        )

    if not turns:
        raise TranscriptParseError("No conversation turns found")

    mismatch = False
    if validate_message_count and declared_count is not None:
        mismatch = declared_count != len(turns)

    return Call(
        call_sid=call_sid,
        source_path=source_path or Path(""),
        turns=turns,
        declared_message_count=declared_count,
        message_count_mismatch=mismatch,
    )


def parse_transcript_file(
    path: Path,
    *,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    validate_message_count: bool = True,
) -> Call:
    """Parse a single transcript file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    call = parse_transcript_text(
        text,
        source_path=path,
        timestamp_format=timestamp_format,
        validate_message_count=validate_message_count,
    )
    return call


def parse_transcript_dir(
    directory: Path,
    *,
    timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    validate_message_count: bool = True,
) -> tuple[list[Call], ParseSummary]:
    """Parse all .txt transcripts in a directory."""
    files = sorted(directory.glob("*.txt"))
    calls: list[Call] = []
    failed: list[str] = []
    message_counts: list[int] = []
    mismatch_count = 0

    for path in files:
        try:
            call = parse_transcript_file(
                path,
                timestamp_format=timestamp_format,
                validate_message_count=validate_message_count,
            )
        except TranscriptParseError:
            failed.append(path.name)
            continue

        calls.append(call)
        message_counts.append(call.message_count)
        if call.message_count_mismatch:
            mismatch_count += 1

    summary = ParseSummary(
        total_files=len(files),
        parsed_calls=len(calls),
        failed_files=failed,
        message_counts=message_counts,
        mismatch_count=mismatch_count,
    )
    return calls, summary


def format_parse_report(summary: ParseSummary) -> str:
    """Human-readable parsing summary."""
    counts = summary.message_counts
    lines = [
        "Transcript parsing complete",
        f"  Files found:     {summary.total_files}",
        f"  Parsed:          {summary.parsed_calls}",
        f"  Failed:          {len(summary.failed_files)}",
        f"  Count mismatches:{summary.mismatch_count}",
    ]
    if counts:
        lines.extend(
            [
                f"  Turns/call min:  {min(counts)}",
                f"  Turns/call max:  {max(counts)}",
                f"  Turns/call mean: {statistics.mean(counts):.1f}",
                f"  Turns/call med:  {statistics.median(counts):.1f}",
            ]
        )
    if summary.failed_files:
        lines.append(f"  Failed files:    {', '.join(summary.failed_files[:5])}")
        if len(summary.failed_files) > 5:
            lines.append(f"                   ... and {len(summary.failed_files) - 5} more")
    return "\n".join(lines)
