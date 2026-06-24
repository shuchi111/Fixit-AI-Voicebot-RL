"""Tests for transcript parsing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.models import Speaker
from src.parser import (
    TranscriptParseError,
    format_parse_report,
    parse_transcript_dir,
    parse_transcript_file,
    parse_transcript_text,
)

TRANSCRIPTS_DIR = Path("data/transcripts")

SAMPLE_SHORT = """============================================================
CALL TRANSCRIPT
Call SID : 0a48de35-5722-f494-6991-e2dc09b67c76
============================================================

[2026-05-28 15:19:00 UTC]  AI ASSISTANT : Hello! This is Arya. Do you have a quick minute?

[2026-05-28 15:19:08 UTC]  CUSTOMER    : Tomorrow morning, after 10.

[2026-05-28 15:19:06 UTC]  AI ASSISTANT : And what's your budget range for this?

[2026-05-28 15:19:13 UTC]  CUSTOMER    : अभी टाइम नहीं है।

[2026-05-28 15:19:16 UTC]  AI ASSISTANT : And what's your budSure - it's ten percent to book.

[2026-05-28 15:19:21 UTC]  CUSTOMER    : What is this meanwhile?

[2026-05-28 15:19:25 UTC]  AI ASSISTANT : What type of property are you looking for - apartment or villa?

[2026-05-28 15:19:32 UTC]  CUSTOMER    : अभी टाइम नहीं

[2026-05-28 15:19:31 UTC]  AI ASSISTANT : No problem at all. Have a nice day.

============================================================
END OF TRANSCRIPT  (9 messages)
============================================================
"""


class TestParseTranscriptText:
    def test_parses_call_sid_and_turn_count(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        assert call.call_sid == "0a48de35-5722-f494-6991-e2dc09b67c76"
        assert call.message_count == 9
        assert call.declared_message_count == 9
        assert call.message_count_mismatch is False

    def test_preserves_file_order(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        assert call.turns[0].speaker is Speaker.AI_ASSISTANT
        assert "Arya" in call.turns[0].text
        assert call.turns[1].speaker is Speaker.CUSTOMER
        assert "Tomorrow morning" in call.turns[1].text

    def test_parses_timestamps(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        first = call.turns[0]
        assert first.timestamp == datetime(2026, 5, 28, 15, 19, 0)
        assert first.timestamp_raw == "2026-05-28 15:19:00 UTC"

    def test_out_of_order_timestamps_kept_in_file_order(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        # Turn index 2 is AI at 15:19:06, after customer at 15:19:08 (index 1)
        assert call.turns[2].timestamp < call.turns[1].timestamp

    def test_hindi_text_preserved(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        hindi_turn = call.turns[3]
        assert hindi_turn.speaker is Speaker.CUSTOMER
        assert "टाइम" in hindi_turn.text

    def test_ai_and_customer_turn_lists(self) -> None:
        call = parse_transcript_text(SAMPLE_SHORT)
        assert len(call.ai_turns) == 5
        assert len(call.customer_turns) == 4

    def test_message_count_mismatch_flagged(self) -> None:
        bad_footer = SAMPLE_SHORT.replace("(9 messages)", "(99 messages)")
        call = parse_transcript_text(bad_footer)
        assert call.message_count_mismatch is True

    def test_missing_header_raises(self) -> None:
        with pytest.raises(TranscriptParseError, match="Call SID"):
            parse_transcript_text("no header here")

    def test_no_turns_raises(self) -> None:
        text = """Call SID : abc-123\nEND OF TRANSCRIPT  (0 messages)"""
        with pytest.raises(TranscriptParseError, match="No conversation turns"):
            parse_transcript_text(text)


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestRealTranscripts:
    KNOWN = {
        "0a48de35-5722-f494-6991-e2dc09b67c76.txt": 9,
        "2e2fabba-cac4-5422-2f7b-85917c5528b8.txt": 23,
        "ba1828dc-fc95-db80-cb36-b07e99a474de.txt": 5,
    }

    @pytest.mark.parametrize("filename,expected_count", list(KNOWN.items()))
    def test_known_files(self, filename: str, expected_count: int) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / filename)
        assert call.message_count == expected_count
        assert call.message_count_mismatch is False

    def test_parse_entire_directory(self) -> None:
        calls, summary = parse_transcript_dir(TRANSCRIPTS_DIR)
        assert summary.parsed_calls == 1500
        assert summary.total_files == 1500
        assert not summary.failed_files
        assert len(calls) == 1500
        assert 5 <= min(summary.message_counts) <= 31

    def test_all_calls_have_both_speakers(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        for call in calls[:50]:
            assert call.ai_turns, f"{call.call_sid} has no AI turns"
            assert call.customer_turns, f"{call.call_sid} has no customer turns"

    def test_parse_report_format(self) -> None:
        _, summary = parse_transcript_dir(TRANSCRIPTS_DIR)
        report = format_parse_report(summary)
        assert "1500" in report
        assert "Parsed" in report
