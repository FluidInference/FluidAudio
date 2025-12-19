#!/usr/bin/env python3
"""
Runs FluidAudio transcriptions and verifies that dictionary entries appear
in the normalized transcription output. Compares greedy vs beam search.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from normalizer import EnglishTextNormalizer

FLUIDAUDIO_DEFAULT = "../.build/release/fluidaudio"
DATA_DIR_DEFAULT = "test-dataset"
DEFAULT_FILE_ID = "4468654_chunk45"

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

# Column width for side-by-side display
COL_WIDTH = 45


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{NC}"


def pad_or_truncate(text: str, width: int) -> str:
    """Pad or truncate text to exact width, handling ANSI codes."""
    # Strip ANSI codes for length calculation
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    visible_text = ansi_escape.sub('', text)
    visible_len = len(visible_text)

    if visible_len >= width:
        # Truncate - need to be careful with ANSI codes
        return text[:width-3] + "..." if visible_len > width else text
    else:
        # Pad
        return text + " " * (width - visible_len)


def calculate_wer(reference: list[str], hypothesis: list[str]) -> float:
    """Calculate Word Error Rate using Levenshtein distance."""
    if not reference:
        return 100.0 if hypothesis else 0.0

    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return (dp[m][n] / m) * 100.0


@dataclass
class TestResult:
    """Rich result capturing word-level analysis for a single test."""

    file_id: str
    mode: str  # "greedy" or "beam"
    expected_words: list[str]
    transcribed_words: list[str]
    transcription_raw: str
    matching_words: list[str]
    missing_words: list[str]
    extra_words: list[str]
    dictionary_found: list[str]
    dictionary_missing: list[str]
    wer: float


@dataclass
class TestStats:
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[TestResult] = field(default_factory=list)
    all_missing_words: Counter = field(default_factory=Counter)

    def register_result(self, passed: bool) -> None:
        self.total_checks += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def register_skip(self) -> None:
        self.skipped += 1

    def add_test_result(self, result: TestResult) -> None:
        self.results.append(result)
        self.all_missing_words.update(result.missing_words)

    @property
    def average_wer(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.wer for r in self.results) / len(self.results)

    @property
    def perfect_count(self) -> int:
        return sum(1 for r in self.results if r.wer == 0.0)


class TranscriptionTester:
    def __init__(self, fluidaudio: Path, data_dir: Path):
        self.fluidaudio = fluidaudio
        self.data_dir = data_dir
        self.greedy_stats = TestStats()
        self.beam_stats = TestStats()
        self.normalizer = EnglishTextNormalizer()

    def run(self, file_id: str) -> None:
        wav_file = self.data_dir / f"{file_id}.wav"
        keywords_file = self.data_dir / f"{file_id}.keywords.txt"
        dictionary_file = self.data_dir / f"{file_id}.dictionary.txt"
        text_file = self.data_dir / f"{file_id}.text.txt"

        print("=" * 95)
        print(f"Testing: {colorize(file_id, BOLD)}")
        print("=" * 95)

        missing_reason = self._validate_files(wav_file, keywords_file, dictionary_file)
        if missing_reason:
            print(colorize(f"SKIPPED: {missing_reason}", YELLOW))
            self.greedy_stats.register_skip()
            self.beam_stats.register_skip()
            return

        dictionary_contents = dictionary_file.read_text(encoding="utf-8")
        expected_text = text_file.read_text(encoding="utf-8").strip() if text_file.exists() else ""
        expected_words = self._tokenize(expected_text)

        # Run both greedy and beam search
        greedy_output = self._transcribe(wav_file, keywords_file, beam_search=False)
        beam_output = self._transcribe(wav_file, keywords_file, beam_search=True)

        # Build results for both
        greedy_result = self._build_result(file_id, "greedy", expected_words, greedy_output, dictionary_contents)
        beam_result = self._build_result(file_id, "beam", expected_words, beam_output, dictionary_contents)

        # Update stats
        self._update_stats(self.greedy_stats, greedy_result, dictionary_contents)
        self._update_stats(self.beam_stats, beam_result, dictionary_contents)

        # Display side-by-side results
        self._display_comparison(expected_text, greedy_result, beam_result)

    def _build_result(
        self, file_id: str, mode: str, expected_words: list[str], transcription: str, dictionary_contents: str
    ) -> TestResult:
        """Build a TestResult from transcription output."""
        transcribed_words = self._tokenize(transcription)
        normalized_transcription = self._normalize(transcription)

        expected_set = set(expected_words)
        transcribed_set = set(transcribed_words)

        matching_words = sorted(expected_set & transcribed_set)
        missing_words = sorted(expected_set - transcribed_set)
        extra_words = sorted(transcribed_set - expected_set)

        # Dictionary word check
        dictionary_found = []
        dictionary_missing = []
        for dict_word in self._non_empty_lines(dictionary_contents):
            normalized_word = self._normalize(dict_word)
            if not normalized_word:
                continue
            if normalized_word in normalized_transcription:
                dictionary_found.append(dict_word)
            else:
                dictionary_missing.append(dict_word)

        wer = calculate_wer(expected_words, transcribed_words) if expected_words else 0.0

        return TestResult(
            file_id=file_id,
            mode=mode,
            expected_words=expected_words,
            transcribed_words=transcribed_words,
            transcription_raw=transcription,
            matching_words=matching_words,
            missing_words=missing_words,
            extra_words=extra_words,
            dictionary_found=dictionary_found,
            dictionary_missing=dictionary_missing,
            wer=wer,
        )

    def _update_stats(self, stats: TestStats, result: TestResult, dictionary_contents: str) -> None:
        """Update stats with result."""
        stats.add_test_result(result)
        for dict_word in self._non_empty_lines(dictionary_contents):
            normalized_word = self._normalize(dict_word)
            if not normalized_word:
                continue
            stats.register_result(dict_word in result.dictionary_found)

    def _display_comparison(self, expected_text: str, greedy: TestResult, beam: TestResult) -> None:
        """Display greedy vs beam results side-by-side."""
        w = COL_WIDTH

        print(f"\n{colorize('Expected:', CYAN)}")
        print(f"  {expected_text}")

        # Transcription outputs (full content)
        print(f"\n{colorize('GREEDY:', CYAN)}")
        print(f"  {greedy.transcription_raw}")
        print(f"\n{colorize('BEAM:', CYAN)}")
        print(f"  {beam.transcription_raw}")

        # Side-by-side header
        print(f"\n{'─' * w} │ {'─' * w}")
        print(f"{pad_or_truncate(colorize('GREEDY', BOLD), w)} │ {colorize('BEAM + VOCAB', BOLD)}")
        print(f"{'─' * w} │ {'─' * w}")

        # WER
        greedy_wer_color = GREEN if greedy.wer == 0 else (YELLOW if greedy.wer < 20 else RED)
        beam_wer_color = GREEN if beam.wer == 0 else (YELLOW if beam.wer < 20 else RED)
        greedy_wer = f"WER: {colorize(f'{greedy.wer:.1f}%', greedy_wer_color)}"
        beam_wer = f"WER: {colorize(f'{beam.wer:.1f}%', beam_wer_color)}"

        # Compare WER and show delta
        wer_delta = beam.wer - greedy.wer
        if abs(wer_delta) > 0.1:
            delta_color = GREEN if wer_delta < 0 else RED
            delta_str = f" ({'+' if wer_delta > 0 else ''}{wer_delta:.1f}%)"
            beam_wer += colorize(delta_str, delta_color)

        print(f"{pad_or_truncate(greedy_wer, w)} │ {beam_wer}")

        # Word counts
        greedy_counts = f"Words: {len(greedy.transcribed_words)}/{len(greedy.expected_words)}"
        beam_counts = f"Words: {len(beam.transcribed_words)}/{len(beam.expected_words)}"
        print(f"{pad_or_truncate(greedy_counts, w)} │ {beam_counts}")

        # Missing words (truncated)
        greedy_missing = f"Missing: {', '.join(greedy.missing_words[:5])}" if greedy.missing_words else "Missing: None"
        beam_missing = f"Missing: {', '.join(beam.missing_words[:5])}" if beam.missing_words else "Missing: None"
        if len(greedy.missing_words) > 5:
            greedy_missing += f" +{len(greedy.missing_words)-5}"
        if len(beam.missing_words) > 5:
            beam_missing += f" +{len(beam.missing_words)-5}"
        print(f"{pad_or_truncate(greedy_missing, w)} │ {beam_missing}")

        # Extra words (truncated)
        greedy_extra = f"Extra: {', '.join(greedy.extra_words[:5])}" if greedy.extra_words else "Extra: None"
        beam_extra = f"Extra: {', '.join(beam.extra_words[:5])}" if beam.extra_words else "Extra: None"
        if len(greedy.extra_words) > 5:
            greedy_extra += f" +{len(greedy.extra_words)-5}"
        if len(beam.extra_words) > 5:
            beam_extra += f" +{len(beam.extra_words)-5}"
        print(f"{pad_or_truncate(greedy_extra, w)} │ {beam_extra}")

        # Dictionary check
        print(f"{'─' * w} │ {'─' * w}")
        greedy_dict_pass = len(greedy.dictionary_found)
        greedy_dict_total = len(greedy.dictionary_found) + len(greedy.dictionary_missing)
        beam_dict_pass = len(beam.dictionary_found)
        beam_dict_total = len(beam.dictionary_found) + len(beam.dictionary_missing)

        greedy_dict_color = GREEN if greedy_dict_pass == greedy_dict_total else (YELLOW if greedy_dict_pass > 0 else RED)
        beam_dict_color = GREEN if beam_dict_pass == beam_dict_total else (YELLOW if beam_dict_pass > 0 else RED)

        greedy_dict = f"Dict: {colorize(f'{greedy_dict_pass}/{greedy_dict_total}', greedy_dict_color)}"
        beam_dict = f"Dict: {colorize(f'{beam_dict_pass}/{beam_dict_total}', beam_dict_color)}"
        print(f"{pad_or_truncate(greedy_dict, w)} │ {beam_dict}")

        # Show dictionary words with results
        all_dict_words = sorted(set(greedy.dictionary_found + greedy.dictionary_missing))
        for word in all_dict_words[:6]:  # Limit to 6 words
            g_mark = colorize('✓', GREEN) if word in greedy.dictionary_found else colorize('✗', RED)
            b_mark = colorize('✓', GREEN) if word in beam.dictionary_found else colorize('✗', RED)
            print(f"{pad_or_truncate(f'  {g_mark} {word}', w)} │   {b_mark} {word}")
        if len(all_dict_words) > 6:
            print(f"{pad_or_truncate(f'  ... +{len(all_dict_words)-6} more', w)} │   ... +{len(all_dict_words)-6} more")

        print(f"{'─' * w} │ {'─' * w}")

    def _validate_files(self, wav: Path, keywords: Path, dictionary: Path) -> str | None:
        if not wav.is_file():
            return "WAV file not found"
        if not dictionary.is_file() or dictionary.stat().st_size == 0:
            return "Dictionary file empty or not found"
        if not keywords.is_file():
            return "Keywords file not found"
        return None

    def _transcribe(self, wav_file: Path, keywords_file: Path, beam_search: bool = True) -> str:
        cmd = [
            str(self.fluidaudio),
            "transcribe",
            str(wav_file),
            "--custom-vocab",
            str(keywords_file),
        ]
        if beam_search:
            cmd.append("--beam-search")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout.strip()

    def _normalize(self, text: str) -> str:
        return self.normalizer(text).strip()

    def _tokenize(self, text: str) -> list[str]:
        """Normalize and split text into word tokens for comparison."""
        normalized = self._normalize(text)
        return normalized.split() if normalized else []

    @staticmethod
    def _non_empty_lines(blob: str) -> Iterable[str]:
        for line in blob.splitlines():
            line = line.strip()
            if line:
                yield line

    def summary(self) -> None:
        w = COL_WIDTH
        print("\n" + "=" * 95)
        print(colorize("TEST SUMMARY", BOLD))
        print("=" * 95)

        # Side-by-side header
        print(f"\n{'─' * w} │ {'─' * w}")
        print(f"{pad_or_truncate(colorize('GREEDY', BOLD), w)} │ {colorize('BEAM + VOCAB', BOLD)}")
        print(f"{'─' * w} │ {'─' * w}")

        # WER Statistics
        if self.greedy_stats.results:
            g_avg = self.greedy_stats.average_wer
            b_avg = self.beam_stats.average_wer
            g_color = GREEN if g_avg < 10 else (YELLOW if g_avg < 25 else RED)
            b_color = GREEN if b_avg < 10 else (YELLOW if b_avg < 25 else RED)

            g_wer = f"Avg WER: {colorize(f'{g_avg:.2f}%', g_color)}"
            b_wer = f"Avg WER: {colorize(f'{b_avg:.2f}%', b_color)}"

            # Delta
            wer_delta = b_avg - g_avg
            if abs(wer_delta) > 0.1:
                delta_color = GREEN if wer_delta < 0 else RED
                b_wer += colorize(f" ({'+' if wer_delta > 0 else ''}{wer_delta:.2f}%)", delta_color)

            print(f"{pad_or_truncate(g_wer, w)} │ {b_wer}")

            g_perfect = f"Perfect: {self.greedy_stats.perfect_count}/{len(self.greedy_stats.results)}"
            b_perfect = f"Perfect: {self.beam_stats.perfect_count}/{len(self.beam_stats.results)}"
            print(f"{pad_or_truncate(g_perfect, w)} │ {b_perfect}")

        # Dictionary Check Statistics
        print(f"{'─' * w} │ {'─' * w}")

        g_total = self.greedy_stats.total_checks
        b_total = self.beam_stats.total_checks
        g_passed = self.greedy_stats.passed
        b_passed = self.beam_stats.passed

        g_rate = (g_passed * 100 / g_total) if g_total else 0
        b_rate = (b_passed * 100 / b_total) if b_total else 0

        g_rate_color = GREEN if g_rate >= 90 else (YELLOW if g_rate >= 70 else RED)
        b_rate_color = GREEN if b_rate >= 90 else (YELLOW if b_rate >= 70 else RED)

        g_dict = f"Dict Pass: {colorize(f'{g_passed}/{g_total} ({g_rate:.1f}%)', g_rate_color)}"
        b_dict = f"Dict Pass: {colorize(f'{b_passed}/{b_total} ({b_rate:.1f}%)', b_rate_color)}"

        # Delta
        dict_delta = b_rate - g_rate
        if abs(dict_delta) > 0.1:
            delta_color = GREEN if dict_delta > 0 else RED
            b_dict += colorize(f" ({'+' if dict_delta > 0 else ''}{dict_delta:.1f}%)", delta_color)

        print(f"{pad_or_truncate(g_dict, w)} │ {b_dict}")

        g_skip = f"Skipped: {self.greedy_stats.skipped}"
        b_skip = f"Skipped: {self.beam_stats.skipped}"
        print(f"{pad_or_truncate(g_skip, w)} │ {b_skip}")

        print(f"{'─' * w} │ {'─' * w}")

    def exit_code(self) -> int:
        # Return 1 if beam search (the main test) has any failures
        return 1 if self.beam_stats.failed else 0


def collect_file_ids(data_dir: Path) -> list[str]:
    ids: list[str] = []
    suffix = ".dictionary.txt"
    for dict_file in sorted(data_dir.glob(f"*{suffix}")):
        if dict_file.stat().st_size == 0:
            continue
        name = dict_file.name
        if name.endswith(suffix):
            ids.append(name[: -len(suffix)])
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FluidAudio transcriptions against dictionary words.")
    parser.add_argument("file_id", nargs="?", default=DEFAULT_FILE_ID, help="File ID to test when --all is not set")
    parser.add_argument("--all", action="store_true", help="Run tests on all files with non-empty dictionaries")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, help="Directory containing extracted files")
    parser.add_argument("--fluidaudio", default=FLUIDAUDIO_DEFAULT, help="Path to the FluidAudio binary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    tester = TranscriptionTester(Path(args.fluidaudio), data_dir)

    #print("FluidAudio Keyword Test")
    #print("=============================")
    #print("")

    if args.all:
        file_ids = collect_file_ids(data_dir)
        if not file_ids:
            print(colorize("No dictionary files found to test.", YELLOW))
            tester.summary()
            return tester.exit_code()
        print("Running tests on all files with dictionary entries...\n")
        for file_id in file_ids:
            tester.run(file_id)
    else:
        tester.run(args.file_id)

    tester.summary()
    return tester.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
