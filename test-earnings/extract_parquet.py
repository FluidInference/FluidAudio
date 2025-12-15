#!/usr/bin/env python3
"""
Extract audio and text fields from an ASR parquet file.
Usage: python extract_parquet.py <parquet_file> <output_directory>
"""

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _iterable_to_text(value: Any) -> str:
    """Convert iterable metadata to newline-delimited text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    if isinstance(value, Iterable):
        try:
            return "\n".join(str(item) for item in value)
        except TypeError:
            pass
    return str(value)


def _extract_audio_bytes(audio_field: Any) -> bytes:
    """Return raw audio bytes from the parquet row."""
    if isinstance(audio_field, dict) and "bytes" in audio_field:
        return audio_field["bytes"]
    if isinstance(audio_field, (bytes, bytearray, memoryview)):
        return bytes(audio_field)
    raise TypeError("Audio field does not contain bytes")


def extract(parquet_path: Path, output_dir: Path) -> None:
    df = pd.read_parquet(parquet_path)
    print(f"Found {len(df)} rows")

    for row in df.to_dict(orient="records"):
        file_id = row["file_id"]

        wav_path = output_dir / f"{file_id}.wav"
        wav_path.write_bytes(_extract_audio_bytes(row["audio"]))

        (output_dir / f"{file_id}.text.txt").write_text(str(row.get("text", "")))
        (output_dir / f"{file_id}.norm_text.txt").write_text(str(row.get("norm_text", "")))
        (output_dir / f"{file_id}.dictionary.txt").write_text(_iterable_to_text(row.get("dictionary")))
        (output_dir / f"{file_id}.keywords.txt").write_text(_iterable_to_text(row.get("keywords")))

        print(f"Extracted: {file_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio and metadata from ASR parquet files.")
    parser.add_argument("parquet_file", help="Input parquet file to read")
    parser.add_argument("output_directory", help="Directory to write extracted contents into")
    args = parser.parse_args()

    parquet_path = Path(args.parquet_file)
    output_dir = Path(args.output_directory)

    if not parquet_path.is_file():
        raise SystemExit(f"Error: Parquet file not found: {parquet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting data from: {parquet_path}")
    print(f"Output directory: {output_dir}")
    extract(parquet_path, output_dir)
    print(f"Done! Extracted files to {output_dir}")


if __name__ == "__main__":
    main()
