#!/usr/bin/env python3
"""Materialize full-length Earnings-22 recordings for the long-form ASR benchmark.

Earnings-22 is the standard long-form ASR benchmark (≈1 h files of accented
earnings calls). This pulls full-length audio + reference transcripts from the
`AudioLLMs/earnings22_test` dataset (16 kHz mono WAV embedded in parquet) and
writes them as `{id}.wav` + `{id}.txt` into FluidAudio's Application Support
directory, where `fluidaudiocli unified-benchmark --longform-dir <dir>` reads
them.

Usage:
    pip install pyarrow huggingface_hub
    python3 Scripts/materialize_earnings22_longform.py [num_shards]

Each parquet shard holds ~5 full-length files; pass the shard count to control
how much audio to materialize (default: 1 shard ≈ 5 files ≈ 5 h).
"""

import os
import re
import sys

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "AudioLLMs/earnings22_test"
REVISION = "refs/convert/parquet"
OUT = os.path.expanduser(
    "~/Library/Application Support/FluidAudio/earnings22-longform"
)


def main() -> None:
    nshards = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(OUT, exist_ok=True)

    count = 0
    for shard in range(nshards):
        path = hf_hub_download(
            REPO,
            f"default/partial-test/{shard:04d}.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            context = table.column("context")
            answer = table.column("answer")
            for i in range(table.num_rows):
                entry = context[i].as_py()
                src = entry.get("path") or f"shard{shard}_rg{rg}_{i}"
                file_id = re.sub(
                    r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(src))[0]
                )
                wav_bytes = entry["bytes"]
                # The .nlp reconstruction prefixes a junk "TOKEN" header word.
                text = re.sub(r"^\s*TOKEN\s+", "", answer[i].as_py() or "").strip()

                with open(os.path.join(OUT, file_id + ".wav"), "wb") as f:
                    f.write(wav_bytes)
                with open(os.path.join(OUT, file_id + ".txt"), "w") as f:
                    f.write(text)
                count += 1
                print(
                    f"  {file_id}: wav={len(wav_bytes) // 1024}KB "
                    f"ref_words={len(text.split())}"
                )

    print(f"materialized {count} files to {OUT}")


if __name__ == "__main__":
    main()
