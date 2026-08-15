#!/usr/bin/env python3
"""Build FLEURS speech-translation pair manifests for `canary-transcribe --translate-benchmark`.

The FluidInference/fleurs-full cache (as downloaded by the fleurs benchmarks)
stores per-language wavs + normalized transcripts but loses the cross-language
alignment ids, so this script re-derives alignment from the original
google/fleurs test TSVs: local wav -> TSV row (matched on normalized
transcript) -> FLEURS sentence id -> target-language raw (cased, punctuated)
transcription as the translation reference.

Usage:
    python3 Scripts/canary_fleurs_translation_pairs.py --source en_us --target de_de \
        --output pairs_en_de.json
    swift run fluidaudiocli canary-transcribe --translate-benchmark pairs_en_de.json \
        --source-lang en --translate-to de

Score exactly afterwards:  uv run --with sacrebleu python -c "
import json, sacrebleu; d = json.load(open('pairs_en_de.hyps.json'))
print(sacrebleu.corpus_bleu(d['hypotheses'], [d['references']]).format())"
"""

import argparse
import csv
import json
import os
import urllib.request

FLEURS_CACHE = os.path.expanduser("~/Library/Application Support/FluidAudio/FLEURS-full")
TSV_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{lang}/test.tsv"


def load_tsv(lang: str):
    path = f"/tmp/fleurs_{lang}_test.tsv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        # Download to a temp name and rename atomically so an interrupted
        # download can't leave a truncated file that later runs trust.
        tmp = path + ".part"
        urllib.request.urlretrieve(TSV_URL.format(lang=lang), tmp)
        os.replace(tmp, path)
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) >= 4:
                rows.append({"id": r[0], "raw": r[2], "norm": r[3].strip()})
    if not rows:
        raise SystemExit(f"{path} parsed to 0 rows — delete it and re-run to re-download")
    return rows


def load_local_transcripts(lang: str):
    out = {}
    with open(f"{FLEURS_CACHE}/{lang}/{lang}.trans.txt") as f:
        for line in f:
            line = line.strip()
            if line:
                fid, _, text = line.partition(" ")
                out[fid] = text.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="FLEURS code of the spoken language, e.g. en_us")
    ap.add_argument("--target", required=True, help="FLEURS code of the reference language, e.g. de_de")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    src_rows = load_tsv(args.source)
    tgt_rows = load_tsv(args.target)
    local = load_local_transcripts(args.source)

    by_norm = {}
    for r in src_rows:
        by_norm.setdefault(r["norm"], []).append(r)
    tgt_by_id = {}
    for r in tgt_rows:
        tgt_by_id.setdefault(r["id"], r)

    pairs, unmatched, no_ref = [], 0, 0
    for fid, text in sorted(local.items()):
        rows = by_norm.get(text)
        if not rows:
            unmatched += 1
            continue
        ref = tgt_by_id.get(rows[0]["id"])
        if not ref:
            no_ref += 1
            continue
        pairs.append(
            {
                "audio": f"{FLEURS_CACHE}/{args.source}/{fid}.wav",
                "reference": ref["raw"],
                "id": rows[0]["id"],
            }
        )

    with open(args.output, "w") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(
        f"{args.source} -> {args.target}: {len(pairs)} pairs "
        f"({unmatched} local files unmatched in TSV, {no_ref} without target reference)"
    )


if __name__ == "__main__":
    main()
