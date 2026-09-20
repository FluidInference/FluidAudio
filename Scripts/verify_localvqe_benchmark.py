#!/usr/bin/env python3
"""Independently verify enhance-benchmark's versioned JSON; no inference needed."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def edits(hypothesis, reference):
    """Levenshtein S/D/I, with the benchmark's substitution-first tie break."""
    table = [list(range(len(reference) + 1))]
    for i, word in enumerate(hypothesis, 1):
        row = [i]
        for j, target in enumerate(reference, 1):
            row.append(table[i - 1][j - 1] if word == target else
                       1 + min(table[i - 1][j - 1], table[i - 1][j], row[j - 1]))
        table.append(row)
    i, j = len(hypothesis), len(reference)
    substitutions = deletions = insertions = 0
    while i or j:
        if i and j and hypothesis[i - 1] == reference[j - 1]:
            i, j = i - 1, j - 1
        elif i and j and table[i][j] == table[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            insertions += 1
            i -= 1
        else:
            deletions += 1
            j -= 1
    return substitutions, deletions, insertions


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: expected {expected}, got {actual}")


def verify(report, expected_files, require_improvement=False):
    require(report["schema_version"] == 2 and report["protocol"] == "localvqe-asr-v2", "Unsupported protocol")
    config = report["configuration"]
    require(config["asr"] == "parakeet-tdt-v3-int8" and config["asr_compute_units"] == "cpu-only",
            "Unexpected ASR configuration")
    require(config["sample_rate"] == 16000 and config["aggregation"] == "micro"
            and config["normalization"] == "TextNormalizer.normalize"
            and config["empty_reference_policy"] == "exclude-from-all-conditions", "Unexpected scoring protocol")
    require(report["chunk"] == "256ms", "This reference run requires the 256ms export")
    require(config["enhancement_compute_units"] == 0, "This reference run requires CPU-only enhancement")
    dataset = report["dataset"]
    require(dataset["repository"] == "FluidInference/aec-challenge-synthetic-mini", "Unexpected dataset")
    require(dataset["revision"] == "1f3714b5a3f98cedef1bbb017f21bbd7ae688596", "Unexpected dataset revision")
    require(dataset["archive_sha256"] == "45ff5d7acfce499558c25a0eace45eb819cec8aa76420fe733de7ee116ae548d",
            "Unexpected archive hash")
    require(dataset["metadata_sha256"] == "865aff8e66eb682c292f42a9d747d931f3a2f71f18e16fec53aea80dbdc2eacc",
            "Unexpected metadata hash")
    require(dataset["selection_order"] == "numeric fileid", "Unexpected selection order")
    selected = dataset["selected_fileids"]
    canonical = json.loads(Path(__file__).with_name("localvqe-dataset.json").read_text())
    require(0 < expected_files <= len(canonical["fileids"]), "Invalid expected file count")
    require(selected == canonical["fileids"][:expected_files], "Incomplete or reordered input selection")
    excluded = report["excluded_empty_reference_fileids"]
    require(len(excluded) == len(set(excluded)), "Duplicate excluded IDs")
    rows = report["files"]
    require(bool(rows), "No scored examples")
    scored = [row["fileid"] for row in rows]
    require(scored == [i for i in selected if i not in excluded], "Scored rows do not match selected IDs")
    require(set(scored).isdisjoint(excluded) and set(scored) | set(excluded) == set(selected),
            "Unaccounted or overlapping exclusions")
    audio_hashes = dataset["audio_files_sha256"]
    require(set(audio_hashes) == {f"fileid_{i}_{kind}.wav" for i in selected for kind in ("mic", "lpb", "clean")},
            "Incomplete audio fingerprints")
    for name, digest in audio_hashes.items():
        require(digest == canonical["audio_files_sha256"][name], f"Audio differs from pinned archive: {name}")
    model_hashes = report["model_files_sha256"]
    model_roots = {"/".join(name.split("/")[:2]) for name in model_hashes}
    require(model_roots == {
        "parakeet-v3/Preprocessor.mlmodelc", "parakeet-v3/Encoder.mlmodelc",
        "parakeet-v3/Decoder.mlmodelc", "parakeet-v3/JointDecisionv3.mlmodelc",
        "parakeet-v3/parakeet_vocab.json", "localvqe/localvqe-v1.3-4.8M-256ms.mlmodelc",
        "localvqe/localvqe-v1.2-1.3M-256ms.mlmodelc",
    }, "Incomplete model fingerprints")
    for name, digest in {**audio_hashes, **model_hashes}.items():
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"Invalid SHA256 for {name}")
    conditions = ["unprocessed", "localvqe-v1.3", "localvqe-v1.2"]
    require(report["conditions"] == conditions and set(report["summary"]) == set(conditions),
            "Missing, extra or duplicate conditions")
    totals = {condition: Counter() for condition in conditions}
    for row in rows:
        reference = row["reference"].split()
        far = row["far_reference"].split()
        require(bool(reference), f"Empty reference in scored file {row['fileid']}")
        close(row["ref_words"], len(reference), "reference word count")
        close(row["far_words"], len(far), "far-end word count")
        duration = row["audio_seconds"]
        require(math.isfinite(duration) and duration > 0, "Invalid audio duration")
        for condition in conditions:
            hypothesis = row[f"{condition}_hyp"].split()
            substitutions, deletions, insertions = edits(hypothesis, reference)
            hits = len(reference) - deletions - substitutions
            errors = substitutions + deletions + insertions
            leakage = sum(((Counter(hypothesis) - Counter(reference)) & Counter(far)).values())
            for field, value in (("substitutions", substitutions), ("deletions", deletions),
                                 ("insertions", insertions), ("hits", hits), ("errors", errors), ("leaked", leakage),
                                 ("recall", hits / len(reference)), ("wer", errors / len(reference))):
                close(row[f"{condition}_{field}"], value, f"{row['fileid']} {condition}.{field}")
            elapsed = row[f"{condition}_enhancement_seconds"]
            require(math.isfinite(elapsed) and (elapsed == 0 if condition == "unprocessed" else elapsed > 0),
                    f"Invalid enhancement timing for {condition}")
            totals[condition].update({
                "files": 1, "reference_words": len(reference), "far_end_words": len(far),
                "hits": hits, "errors": errors, "leaked_words": leakage,
                "audio_seconds": duration, "enhancement_seconds": elapsed,
            })
    for condition, total in totals.items():
        total["recall"] = total["hits"] / total["reference_words"]
        total["wer"] = total["errors"] / total["reference_words"]
        total["leakage"] = total["leaked_words"] / total["far_end_words"] if total["far_end_words"] else 0
        total["rtfx"] = total["audio_seconds"] / total["enhancement_seconds"] if total["enhancement_seconds"] else 0
        for field, value in total.items():
            close(report["summary"][condition][field], value, f"{condition}.{field}")
    if require_improvement:
        require(expected_files == 200, "Quality comparison requires the full 200-file selection")
        baseline = totals["unprocessed"]
        require(baseline["far_end_words"] > 0, "Cannot evaluate leakage without far-end reference words")
        for condition in conditions[1:]:
            require(totals[condition]["recall"] > baseline["recall"], f"{condition} did not improve recall")
            require(totals[condition]["leakage"] < baseline["leakage"], f"{condition} did not reduce leakage")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-files", type=int, default=200)
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    totals = verify(report, args.expected_files, args.require_improvement)
    lines = ["## LocalVQE ASR benchmark", "",
             f"Verified {args.expected_files} selected examples; {len(report['files'])} scored; "
             f"{len(report['excluded_empty_reference_fileids'])} empty references excluded.", "",
             "| Condition | Recall | WER | Leakage | Enhancement RTFx |",
             "|---|---:|---:|---:|---:|"]
    for condition, total in totals.items():
        speed = f"{total['rtfx']:.2f}x" if condition != "unprocessed" else "—"
        lines.append(f"| {condition} | {total['recall']:.2%} | {total['wer']:.2%} | {total['leakage']:.2%} | {speed} |")
    lines += ["", "Protocol: `localvqe-asr-v2`, Parakeet v3 int8, CPU-only, 256ms LocalVQE exports.",
              "Exploratory training-shard study with machine transcripts. CI timing is not device performance.",
              "Model/audio fingerprints, raw counts and excluded IDs are in the JSON artifact."]
    output = "\n".join(lines) + "\n"
    print(output)
    if args.markdown:
        args.markdown.write_text(output)


if __name__ == "__main__":
    main()
