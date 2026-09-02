#!/usr/bin/env python3
"""Materialize NOTSOFAR1 eval audio + references for Nemotron 3 card-style rows.

Card conditions (NVIDIA model card, Evaluation Datasets):
  - NOTSOFAR1 Eval MHM = "mix of headset microphones" -> equal-weight average of
    close_talk/CT_*.wav per meeting
  - NOTSOFAR1 Eval SC  = "far-field single-channel"   -> ch0.wav of one
    single-channel device per meeting (first sc_* directory sorted by name;
    NVIDIA's exact device/session list is unpublished)

References: NVIDIA scored against unpublished FastMSS forced alignments. We build
RTTMs from the released gt_transcription.json word timings, merging consecutive
same-speaker words when the inter-word gap is <= 0.2 s (mirrors the
nttcslab-sp/diar-forced-alignment word-alignment convention used for AMI and
AliMeeting). Our NOTSOFAR rows are therefore protocol-adjacent, not
protocol-identical — same audio conditions and scoring settings, different
reference timing source.

Inputs  : ~/FluidAudioDatasets/notsofar/hf/benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG/MTG_*
Outputs : ~/FluidAudioDatasets/notsofar/card/{eval_mhm,eval_sc}/<meeting>.wav
          ~/FluidAudioDatasets/notsofar/card/rttm/<meeting>.rttm
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "FluidAudioDatasets" / "notsofar"
MTG_ROOT = ROOT / "hf" / "benchmark-datasets" / "eval_set" / "240825.1_eval_full_with_GT" / "MTG"
MHM_OUT = ROOT / "card" / "eval_mhm"
SC_OUT = ROOT / "card" / "eval_sc"
RTTM_OUT = ROOT / "card" / "rttm"

WORD_MERGE_GAP = 0.2  # seconds


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def build_rttm(meeting_dir: Path, meeting: str) -> bool:
    gt_path = meeting_dir / "gt_transcription.json"
    if not gt_path.exists():
        return False
    utterances = json.loads(gt_path.read_text())

    # Word-level segments per speaker, merged at <= WORD_MERGE_GAP gaps.
    words = []
    for utt in utterances:
        spk = utt["speaker_id"]
        timing = utt.get("word_timing") or []
        for _, start, end in timing:
            words.append((spk, float(start), float(end)))
        if not timing:
            words.append((spk, float(utt["start_time"]), float(utt["end_time"])))
    words.sort(key=lambda w: (w[0], w[1]))

    segments = []
    for spk, start, end in words:
        if segments and segments[-1][0] == spk and start - segments[-1][2] <= WORD_MERGE_GAP:
            segments[-1][2] = max(segments[-1][2], end)
        else:
            segments.append([spk, start, end])
    segments.sort(key=lambda s: s[1])

    lines = [
        f"SPEAKER {meeting} 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>"
        for spk, start, end in segments
        if end > start
    ]
    (RTTM_OUT / f"{meeting}.rttm").write_text("\n".join(lines) + "\n")
    return True


def materialize():
    for d in (MHM_OUT, SC_OUT, RTTM_OUT):
        d.mkdir(parents=True, exist_ok=True)

    meetings = sorted(p for p in MTG_ROOT.glob("MTG_*") if p.is_dir())
    if not meetings:
        sys.exit(f"no meetings found under {MTG_ROOT}")

    n_mhm = n_sc = 0
    for meeting_dir in meetings:
        meeting = meeting_dir.name
        if not build_rttm(meeting_dir, meeting):
            print(f"skip {meeting}: no gt_transcription.json")
            continue

        mhm_out = MHM_OUT / f"{meeting}.wav"
        ct_wavs = sorted((meeting_dir / "close_talk").glob("CT_*.wav"))
        if ct_wavs and not mhm_out.exists():
            cmd = ["ffmpeg", "-nostdin", "-v", "error"]
            for wav in ct_wavs:
                cmd += ["-i", str(wav)]
            cmd += [
                "-filter_complex", f"amix=inputs={len(ct_wavs)}:duration=longest",
                "-ar", "16000", "-c:a", "pcm_s16le", str(mhm_out),
            ]
            run(cmd)
            n_mhm += 1

        sc_out = SC_OUT / f"{meeting}.wav"
        sc_devices = sorted(d for d in meeting_dir.glob("sc_*") if (d / "ch0.wav").exists())
        if sc_devices and not sc_out.exists():
            run([
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(sc_devices[0] / "ch0.wav"),
                "-ar", "16000", "-c:a", "pcm_s16le", str(sc_out),
            ])
            n_sc += 1

    print(
        f"meetings {len(meetings)}: mhm {len(list(MHM_OUT.glob('*.wav')))} "
        f"sc {len(list(SC_OUT.glob('*.wav')))} rttm {len(list(RTTM_OUT.glob('*.rttm')))}"
    )


if __name__ == "__main__":
    materialize()
