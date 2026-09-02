#!/usr/bin/env python3
"""Materialize AliMeeting Test audio for the NVIDIA Nemotron 3 card protocol.

Card conditions (model card, Evaluation Datasets):
  - AliMeeting Test Far  = far-field array audio  -> channel 0 of the 8-channel wav
  - AliMeeting Test Near = "mix of headset microphones" -> equal-weight average of
    the per-speaker N_SPK*.wav headset channels

Inputs  : ~/FluidAudioDatasets/alimeeting/Test_Ali/Test_Ali_{far,near}/audio_dir
Outputs : ~/FluidAudioDatasets/alimeeting/card/{far_ch0,near_mix}/<meeting>.wav
          where <meeting> matches the nttcslab-sp/diar-forced-alignment RTTM names
          (e.g. R8002_M8002).

Idempotent: existing outputs are skipped.
"""

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "FluidAudioDatasets" / "alimeeting"
FAR_IN = ROOT / "Test_Ali" / "Test_Ali_far" / "audio_dir"
NEAR_IN = ROOT / "Test_Ali" / "Test_Ali_near" / "audio_dir"
FAR_OUT = ROOT / "card" / "far_ch0"
NEAR_OUT = ROOT / "card" / "near_mix"

MEETING_RE = re.compile(r"^(R\d+_M\d+)")


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def materialize_far():
    FAR_OUT.mkdir(parents=True, exist_ok=True)
    for wav in sorted(FAR_IN.glob("*.wav")):
        m = MEETING_RE.match(wav.stem)
        if not m:
            print(f"skip (unrecognized name): {wav.name}")
            continue
        out = FAR_OUT / f"{m.group(1)}.wav"
        if out.exists():
            continue
        # Channel 0 of the far-field array, 16 kHz mono.
        run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
            "-af", "pan=mono|c0=c0", "-ar", "16000", "-c:a", "pcm_s16le", str(out),
        ])
        print(f"far  {out.name}")


def materialize_near():
    NEAR_OUT.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for wav in sorted(NEAR_IN.glob("*.wav")):
        m = MEETING_RE.match(wav.stem)
        if m:
            groups[m.group(1)].append(wav)
    for meeting, wavs in sorted(groups.items()):
        out = NEAR_OUT / f"{meeting}.wav"
        if out.exists():
            continue
        # Equal-weight average of the headset channels: amix with default
        # normalize=1 divides the sum by the input count.
        cmd = ["ffmpeg", "-nostdin", "-v", "error"]
        for wav in wavs:
            cmd += ["-i", str(wav)]
        cmd += [
            "-filter_complex", f"amix=inputs={len(wavs)}:duration=longest",
            "-ar", "16000", "-c:a", "pcm_s16le", str(out),
        ]
        run(cmd)
        print(f"near {out.name} ({len(wavs)} headsets)")


if __name__ == "__main__":
    if not FAR_IN.is_dir() or not NEAR_IN.is_dir():
        sys.exit(f"AliMeeting Test_Ali audio not found under {ROOT}")
    materialize_far()
    materialize_near()
    print(f"done: {len(list(FAR_OUT.glob('*.wav')))} far, {len(list(NEAR_OUT.glob('*.wav')))} near")
