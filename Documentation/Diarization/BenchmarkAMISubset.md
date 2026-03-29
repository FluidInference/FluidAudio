# Diarization Benchmarks

Hardware: 2024 MacBook Pro, 48GB RAM, M4 Pro, macOS Tahoe 26.0

Dataset: AMI SDM (Single Distant Microphone), 4-meeting subset — one session per speaker group for diversity.

All results use collar=0.25s, ignoreOverlap=true.

## Summary

| System | Avg DER | Avg RTFx | Mode |
|---|---|---|---|
| LS-EEND (AMI) | 25.7% | 53.9x | Streaming |
| Offline VBx | 21.8% | 97.5x | Offline |
| Streaming 5s/0.8 | 29.9% | 96.2x | Streaming |
| Sortformer (high-lat) | 34.3% | 120.3x | Streaming |

## Offline VBx

Pyannote segmentation + WeSpeaker embeddings + PLDA scoring + VBx clustering.

Default configuration: step ratio 0.2, minSegmentDurationSeconds 1.0, clustering threshold 0.7.

```bash
Scripts/diarizer_subset_benchmark.sh
# or manually:
swift run -c release fluidaudiocli diarization-benchmark --mode offline \
    --dataset ami-sdm --auto-download
```

```text
----------------------------------------------------------------------
Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
ES2004a          14.5      7.6      1.7      5.2     5/4        98.2
IS1009a          17.7      3.6      3.0     11.1     6/4        99.1
TS3003a          21.2     11.7      1.4      8.1     2/4        98.4
EN2002a          33.9      4.5      1.4     28.0     4/4        94.2
----------------------------------------------------------------------
AVERAGE          21.8      6.9      1.9     13.1      -         97.5
======================================================================
```

Full VoxConverse results (232 clips): 15.07% DER, 122x RTFx. See [Benchmarks.md](../Benchmarks.md) for details.

## Streaming (5s chunks, 0.8 threshold)

Pyannote segmentation + WeSpeaker embeddings + online SpeakerManager clustering.

Best streaming configuration: 5s chunks, 0s overlap, 0.8 clustering threshold.

```bash
Scripts/diarizer_subset_benchmark.sh
# or manually:
swift run -c release fluidaudiocli diarization-benchmark --mode streaming \
    --dataset ami-sdm --chunk-seconds 5.0 --overlap-seconds 0.0 \
    --threshold 0.8 --auto-download
```

```text
----------------------------------------------------------------------
Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
ES2004a          17.0      9.0      1.3      6.7     7/4        99.2
IS1009a          18.1      4.7      2.7     10.8     4/4       101.0
TS3003a          21.0     12.7      1.4      6.8     2/4       104.3
EN2002a          63.4      9.2      1.1     53.0     7/4        80.1
----------------------------------------------------------------------
AVERAGE          29.9      8.9      1.6     19.3      -         96.2
======================================================================
```

Full 7-meeting results: 26.2% DER, 223x RTFx. See [Benchmarks.md](../Benchmarks.md) for details.

EN2002a is a known difficult meeting for the streaming pipeline — aggressive speaker error (53%) due to over-fragmentation.

## Sortformer (NVIDIA High-Latency)

NVIDIA end-to-end Sortformer model, 30.4s chunk config.

Model: [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml)

```bash
Scripts/diarizer_subset_benchmark.sh
# or manually:
swift run -c release fluidaudiocli sortformer-benchmark \
    --nvidia-high-latency --hf --auto-download
```

```text
----------------------------------------------------------------------
Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
IS1009a          26.5     15.9      1.4      9.3     4/4       122.9
ES2004a          33.4     24.5      0.1      8.8     4/4       117.9
EN2002a          35.7     20.0      0.4     15.2     4/4       121.5
TS3003a          41.8     36.8      0.7      4.3     4/4       119.0
----------------------------------------------------------------------
AVERAGE          34.3     24.3      0.7      9.4      -        120.3
======================================================================
```

Full 16-meeting results: 31.7% DER, 126.7x RTFx. See [Benchmarks.md](../Benchmarks.md) for details.

## LS-EEND (AMI variant)

Linear Streaming End-to-End Neural Diarization from Westlake University.

Model: [GradientDescent2718/ls-eend-coreml](https://huggingface.co/GradientDescent2718/ls-eend-coreml)

```bash
Scripts/diarizer_subset_benchmark.sh
# or manually:
swift run -c release fluidaudiocli lseend-benchmark \
    --variant ami --auto-download
```

```text
----------------------------------------------------------------------
Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
TS3003a          19.0     16.6      0.8      1.6     4/4        47.5
IS1009a          23.4      8.0      2.6     12.8     4/4        57.7
EN2002a          24.5     19.7      1.1      3.6     4/4        53.2
ES2004a          35.8     13.3     19.2      3.2     4/4        57.2
----------------------------------------------------------------------
AVERAGE          25.7     14.4      5.9      5.3      -         53.9
======================================================================
```

Full 16-meeting results: 20.7% DER, 74.5x RTFx. See [Benchmarks.md](../Benchmarks.md) for details.

## Reproducing

Run all 4 systems on the default 4-meeting subset:

```bash
./Scripts/diarizer_subset_benchmark.sh
```

Run on all 16 AMI meetings:

```bash
./Scripts/diarizer_subset_benchmark.sh --all
```

Results are saved to `benchmark_results/` with timestamps. The script uses `caffeinate` to prevent sleep during long runs.
