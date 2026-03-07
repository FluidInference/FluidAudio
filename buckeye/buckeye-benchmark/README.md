# Buckeye Corpus - Forced Alignment Benchmark

Segmented utterances from the [Buckeye Corpus of Conversational Speech](https://buckeyecorpus.osu.edu/) (v2.0) with **human-annotated word-level timestamps**, prepared for forced alignment benchmarking.

## Dataset Stats

| Metric | Value |
|--------|-------|
| Speakers | 20 (s01-s20) |
| Segments | 2,478 |
| Total words | 145,762 |
| Total audio | 16.1 hours |
| Audio format | 16kHz mono PCM WAV |
| Avg segment | 23.3s |
| Words per segment | 3-123 (avg 59) |

## Structure

```
buckeye-forced-alignment-benchmark/
  manifest.json       # metadata + ground truth word timestamps
  audio/
    s0101a_000.wav    # segmented audio clips
    s0101a_001.wav
    ...
```

## manifest.json Format

```json
{
  "dataset": "Buckeye Corpus v2.0",
  "speakers": 20,
  "total_segments": 2478,
  "total_words": 145762,
  "samples": [
    {
      "id": "s0101a_000",
      "speaker": "s01",
      "audio": "audio/s0101a_000.wav",
      "transcript": "okay um i'm lived in columbus...",
      "duration_s": 24.96,
      "num_words": 25,
      "words": [
        {"word": "okay", "start_ms": 100.0, "end_ms": 505.5},
        {"word": "um", "start_ms": 12501.4, "end_ms": 12830.3},
        ...
      ]
    }
  ]
}
```

## Ground Truth

Word timestamps are from Buckeye's **human phonetic labeling** — hand-corrected by trained annotators. The `start_ms` and `end_ms` for each word are derived from the `.words` annotation files.

## Segmentation

Long conversation recordings (5-12 min each) were segmented into utterances at silence/noise boundaries (`<SIL>`, `<IVER>`, `<VOCNOISE>` tokens with gaps > 0.3s). Segments are 2-25 seconds with minimum 3 real words.

## Intended Use

Benchmarking forced alignment systems by comparing predicted word timestamps against human ground truth. Primary metric: **AAS (Accumulated Average Shift)** — mean absolute boundary error in milliseconds.

## Citation

```
Pitt, M.A., Johnson, K., Hume, E., Kiesling, S., & Raymond, W. (2005).
The Buckeye corpus of conversational speech: labeling conventions and a test of transcriber reliability.
Speech Communication, 45, 89-95.
```

## License

The Buckeye Corpus is free for noncommercial use. See [buckeyecorpus.osu.edu](https://buckeyecorpus.osu.edu/) for terms.
