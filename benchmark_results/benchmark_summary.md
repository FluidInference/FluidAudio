# Earnings22 Keyword Benchmarks (file-keywords)

## How to read these metrics
- WER: Word error rate of the transcript (lower is better).
- Keyword Precision: Of the keywords predicted in the transcript, the fraction that are correct.
- Keyword Recall: Of the ground-truth keywords in the reference, the fraction found in the transcript.
- F1: Harmonic mean of precision and recall.
- TP/FP/FN: Alignment-based true positives (keyword matched in the correct position), false positives (keyword inserted in the wrong position), false negatives (missed keywords).

## Benchmark types
- Hybrid 110M - CTC: Single encoder; CTC decoding for transcript + CTC keyword detections.
- Hybrid 110M - TDT: Single encoder; TDT decoding for transcript + CTC keyword detections.
- Hybrid 110M - TDT + Rescore: TDT transcript with CTC-based corrections applied.
- Two-encoder (TDT 0.6B + CTC 110M): TDT 0.6B for transcript + separate CTC 110M for keyword spotting.

## Hybrid 110M - CTC
- Output: benchmark_results/hybrid_ctc_file.json
- WER: 19.25
- Keyword Precision/Recall/F1: 0.977 / 0.343 / 0.508
- TP/FP/FN: 475 / 11 / 910
- Total audio (s): 11579.52
- Processing time (s): 366.33

## Hybrid 110M - TDT
- Output: benchmark_results/hybrid_tdt_file.json
- WER: 17.67
- Keyword Precision/Recall/F1: 0.974 / 0.381 / 0.548
- TP/FP/FN: 528 / 14 / 857
- Total audio (s): 11579.52
- Processing time (s): 450.47

## Hybrid 110M - TDT + Rescore
- Output: benchmark_results/hybrid_tdt_rescore_file.json
- WER: 18.07
- Keyword Precision/Recall/F1: 0.614 / 0.617 / 0.615
- TP/FP/FN: 854 / 537 / 531
- Total audio (s): 11579.52
- Processing time (s): 446.84

## Two-encoder (TDT 0.6B + CTC 110M)
- Output: benchmark_results/ctc_earnings_file.json
- WER: 15.92
- Keyword Precision/Recall/F1: 0.674 / 0.741 / 0.706
- TP/FP/FN: 1011 / 489 / 354
- Total audio (s): 11564.52
- Processing time (s): 1032.7
- Note: totalTests=771 (one file skipped due to empty TDT transcription)
- Note: uses Parakeet TDT 0.6B for transcription and Parakeet CTC 110M for keyword spotting.
