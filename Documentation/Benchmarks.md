# Benchmarks

2024 MacBook Pro, 48GB Ram, M4 Pro, Tahoe 26.0

## Transcription

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```bash
swift run fluidaudio fleurs-benchmark --languages en_us,it_it,es_419,fr_fr,de_de,ru_ru,uk_ua --samples all
```

```text
[21:26:04.328] [INFO] [FLEURSBenchmark] ----------------------------------------
[21:26:04.328] [INFO] [FLEURSBenchmark] ================================================================================
[21:26:04.328] [INFO] [FLEURSBenchmark] ✓ Results saved to fleurs_benchmark_results.json
[21:26:04.328] [INFO] [FLEURSBenchmark] ================================================================================
[21:26:04.328] [INFO] [FLEURSBenchmark] FLEURS BENCHMARK SUMMARY
[21:26:04.328] [INFO] [FLEURSBenchmark] ================================================================================
[21:26:04.328] [INFO] [FLEURSBenchmark]
[21:26:04.328] [INFO] [FLEURSBenchmark] Language                  | WER%   | CER%   | RTFx    | Duration | Processed | Skipped
[21:26:04.328] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[21:26:04.328] [INFO] [FLEURSBenchmark] English (US)              | 5.8    | 2.9    | 178.2   | 3442.9s  | 350       | -
[21:26:04.328] [INFO] [FLEURSBenchmark] French (France)           | 8.8    | 3.8    | 163.8   | 560.8s   | 52        | 298
[21:26:04.328] [INFO] [FLEURSBenchmark] German (Germany)          | 4.2    | 1.2    | 191.4   | 62.1s    | 5         | -
[21:26:04.328] [INFO] [FLEURSBenchmark] Italian (Italy)           | 2.8    | 1.0    | 194.3   | 743.3s   | 50        | -
[21:26:04.328] [INFO] [FLEURSBenchmark] Russian (Russia)          | 7.0    | 2.3    | 173.6   | 621.2s   | 50        | -
[21:26:04.328] [INFO] [FLEURSBenchmark] Spanish (Spain)           | 4.0    | 1.8    | 195.3   | 586.9s   | 50        | -
[21:26:04.328] [INFO] [FLEURSBenchmark] Ukrainian (Ukraine)       | 7.2    | 2.1    | 169.7   | 528.2s   | 50        | -
[21:26:04.328] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[21:26:04.328] [INFO] [FLEURSBenchmark] AVERAGE                   | 5.7    | 2.2    | 180.9   | 6545.5s  | 607       | 298
```

```text
2620 files per dataset • Test runtime: 4m 1s • 09/04/2025, 1:55 AM EDT
[21:36:14.730] [INFO] [Benchmark] --- Benchmark Results ---
[21:36:14.730] [INFO] [Benchmark]    Dataset: librispeech test-clean
[21:36:14.730] [INFO] [Benchmark]    Files processed: 2620
[21:36:14.730] [INFO] [Benchmark]    Average WER: 2.7%
[21:36:14.730] [INFO] [Benchmark]    Median WER: 0.0%
[21:36:14.730] [INFO] [Benchmark]    Average CER: 1.1%
[21:36:14.730] [INFO] [Benchmark]    Median RTFx: 124.7x
[21:36:14.730] [INFO] [Benchmark]    Overall RTFx: 137.4x (19452.5s / 141.6s)
```

## Voice Activity Detection

Model is nearly identical to the base model in terms of quality, perforamnce wise we see an up to ~3.5x improvement compared to the silero Pytorch VAD model with the 256ms batch model (8 chunks of 32ms)

![VAD/speed.png](VAD/speed.png)
![VAD/correlation.png](VAD/correlation.png)

Dataset: https://github.com/Lab41/VOiCES-subset

```text
swift run fluidaudio vad-benchmark --dataset voices-subset --all-files --threshold 0.85
...
Timing Statistics:
[18:56:31.208] [INFO] [VAD]    Total processing time: 0.29s
[18:56:31.208] [INFO] [VAD]    Total audio duration: 351.05s
[18:56:31.208] [INFO] [VAD]    RTFx: 1230.6x faster than real-time
[18:56:31.208] [INFO] [VAD]    Audio loading time: 0.00s (0.6%)
[18:56:31.208] [INFO] [VAD]    VAD inference time: 0.28s (98.7%)
[18:56:31.208] [INFO] [VAD]    Average per file: 0.011s
[18:56:31.208] [INFO] [VAD]    Min per file: 0.001s
[18:56:31.208] [INFO] [VAD]    Max per file: 0.020s
[18:56:31.208] [INFO] [VAD]
VAD Benchmark Results:
[18:56:31.208] [INFO] [VAD]    Accuracy: 96.0%
[18:56:31.208] [INFO] [VAD]    Precision: 100.0%
[18:56:31.208] [INFO] [VAD]    Recall: 95.8%
[18:56:31.208] [INFO] [VAD]    F1-Score: 97.9%
[18:56:31.208] [INFO] [VAD]    Total Time: 0.29s
[18:56:31.208] [INFO] [VAD]    RTFx: 1230.6x faster than real-time
[18:56:31.208] [INFO] [VAD]    Files Processed: 25
[18:56:31.208] [INFO] [VAD]    Avg Time per File: 0.011s
```

```text
swift run fluidaudio vad-benchmark --dataset musan-full --num-files all --threshold 0.8
...
[23:02:35.539] [INFO] [VAD] Total processing time: 322.31s
[23:02:35.539] [INFO] [VAD] Timing Statistics:
[23:02:35.539] [INFO] [VAD] RTFx: 1220.7x faster than real-time
[23:02:35.539] [INFO] [VAD] Audio loading time: 1.20s (0.4%)
[23:02:35.539] [INFO] [VAD] VAD inference time: 319.57s (99.1%)
[23:02:35.539] [INFO] [VAD] Average per file: 0.160s
[23:02:35.539] [INFO] [VAD] Total audio duration: 393442.58s
[23:02:35.539] [INFO] [VAD] Min per file: 0.000s
[23:02:35.539] [INFO] [VAD] Max per file: 0.873s
[23:02:35.711] [INFO] [VAD] VAD Benchmark Results:
[23:02:35.711] [INFO] [VAD] Accuracy: 94.2%
[23:02:35.711] [INFO] [VAD] Precision: 92.6%
[23:02:35.711] [INFO] [VAD] Recall: 78.9%
[23:02:35.711] [INFO] [VAD] F1-Score: 85.2%
[23:02:35.711] [INFO] [VAD] Total Time: 322.31s
[23:02:35.711] [INFO] [VAD] RTFx: 1220.7x faster than real-time
[23:02:35.711] [INFO] [VAD] Files Processed: 2016
[23:02:35.711] [INFO] [VAD] Avg Time per File: 0.160s
[23:02:35.744] [INFO] [VAD] Results saved to: vad_benchmark_results.json
```


## Speaker Diarization