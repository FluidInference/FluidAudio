# Benchmarks

2024 MacBook Pro, 48GB Ram, M4 Pro, Tahoe 26.0

## Transcription

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```bash
swift run fluidaudio fleurs-benchmark --languages en_us,it_it,es_419,fr_fr,de_de,ru_ru,uk_ua --samples all
```

```text
[22:00:56.652] [INFO] [FLEURSBenchmark] ================================================================================
[22:00:56.652] [INFO] [FLEURSBenchmark] FLEURS BENCHMARK SUMMARY
[22:00:56.652] [INFO] [FLEURSBenchmark] ================================================================================
[22:00:56.652] [INFO] [FLEURSBenchmark]
[22:00:56.652] [INFO] [FLEURSBenchmark] Language                  | WER%   | CER%   | RTFx    | Duration | Processed | Skipped
[22:00:56.652] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[22:00:56.652] [INFO] [FLEURSBenchmark] English (US)              | 5.8    | 2.9    | 189.4   | 3442.9s  | 350       | -
[22:00:56.652] [INFO] [FLEURSBenchmark] French (France)           | 8.8    | 3.8    | 181.3   | 560.8s   | 52        | 298
[22:00:56.652] [INFO] [FLEURSBenchmark] German (Germany)          | 4.2    | 1.2    | 211.2   | 62.1s    | 5         | -
[22:00:56.652] [INFO] [FLEURSBenchmark] Italian (Italy)           | 2.8    | 1.0    | 206.6   | 743.3s   | 50        | -
[22:00:56.652] [INFO] [FLEURSBenchmark] Russian (Russia)          | 7.0    | 2.3    | 185.3   | 621.2s   | 50        | -
[22:00:56.652] [INFO] [FLEURSBenchmark] Spanish (Spain)           | 4.0    | 1.8    | 207.9   | 586.9s   | 50        | -
[22:00:56.652] [INFO] [FLEURSBenchmark] Ukrainian (Ukraine)       | 7.2    | 2.1    | 182.8   | 528.2s   | 50        | -
[22:00:56.652] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[22:00:56.652] [INFO] [FLEURSBenchmark] AVERAGE                   | 5.7    | 2.2    | 194.9   | 6545.5s  | 607       | 298
```

```text
[22:06:25.813] [INFO] [Benchmark] 2620 files per dataset • Test runtime: 3m 12s • 09/19/2025, 10:06 PM EDT
[22:06:25.813] [INFO] [Benchmark] --- Benchmark Results ---
[22:06:25.813] [INFO] [Benchmark]    Dataset: librispeech test-clean
[22:06:25.813] [INFO] [Benchmark]    Files processed: 2620
[22:06:25.813] [INFO] [Benchmark]    Average WER: 2.7%
[22:06:25.813] [INFO] [Benchmark]    Median WER: 0.0%
[22:06:25.813] [INFO] [Benchmark]    Average CER: 1.1%
[22:06:25.813] [INFO] [Benchmark]    Median RTFx: 132.0x
[22:06:25.813] [INFO] [Benchmark] Results saved to: asr_benchmark_results.json
[22:06:25.813] [INFO] [Benchmark] ASR benchmark completed successfully
[22:06:25.813] [INFO] [Benchmark]    Overall RTFx: 146.5x (19452.5s / 132.8s)
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