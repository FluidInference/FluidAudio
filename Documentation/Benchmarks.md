# Benchmarks

2024 MacBook Pro, 48GB Ram, M4 Pro, Tahoe 26.0

## Transcription

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```bash
swift run fluidaudio fleurs-benchmark --languages en_us,it_it,es_419,fr_fr,de_de,ru_ru,uk_ua --samples all
```

```text
[01:58:26.666] [INFO] [FLEURSBenchmark] ================================================================================
[01:58:26.666] [INFO] [FLEURSBenchmark] FLEURS BENCHMARK SUMMARY
[01:58:26.666] [INFO] [FLEURSBenchmark] ================================================================================
[01:58:26.666] [INFO] [FLEURSBenchmark]
[01:58:26.666] [INFO] [FLEURSBenchmark] Language                  | WER%   | CER%   | RTFx    | Duration | Processed | Skipped
[01:58:26.666] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[01:58:26.666] [INFO] [FLEURSBenchmark] English (US)              | 5.7    | 2.8    | 197.8   | 3442.9s  | 350       | -
[01:58:26.666] [INFO] [FLEURSBenchmark] French (France)           | 6.3    | 3.0    | 191.3   | 560.8s   | 52        | 298
[01:58:26.667] [INFO] [FLEURSBenchmark] German (Germany)          | 3.1    | 1.2    | 216.7   | 62.1s    | 5         | -
[01:58:26.667] [INFO] [FLEURSBenchmark] Italian (Italy)           | 4.3    | 2.0    | 213.5   | 743.3s   | 50        | -
[01:58:26.667] [INFO] [FLEURSBenchmark] Russian (Russia)          | 7.8    | 2.8    | 186.3   | 621.2s   | 50        | -
[01:58:26.667] [INFO] [FLEURSBenchmark] Spanish (Spain)           | 5.6    | 2.7    | 214.6   | 586.9s   | 50        | -
[01:58:26.667] [INFO] [FLEURSBenchmark] Ukrainian (Ukraine)       | 7.2    | 2.1    | 192.8   | 528.2s   | 50        | -
[01:58:26.667] [INFO] [FLEURSBenchmark] -----------------------------------------------------------------------------------------
[01:58:26.667] [INFO] [FLEURSBenchmark] AVERAGE                   | 5.7    | 2.4    | 201.9   | 6545.5s  | 607       | 298
```

```text
[02:01:49.655] [INFO] [Benchmark] 2620 files per dataset • Test runtime: 3m 2s • 09/25/2025, 2:01 AM EDT
[02:01:49.655] [INFO] [Benchmark] --- Benchmark Results ---
[02:01:49.655] [INFO] [Benchmark]    Dataset: librispeech test-clean
[02:01:49.655] [INFO] [Benchmark]    Files processed: 2620
[02:01:49.655] [INFO] [Benchmark]    Average WER: 2.6%
[02:01:49.655] [INFO] [Benchmark]    Median WER: 0.0%
[02:01:49.655] [INFO] [Benchmark]    Average CER: 1.1%
[02:01:49.655] [INFO] [Benchmark]    Median RTFx: 137.8x
[02:01:49.655] [INFO] [Benchmark]    Overall RTFx: 153.4x (19452.5s / 126.8s)
[02:01:49.655] [INFO] [Benchmark] Results saved to: asr_benchmark_results.json
[02:01:49.655] [INFO] [Benchmark] ASR benchmark completed successfully
```

`swift run fluidaudio asr-benchmark --max-files all --model-version v2`

Use v2 if you only need English, it is a bit more accurate

```text
ansient day, like music in the air. Ah
[01:35:16.880] [INFO] [Benchmark] File: 908-157963-0010.flac (WER: 15.4%) (Duration: 6.28s)
[01:35:16.880] [INFO] [Benchmark] ------------------------------------------------------------
[01:35:16.894] [INFO] [Benchmark] Normalized Reference: she ceasd and smild in tears then sat down in her silver shrine
[01:35:16.894] [INFO] [Benchmark] Normalized Hypothesis:        she ceased and smiled in tears then sat down in her silver shrine
[01:35:16.894] [INFO] [Benchmark] Original Hypothesis:  She ceased and smiled in tears, Then sat down in her silver shrine,
[01:35:16.894] [INFO] [Benchmark] 2620 files per dataset • Test runtime: 3m 25s • 09/26/2025, 1:35 AM EDT
[01:35:16.894] [INFO] [Benchmark] --- Benchmark Results ---
[01:35:16.894] [INFO] [Benchmark]    Dataset: librispeech test-clean
[01:35:16.894] [INFO] [Benchmark]    Files processed: 2620
[01:35:16.894] [INFO] [Benchmark]    Average WER: 2.2%
[01:35:16.894] [INFO] [Benchmark]    Median WER: 0.0%
[01:35:16.894] [INFO] [Benchmark]    Average CER: 0.7%
[01:35:16.894] [INFO] [Benchmark]    Median RTFx: 125.6x
[01:35:16.894] [INFO] [Benchmark]    Overall RTFx: 141.2x (19452.5s / 137.7s)
[01:35:16.894] [INFO] [Benchmark] Results saved to: asr_benchmark_results.json
[01:35:16.894] [INFO] [Benchmark] ASR benchmark completed successfully
```

### ASR Model Compilation

Core ML first-load compile times captured on iPhone 16 Pro Max and iPhone 13 running the
parakeet-tdt-0.6b-v3-coreml bundle. Cold-start compilation happens the first time each Core ML model
is loaded; subsequent loads hit the cached binaries. Warm compile metrics were collected only on the
iPhone 16 Pro Max run, and only for models that were reloaded during the session.

| Model         | iPhone 16 Pro Max cold (ms) | iPhone 16 Pro Max warm (ms) | iPhone 13 cold (ms) | Compute units               |
| ------------- | --------------------------: | ---------------------------: | ------------------: | --------------------------- |
| Preprocessor  |                        9.15 |                           - |              632.63 | MLComputeUnits(rawValue: 2) |
| Encoder       |                     3361.23 |                      162.05 |             4396.00 | MLComputeUnits(rawValue: 1) |
| Decoder       |                       88.49 |                        8.11 |              146.01 | MLComputeUnits(rawValue: 1) |
| JointDecision |                       48.46 |                        7.97 |               71.85 | MLComputeUnits(rawValue: 1) |

## Text-to-Speech

### Kokoro-82M PyTorch Pipeline

```bash
uv run python benchmark_kokoro_pip.py
```

```text
WARNING: Defaulting repo_id to hexgrad/Kokoro-82M. Pass repo_id='hexgrad/Kokoro-82M' to suppress this warning.
/Users/brandonweng/code/fluid/models/kokoro-82m/.venv/lib/python3.10/site-packages/torch/nn/modules/rnn.py:123: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  warnings.warn(
/Users/brandonweng/code/fluid/models/kokoro-82m/.venv/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)

KPipeline benchmark for voice af_heart (warm-up took 0.220s) using pip package
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.224        12.250x    1.47
2      129      8.625        0.539        16.002x    1.89
3      254      15.525       0.922        16.846x    2.69
4      93       6.125        0.346        17.725x    2.70
5      104      7.200        0.403        17.875x    2.72
6      130      9.300        0.499        18.619x    2.72
7      197      12.850       0.768        16.733x    2.83
8      6        1.350        0.095        14.270x    2.83
9      1228     76.200       4.247        17.940x    3.19
10     567      35.200       2.052        17.156x    4.85
11     4615     286.525      18.347       15.617x    4.79
Total  -        461.650      28.442       16.231x    4.85
```

### Kokoro-82M MLX Pipeline

```bash
uv run python benchmark_kokoro_mlx.py
```

```text
Fetching 2 files: 100%|##################################################| 2/2 [00:00<00:00, 41734.37it/s]
2025-09-26 20:13:39.173 | INFO     | mlx_audio.tts.models.kokoro.kokoro:_get_pipeline:261 - Creating new KokoroPipeline for language: a

TTS benchmark for voice af_heart (warm-up took an extra 3.343s) using model prince-canuma/Kokoro-82M
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.796        3.456x     1.12
2      129      8.650        1.204        7.186x     2.47
3      254      15.525       2.589        5.996x     2.65
4      93       6.125        1.100        5.566x     2.65
5      104      7.200        1.211        5.944x     2.65
6      130      9.300        1.416        6.566x     2.65
7      197      12.850       0.692        18.567x    2.65
8      6        1.350        0.112        12.099x    2.65
9      1228     76.200       2.787        27.344x    3.29
10     567      35.200       1.846        19.068x    3.37
11     4615     286.500      11.121       25.762x    3.37
Total  -        461.650      24.874       18.559x    3.37
```

### FluidAudio CLI TTS Benchmark

```text
Build of product 'fluidaudio' complete! (16.37s)

FluidAudio TTS benchmark for voice af_heart (warm-up took an extra 7.173s) using model prince-canuma/Kokoro-82M
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.411        6.684x     1.01
2      129      8.650        0.921        9.397x     1.97
3      254      15.525       1.784        8.705x     2.39
4      93       6.125        0.824        7.433x     2.39
5      104      7.200        0.825        8.723x     2.39
6      130      9.300        0.952        9.764x     2.39
7      197      12.850       1.288        9.980x     2.39
8      6        1.350        0.233        5.792x     2.39
9      1228     76.200       7.376        10.331x    2.98
10     567      35.200       3.579        9.836x     3.16
11     4615     286.500      27.551       10.399x    3.16
Total  -        461.650      45.744       10.092x    3.16

Peak memory usage (process-wide): 3.16 GB
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
