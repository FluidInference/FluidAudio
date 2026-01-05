# Benchmarks

2024 MacBook Pro, 48GB Ram, M4 Pro, Tahoe 26.0

## Transcription

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```bash
swift run fluidaudio fleurs-benchmark --languages all --samples all
```

```text
Language                  | WER%   | CER%   | RTFx    | Duration | Processed | Skipped
-----------------------------------------------------------------------------------------
Bulgarian (Bulgaria)      | 12.8   | 4.1    | 195.2   | 3468.0s  | 350       | -
Croatian (Croatia)        | 14.0   | 4.3    | 204.9   | 3647.0s  | 350       | -
Czech (Czechia)           | 12.0   | 3.8    | 214.2   | 4247.4s  | 350       | -
Danish (Denmark)          | 20.2   | 7.4    | 214.4   | 10579.1s | 930       | -
Dutch (Netherlands)       | 7.8    | 2.6    | 191.7   | 3337.7s  | 350       | -
English (US)              | 5.4    | 2.5    | 207.4   | 3442.9s  | 350       | -
Estonian (Estonia)        | 20.1   | 4.2    | 225.3   | 10825.4s | 893       | -
Finnish (Finland)         | 14.8   | 3.1    | 222.0   | 11894.4s | 918       | -
French (France)           | 5.9    | 2.2    | 199.9   | 3667.3s  | 350       | -
German (Germany)          | 5.9    | 1.9    | 220.9   | 4684.6s  | 350       | -
Greek (Greece)            | 36.9   | 13.7   | 183.0   | 6862.0s  | 650       | -
Hungarian (Hungary)       | 17.6   | 5.2    | 213.6   | 11050.9s | 905       | -
Italian (Italy)           | 4.0    | 1.3    | 236.7   | 5098.7s  | 350       | -
Latvian (Latvia)          | 27.1   | 7.5    | 217.8   | 10218.6s | 851       | -
Lithuanian (Lithuania)    | 25.0   | 6.8    | 202.8   | 10686.5s | 986       | -
Maltese (Malta)           | 25.2   | 9.3    | 217.4   | 12770.6s | 926       | -
Polish (Poland)           | 8.6    | 2.8    | 190.2   | 3409.6s  | 350       | -
Romanian (Romania)        | 14.4   | 4.7    | 200.4   | 9099.4s  | 883       | -
Russian (Russia)          | 7.2    | 2.2    | 209.7   | 3974.6s  | 350       | -
Slovak (Slovakia)         | 12.6   | 4.4    | 227.6   | 4169.6s  | 350       | -
Slovenian (Slovenia)      | 27.4   | 9.2    | 197.1   | 8173.1s  | 834       | -
Spanish (Spain)           | 4.5    | 2.2    | 221.7   | 4258.9s  | 350       | -
Swedish (Sweden)          | 16.8   | 5.0    | 219.5   | 8399.2s  | 759       | -
Ukrainian (Ukraine)       | 7.2    | 2.5    | 201.9   | 3853.7s  | 350       | -
-----------------------------------------------------------------------------------------
AVERAGE                   | 14.7   | 4.7    | 209.8   | 161819.2 | 14085     | -
```

```text
Dataset: librispeech test-clean
Files processed: 2620
Average WER: 2.5%
Median WER: 0.0%
Average CER: 1.0%
Median RTFx: 139.6x
Overall RTFx: 155.6x (19452.5s / 125.0s)
```

`swift run fluidaudio asr-benchmark --max-files all --model-version v2`

Use v2 if you only need English, it is a bit more accurate

```text
--- Benchmark Results ---
   Dataset: librispeech test-clean
   Files processed: 2620
   Average WER: 2.1%
   Median WER: 0.0%
   Average CER: 0.7%
   Median RTFx: 128.6x
   Overall RTFx: 145.8x (19452.5s / 133.4s)
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

We generated the same strings with to gerneate audio between 1s to ~300s in order to test the speed across a range of varying inputs on Pytorch CPU, MPS, and MLX pipeline, and compared it against the native Swift version with Core ML models.

Each pipeline warmed up the models by running through it once with pesudo inputs, and then comparing the raw inference time with the model already loaded. You can see that for the Core ML model, we traded lower memory and very slightly faster inference for longer initial warm-up.

Note that the Pytorch kokoro model in Pytorch has a memory leak issue: https://github.com/hexgrad/kokoro/issues/152

The following tests were ran on M4 Pro, 48GB RAM, Macbook Pro. If you have another device, please do try replicating it as well!

### Kokoro-82M PyTorch (CPU)

```bash
KPipeline benchmark for voice af_heart (warm-up took 0.175s) using hexgrad/kokoro
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.187        14.737x    1.44
2      129      8.625        0.530        16.264x    1.85
3      254      15.525       0.923        16.814x    2.65
4      93       6.125        0.349        17.566x    2.66
5      104      7.200        0.410        17.567x    2.70
6      130      9.300        0.504        18.443x    2.72
7      197      12.850       0.726        17.711x    2.83
8      6        1.350        0.098        13.823x    2.83
9      1228     76.200       4.342        17.551x    3.19
10     567      35.200       2.069        17.014x    4.85
11     4615     286.525      17.041       16.814x    4.78
Total  -        461.650      27.177       16.987x    4.85    
```

### Kokoro-82M PyTorch (MPS)

I wasn't able to run the MPS model for longer durations, even with `PYTORCH_ENABLE_MPS_FALLBACK=1` enabled, it kept crashing for the longer strings.

```bash
KPipeline benchmark for voice af_heart (warm-up took 0.568s) using pip package
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.414        6.649x     1.41
2      129      8.625        0.729        11.839x    1.54
Total  -        11.375       1.142        9.960x     1.54    
```

### Kokoro-82M MLX Pipeline

```bash
TTS benchmark for voice af_heart (warm-up took an extra 2.155s) using model prince-canuma/Kokoro-82M
Test   Chars    Output (s)   Inf(s)       RTFx       Peak GB
1      42       2.750        0.347        7.932x     1.12
2      129      8.650        0.597        14.497x    2.47
3      254      15.525       0.825        18.829x    2.65
4      93       6.125        0.306        20.039x    2.65
5      104      7.200        0.343        21.001x    2.65
6      130      9.300        0.560        16.611x    2.65
7      197      12.850       0.596        21.573x    2.65
8      6        1.350        0.364        3.706x     2.65
9      1228     76.200       2.979        25.583x    3.29
10     567      35.200       1.374        25.615x    3.37
11     4615     286.500      11.112       25.783x    3.37
Total  -        461.650      19.401       23.796x    3.37
```

#### Swift + Fluid Audio Core ML models

Note that it does take `~15s` to compile the model on the first run, subsequent runs are shorter, we expect ~2s to load. 

```bash
> swift run fluidaudio tts --benchmark
...
FluidAudio TTS benchmark for voice af_heart (warm-up took an extra 2.348s)
Test   Chars    Ouput (s)    Inf(s)       RTFx
1      42       2.825        0.440        6.424x
2      129      7.725        0.594        13.014x
3      254      13.400       0.776        17.278x
4      93       5.875        0.587        10.005x
5      104      6.675        0.613        10.889x
6      130      8.075        0.621        13.008x
7      197      10.650       0.627        16.983x
8      6        0.825        0.360        2.290x
9      1228     67.625       2.362        28.625x
10     567      33.025       1.341        24.619x
11     4269     247.600      9.087        27.248x
Total  -        404.300      17.408       23.225

Peak memory usage (process-wide): 1.503 GB
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

## Streaming ASR (Parakeet EOU)

Real-time streaming ASR with End-of-Utterance detection using the Parakeet EOU 120M CoreML model.

Model: [FluidInference/parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml)

Hardware: Apple M2, 2022, macOS 26

### LibriSpeech test-clean (2620 files, 5.40h audio)

| Chunk Size | WER (Avg) | RTFx | Total Time |
|------------|-----------|------|------------|
| 320ms      | 4.87%     | 12.48x | 1558s (26m) |
| 160ms      | 8.29%     | 4.78x  | 4070s (68m) |


```bash
# Run 320ms benchmark
swift run -c release fluidaudio parakeet-eou --benchmark --chunk-size 320 --use-cache

# Run 160ms benchmark
swift run -c release fluidaudio parakeet-eou --benchmark --chunk-size 160 --use-cache
```

## Speaker Diarization

The offline version uses the community-1 model, the online version uses the legacy speaker-diarization-3.1 model.

### Offline diarzing pipeline

For slightly ~1.2% worse DER we default to a higher step ratio segmentation duration than the baseline community-1 pipeline. This allows us to get nearly ~2x the speed (as expected because we're processing 1/2 of the embeddings). For highly critical use cases, one may should use step ratio = 0.1 and minSegmentDurationSeconds = 0.0

Running on the full voxconverse benchmark:

```bash
StepRatio = 0.2, minSegmentDurationSeconds= 1.0
Average DER: 15.07% | Median DER: 10.70% | Average JER: 39.40% | Median JER: 40.95% (collar=0.25s, ignoreOverlap=True)
Average RTFx: 122.06 (from 232 clips)
Completed. New results: 232, Skipped existing: 0, Total attempted: 232
Step Ratio 2, min turation 1.0


StepRatio = 0.1, minSegmentDurationSeconds= 0
Average DER: 13.89% | Median DER: 10.49% | Average JER: 42.84% | Median JER: 43.30% (collar=0.25s, ignoreOverlap=True)
Average RTFx: 64.75 (from 232 clips)
Completed. New results: 232, Skipped existing: 0, Total attempted: 232
Step Ratio 1, min duration 0 (edited) 
```

Note that the baseline pytorch version is ~11% DER, we lost some precision dropping down to fp16 precision in order to run most of the emebdding model on neural engine. But as a result, we significantly out perform the baseline `mps` backend as well. the pyannote-community-1 on cpu is ~1.5-2 RTFx, on mps, it's ~20-25 RTFx.

### Streaming/online Diarization

This is more tricky and honestly a lot more fragile to clustering. Expect +10-15% worse DER for the streaming implementation. Only use this when you critically need realtime streaming speaker diarization. In most cases, offline is more than enough for most applications.

Running a near real-time diarization benchmark for 3s chunks, 1s overlap, and 0.85 clustering threshold:
```bash
swift run fluidaudio diarization-benchmark --mode streaming \
    --dataset ami-sdm \
    --threshold 0.85 \
    --auto-download \
    --chunk-seconds 3.0 \
    --overlap-seconds 1.0
    
...

------------------------------------------------------------------------------------------
Meeting        DER %    JER %    Miss %     FA %     SE %   Speakers     RTFx
------------------------------------------------------------------------------------------
ES2004a          31.6     41.6      6.7      2.1     22.7 7/4            49.8
ES2005a          39.7     65.0      6.9      7.3     25.5 5/4            59.1
IS1002b          40.4     51.3      1.1      5.2     34.1 9/4            45.3
ES2002a          41.5     56.0      5.3     10.1     26.1 6/4            48.6
ES2003a          53.1     78.7      5.3      2.3     45.5 5/4            57.1
IS1000a          66.7     74.0      6.1      7.6     53.0 7/4            50.7
IS1001a          75.0     88.6      7.1      4.7     63.2 10/4           48.8
------------------------------------------------------------------------------------------
AVERAGE          49.7     65.0      5.5      5.6     38.6         -     51.4
==========================================================================================
```


Diarization benchmark with 10s chunks, 0s overlap, and 0.7 clustering threshold:
```bash
swift run fluidaudio diarization-benchmark --mode streaming \
    --dataset ami-sdm
    --threshold 0.7
    --auto-download
    --chunk-seconds 10.0
    --overlap-seconds 0.0

...

------------------------------------------------------------------------------------------
Meeting        DER %    JER %    Miss %     FA %     SE %   Speakers     RTFx
------------------------------------------------------------------------------------------
ES2003a          12.0     19.5      6.9      1.2      3.9 4/4           477.0
ES2004a          15.1     24.8      9.2      1.2      4.7 4/4           367.4
ES2002a          17.8     26.8      8.6      5.8      3.4 6/4           356.8
IS1002b          38.0     41.8      3.1      3.1     31.8 5/4           361.9
ES2005a          22.5     36.8      7.7      6.8      8.0 4/4           460.8
IS1000a          57.7     80.6     11.9      3.9     41.9 8/4           352.1
IS1001a          70.1     85.4     11.2      2.4     56.5 7/4           370.9
------------------------------------------------------------------------------------------
AVERAGE          33.3     45.1      8.4      3.5     21.5         -    392.4
==========================================================================================
```


Diarization benchmark with 5s chunks, 0s overlap, and 0.8 clustering threshold (best configuration found):
```bash
swift run fluidaudio diarization-benchmark --mode streaming \
    --dataset ami-sdm
    --threshold 0.8
    --auto-download
    --chunk-seconds 5.0
    --overlap-seconds 0.0

...

------------------------------------------------------------------------------------------
Meeting        DER %    JER %    Miss %     FA %     SE %   Speakers     RTFx
------------------------------------------------------------------------------------------
IS1002b           9.8     11.7      3.5      3.8      2.6 5/4           205.2
ES2003a          14.4     23.3      7.4      1.6      5.3 4/4           260.9
ES2004a          17.0     26.0      9.0      1.3      6.7 7/4           218.1
ES2005a          18.4     31.0      9.2      5.8      3.4 4/4           259.8
ES2002a          20.8     30.5      9.5      7.4      3.9 5/4           198.0
IS1000a          24.7     35.7     12.1      4.3      8.3 6/4           204.2
IS1001a          78.0     94.5     13.3      3.0     61.6 6/4           215.7
------------------------------------------------------------------------------------------
AVERAGE          26.2     36.1      9.2      3.9     13.1         -    223.1
==========================================================================================
```


Diarization benchmark with 5s chunks, 2s overlap, and 0.8 clustering threshold:
```bash
swift run fluidaudio diarization-benchmark --mode streaming \
    --dataset ami-sdm
    --threshold 0.8
    --auto-download
    --chunk-seconds 5.0
    --overlap-seconds 2.0

...

------------------------------------------------------------------------------------------
Meeting        DER %    JER %    Miss %     FA %     SE %   Speakers     RTFx
------------------------------------------------------------------------------------------
ES2003a          24.5     42.1      4.7      1.9     18.0 6/4            81.4
ES2005a          27.5     50.6      5.5      7.6     14.4 5/4            76.8
ES2004a          31.6     54.8      6.4      2.3     23.0 5/4            66.9
IS1002b          39.6     57.0      0.8      5.1     33.7 6/4            63.7
ES2002a          41.1     57.2      4.7      9.8     26.7 5/4            65.5
IS1000a          57.4     54.2      6.1      7.7     43.6 9/4            67.2
IS1001a          79.0     86.8      7.0      5.0     66.9 10/4           64.5
------------------------------------------------------------------------------------------
AVERAGE          43.0     57.5      5.0      5.6     32.3         -     69.4
==========================================================================================
```

## Sortformer Streaming Diarization

NVIDIA's Sortformer model for streaming speaker diarization, converted to CoreML.

Model: [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml)

Hardware: 2024 MacBook Pro, M4 Pro, 48GB RAM

### AMI SDM Dataset (NVIDIA High-Latency Config - 30.4s chunks)

```bash
swift run fluidaudio sortformer-benchmark --nvidia-high-latency --hf --auto-download
```

```text
================================================================================
SORTFORMER BENCHMARK SUMMARY
================================================================================
Results Sorted by DER:
----------------------------------------------------------------------
Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx
----------------------------------------------------------------------
IS1009b          16.2     10.5      0.5      5.2 4/4           127.7
IS1009d          22.7     15.3      1.1      6.3 4/4           126.5
ES2004b          23.8     18.7      0.2      5.0 4/4           126.6
ES2004c          23.9     17.9      0.3      5.7 4/4           126.7
IS1009a          26.9     15.9      1.6      9.4 4/4           125.7
ES2004d          28.9     19.5      0.3      9.1 4/4           125.2
TS3003b          31.1     27.2      0.6      3.3 4/4           124.6
EN2002c          31.3     20.0      0.2     11.1 4/3           125.8
ES2004a          33.6     24.5      0.2      8.9 4/4           123.2
EN2002b          33.8     20.4      0.6     12.9 4/4           126.6
TS3003c          34.6     31.1      0.3      3.1 4/4           124.0
EN2002a          35.8     20.3      0.4     15.0 4/4           122.1
EN2002d          37.0     20.1      0.4     16.5 4/4           126.5
IS1009c          37.3     12.7      0.9     23.7 4/4           125.1
TS3003d          38.4     31.9      0.2      6.3 4/4           124.1
TS3003a          41.8     36.8      0.6      4.4 4/4           123.7
----------------------------------------------------------------------
AVERAGE          31.1     21.4      0.5      9.1         -    125.3
======================================================================
```
