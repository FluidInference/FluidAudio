## Transcription

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```bash
swift run fluidaudio fleurs-benchmark --languages en_us,it_it,es_419,fr_fr,de_de,ru_ru,uk_ua --samples all
```

```text
================================================================================
FLEURS BENCHMARK SUMMARY
================================================================================

Language                  | WER%   | CER%   | RTFx    | Duration | Processed | Skipped
-----------------------------------------------------------------------------------------
English (US)              | 5.7    | 2.8    | 136.7   | 3442.9s  | 350       | -
French (France)           | 5.8    | 2.4    | 136.5   | 560.8s   | 52        | 298
German (Germany)          | 3.1    | 1.2    | 152.2   | 62.1s    | 5         | -
Italian (Italy)           | 4.3    | 2.0    | 153.7   | 743.3s   | 50        | -
Russian (Russia)          | 7.7    | 2.8    | 134.1   | 621.2s   | 50        | -
Spanish (Spain)           | 6.5    | 3.0    | 152.3   | 586.9s   | 50        | -
Ukrainian (Ukraine)       | 6.5    | 1.9    | 132.5   | 528.2s   | 50        | -
-----------------------------------------------------------------------------------------
AVERAGE                   | 5.6    | 2.3    | 142.6   | 6545.5s  | 607       | 298
```

```text
2620 files per dataset • Test runtime: 4m 1s • 09/04/2025, 1:55 AM EDT
--- Benchmark Results ---
   Dataset: librispeech test-clean
   Files processed: 2620
   Average WER: 2.7%
   Median WER: 0.0%
   Average CER: 1.1%
   Median RTFx: 99.3x
   Overall RTFx: 109.6x (19452.5s / 177.5s)
```


## Speaker Diarization

https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml 

```text

```
