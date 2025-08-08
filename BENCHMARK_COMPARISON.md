# Speaker Diarization Benchmark Comparison

## Hungarian Algorithm (Batch Mode) vs Streaming Mode

### Test Results on ES2004a

| Mode | Algorithm | DER | JER | RTFx | Speaker Mapping |
|------|-----------|-----|-----|------|----------------|
| **Batch** | Hungarian (Optimal) | 17.7% | 22.1% | 142.4x | Globally optimal assignment |
| **Streaming** | First-occurrence | 17.7% | ~22% | 141.5x | Chronological assignment |

### Key Differences

#### 1. Hungarian Algorithm (Batch Mode)
- **Algorithm**: Uses Hungarian algorithm for globally optimal speaker-to-speaker mapping
- **Mapping**: Finds the assignment that maximizes total overlap between predicted and ground truth
- **When to use**: Post-processing, evaluation, offline analysis
- **Advantages**: 
  - Lowest possible DER given the predictions
  - Optimal speaker assignment
  - Better for benchmarking model quality
- **Example mapping**:
  ```
  Speaker_1 → FEE013 (30783 frames overlap)
  Speaker_2 → FEE016 (19202 frames overlap)
  Speaker_3 → MEO015 (5365 frames overlap)
  Speaker_4 → MEE014 (10431 frames overlap)
  Total overlap: 65781 frames (optimal)
  ```

#### 2. Streaming Mode (Real-time)
- **Algorithm**: First-occurrence mapping based on chronological appearance
- **Mapping**: Assigns speakers based on first significant overlap encountered
- **When to use**: Real-time processing, live transcription, streaming applications
- **Advantages**:
  - No retroactive changes
  - Consistent speaker IDs throughout stream
  - Lower latency
- **Disadvantages**:
  - Higher fragmentation (11.0 vs 1.0 ideal)
  - May not find optimal mapping
  - Speaker IDs can be inconsistent across chunks

### Performance Metrics Explained

- **DER (Diarization Error Rate)**: Sum of missed speech, false alarms, and speaker confusion
  - Both modes: ~17.7% (excellent performance)
  - Research comparison: Powerset BCE achieves 18.5%

- **JER (Jaccard Error Rate)**: Measures speaker segment overlap accuracy
  - Both modes: ~22%
  - More sensitive to boundary errors

- **RTFx (Real-Time Factor)**: Processing speed relative to audio duration
  - Both modes: >140x real-time
  - Far exceeds real-time requirement (>1x)

- **Fragmentation**: Speaker ID consistency (1.0 = perfect)
  - Batch: Near 1.0 (consistent IDs)
  - Streaming: 11.0 (fragmented IDs across chunks)

### Usage Examples

#### Batch Mode with Hungarian Algorithm
```bash
# Standard benchmark with optimal speaker mapping
fluidaudio diarization-benchmark --single-file ES2004a --threshold 0.7

# Shows Hungarian mapping in output:
# 🔍 HUNGARIAN MAPPING: 'Speaker_1' → 'FEE013' (overlap: 30783 frames)
# 🔍 HUNGARIAN RESULT: Total assignment cost: 57351.0, Total overlap: 65781 frames
```

#### Streaming Mode
```bash
# Real-time simulation without retroactive remapping
fluidaudio stream-diarization-benchmark --single-file ES2004a --threshold 0.7

# Shows first-occurrence mapping:
# 🔄 STREAMING MAPPING (first-occurrence): ["Speaker_1": "FEE013", ...]
```

### Recommendations

1. **For evaluation/benchmarking**: Use batch mode with Hungarian algorithm
   - Provides true model performance metrics
   - Enables fair comparison with research papers

2. **For production/real-time**: Use streaming mode
   - Maintains consistency for live applications
   - No retroactive speaker ID changes

3. **For hybrid approaches**: Consider buffered streaming
   - Process in larger chunks (30-60s)
   - Apply local Hungarian optimization within buffer
   - Balance between accuracy and latency

### Conclusion

Both modes achieve excellent 17.7% DER, demonstrating that the underlying models are highly accurate. The choice between Hungarian (batch) and streaming modes depends on your use case:
- Choose Hungarian for optimal accuracy in offline scenarios
- Choose streaming for real-time applications with consistent speaker tracking