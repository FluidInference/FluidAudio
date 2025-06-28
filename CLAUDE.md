# FluidAudioSwift - Claude Code Instructions

## Project Overview
FluidAudioSwift is a speaker diarization system for Apple platforms using Core ML models. The system processes audio to identify "who spoke when" by segmenting audio and clustering speaker embeddings.

## Current Performance Baseline (AMI Benchmark)
- **Dataset**: AMI SDM (Single Distant Microphone)
- **Current Results**: DER: 81.0%, JER: 24.4%, RTF: 0.02x
- **Research Benchmarks**: 
  - Powerset BCE (2023): 18.5% DER
  - EEND (2019): 25.3% DER
  - x-vector clustering: 28.7% DER
- **Performance Gap**: Our current 81% DER indicates significant room for optimization

## Optimization Goals
- **Primary**: Reduce DER from 81% to < 30% (competitive with research benchmarks)
- **Secondary**: Maintain JER < 25%
- **Constraint**: Keep RTF reasonable (< 1.0x for real-time capability)

## DiarizerConfig Parameters for Tuning

### Current Default Values
```swift
DiarizerConfig(
    clusteringThreshold: 0.7,     // Similarity threshold for grouping speakers (0.0-1.0)
    minDurationOn: 1.0,           // Minimum speaker segment duration (seconds)
    minDurationOff: 0.5,          // Minimum silence between speakers (seconds)
    numClusters: -1,              // Number of speakers (always -1 for auto-detect)
    minActivityThreshold: 10.0,   // Minimum activity threshold (frames)
    debugMode: false
)
```

### Parameter Effects and Ranges
1. **clusteringThreshold** (0.0-1.0): Higher = stricter speaker separation, fewer speakers
   - Range to test: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
   - Impact: High impact on speaker confusion errors

2. **minDurationOn** (seconds): Filters out very short speech segments
   - Range to test: [0.5, 1.0, 1.5, 2.0, 3.0]
   - Impact: Affects false alarm rate

3. **minDurationOff** (seconds): Minimum gap between different speakers
   - Range to test: [0.1, 0.25, 0.5, 0.75, 1.0]
   - Impact: Affects speaker change detection

4. **minActivityThreshold** (frames): Minimum activity for speaker detection
   - Range to test: [5.0, 8.0, 10.0, 15.0, 20.0, 25.0]
   - Impact: Affects missed speech detection


## CLI Commands Needed

### Current Available
```bash
swift run fluidaudio benchmark --auto-download --threshold 0.7 --output results.json
```

### Additional Parameters Needed
The CLI needs to be extended to support:
- `--min-duration-on <float>`
- `--min-duration-off <float>`
- `--min-activity <float>`

## Optimization Strategy - Intelligent Parameter Tuning

### Phase 1: Baseline Assessment
1. Run current baseline to confirm DER: 81.0%
2. Analyze baseline results to identify primary error types:
   - High DER suggests speaker confusion, missed speech, or false alarms
   - Use error breakdown to guide parameter adjustments

### Phase 2: Adaptive Parameter Search
**Smart Optimization Approach:**
1. **Start with most impactful parameter**: clusteringThreshold
   - If DER > 50%: Try lower threshold (0.5, 0.4) for more aggressive clustering
   - If many false speakers detected: Try higher threshold (0.8, 0.9)
   
2. **Adjust based on results**:
   - **If DER improves significantly (>10%)**: Continue in same direction
   - **If DER worsens**: Reverse direction or try smaller steps
   - **If DER plateaus**: Move to next parameter

3. **Sequential parameter optimization**:
   - Once clusteringThreshold converges, optimize minActivityThreshold
   - Then optimize duration parameters (minDurationOn, minDurationOff)
   - Each parameter builds on previous best configuration

### Phase 3: Gradient-Based Fine-Tuning
1. **Binary search refinement**: Narrow down optimal ranges
2. **Cross-parameter interaction**: Test parameter combinations that showed promise
3. **Stability testing**: Ensure improvements are consistent across multiple files

### Phase 4: Validation and Analysis
1. Run best configuration on full AMI test set
2. Compare against research benchmarks
3. Document parameter sensitivity and final recommendations

### Optimization Stopping Criteria
- DER improvement < 1% for 3 consecutive tests
- DER reaches target < 30%
- Parameter has been tested in both directions without improvement

## Optimization Log

| Date | Phase | Parameters | DER | JER | RTF | Notes |
|------|-------|------------|-----|-----|-----|-------|
| 2024-06-28 | Baseline | threshold=0.7, defaults | 81.0% | 24.4% | 0.02x | Initial measurement |
| | | | | | | |

## Best Configurations Found

*To be updated during optimization*

## Parameter Sensitivity Insights

*To be documented during optimization*

## Final Recommendations

*To be determined after optimization completion*

## Instructions for Claude Code

When asked to optimize DiarizerConfig parameters:

1. **First, add missing CLI parameters** to support full tuning
2. **Start with Phase 1**: Test high-impact parameters systematically
3. **Document all results** in the Optimization Log table above
4. **Update Best Configurations** as better parameters are found
5. **Move through phases** based on results and convergence
6. **Save final config** to both CLAUDE.md and separate JSON file

### Running Benchmarks
Always use:
```bash
swift run fluidaudio benchmark --auto-download --output results_[timestamp].json [parameters]
```

### Result Analysis
- DER (Diarization Error Rate): Primary metric to minimize
- JER (Jaccard Error Rate): Secondary metric
- Look for parameter combinations that reduce both
- Consider RTF (Real-Time Factor) for practical deployment

### Stopping Criteria
- DER improvements < 1% for 3 consecutive parameter tests
- DER reaches target of < 30%
- All parameter combinations in current phase tested