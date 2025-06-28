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

## Optimization Strategy - Expert ML Engineer Parameter Tuning

### Phase 1: Baseline Assessment & Anomaly Detection
1. **Run baseline multiple times** to establish statistical significance
   - Run 3-5 iterations of same config to measure stability
   - Calculate mean ± std deviation for DER, JER, RTF
   - **RED FLAG**: If std deviation > 5%, investigate non-deterministic behavior
   
2. **Deep error analysis** (act like forensic ML engineer):
   - **If DER > 60%**: Likely clustering failure - speakers being confused
   - **If JER > DER**: Timeline alignment issues - check duration parameters
   - **If RTF varies significantly**: Resource contention or memory issues
   - **If same results across different parameters**: Model may be broken/not using params

### Phase 2: Intelligent Anomaly-Aware Parameter Search

**Expert-Level Optimization with Consistency Checks:**

1. **Pre-flight validation**:
   ```
   BEFORE each parameter test:
   - Verify parameter actually changed in logs/debug output
   - Confirm model is using new parameters (not cached)
   - Check if audio files are being processed correctly
   ```

2. **Smart parameter testing with anomaly detection**:
   - **Test parameter extremes first**: (0.3, 0.9) for clusteringThreshold
   - **CONSISTENCY CHECK**: If extreme values give identical results → INVESTIGATE
   - **SANITY CHECK**: If threshold=0.9 gives same DER as threshold=0.3 → MODEL ISSUE
   
3. **Expert troubleshooting triggers**:
   ```
   IF (same DER across 3+ different parameter values):
       → Check if parameters are actually being used
       → Verify model isn't using cached/default values
       → Add debug logging to confirm parameter propagation
   
   IF (DER increases when it should decrease):
       → Analyze what type of errors increased
       → Check if we're optimizing the wrong bottleneck
       → Verify ground truth data integrity
   
   IF (improvement then sudden degradation):
       → Look for parameter interaction effects
       → Check if we hit a threshold/boundary condition
       → Analyze if overfitting to specific audio characteristics
   ```

4. **Gradient analysis like an expert**:
   - **Calculate parameter sensitivity**: ΔDER / Δparameter
   - **Detect non-monotonic behavior**: When increasing parameter sometimes helps, sometimes hurts
   - **Identify parameter interactions**: When two parameters must be tuned together

### Phase 3: Expert Debugging & Deep Analysis

**When things don't make sense (expert troubleshooting):**

1. **Identical results debugging**:
   ```
   IF multiple different parameters → same DER:
   THEN investigate:
   - Are parameters reaching the model layer?
   - Is there parameter clamping/saturation?
   - Are we testing on different audio files accidentally?
   - Is there a bug in parameter passing?
   ```

2. **Counterintuitive results analysis**:
   ```
   IF (lower clustering threshold → worse DER):
   THEN analyze:
   - Are we creating too many micro-clusters?
   - Is the similarity metric broken?
   - Are we hitting edge cases in clustering algorithm?
   
   IF (longer minDurationOn → worse performance):
   THEN check:
   - Are we filtering out too much real speech?
   - Is ground truth data very granular?
   - Are we introducing boundary artifacts?
   ```

3. **Expert validation techniques**:
   - **A/B testing**: Run same config twice to verify reproducibility
   - **Parameter sweep validation**: Test 3 values around best config
   - **Cross-validation**: Test best config on different AMI files
   - **Ablation studies**: Remove one optimization at a time

### Phase 4: Advanced Optimization Strategies

1. **Multi-objective optimization** (expert approach):
   - Don't just minimize DER - analyze DER vs JER trade-offs
   - Consider parameter stability (configs that work across multiple files)
   - Factor in computational cost (RTF) as constraint

2. **Adaptive search strategies**:
   ```
   IF (DER variance > 10% across files):
       → Need more robust parameters, not just lowest DER
   
   IF (no improvement after 5 tests):
       → Switch to different parameter or try combinations
   
   IF (improvements < 2% but consistent):
       → Continue fine-tuning in smaller steps
   ```

3. **Expert stopping criteria**:
   - **Statistical significance**: Need 3+ runs showing improvement
   - **Diminishing returns**: When improvement rate < 0.5% per iteration
   - **Validation consistency**: Best config must work on multiple test files

### Phase 5: Expert Validation & Forensics

1. **Results validation**:
   - Run best config 5 times to confirm stability
   - Test on different AMI files to verify generalization
   - Compare against original baseline to quantify total improvement

2. **Expert forensics** (if results seem weird):
   - **Parameter correlation analysis**: Which parameters interact?
   - **Error pattern analysis**: What types of errors decreased/increased?
   - **Audio characteristics**: Do improvements work on all meeting types?
   - **Boundary condition testing**: What happens at parameter extremes?

### Expert Troubleshooting Decision Tree

```
START optimization iteration:
├── Results identical to previous? 
│   ├── YES → INVESTIGATE: Parameter not being used / Model caching
│   └── NO → Continue
├── Results worse than expected?
│   ├── YES → ANALYZE: Wrong direction / Parameter interaction / Bad config
│   └── NO → Continue
├── Results too good to believe?
│   ├── YES → VALIDATE: Run multiple times / Check different files
│   └── NO → Continue
└── Results make sense? → Document and continue
```

### Expert Anomaly Red Flags

**Immediately investigate if you see:**
- Same DER across 4+ different parameter values
- DER improvement then sudden 20%+ degradation  
- RTF varying by >50% with same parameters
- JER > DER consistently (suggests timeline issues)
- Parameters having opposite effect than expected
- No improvement despite testing full parameter range

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