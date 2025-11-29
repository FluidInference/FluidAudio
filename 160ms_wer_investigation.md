# 160ms WER Investigation Plan

## Current Status
- **160ms WER**: 46.43%
- **1000ms WER**: 25%
- **Gap**: 21.43% (need to close this)

## Error Analysis (from benchmark)

### Reference:
"HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE"

### Hypothesis (160ms):
"he hoped there would be souvenir turnips and carrots and brews potatoes and fat as below in thick peppered flower fat sa"

### Error Patterns:
1. **"souvenir"** ← "stew for dinner" (phonetically similar, wrong segmentation)
2. **"brews"** ← "bruised" (phonetic confusion)
3. **"fat as below"** ← "fat mutton pieces to be ladled out" (massive context loss)
4. **"flower fat sa"** ← "flour fattened sauce" (phonetic + truncation)

## Root Causes
1. **Greedy Decoding**: No beam search = poor choices at every step
2. **Short Context**: 160ms chunks don't have enough acoustic info
3. **No Language Model**: Can't leverage word co-occurrence probabilities
4. **Cache Degradation**: Possible precision loss in stateful cache

## Improvement Strategies (Priority Order)

### 1. Use Native Chunk Size (16 frames) ✅ DONE - 4.3% improvement
- **Implementation**: Re-exported encoder with 16 frames (native chunk_size per streaming_cfg)
- **Result**: WER 46.43% → 42.13% (4.3% improvement)
- **Tradeoff**: Latency 8.7ms → 14.2ms (but still real-time capable)
- **Status**: Successfully deployed

### 2. Add Overlap Buffering (BLOCKED)
- **Hypothesis**: Clean boundaries = better acoustic features
- **Blocker**: Would need variable-shape preprocessing or re-export all models
- **Expected improvement**: 5-10% WER reduction
- **Status**: Deferred due to complexity

### 3. Implement Beam Search (NEXT PRIORITY)
- **Hypothesis**: Exploring top-k paths = better token choices
- **Implementation**: Modify RnntDecoder to use beam search (k=5-10)
- **Expected improvement**: 10-15% WER reduction
- **Time**: 2-3 hours

### 3. Add Language Model Rescoring (HARD)
- **Hypothesis**: LM can fix phonetic confusion ("brews" vs "bruised")
- **Implementation**: Integrate n-gram or neural LM for rescoring
- **Expected improvement**: 15-20% WER reduction
- **Time**: 1-2 days

### 4. Optimize Cache Precision (MEDIUM)
- **Hypothesis**: Float16 precision loss in cache
- **Implementation**: Force float32 for cache tensors
- **Expected improvement**: 2-5% WER reduction
- **Time**: 1 hour

## Testing Plan
1. Test each improvement individually
2. Measure WER on 10-20 files minimum
3. Combine successful strategies
4. Target: Get 160ms WER to ~30% (competitive with 1000ms baseline)
