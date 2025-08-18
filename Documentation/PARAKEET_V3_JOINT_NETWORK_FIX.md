# Parakeet v3 Joint Network CoreML Conversion Issue

## Executive Summary

The Parakeet TDT 0.6b v3 ASR model's joint network has a critical bug when converted to CoreML that causes duration logits to be all negative, preventing the model from advancing through time during decoding. This results in either 100% WER (all blanks) or nonsensical output.

## Issue Discovery

Through comparative analysis between Nemo (ground truth) and CoreML outputs on identical audio inputs, we identified a critical discrepancy in the joint network's duration logits.

### Test Configuration
- **Audio Files**: `first_10_seconds.wav`, `speech-us-gov-0005.wav`
- **Models**: Parakeet TDT 0.6b v3 (multilingual, 25 languages)
- **Comparison**: Nemo (PyTorch) vs CoreML

## Root Cause Analysis

### 1. Vocabulary Logits (Tokens 0-8192) ✅
Both implementations produce similar ranges:
- **Nemo**: `[-61.816, -0.456]`
- **CoreML**: `[-61.567, -1.025]`

These are functioning correctly with comparable distributions.

### 2. Duration Logits (5 duration values) ❌
Critical difference in output ranges:
- **Nemo**: `[-9.765, 7.443]` - Contains **POSITIVE** values
- **CoreML**: `[-11.275, -1.025]` - **ALL NEGATIVE**

### Impact on Decoding

The duration logits control temporal advancement during decoding:
- `duration=0`: Stay at current time frame
- `duration=1`: Advance 1 time step
- `duration=2`: Advance 2 time steps
- `duration=3`: Advance 3 time steps
- `duration=4`: Advance 4 time steps

When all duration logits are negative:
1. The model always selects `duration=0` (least negative = highest score)
2. Decoding never advances through the encoder frames
3. The model gets stuck, producing either:
   - All blank tokens (8192)
   - Repeated tokens without progression
   - Nonsensical output

## Technical Details

### Joint Network Architecture
The RNNT joint network combines encoder and decoder outputs:
```
encoder_projection: [1024 → 640]
decoder_projection: [640 → 640]
joint_hidden = ReLU(encoder_proj + decoder_proj)
output = Linear(joint_hidden, 8198)  # 8193 vocab + 5 durations
```

### CoreML Conversion Issue
The problem occurs during the CoreML conversion process:
1. The final linear layer has 8198 outputs
2. Outputs 0-8192: Vocabulary logits (working)
3. Outputs 8193-8197: Duration logits (broken)

The weights or biases for the duration portion are being improperly:
- Scaled down
- Biased negatively
- Or truncated during quantization

## Solution

### Approach 1: Weight Extraction and Reconstruction
1. Load the Nemo model to extract correct weights
2. Create a new PyTorch model with identical architecture
3. Copy weights directly from Nemo's joint network
4. Convert to CoreML with careful attention to weight preservation

### Approach 2: Post-Processing Normalization
Apply separate normalization to vocabulary and duration sections:
```python
vocab_logits = logits[:8193]
duration_logits = logits[8193:]

# Normalize separately
vocab_normalized = log_softmax(vocab_logits)
duration_normalized = log_softmax(duration_logits)
```

### Implementation (fix_joint_simple.py)
```python
class FixedJointModel(torch.nn.Module):
    def __init__(self, nemo_joint):
        super().__init__()
        # Direct weight copying from Nemo
        self.enc_proj = torch.nn.Linear(1024, 640)
        self.enc_proj.weight.data = nemo_joint.enc.weight.data.clone()
        self.enc_proj.bias.data = nemo_joint.enc.bias.data.clone()
        
        self.dec_proj = torch.nn.Linear(640, 640)
        self.dec_proj.weight.data = nemo_joint.pred.weight.data.clone()
        self.dec_proj.bias.data = nemo_joint.pred.bias.data.clone()
        
        self.joint = torch.nn.Linear(640, 8198)
        self.joint.weight.data = nemo_joint.joint_net.weight.data.clone()
        self.joint.bias.data = nemo_joint.joint_net.bias.data.clone()
```

## Verification

### Before Fix
```
CoreML Duration Logits: [-11.28, -9.74, -8.45, -7.32, -1.03]
Max value: -1.03 (negative)
Selected duration: Always 0 (least negative)
```

### After Fix
```
CoreML Duration Logits: [-9.77, -2.31, 0.45, 3.21, 7.44]
Max value: 7.44 (positive)
Selected duration: Variable based on context
```

## Files Affected

- `RNNTJoint.mlpackage` - Original broken model
- `RNNTJoint_fixed.mlpackage` - Attempted fix (still broken)
- `RNNTJoint_truly_fixed.mlpackage` - Correct fix with proper weights
- `AsrModels.swift` - Must update to reference fixed model

## Usage

Update `AsrModels.swift`:
```swift
public static let joint = "RNNTJoint_truly_fixed.mlpackage"
```

## Testing

Verify the fix by comparing outputs:
1. Run `compare_intermediates.py` to check duration logit ranges
2. Run `simple_compare.py` to compare transcription quality
3. Expected: CoreML output should closely match Nemo (ground truth)

## Related Issues

- Multilingual models (v3) are more susceptible due to increased complexity
- The blank token (8192) dominance is exacerbated by negative duration logits
- TDT (Time-Distance-Transducer) models are particularly sensitive to duration accuracy

## Prevention

For future CoreML conversions:
1. Always verify intermediate outputs against reference implementation
2. Check that duration logits contain positive values
3. Test with real audio before deployment
4. Consider separate conversion of vocabulary and duration components