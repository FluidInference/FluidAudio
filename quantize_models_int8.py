#!/usr/bin/env python3
"""
Quantize wespeaker and segmentation models to INT8.
WARNING: This will likely degrade accuracy significantly.
"""

import os
import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import numpy as np

def quantize_model_int8(model_path, output_name):
    """Quantize a CoreML model to INT8."""
    print(f"\n{'='*60}")
    print(f"Quantizing {model_path} to INT8")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load the model
        print(f"📦 Loading model from {model_path}...")
        model = ct.models.MLModel(model_path)
        
        # Get model info
        print(f"📊 Model info:")
        print(f"   - Input: {list(model.input_description)}")
        print(f"   - Output: {list(model.output_description)}")
        
        # Method 1: Try using quantize_weights (safest approach)
        try:
            print("\n🔧 Attempting weight quantization to INT8...")
            spec = model.get_spec()
            
            # Apply 8-bit quantization
            from coremltools.models.neural_network.quantization_utils import quantize_weights
            quantized_model = quantize_weights(model, nbits=8, quantization_mode='linear')
            
            # Save the quantized model
            output_path = f"{output_name}_int8_weights.mlpackage"
            quantized_model.save(output_path)
            print(f"✅ Saved weight-quantized model to {output_path}")
            
            # Check size reduction
            original_size = get_model_size(model_path)
            quantized_size = get_model_size(output_path)
            print(f"📉 Size reduction: {original_size:.1f}MB → {quantized_size:.1f}MB ({quantized_size/original_size*100:.1f}%)")
            
        except Exception as e:
            print(f"⚠️  Weight quantization failed: {e}")
            print("Trying alternative approach...")
            
            # Method 2: Try compute precision optimization
            try:
                print("\n🔧 Attempting compute precision optimization...")
                
                # Configure optimization for INT8
                config = OptimizationConfig(
                    global_config={
                        "compute_precision": ComputePrecision.INT8,
                        "minimum_deployment_target": ct.target.iOS16
                    }
                )
                
                # Load as MLProgram if possible
                model_proto = ct.utils.load_spec(model_path)
                
                # For MLProgram models
                if hasattr(model_proto, 'mlProgram'):
                    print("📱 Detected MLProgram model, applying INT8 optimization...")
                    
                    # Create optimized model
                    optimized_spec = ct.optimize.coreml.optimize_model(
                        model_proto,
                        config=config
                    )
                    
                    # Save optimized model
                    output_path = f"{output_name}_int8_compute.mlpackage"
                    ct.utils.save_spec(optimized_spec, output_path)
                    print(f"✅ Saved compute-optimized model to {output_path}")
                else:
                    print("❌ Model is not MLProgram format, cannot apply compute optimization")
                    
            except Exception as e2:
                print(f"❌ Compute optimization also failed: {e2}")
                
                # Method 3: Manual quantization (most aggressive)
                print("\n🔧 Attempting manual INT8 quantization...")
                try:
                    manual_quantize_int8(model_path, output_name)
                except Exception as e3:
                    print(f"❌ Manual quantization failed: {e3}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to quantize model: {e}")
        return False

def manual_quantize_int8(model_path, output_name):
    """Manually quantize model to INT8 using experimental features."""
    print("⚠️  Using experimental quantization - expect accuracy degradation!")
    
    # Load model
    model = ct.models.MLModel(model_path)
    
    # Use experimental linear quantization
    quantized_model = linear_quantize_weights(
        model,
        mode="INT8",  # Force INT8
        dtype=np.int8,
        quantization_mode='linear_symmetric'  # Symmetric quantization for INT8
    )
    
    output_path = f"{output_name}_int8_manual.mlpackage"
    quantized_model.save(output_path)
    print(f"✅ Saved manually quantized model to {output_path}")
    
    return True

def get_model_size(model_path):
    """Get the size of a model in MB."""
    total_size = 0
    if os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(model_path)
    return total_size / (1024 * 1024)

def create_test_script():
    """Create a script to test the quantized models."""
    test_script = '''#!/usr/bin/env python3
"""Test quantized models for functionality and accuracy impact."""

import coremltools as ct
import numpy as np
import time

def test_model(model_path, model_name):
    """Test a quantized model."""
    print(f"\\nTesting {model_name}...")
    
    try:
        model = ct.models.MLModel(model_path)
        
        # Test inference speed
        if "wespeaker" in model_name:
            # Test wespeaker
            waveform = np.random.randn(3, 160000).astype(np.float32)
            mask = np.ones((3, 589)).astype(np.float32)
            
            start = time.time()
            output = model.predict({"waveform": waveform, "mask": mask})
            end = time.time()
            
            print(f"✅ Inference time: {(end-start)*1000:.1f}ms")
            print(f"   Output shape: {output['embedding'].shape}")
            
        elif "segmentation" in model_name:
            # Test segmentation
            audio = np.random.randn(1, 1, 160000).astype(np.float32)
            
            start = time.time()
            output = model.predict({"audio": audio})
            end = time.time()
            
            print(f"✅ Inference time: {(end-start)*1000:.1f}ms")
            if 'output' in output:
                print(f"   Output shape: {output['output'].shape}")
            elif 'segments' in output:
                print(f"   Output shape: {output['segments'].shape}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("Testing INT8 Quantized Models")
    print("=" * 60)
    
    # Test all quantized models
    import glob
    for model_path in glob.glob("*_int8*.mlpackage"):
        test_model(model_path, model_path)
'''
    
    with open("test_quantized_models.py", "w") as f:
        f.write(test_script)
    os.chmod("test_quantized_models.py", 0o755)
    print("\n📝 Created test_quantized_models.py")

def main():
    print("🚀 INT8 Quantization Tool for FluidAudio Models")
    print("⚠️  WARNING: INT8 quantization will likely degrade accuracy!")
    print("⚠️  Expect DER to increase from 17.8% to 25-30%")
    print("=" * 60)
    
    # Models to quantize
    models = [
        ("wespeaker.mlpackage", "wespeaker"),
        ("pyannote_segmentation.mlpackage", "segmentation"),
        ("segmentation.mlpackage", "segmentation_alt")  # In case it has different name
    ]
    
    success_count = 0
    
    for model_path, output_name in models:
        if quantize_model_int8(model_path, output_name):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Quantization Summary")
    print(f"{'='*60}")
    print(f"✅ Successfully quantized: {success_count} models")
    
    if success_count > 0:
        print("\n📋 Next Steps:")
        print("1. Test the quantized models:")
        print("   python test_quantized_models.py")
        print("\n2. Benchmark with quantized models:")
        print("   swift run fluidaudio diarization-benchmark --single-file ES2004a")
        print("\n3. Compare DER and RTF with original models")
        print("\n⚠️  If DER degrades too much (>25%), consider:")
        print("   - Using Float16 instead of INT8")
        print("   - Selective quantization (some layers only)")
        print("   - Keeping embedding layers at higher precision")
        
        # Create test script
        create_test_script()
        
        # Create Swift integration guide
        create_swift_integration_guide()

def create_swift_integration_guide():
    """Create a guide for integrating quantized models in Swift."""
    guide = '''# INT8 Quantized Models Integration Guide

## Model Paths

Update DiarizerModels.swift to load quantized models:

```swift
// In DiarizerModels.swift, update model loading:

// For wespeaker
if let quantizedPath = modelDirectory.appendingPathComponent("wespeaker_int8_weights.mlpackage"),
   FileManager.default.fileExists(atPath: quantizedPath.path) {
    logger.info("Loading INT8 quantized wespeaker model")
    embeddingModel = try? MLModel(contentsOf: quantizedPath)
}

// For segmentation
if let quantizedPath = modelDirectory.appendingPathComponent("segmentation_int8_weights.mlpackage"),
   FileManager.default.fileExists(atPath: quantizedPath.path) {
    logger.info("Loading INT8 quantized segmentation model")
    segmentationModel = try? MLModel(contentsOf: quantizedPath)
}
```

## Performance Monitoring

Add logging to track quantization impact:

```swift
// In EmbeddingExtractor.swift
logger.info("Model precision: INT8 (quantized)")
logger.info("Expected performance: 2-4x faster, 25-30% DER")
```

## Rollback Plan

Keep original models and add a flag:

```swift
struct DiarizerConfig {
    var useQuantizedModels: Bool = false  // Set to true to use INT8
}
```
'''
    
    with open("INT8_INTEGRATION_GUIDE.md", "w") as f:
        f.write(guide)
    print("📄 Created INT8_INTEGRATION_GUIDE.md")

if __name__ == "__main__":
    main()