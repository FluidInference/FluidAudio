#!/usr/bin/env python3
"""
Add post-processing layers to CoreML models to move array manipulations from Swift.
"""

import os
import sys
import torch
import coremltools as ct
import numpy as np


def add_powerset_postprocessing_to_segmentation():
    """
    Add powerset conversion post-processing to existing segmentation model.
    """
    print("Adding powerset post-processing to segmentation model...")
    
    # Load the existing CoreML model
    home = os.path.expanduser("~")
    input_model_path = os.path.join(home, "Library/Application Support/FluidAudio/Models/speaker-diarization-coreml/pyannote_segmentation.mlmodelc")
    if not os.path.exists(input_model_path):
        print(f"Error: {input_model_path} not found.")
        return False
    
    # Load model
    model = ct.models.MLModel(input_model_path)
    
    # Create a wrapper model that adds post-processing
    import coremltools.converters.mil as mil
    from coremltools.converters.mil import Builder as mb
    
    @mb.program(input_specs=[ct.TensorType(shape=(1, 1, 160000), name="audio")])
    def segmentation_with_powerset(audio):
        # Run the original model
        segments_flat = mb.coreml_predict(
            mlmodel=model,
            inputs={"audio": audio},
            outputs=["segments"]
        )["segments"]
        
        # Reshape from (1, 4123) to (1, 589, 7)
        segments_reshaped = mb.reshape(x=segments_flat, shape=[1, 589, 7])
        
        # Get argmax for each frame
        argmax_indices = mb.reduce_argmax(x=segments_reshaped, axes=[2], keep_dims=False)
        
        # Create powerset mapping matrix
        powerset_mapping = mb.const(val=np.array([
            [0, 0, 0],  # 0: []
            [1, 0, 0],  # 1: [0]
            [0, 1, 0],  # 2: [1]
            [0, 0, 1],  # 3: [2]
            [1, 1, 0],  # 4: [0, 1]
            [1, 0, 1],  # 5: [0, 2]
            [0, 1, 1],  # 6: [1, 2]
        ], dtype=np.float32))
        
        # One-hot encode the argmax indices
        one_hot = mb.one_hot(
            indices=argmax_indices,
            one_hot_vector_size=7,
            axis=-1,
            on_value=1.0,
            off_value=0.0
        )
        
        # Reshape for matrix multiplication
        one_hot_2d = mb.reshape(x=one_hot, shape=[589, 7])
        
        # Apply powerset mapping
        speaker_activations = mb.matmul(x=one_hot_2d, y=powerset_mapping)
        
        # Reshape to final output
        output = mb.reshape(x=speaker_activations, shape=[1, 589, 3])
        
        return output
    
    # Convert to CoreML
    try:
        coreml_model = ct.convert(
            segmentation_with_powerset,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32
        )
        
        output_path = "pyannote_segmentation_with_powerset.mlpackage"
        coreml_model.save(output_path)
        print(f"✓ Saved segmentation model with powerset post-processing to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to convert segmentation model: {e}")
        return False


def add_clean_mask_preprocessing_to_embedding():
    """
    Add clean mask preprocessing to existing embedding model.
    """
    print("Adding clean mask preprocessing to embedding model...")
    
    # Load the existing CoreML model
    home = os.path.expanduser("~")
    input_model_path = os.path.join(home, "Library/Application Support/FluidAudio/Models/speaker-diarization-coreml/wespeaker.mlmodelc")
    if not os.path.exists(input_model_path):
        print(f"Error: {input_model_path} not found.")
        return False
    
    # Load model
    model = ct.models.MLModel(input_model_path)
    
    # Create a wrapper model that adds preprocessing
    import coremltools.converters.mil as mil
    from coremltools.converters.mil import Builder as mb
    
    @mb.program(input_specs=[
        ct.TensorType(shape=(3, 160000), name="waveform"),
        ct.TensorType(shape=(3, 589), name="mask")
    ])
    def embedding_with_clean_masks(waveform, mask):
        # Create clean masks (filter frames with multiple speakers)
        # Sum across speakers dimension
        speaker_sum = mb.reduce_sum(x=mask, axes=[0], keep_dims=True)
        
        # Create clean mask: 1.0 where sum < 2, 0.0 otherwise
        clean_mask_bool = mb.less(x=speaker_sum, y=2.0)
        clean_mask_float = mb.cast(x=clean_mask_bool, dtype="fp32")
        
        # Broadcast clean mask to all speakers
        clean_masks = mb.mul(x=mask, y=clean_mask_float)
        
        # Run the original model with clean masks
        outputs = mb.coreml_predict(
            mlmodel=model,
            inputs={
                "waveform": waveform,
                "mask": clean_masks
            },
            outputs=["constant", "embedding"]
        )
        
        return outputs["constant"], outputs["embedding"]
    
    # Convert to CoreML
    try:
        coreml_model = ct.convert(
            embedding_with_clean_masks,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32
        )
        
        output_path = "wespeaker_with_clean_masks.mlpackage"
        coreml_model.save(output_path)
        print(f"✓ Saved embedding model with clean mask preprocessing to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to convert embedding model: {e}")
        return False


def test_models():
    """
    Test the models with post-processing.
    """
    print("\nTesting models with post-processing...")
    
    # Test segmentation
    try:
        model_path = "pyannote_segmentation_with_powerset.mlpackage"
        if os.path.exists(model_path):
            model = ct.models.MLModel(model_path)
            dummy_audio = np.random.randn(1, 1, 160000).astype(np.float32)
            output = model.predict({"audio": dummy_audio})
            print(f"✓ Segmentation output shape: {list(output.values())[0].shape}")
            print(f"  Output contains binary speaker activations for 3 speakers")
    except Exception as e:
        print(f"✗ Segmentation test failed: {e}")
    
    # Test embedding
    try:
        model_path = "wespeaker_with_clean_masks.mlpackage"
        if os.path.exists(model_path):
            model = ct.models.MLModel(model_path)
            dummy_waveform = np.random.randn(3, 160000).astype(np.float32)
            dummy_mask = np.zeros((3, 589), dtype=np.float32)
            dummy_mask[0, :100] = 1.0  # Single speaker in first 100 frames
            
            output = model.predict({
                "waveform": dummy_waveform,
                "mask": dummy_mask
            })
            print(f"✓ Embedding output shape: {output['embedding'].shape}")
            print(f"  Clean masks applied automatically")
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")


def main():
    print("CoreML Model Post-processing Tool")
    print("=" * 50)
    print("This tool adds post-processing layers to existing CoreML models")
    print("to move array manipulations from Swift into the models.\n")
    
    # Check if models exist
    home = os.path.expanduser("~")
    models_dir = os.path.join(home, "Library/Application Support/FluidAudio/Models/speaker-diarization-coreml")
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        print("Please download the models first.")
        return
    
    # Add post-processing to models
    segmentation_success = add_powerset_postprocessing_to_segmentation()
    embedding_success = add_clean_mask_preprocessing_to_embedding()
    
    if segmentation_success or embedding_success:
        print("\n" + "=" * 50)
        test_models()
        
        print("\n" + "=" * 50)
        print("Summary:")
        print("--------")
        if segmentation_success:
            print("✓ Segmentation model: Powerset conversion integrated")
            print("  - Input: audio (1, 1, 160000)")
            print("  - Output: speaker_activations (1, 589, 3)")
            print("  - No Swift array manipulation needed!")
        
        if embedding_success:
            print("\n✓ Embedding model: Clean mask preprocessing integrated")
            print("  - Inputs: waveform (3, 160000), mask (3, 589)")
            print("  - Outputs: constant, embedding")
            print("  - Clean masks applied automatically!")
        
        print("\nNext steps:")
        print("1. Update Swift code to use the new models")
        print("2. Remove array manipulation code from Swift")
        print("3. Test with real audio data")
    else:
        print("\nNo models were successfully processed.")
        print("Please ensure the original models exist in the 'conversion' directory.")


if __name__ == "__main__":
    main()