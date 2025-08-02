#!/usr/bin/env python3
"""
Create an embedding preprocessor model that performs clean mask generation and waveform duplication on GPU.
This eliminates the need for CPU-based array manipulations in Swift.
"""

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


class EmbeddingPreprocessor(nn.Module):
    """
    PyTorch model that preprocesses inputs for the embedding model.
    
    Operations:
    1. Calculate clean frames (sum < 2.0)
    2. Apply clean mask to speaker segments
    3. Duplicate waveform for each speaker
    4. Transpose masks to correct format
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, audio, speaker_masks):
        # audio: (1, 1, 160000)
        # speaker_masks: (1, 589, 3)
        
        # Remove batch dimension from masks for easier processing
        masks = speaker_masks.squeeze(0)  # (589, 3)
        
        # Calculate clean frames: sum speaker activities and check if < 2.0
        speaker_sum = masks.sum(dim=1, keepdim=True)  # (589, 1)
        clean_mask = (speaker_sum < 2.0).float()  # (589, 1)
        
        # Apply clean mask to speaker segments
        clean_segments = masks * clean_mask  # (589, 3)
        
        # Transpose from (frames, speakers) to (speakers, frames) for embedding model
        masks_transposed = clean_segments.transpose(0, 1)  # (3, 589)
        
        # Duplicate audio waveform for each speaker
        # First, reshape audio from (1, 1, 160000) to (160000,)
        audio_flat = audio.squeeze()  # (160000,)
        
        # Create 3 copies
        waveforms = audio_flat.unsqueeze(0).repeat(3, 1)  # (3, 160000)
        
        return waveforms, masks_transposed


def create_embedding_preprocessor():
    """Create and convert the embedding preprocessor to CoreML."""
    
    # Create PyTorch model
    model = EmbeddingPreprocessor()
    model.eval()
    
    # Create example inputs
    example_audio = torch.randn(1, 1, 160000)
    example_masks = torch.randn(1, 589, 3)
    
    # Trace the model
    traced_model = torch.jit.trace(model, (example_audio, example_masks))
    
    # Convert to CoreML
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="audio",
                shape=(1, 1, 160000),
                dtype=np.float32
            ),
            ct.TensorType(
                name="speaker_masks", 
                shape=(1, 589, 3),
                dtype=np.float32
            )
        ],
        outputs=[
            ct.TensorType(name="waveforms", dtype=np.float32),
            ct.TensorType(name="masks_transposed", dtype=np.float32)
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS13
    )
    
    # Add metadata
    coreml_model.author = "FluidAudio"
    coreml_model.short_description = "Embedding preprocessor for speaker diarization"
    coreml_model.version = "1.0"
    
    # Add input/output descriptions
    coreml_model.input_description["audio"] = "Audio waveform (1, 1, 160000)"
    coreml_model.input_description["speaker_masks"] = "Binary speaker masks from segmentation (1, 589, 3)"
    coreml_model.output_description["waveforms"] = "Duplicated waveforms for each speaker (3, 160000)"
    coreml_model.output_description["masks_transposed"] = "Clean speaker masks (3, 589)"
    
    return coreml_model


def test_embedding_preprocessor(model):
    """Test the embedding preprocessor with sample data."""
    print("\nTesting embedding preprocessor...")
    
    # Create test inputs
    test_audio = np.random.randn(1, 1, 160000).astype(np.float32)
    test_masks = np.zeros((1, 589, 3), dtype=np.float32)
    
    # Simulate speaker activity patterns
    # Speaker 0 active in frames 0-100
    test_masks[0, 0:100, 0] = 1.0
    # Speaker 1 active in frames 150-250
    test_masks[0, 150:250, 1] = 1.0
    # Overlap in frames 200-210 (should be filtered out)
    test_masks[0, 200:210, 0] = 1.0
    
    # Run prediction
    output = model.predict({
        "audio": test_audio,
        "speaker_masks": test_masks
    })
    
    waveforms = output["waveforms"]
    masks_out = output["masks_transposed"]
    
    print(f"✓ Input audio shape: {test_audio.shape}")
    print(f"✓ Input masks shape: {test_masks.shape}")
    print(f"✓ Output waveforms shape: {waveforms.shape}")
    print(f"✓ Output masks shape: {masks_out.shape}")
    
    # Verify waveform duplication
    audio_flat = test_audio[0, 0, :]
    waveform_match = True
    for i in range(3):
        if not np.allclose(waveforms[i, :], audio_flat):
            waveform_match = False
            break
    
    if waveform_match:
        print("✓ Waveform duplication verified")
    else:
        print("✗ Waveform duplication failed")
    
    # Verify clean mask application
    # Frames 200-210 should be zero for all speakers (overlap)
    overlap_frames = masks_out[:, 200:210]
    if np.all(overlap_frames == 0):
        print("✓ Clean mask filtering verified (overlapping frames zeroed)")
    else:
        print("✗ Clean mask filtering failed")
    
    # Check non-overlapping frames preserved
    speaker0_early = masks_out[0, 50:60]
    speaker1_mid = masks_out[1, 180:190]
    if np.all(speaker0_early == 1.0) and np.all(speaker1_mid == 1.0):
        print("✓ Non-overlapping frames preserved")
    else:
        print("✗ Non-overlapping frame preservation failed")
    
    return True


def main():
    print("Creating Embedding Preprocessor for FluidAudio")
    print("=" * 50)
    
    # Create the model
    model = create_embedding_preprocessor()
    
    # Save the model
    output_path = "embedding_preprocessor.mlpackage"
    model.save(output_path)
    print(f"\n✓ Saved embedding preprocessor to {output_path}")
    
    # Test the model
    test_embedding_preprocessor(model)
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("--------")
    print("The embedding preprocessor performs the following GPU-accelerated operations:")
    print("1. Clean frame calculation (sum < 2.0)")
    print("2. Clean mask application to speaker segments")
    print("3. Audio waveform duplication (3x)")
    print("4. Mask transposition for embedding model")
    print("\nThis replaces ~80 lines of Swift array manipulation with a single GPU call.")
    print("\nNext steps:")
    print("1. Copy the model to the FluidAudio models directory")
    print("2. Update DiarizerManager to load and use this preprocessor")
    print("3. Update EmbeddingExtractor to use the preprocessor when available")


if __name__ == "__main__":
    main()