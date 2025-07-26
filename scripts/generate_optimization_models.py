#!/usr/bin/env python3
"""
Generate CoreML optimization models for ASR pipeline
"""

import coremltools as ct
import numpy as np
from coremltools.models.neural_network import NeuralNetworkBuilder
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models import MLModel
import os

def create_transpose_model():
    """Create a model that transposes encoder output from [1, hidden_dim, T] to [1, T, hidden_dim]"""
    
    # Create flexible transpose using PyTorch approach
    import torch
    import torch.nn as nn
    
    class TransposeModel(nn.Module):
        def forward(self, x):
            # x shape: [batch, hidden_dim, time] -> [batch, time, hidden_dim]
            return x.transpose(1, 2)
    
    # Create and trace the model
    model = TransposeModel()
    model.eval()
    
    # Use flexible shapes
    example_input = torch.randn(1, 1024, 100)  # Example with actual encoder dimensions
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to Core ML with flexible shapes
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="x", shape=(1, ct.RangeDim(1, 2048), ct.RangeDim(1, 2048)))],
        minimum_deployment_target=ct.target.macOS13
    )
    
    # Set metadata
    coreml_model.author = 'FluidAudio'
    coreml_model.short_description = 'Efficient tensor transpose for ASR encoder output'
    coreml_model.version = '1.0'
    
    return coreml_model

def create_argmax_model(vocab_size=1025):
    """Create a model for argmax operation on logits"""
    
    input_features = [
        ('logits', ct.models.datatypes.Array(vocab_size))
    ]
    
    output_features = [
        ('index', ct.models.datatypes.Array(1)),
        ('score', ct.models.datatypes.Array(1))
    ]
    
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    # Add argmax to get index
    builder.add_argmax(
        name='argmax',
        input_name='logits',
        output_name='index',
        axis=0,
        keepdims=True
    )
    
    # Add reduce_max to get score
    builder.add_reduce_max(
        name='max_score',
        input_name='logits',
        output_name='score',
        axes=[0],
        keepdims=True
    )
    
    model = MLModel(builder.spec)
    model.author = 'FluidAudio'
    model.short_description = 'Argmax operation for token prediction'
    model.version = '1.0'
    
    return model

def create_padding_model(max_length=160000):
    """Create a model that pads variable length audio to fixed length"""
    
    # Variable length input
    input_features = [
        ('audio', ct.models.datatypes.Array(shape=(ct.RangeDim(1, max_length),)))
    ]
    
    output_features = [
        ('padded_output', ct.models.datatypes.Array(shape=(max_length,)))
    ]
    
    # Use Core ML's newer unified conversion API for padding
    import torch
    import torch.nn as nn
    
    class PaddingModel(nn.Module):
        def __init__(self, max_length):
            super().__init__()
            self.max_length = max_length
            
        def forward(self, x):
            # x shape: (length,)
            current_length = x.shape[0]
            if current_length >= self.max_length:
                return x[:self.max_length]
            else:
                padding = self.max_length - current_length
                return torch.nn.functional.pad(x, (0, padding), mode='constant', value=0.0)
    
    # Create PyTorch model
    pytorch_model = PaddingModel(max_length)
    pytorch_model.eval()
    
    # Example input for tracing
    example_input = torch.randn(80000)  # Example audio length
    
    # Trace the model
    traced_model = torch.jit.trace(pytorch_model, example_input)
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="audio", shape=(ct.RangeDim(1, max_length),))],
        outputs=[ct.TensorType(name="padded_output", shape=(max_length,))],
        minimum_deployment_target=ct.target.macOS13
    )
    
    model.author = 'FluidAudio'
    model.short_description = 'Audio padding for ASR preprocessing'
    model.version = '1.0'
    
    return model

def create_token_duration_model(vocab_size=1025, duration_count=5):
    """Create a model that splits logits and predicts token and duration"""
    
    total_size = vocab_size + duration_count
    
    input_features = [
        ('logits', ct.models.datatypes.Array(total_size))
    ]
    
    output_features = [
        ('token_id', ct.models.datatypes.Array(1)),
        ('token_score', ct.models.datatypes.Array(1)),
        ('duration_index', ct.models.datatypes.Array(1))
    ]
    
    builder = NeuralNetworkBuilder(input_features, output_features)
    
    # Slice token logits
    builder.add_slice(
        name='slice_tokens',
        input_name='logits',
        output_name='token_logits',
        axis=0,
        start_index=0,
        end_index=vocab_size
    )
    
    # Slice duration logits
    builder.add_slice(
        name='slice_durations',
        input_name='logits',
        output_name='duration_logits',
        axis=0,
        start_index=vocab_size,
        end_index=total_size
    )
    
    # Argmax for token
    builder.add_argmax(
        name='token_argmax',
        input_name='token_logits',
        output_name='token_id',
        axis=0,
        keepdims=True
    )
    
    # Max score for token
    builder.add_reduce_max(
        name='token_max_score',
        input_name='token_logits',
        output_name='token_score',
        axes=[0],
        keepdims=True
    )
    
    # Argmax for duration
    builder.add_argmax(
        name='duration_argmax',
        input_name='duration_logits',
        output_name='duration_index',
        axis=0,
        keepdims=True
    )
    
    model = MLModel(builder.spec)
    model.author = 'FluidAudio'
    model.short_description = 'Token and duration prediction for TDT decoder'
    model.version = '1.0'
    
    return model

def main():
    """Generate all optimization models"""
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Resources', 'OptimizationModels')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating CoreML optimization models...")
    
    # Generate transpose model
    print("1. Creating transpose model...")
    transpose_model = create_transpose_model()
    transpose_path = os.path.join(output_dir, 'TransposeEncoder.mlpackage')
    transpose_model.save(transpose_path)
    print(f"   Saved to: {transpose_path}")
    
    # Generate argmax model
    print("2. Creating argmax model...")
    argmax_model = create_argmax_model()
    argmax_path = os.path.join(output_dir, 'Argmax.mlmodel')
    argmax_model.save(argmax_path)
    print(f"   Saved to: {argmax_path}")
    
    # Generate padding model
    print("3. Skipping padding model (not needed for current issue)")
    
    # Generate token/duration model  
    print("4. Skipping token/duration model (not needed for current issue)")
    
    print("\nAll models generated successfully!")
    print(f"Models saved to: {output_dir}")
    
    # Compile models for faster loading
    print("\nCompiling models...")
    for model_file in os.listdir(output_dir):
        if model_file.endswith('.mlmodel') or model_file.endswith('.mlpackage'):
            model_path = os.path.join(output_dir, model_file)
            
            # Use coremlcompiler to compile
            result = os.system(f'xcrun coremlcompiler compile "{model_path}" "{output_dir}"')
            if result == 0:
                print(f"   Compiled: {model_file}")
            else:
                print(f"   Failed to compile: {model_file}")

if __name__ == '__main__':
    main()