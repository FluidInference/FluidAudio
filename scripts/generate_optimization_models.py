#!/usr/bin/env python3
"""
Generate CoreML optimization models for ASR pipeline
Note: Transpose model removed as it's now handled in the CoreML encoder
"""

import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models import MLModel
import os


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

    import torch
    import torch.nn as nn

    total_size = vocab_size + duration_count

    class TokenDurationModel(nn.Module):
        def __init__(self, vocab_size, duration_count):
            super().__init__()
            self.vocab_size = vocab_size
            self.duration_count = duration_count

        def forward(self, logits):
            # Flatten 4D input to 1D: (1, 1, 1, total_size) -> (total_size,)
            flattened = logits.view(-1)

            # Split into token and duration logits
            token_logits = flattened[:self.vocab_size]
            duration_logits = flattened[self.vocab_size:self.vocab_size + self.duration_count]

            # Get argmax and max score for tokens
            token_id = torch.argmax(token_logits, dim=0, keepdim=True)
            token_score = torch.max(token_logits, dim=0, keepdim=True)[0]

            # Get argmax for duration
            duration_index = torch.argmax(duration_logits, dim=0, keepdim=True)

            return token_id, token_score, duration_index

    # Create and trace the model
    model = TokenDurationModel(vocab_size, duration_count)
    model.eval()

    # Example input with 4D shape as expected from joint network
    example_input = torch.randn(1, 1, 1, total_size)
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML with 4D input shape - let CoreML handle the flattening internally
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="logits", shape=(1, 1, 1, total_size))],
        minimum_deployment_target=ct.target.macOS13
    )

    # Set metadata
    coreml_model.author = 'FluidAudio'
    coreml_model.short_description = 'Token and duration prediction for TDT decoder'
    coreml_model.version = '1.0'

    return coreml_model

def main():
    """Generate all optimization models"""

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Resources', 'OptimizationModels')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating CoreML optimization models...")

    # Generate padding model
    print("1. Skipping padding model (not needed for current issue)")

    # Generate token/duration model
    print("2. Creating token/duration model...")
    token_duration_model = create_token_duration_model()
    token_duration_path = os.path.join(output_dir, 'TokenDurationPrediction.mlpackage')
    token_duration_model.save(token_duration_path)
    print(f"   Saved to: {token_duration_path}")

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
