#!/bin/bash

# Setup script for StyleTTS2 visualization
set -e

echo "🚀 Setting up StyleTTS2 visualization..."

# Navigate to StyleTTS2
cd /Users/kikow/brandon/StyleTTS2

# Create directory for models
mkdir -p Models/LJSpeech

# Download config
echo "📥 Downloading config..."
curl -L https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/config.yml -o Models/LJSpeech/config.yml

# Check if download succeeded
if [ -f "Models/LJSpeech/config.yml" ]; then
    echo "✅ Config downloaded successfully"
else
    echo "❌ Failed to download config"
    exit 1
fi

# Create simple visualization script
cat > viz.py << 'EOF'
import torch
import yaml
from munch import Munch
from models import build_model

# Load config
print("Loading config...")
with open('Models/LJSpeech/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Build model
print("Building model...")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model_params = Munch(config['model_params'])
slm_params = Munch(config.get('slm_params', {}))

nets = build_model(model_params, slm_params, device)

# Print what we have
print("\n" + "="*50)
print("StyleTTS2 Components Found:")
print("="*50)

for name in dir(nets):
    if not name.startswith('_'):
        attr = getattr(nets, name)
        if hasattr(attr, 'parameters'):
            param_count = sum(p.numel() for p in attr.parameters())
            print(f"\n{name}:")
            print(f"  Parameters: {param_count:,}")

# Visualize text encoder
print("\n" + "="*50)
print("Text Encoder Architecture:")
print("="*50)
if hasattr(nets, 'text_encoder'):
    print(nets.text_encoder)

# Try torchview
try:
    from torchview import draw_graph
    import os
    os.makedirs('graphs', exist_ok=True)
    
    print("\nCreating visualization...")
    dummy = torch.randn(1, 50, model_params.n_tokens).to(device)
    graph = draw_graph(
        nets.text_encoder,
        input_data=dummy,
        graph_name='TextEncoder',
        save_graph=True,
        filename='graphs/text_encoder',
        expand_nested=False,
        device=device
    )
    print("✅ Saved to graphs/text_encoder.png")
    
    # Also create a detailed summary
    from torchinfo import summary
    print("\n" + "="*50)
    print("Detailed Model Summary:")
    print("="*50)
    summary(nets.text_encoder, input_size=(1, 50, model_params.n_tokens), device=device, depth=3)
    
except Exception as e:
    print(f"Visualization failed: {e}")
    print("\nBut you can still see the model structure printed above!")

print("\n✅ Done! Check 'graphs' folder for visualizations")
EOF

echo "✅ Created viz.py"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated venv"
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import torch" 2>/dev/null || pip install torch torchvision torchaudio
python -c "import yaml" 2>/dev/null || pip install pyyaml
python -c "import munch" 2>/dev/null || pip install munch
python -c "import torchview" 2>/dev/null || pip install torchview
python -c "import torchinfo" 2>/dev/null || pip install torchinfo

echo ""
echo "✅ Setup complete! Now run:"
echo "  cd /Users/kikow/brandon/StyleTTS2"
echo "  python viz.py"