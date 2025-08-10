#!/bin/bash

echo "🚀 Running StyleTTS2 visualization setup..."

# Commands to run
cat << 'EOF'

Please copy and run these commands in your terminal:

# 1. Go to StyleTTS2 directory
cd /Users/kikow/brandon/StyleTTS2

# 2. Create Models directory
mkdir -p Models/LJSpeech

# 3. Download the config file
curl -L https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/config.yml -o Models/LJSpeech/config.yml

# 4. Activate your virtual environment
source venv/bin/activate

# 5. Install missing dependencies (if needed)
pip install requests

# 6. Create and run the visualization script
cat > viz.py << 'VIZEOF'
import torch
import yaml
from munch import Munch

print("Loading StyleTTS2...")

# Load config
with open('Models/LJSpeech/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Import and build model
from models import build_model

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

model_params = Munch(config['model_params'])
slm_params = Munch(config.get('slm_params', {}))

nets = build_model(model_params, slm_params, device)

# Show all components
print("\nStyleTTS2 Components:")
print("-" * 40)
for name in dir(nets):
    if not name.startswith('_') and hasattr(getattr(nets, name), 'parameters'):
        params = sum(p.numel() for p in getattr(nets, name).parameters())
        print(f"{name}: {params:,} parameters")

# Show text encoder
print("\nText Encoder:")
print("-" * 40)
print(nets.text_encoder)

# Create visualization
try:
    from torchview import draw_graph
    import os
    os.makedirs('graphs', exist_ok=True)
    
    dummy = torch.randn(1, 50, model_params.n_tokens).to(device)
    graph = draw_graph(
        nets.text_encoder,
        input_data=dummy,
        save_graph=True,
        filename='graphs/text_encoder'
    )
    print("\n✅ Visualization saved to graphs/text_encoder.png")
except Exception as e:
    print(f"\n⚠️ Could not create graph: {e}")
VIZEOF

# 7. Run the visualization
python viz.py

EOF