import json
import os

home = os.path.expanduser("~")
path = os.path.join(home, ".cache/fluidaudio/Models/kokoro/zh_vocab_index.json")

if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    with open(path, 'r') as f:
        vocab = json.load(f)
    
    print(f"Vocab size: {len(vocab)}")
    tokens = ["↓", "↗", "→", "↘", "wo", "w", "o"]
    for t in tokens:
        if t in vocab:
            print(f"'{t}': {vocab[t]}")
        else:
            print(f"'{t}': NOT FOUND")
