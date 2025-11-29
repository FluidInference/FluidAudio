import json
import os
from misaki.zh import ZHG2P
from tqdm import tqdm

def generate_ipa_map():
    # Path to existing lexicon to get keys
    home = os.path.expanduser("~")
    existing_lexicon_path = os.path.join(home, ".cache/fluidaudio/Models/kokoro/zh_char_phonemes.json")
    
    if not os.path.exists(existing_lexicon_path):
        print(f"Error: Existing lexicon not found at {existing_lexicon_path}")
        return

    print(f"Loading keys from {existing_lexicon_path}...")
    with open(existing_lexicon_path, 'r') as f:
        existing_map = json.load(f)
    
    chars = list(existing_map.keys())
    print(f"Found {len(chars)} characters.")

    # Initialize G2P
    print("Initializing ZHG2P...")
    g2p = ZHG2P()
    
    new_map = {}
    print("Generating IPA phonemes...")
    
    # Process batch
    # ZHG2P might not support batch list input, so loop is fine as it's pure python/regex/dict lookup
    for char in tqdm(chars):
        try:
            phonemes, _ = g2p(char)
            new_map[char] = phonemes.strip()
        except Exception as e:
            print(f"Error processing {char}: {e}")
            new_map[char] = ""

    output_path = "zh_char_ipa.json"
    print(f"Saving new map to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(new_map, f, ensure_ascii=False, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    generate_ipa_map()
