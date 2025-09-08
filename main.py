from IPython.display import display, Audio
import soundfile as sf
import torch
import coremltools as ct
import numpy as np
import json
    
def get_input_ids(
        phonemes,
        context_length: int = 249,
        speed: float = 1,
    ) -> torch.FloatTensor:
    
    # Load vocabulary from vocab_index.json
    import json
    with open("vocab_index.json", "r") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["vocab"]
    
    print(f"Using vocab_index.json with {len(vocab)} entries")
    
    # Map phonemes to IDs using the correct vocabulary
    input_ids = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    assert len(input_ids)+2 <= context_length, (len(input_ids)+2, context_length)
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    print(f"Using correct vocab - mapped {len(input_ids[0])-2} phonemes to IDs")
    print(f"Input IDs: {input_ids[0].tolist()[:20]}")
    return input_ids


def get_phonemes(
    text: str,
    voice: str = "af_heart",
    speed: int = 1
) -> tuple[torch.FloatTensor, torch.LongTensor]:

    # Load the local af_heart.pt file directly
    pack = torch.load("af_heart.pt", map_location='cpu')
    print(f"Loaded af_heart.pt directly, shape: {pack.shape}")
    
    # Try to use the comprehensive phoneme dictionary first
    with open("word_phonemes.json", "r") as f:
        phoneme_dict = json.load(f)
        word_to_phonemes = phoneme_dict["word_to_phonemes"]
        print(f"Loaded HF phoneme dictionary with {len(word_to_phonemes)} words")
        
    # Convert text to phonemes using dictionary lookup
    words = text.lower().split()
    all_phonemes = []
    for word in words:
        # Remove punctuation
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word in word_to_phonemes:
            phonemes = word_to_phonemes[clean_word]
            all_phonemes.extend(phonemes)
            all_phonemes.append(" ")  # Add space between words

    
    # Remove trailing space
    if all_phonemes and all_phonemes[-1] == " ":
        all_phonemes.pop()
        
    ps = all_phonemes
    print(f"Dictionary lookup: {len(ps)} phonemes: {' '.join(ps[:20])}")
    
    # Get input IDs
    input_ids = get_input_ids(ps, speed=speed)
    
    # Get the voice embedding based on phoneme count
    if isinstance(pack, torch.Tensor):
        index = min(len(ps) - 1, pack.shape[0] - 1)
        refs = pack[index, :, :]
        print(f"Using embedding at index {index} for {len(ps)} phonemes, shape: {refs.shape}")
    else:
        refs = pack[len(ps)-1]
    
    return input_ids, refs
        

text = "I can't believe we finally made it to the summit after climbing for twelve exhausting hours through wind and rain, but wow, this view of the endless mountain ranges stretching to the horizon makes every single difficult step completely worth the journey."

input_ids, ref_s = get_phonemes(text, "af_heart")


import coremltools as ct
import numpy as np
import soundfile as sf
from IPython.display import Audio, display

# Load the saved model
mlmodel = ct.models.MLModel("kokoro.mlpackage")

# Prepare inputs - pad to 249 for CoreML model
input_ids_padded = input_ids.clone()
while input_ids_padded.shape[1] < 249:
    input_ids_padded = torch.cat([input_ids_padded, torch.zeros(1, 1, dtype=torch.long)], dim=1)
input_ids_np = input_ids_padded.numpy().astype(np.int32)
ref_s_np = ref_s.numpy().astype(np.float32)
random_phases_np = np.zeros((1, 9)).astype(np.float32)  # Fixed phases matching Swift

# Run inference
output = mlmodel.predict({ 
    "input_ids": input_ids_np,
    "ref_s": ref_s_np,
    "random_phases": random_phases_np  # Include third input
})

# Get audio
audio = list(output.values())[0]
print(f"Audio shape: {audio.shape}")

# Normalize audio before saving
audio_normalized = audio.squeeze() / (np.abs(audio).max() + 1e-8)

# Save as WAV
sf.write('kokoro5.wav', audio_normalized, 24000)

# Play audio
display(Audio(audio_normalized, rate=24000))
