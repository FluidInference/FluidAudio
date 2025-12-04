import numpy as np
import glob
import os

def load_bin(path, shape=None):
    data = np.fromfile(path, dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data

def compare_outputs():
    nemo_files = sorted(glob.glob("ReferenceOutputs/EncoderDebug/nemo_encoder_step_*.npy"))
    swift_files = sorted(glob.glob("ReferenceOutputs/EncoderDebug/swift_encoder_step_*.bin"))
    
    print(f"Found {len(nemo_files)} NeMo files and {len(swift_files)} Swift files")
    
    for i, (nemo_path, swift_path) in enumerate(zip(nemo_files, swift_files)):
        print(f"\nComparing Step {i}:")
        print(f"  NeMo: {nemo_path}")
        print(f"  Swift: {swift_path}")
        
        nemo_data = np.load(nemo_path)
        # Swift data is flat, we need to reshape it to match NeMo
        # NeMo shape is likely [1, 512, T]
        swift_data = load_bin(swift_path)
        
        print(f"  NeMo Shape: {nemo_data.shape}")
        print(f"  Swift Size: {swift_data.size}")
        
        if swift_data.size != nemo_data.size:
            print(f"  MISMATCH: Size mismatch! {swift_data.size} != {nemo_data.size}")
            continue
            
        swift_data = swift_data.reshape(nemo_data.shape)
        
        # Compare
        diff = np.abs(nemo_data - swift_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"  Max Diff: {max_diff}")
        print(f"  Mean Diff: {mean_diff}")
        
        if max_diff > 1e-3:
            print("  FAIL: Significant difference found")
            # Print some values
            print(f"  NeMo First 10: {nemo_data.flatten()[:10]}")
            print(f"  Swift First 10: {swift_data.flatten()[:10]}")
        else:
            print("  PASS: Outputs match")

if __name__ == "__main__":
    compare_outputs()
