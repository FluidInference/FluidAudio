#!/bin/bash

# AMI Meeting Corpus Download Script for FluidAudioSwift Benchmarks
# Downloads the standard test meetings used in research papers

set -e

echo "🎯 Downloading AMI Meeting Corpus benchmark data..."
echo "   This follows the same evaluation protocol used in research papers"

# Create directory structure
BENCHMARK_DIR="$HOME/FluidAudioSwift_Datasets/ami_official"
mkdir -p "$BENCHMARK_DIR/sdm"
mkdir -p "$BENCHMARK_DIR/ihm"
mkdir -p "$BENCHMARK_DIR/annotations"

echo "📁 Created directory structure at: $BENCHMARK_DIR"

# Download ES2002a (Standard benchmark meeting)
echo "⬇️ Downloading ES2002a..."
mkdir -p "$BENCHMARK_DIR/temp/ES2002a/audio"
wget -P "$BENCHMARK_DIR/temp/ES2002a/audio" https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
wget -P "$BENCHMARK_DIR/temp/" https://groups.inf.ed.ac.uk/ami/download/temp/../CCBY4.0.txt

# Download ES2003a (Standard benchmark meeting)
echo "⬇️ Downloading ES2003a..."
mkdir -p "$BENCHMARK_DIR/temp/ES2003a/audio"
wget -P "$BENCHMARK_DIR/temp/ES2003a/audio" https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2003a/audio/ES2003a.Mix-Headset.wav

# Note: Your first command had ES2001a path but ES2003a directory - using ES2003a consistently
echo "⬇️ Downloading additional meeting..."
wget -P "$BENCHMARK_DIR/temp/ES2003a/audio" https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2003a/audio/ES2003a.Mix-Headset.wav

# Move files to correct benchmark structure
echo "📦 Organizing files for benchmark tests..."

# Move Mix-Headset files to SDM directory (Single Distant Microphone)
if [ -f "$BENCHMARK_DIR/temp/ES2002a/audio/ES2002a.Mix-Headset.wav" ]; then
    mv "$BENCHMARK_DIR/temp/ES2002a/audio/ES2002a.Mix-Headset.wav" "$BENCHMARK_DIR/sdm/"
    echo "   ✅ ES2002a.Mix-Headset.wav → sdm/"
fi

if [ -f "$BENCHMARK_DIR/temp/ES2003a/audio/ES2003a.Mix-Headset.wav" ]; then
    mv "$BENCHMARK_DIR/temp/ES2003a/audio/ES2003a.Mix-Headset.wav" "$BENCHMARK_DIR/sdm/"
    echo "   ✅ ES2003a.Mix-Headset.wav → sdm/"
fi

# Move license file
if [ -f "$BENCHMARK_DIR/temp/CCBY4.0.txt" ]; then
    mv "$BENCHMARK_DIR/temp/CCBY4.0.txt" "$BENCHMARK_DIR/"
    echo "   ✅ License file moved"
fi

# Clean up temp directory
rm -rf "$BENCHMARK_DIR/temp"

# Download AMI annotations (ground truth)
echo "⬇️ Downloading AMI manual annotations (ground truth)..."
cd "$BENCHMARK_DIR/annotations"
wget https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_manual_1.6.2.zip
unzip -q ami_manual_1.6.2.zip
echo "   ✅ Ground truth annotations downloaded and extracted"

# Create placeholder IHM files (for individual headset testing)
echo "📝 Creating IHM placeholders..."
echo "   Note: Add individual headset files to ihm/ directory for IHM testing"
touch "$BENCHMARK_DIR/ihm/.placeholder"

# Summary
echo ""
echo "🎉 AMI benchmark data setup complete!"
echo ""
echo "📊 Downloaded files:"
ls -la "$BENCHMARK_DIR/sdm/"
echo ""
echo "📁 Directory structure:"
echo "   $BENCHMARK_DIR/"
echo "   ├── sdm/                    # Single Distant Microphone (Mix-Headset files)"
echo "   ├── ihm/                    # Individual Headset Microphones (add individual files here)"
echo "   ├── annotations/            # Ground truth annotations"
echo "   └── CCBY4.0.txt            # License"
echo ""
echo "✅ Ready to run benchmark tests with:"
echo "   swift test --filter BenchmarkTests"
echo ""
echo "📝 Research paper comparison:"
echo "   - ES2002a, ES2003a are standard AMI test meetings"
echo "   - Mix-Headset.wav = SDM condition (Single Distant Microphone)"
echo "   - Expected DER: 25-35% for modern systems on SDM"
