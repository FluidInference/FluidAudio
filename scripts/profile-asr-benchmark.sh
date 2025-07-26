#!/bin/bash

# Profile ASR benchmark with Instruments
# Usage: ./profile-asr-benchmark.sh [options]

set -e

rm -rf profiling-results/asr-benchmark.trace


# Default values
SUBSET="test-clean"
MAX_FILES="10"
OUTPUT_DIR="profiling-results"
TRACE_NAME="asr-benchmark"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --trace-name)
            TRACE_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --subset <subset>         LibriSpeech subset (default: test-clean)"
            echo "  --max-files <n>          Max files to process (default: 10)"
            echo "  --output-dir <dir>       Output directory (default: profiling-results)"
            echo "  --trace-name <name>      Trace file name (default: asr-benchmark)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Remove existing trace file if it exists
if [ -f "$OUTPUT_DIR/${TRACE_NAME}.trace" ]; then
    echo "🗑️  Removing existing trace file..."
    rm -rf "$OUTPUT_DIR/${TRACE_NAME}.trace"
fi

echo "🔍 Building release version..."
swift build -c release

echo "📊 Starting comprehensive Instruments profiling..."
echo "   Subset: $SUBSET"
echo "   Max files: $MAX_FILES"
echo ""
echo "🚀 Capturing all metrics in a single trace..."
echo "   - CPU Performance & Time Profiling"
echo "   - GPU & Metal Performance Shaders"
echo "   - Neural Engine (ANE) Activity"
echo "   - Memory Usage & Leaks"
echo "   - System Activity & Power"
echo ""
echo "   Output: $OUTPUT_DIR/${TRACE_NAME}.trace"

# Create a custom template configuration that includes all instruments
# Using Time Profiler as base to ensure we get CPU samples/profiling data
# Note: Some instruments (Neural Engine, Metal GPU Counters) may require admin privileges
echo ""
echo "⚠️  Note: Some instruments may require administrator privileges."
echo "   If prompted, you can either:"
echo "   1. Enter your password when requested"
echo "   2. Run this script with: sudo $0 $@"
echo "   3. Use --no-prompt to skip privileged instruments"
echo ""

# Build the command path
COMMAND_PATH="$(pwd)/.build/release/fluidaudio"

# Check if the binary exists
if [ ! -f "$COMMAND_PATH" ]; then
    echo "❌ Error: Binary not found at $COMMAND_PATH"
    echo "   Please run: swift build -c release"
    exit 1
fi

# Create a simple launcher script that adds a small delay
# This helps ensure the process stays alive long enough for Instruments
LAUNCHER_SCRIPT="$OUTPUT_DIR/launcher.sh"
cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
# Small delay to ensure Instruments can attach
sleep 0.5
exec "$COMMAND_PATH" asr-benchmark --subset "$SUBSET" --max-files "$MAX_FILES" --auto-download
EOF
chmod +x "$LAUNCHER_SCRIPT"

# Run with all relevant instruments including Neural Engine and memory tracking
# Using a launcher script to ensure proper attachment
xcrun xctrace record \
    --output "$OUTPUT_DIR/${TRACE_NAME}.trace" \
    --template "Time Profiler" \
    --instrument "Neural Engine" \
    --instrument "Core ML" \
    --instrument "Metal Application" \
    --instrument "VM Tracker" \
    --instrument "Allocations" \
    --instrument "Leaks" \
    --instrument "os_signpost" \
    --instrument "os_log" \
    --time-limit 300s \
    --launch -- \
    "$LAUNCHER_SCRIPT"

# Clean up launcher script
rm -f "$LAUNCHER_SCRIPT"

echo ""
echo "✅ Profiling complete!"
echo "sudo chown -R $(whoami):staff profiling-results/asr-benchmark.trace"

echo ""
echo "📊 Trace file saved to: $OUTPUT_DIR/${TRACE_NAME}.trace"
echo ""
echo "🔍 To analyze the results:"
echo ""
echo "1. Open the trace file:"
echo "   open $OUTPUT_DIR/${TRACE_NAME}.trace"
echo ""
echo "2. Key metrics to examine:"
echo ""
echo "   📱 Neural Engine (ANE):"
echo "   - Look for 'Neural Engine' track in the timeline"
echo "   - Check 'com.apple.ane' processes for ANE activity"
echo "   - Monitor ANE utilization percentage"
echo ""
echo "   🎮 GPU/Metal Performance:"
echo "   - Check 'GPU' track for utilization"
echo "   - Look for Metal Performance Shaders (MPS) activity"
echo "   - Monitor GPU memory usage"
echo ""
echo "   💾 Memory Usage:"
echo "   - Check 'Memory' track for allocation patterns"
echo "   - Look for spikes during model loading"
echo "   - Monitor total memory footprint"
echo ""
echo "   ⚡ CPU Performance:"
echo "   - Check 'CPU' tracks for each core"
echo "   - Look for CoreML framework activity"
echo "   - Identify performance bottlenecks"
echo ""
echo "   🔋 System Impact:"
echo "   - Monitor thermal state changes"
echo "   - Check power consumption"
echo "   - Look for process priority changes"
echo ""
echo "3. Pro tips:"
echo "   - Use the timeline to correlate different metrics"
echo "   - Filter by process name 'fluidaudio' for focused analysis"
echo "   - Check the 'Points of Interest' track for custom signposts"
echo "   - Use the statistics view for aggregate data"
