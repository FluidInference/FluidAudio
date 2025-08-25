#!/bin/bash

#
# Voice Processing Test Runner
# This script runs the VoiceProcessingTestApp.swift with proper FluidAudio dependencies
#

set -e

echo "🎤 FluidAudio Voice Processing Test"
echo "=================================="
echo ""

# Check if we're in the FluidAudio directory
if [[ ! -f "Package.swift" ]]; then
    echo "❌ Error: Must be run from the FluidAudio root directory"
    exit 1
fi

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "❌ Error: Swift is not installed or not in PATH"
    exit 1
fi

echo "📋 Checking system requirements..."
echo "✅ macOS version: $(sw_vers -productVersion)"
echo "✅ Swift version: $(swift --version | head -n1)"
echo ""

echo "🔨 Building FluidAudio package..."
swift build -c release
echo ""

echo "🚀 Running Voice Processing Test App..."
echo "   (This will compile and run the test app with FluidAudio dependencies)"
echo ""

# Run the test app with package dependencies
swift run voice-processing-test || {
    echo ""
    echo "ℹ️  If the above failed, try building first:"
    echo "   swift build"
    echo "   swift run voice-processing-test"
}