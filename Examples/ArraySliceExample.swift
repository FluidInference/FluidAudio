import FluidAudio
import Foundation

@available(macOS 13.0, iOS 16.0, *)
class StreamingDiarizationBuffer {
    private let bufferDuration: TimeInterval = 3.0  // 3 seconds of audio
    private let updateInterval: TimeInterval = 0.333  // Update every 1/3 second
    private let sampleRate = 16000

    // Circular buffer to hold audio samples
    private var audioBuffer: [Float]
    private var writePosition = 0
    private var bufferSize: Int

    private let diarizerManager: DiarizerManager

    init() {
        self.bufferSize = Int(bufferDuration * Double(sampleRate))
        self.audioBuffer = Array(repeating: 0.0, count: bufferSize)

        // Configure diarizer for streaming
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minActiveFramesCount: 10.0,
            debugMode: false,
            chunkDuration: 3.0,  // Process 3-second chunks
            chunkOverlap: 0.5  // Small overlap for continuity
        )
        self.diarizerManager = DiarizerManager(config: config)
    }

    /// Add new audio samples to the circular buffer
    func appendAudio(_ newSamples: [Float]) {
        for sample in newSamples {
            audioBuffer[writePosition] = sample
            writePosition = (writePosition + 1) % bufferSize
        }
    }

    /// Process the current buffer using ArraySlice (no copy!)
    func processCurrentBuffer() async throws -> DiarizationResult {
        // Create an ArraySlice view of the current buffer state
        // This avoids copying the entire buffer
        let currentSlice: ArraySlice<Float>

        if writePosition == 0 {
            // Buffer hasn't wrapped yet - use the whole buffer
            currentSlice = audioBuffer[...]
        } else {
            // Buffer has wrapped - need to create a contiguous view
            // In a real implementation, you might use a ring buffer data structure
            // For now, we'll create a slice of the most recent data
            let startIdx = max(0, writePosition - bufferSize)
            currentSlice = audioBuffer[startIdx..<writePosition]
        }

        // Process the slice directly - no memory copy!
        return try diarizerManager.performCompleteDiarization(
            currentSlice,
            sampleRate: sampleRate
        )
    }

    /// Example of efficient sliding window processing
    func processSlidingWindow() async throws {
        let windowSize = Int(2.0 * Double(sampleRate))  // 2-second window
        let stepSize = Int(0.333 * Double(sampleRate))  // 1/3 second step

        // Process sliding windows without copying data
        for offset in stride(from: 0, to: audioBuffer.count - windowSize, by: stepSize) {
            // Create ArraySlice views - zero copy!
            let windowSlice = audioBuffer[offset..<(offset + windowSize)]

            // Validate audio quality on the slice
            let validationResult = diarizerManager.validateAudio(windowSlice)

            if validationResult.isValid {
                print("Processing window at offset \(offset)...")
                // Process this window
                _ = try diarizerManager.performCompleteDiarization(
                    windowSlice,
                    sampleRate: sampleRate
                )
            }
        }
    }
}

// Example usage
@available(macOS 13.0, iOS 16.0, *)
func demonstrateArraySliceUsage() async throws {
    print("=== ArraySlice Diarization Example ===\n")

    let buffer = StreamingDiarizationBuffer()

    // Simulate streaming audio input
    let chunkSize = 5333  // ~1/3 second at 16kHz

    print("Simulating streaming audio input...")
    for i in 0..<10 {
        // Generate or receive audio chunk
        let audioChunk = [Float](repeating: Float.random(in: -0.1...0.1), count: chunkSize)

        // Append to buffer (efficient circular buffer update)
        buffer.appendAudio(audioChunk)

        // Every 3rd chunk (~1 second), process the buffer
        if i % 3 == 2 {
            print("\nProcessing buffer after \(i+1) chunks...")

            // Load models if needed
            if let models = try? await DiarizerModels.download() {
                buffer.diarizerManager.initialize(models: models)

                // Process using ArraySlice - no memory copy!
                let result = try await buffer.processCurrentBuffer()
                print("Found \(result.segments.count) speaker segments")
            }
        }
    }

    print("\n=== Memory Efficiency Comparison ===")

    // Traditional approach - creates copies
    func traditionalProcessing(audio: [Float]) -> [Float] {
        // This creates a new array (memory copy)
        let segment = Array(audio[1000..<5000])
        return segment
    }

    // ArraySlice approach - no copies
    func arraySliceProcessing(audio: [Float]) -> ArraySlice<Float> {
        // This creates a view, not a copy
        let segment = audio[1000..<5000]
        return segment
    }

    let testAudio = [Float](repeating: 0.5, count: 160000)

    // Measure memory impact
    print("\nOriginal array size: \(testAudio.count * MemoryLayout<Float>.size) bytes")

    let traditionalSegment = traditionalProcessing(audio: testAudio)
    print("Traditional approach: Created new array of \(traditionalSegment.count * MemoryLayout<Float>.size) bytes")

    let sliceSegment = arraySliceProcessing(audio: testAudio)
    print("ArraySlice approach: Created view (minimal overhead, no data copy)")
    print("Slice references elements \(sliceSegment.startIndex) to \(sliceSegment.endIndex)")

    print("\n✅ ArraySlice Benefits:")
    print("- Zero-copy views of existing data")
    print("- Perfect for sliding windows and overlapping segments")
    print("- Reduces memory pressure in streaming applications")
    print("- Maintains compatibility with existing FluidAudio APIs")
}
