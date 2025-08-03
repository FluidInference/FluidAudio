import Accelerate
import CoreML
import OSLog

/// Optimized wespeaker wrapper that minimizes SliceByIndex overhead
@available(macOS 13.0, iOS 16.0, *)
public class OptimizedWeSpeaker {
    private let wespeakerModel: MLModel
    private let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "OptimizedWeSpeaker")

    // Pre-allocated buffers
    private var frameBuffer: MLMultiArray?
    private var waveformBuffer: MLMultiArray?

    public init(wespeakerPath: URL) throws {
        self.wespeakerModel = try MLModel(contentsOf: wespeakerPath)

        // Pre-allocate buffers
        self.waveformBuffer = try? MLMultiArray(
            shape: [3, 160000] as [NSNumber],
            dataType: .float32
        )
    }

    /// Process with optimizations
    public func getEmbeddings(
        audio: [Float],
        masks: [[Float]]
    ) throws -> [[Float]] {
        // We need to return embeddings for ALL speakers, not just active ones
        // to maintain compatibility with the rest of the pipeline
        var embeddings: [[Float]] = []

        // Process all speakers but optimize for active ones
        for speakerIdx in 0..<masks.count {
            // Check if speaker is active
            let speakerActivity = masks[speakerIdx].reduce(0, +)

            if speakerActivity < 10.0 {
                // For inactive speakers, return zero embedding
                embeddings.append([Float](repeating: 0.0, count: 256))
                continue
            }

            // Process active speaker
            fillWaveformBuffer(
                audio: audio,
                speakerIndex: 0,  // Always use first slot
                buffer: waveformBuffer!
            )

            // Create mask for single speaker
            let singleMask = createSingleSpeakerMask(
                masks: masks,
                speakerIndex: speakerIdx
            )

            // Run model
            let inputs: [String: Any] = [
                "waveform": waveformBuffer!,
                "mask": singleMask,
            ]

            let output = try wespeakerModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: inputs)
            )

            // Extract embedding for first speaker slot
            if let embeddingArray = output.featureValue(for: "embedding")?.multiArrayValue {
                let embedding = extractEmbedding(
                    from: embeddingArray,
                    speakerIndex: 0
                )
                embeddings.append(embedding)
            } else {
                // Fallback to zero embedding
                embeddings.append([Float](repeating: 0.0, count: 256))
            }
        }

        return embeddings
    }

    private func findActiveSpeakers(masks: [[Float]]) -> [Int] {
        var active: [Int] = []
        for (idx, mask) in masks.enumerated() {
            if mask.reduce(0, +) > 10.0 {
                active.append(idx)
            }
        }
        return active
    }

    private func fillWaveformBuffer(
        audio: [Float],
        speakerIndex: Int,
        buffer: MLMultiArray
    ) {
        // Clear buffer
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        memset(ptr, 0, 3 * 160000 * MemoryLayout<Float>.size)

        // Copy audio to specified speaker slot
        audio.withUnsafeBufferPointer { audioPtr in
            vDSP_mmov(
                audioPtr.baseAddress!,
                ptr.advanced(by: speakerIndex * 160000),
                vDSP_Length(min(audio.count, 160000)),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(min(audio.count, 160000))
            )
        }
    }

    private func createSingleSpeakerMask(
        masks: [[Float]],
        speakerIndex: Int
    ) -> MLMultiArray {
        let mask = try! MLMultiArray(
            shape: [3, masks[0].count] as [NSNumber],
            dataType: .float32
        )

        // Clear mask
        let ptr = mask.dataPointer.assumingMemoryBound(to: Float.self)
        memset(ptr, 0, 3 * masks[0].count * MemoryLayout<Float>.size)

        // Copy speaker mask to first slot
        masks[speakerIndex].withUnsafeBufferPointer { maskPtr in
            vDSP_mmov(
                maskPtr.baseAddress!,
                ptr,
                vDSP_Length(masks[0].count),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(masks[0].count)
            )
        }

        return mask
    }

    private func extractEmbedding(
        from multiArray: MLMultiArray,
        speakerIndex: Int
    ) -> [Float] {
        let embeddingDim = 256
        var embedding = [Float](repeating: 0, count: embeddingDim)

        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        let offset = speakerIndex * embeddingDim

        embedding.withUnsafeMutableBufferPointer { buffer in
            vDSP_mmov(
                ptr.advanced(by: offset),
                buffer.baseAddress!,
                vDSP_Length(embeddingDim),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(embeddingDim)
            )
        }

        return embedding
    }
}
