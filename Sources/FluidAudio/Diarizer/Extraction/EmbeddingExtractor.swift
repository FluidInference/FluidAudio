import Accelerate
import CoreML
import OSLog

/// Embedding extractor with ANE-aligned memory and zero-copy operations
public final class EmbeddingExtractor {
    private let wespeakerModel: MLModel
    private let logger = AppLogger(category: "EmbeddingExtractor")
    private let memoryOptimizer = ANEMemoryOptimizer()
    private let embeddingWindowSampleCount = 160_000

    public init(embeddingModel: MLModel) {
        self.wespeakerModel = embeddingModel
        logger.info("EmbeddingExtractor ready with ANE memory optimizer")
    }

    /// Extract speaker embeddings using the CoreML embedding model.
    ///
    /// This is the main model inference method that runs the WeSpeaker embedding model
    /// to convert audio+masks into 256-dimensional speaker embeddings.
    ///
    /// - Parameters:
    ///   - audio: Raw audio samples (16kHz) - accepts any RandomAccessCollection of Float
    ///           (Array, ArraySlice, ContiguousArray, or custom collections)
    ///   - masks: Speaker activity masks from segmentation
    ///   - minActivityThreshold: Minimum frames for valid speaker
    /// - Returns: Array of 256-dim embeddings for each speaker
    public func getEmbeddings<C>(
        audio: C,
        masks: [[Float]],
        minActivityThreshold: Float = 10.0
    ) throws -> [[Float]]
    where C: RandomAccessCollection, C.Element == Float, C.Index == Int {
        guard let firstMask = masks.first else {
            return []
        }

        let waveformShape = [3, embeddingWindowSampleCount] as [NSNumber]
        let maskShape = [3, firstMask.count] as [NSNumber]

        let waveformBuffer = try memoryOptimizer.getPooledBuffer(
            key: "embedding_waveform",
            shape: waveformShape,
            dataType: .float32
        )

        let maskBuffer = try memoryOptimizer.getPooledBuffer(
            key: "embedding_mask_\(firstMask.count)",
            shape: maskShape,
            dataType: .float32
        )

        let audioCopyLength = min(audio.count, embeddingWindowSampleCount)

        memoryOptimizer.optimizedCopy(
            from: audio,
            to: waveformBuffer,
            offset: 0  // First speaker slot
        )

        repeatVoiceSegmentIfNeeded(
            in: waveformBuffer,
            filledLength: audioCopyLength,
            targetLength: embeddingWindowSampleCount,
            offset: 0
        )

        // Prefetch once since waveform is shared across speakers
        waveformBuffer.prefetchToNeuralEngine()

        // We need to return embeddings for ALL speakers, not just active ones
        // to maintain compatibility with the rest of the pipeline
        var embeddings: [[Float]] = []

        let options = MLPredictionOptions()

        // Process all speakers but optimize for active ones
        for speakerIdx in 0..<masks.count {
            // Check if speaker is active
            let speakerActivity = masks[speakerIdx].reduce(0, +)

            if speakerActivity < minActivityThreshold {
                // For inactive speakers, return zero embedding
                embeddings.append([Float](repeating: 0.0, count: 256))
                continue
            }

            // Optimize mask creation with zero-copy view
            fillMaskBufferOptimized(
                masks: masks,
                speakerIndex: speakerIdx,
                buffer: maskBuffer
            )

            // Create zero-copy feature provider
            let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveformBuffer),
                "mask": MLFeatureValue(multiArray: maskBuffer),
            ])

            // Run model with optimal prediction options
            // Prefetch to Neural Engine for better performance
            maskBuffer.prefetchToNeuralEngine()

            let output = try wespeakerModel.prediction(from: featureProvider, options: options)

            // Extract embedding with zero-copy
            if let embeddingArray = output.featureValue(for: "embedding")?.multiArrayValue {
                let embedding = extractEmbeddingOptimized(
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

    private func fillMaskBufferOptimized(
        masks: [[Float]],
        speakerIndex: Int,
        buffer: MLMultiArray
    ) {
        // Clear buffer using vDSP for speed
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        let totalElements = buffer.count
        var zero: Float = 0
        vDSP_vfill(&zero, ptr, 1, vDSP_Length(totalElements))

        // Copy speaker mask to first slot using optimized memory copy
        let maskCount = masks[speakerIndex].count
        masks[speakerIndex].withUnsafeBufferPointer { maskPtr in
            vDSP_mmov(
                maskPtr.baseAddress!,
                ptr,
                vDSP_Length(maskCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(maskCount)
            )
        }
    }

    private func extractEmbeddingOptimized(
        from multiArray: MLMultiArray,
        speakerIndex: Int
    ) -> [Float] {
        let embeddingDim = 256

        // Try to create a zero-copy view if possible
        if let embeddingView = try? memoryOptimizer.createZeroCopyView(
            from: multiArray,
            shape: [embeddingDim as NSNumber],
            offset: speakerIndex * embeddingDim
        ) {
            // Extract directly from the view
            var embedding = [Float](repeating: 0, count: embeddingDim)
            let ptr = embeddingView.dataPointer.assumingMemoryBound(to: Float.self)
            _ = embedding.withUnsafeMutableBufferPointer { buffer in
                // Use optimized memory copy
                memcpy(buffer.baseAddress!, ptr, embeddingDim * MemoryLayout<Float>.size)
            }
            return embedding
        }

        // Fallback to standard extraction
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

    private func repeatVoiceSegmentIfNeeded(
        in buffer: MLMultiArray,
        filledLength: Int,
        targetLength: Int,
        offset: Int
    ) {
        guard buffer.dataType == .float32 else { return }

        let availableLength = min(targetLength, max(0, buffer.count - offset))

        guard filledLength > 0, filledLength < availableLength else { return }

        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self).advanced(by: offset)

        var currentLength = filledLength
        while currentLength < availableLength {
            let copyCount = min(currentLength, availableLength - currentLength)
            vDSP_mmov(
                ptr,
                ptr.advanced(by: currentLength),
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
            currentLength += copyCount
        }
    }
}
