import Accelerate
import CoreML
import OSLog

/// Embedding extractor with ANE-aligned memory and zero-copy operations
public final class EmbeddingExtractor {
    private let wespeakerModel: MLModel
    private let logger = AppLogger(category: "EmbeddingExtractor")
    private let memoryOptimizer = ANEMemoryOptimizer()

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

        let waveformShape = [3, 160_000] as [NSNumber]
        let maskShape = [3, firstMask.count] as [NSNumber]

        let waveformBuffer = try memoryOptimizer.createAlignedArray(
            shape: waveformShape,
            dataType: .float32
        )

        let maskBuffer = try memoryOptimizer.createAlignedArray(
            shape: maskShape,
            dataType: .float32
        )

        // We need to return embeddings for ALL speakers, not just active ones
        // to maintain compatibility with the rest of the pipeline
        var embeddings: [[Float]] = []

        // Fill shared waveform buffer once; reused across speakers
        fillWaveformBuffer(
            audio: audio,
            buffer: waveformBuffer
        )

        // Calculate number of masks that are actually used

        let numMasksInChunk = (firstMask.count * audio.count + 80_000) / 160_000

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
                numMasksInChunk: numMasksInChunk,
                speakerIndex: speakerIdx,
                buffer: maskBuffer
            )

            // Create zero-copy feature provider
            let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveformBuffer),
                "mask": MLFeatureValue(multiArray: maskBuffer),
            ])

            // Run model with optimal prediction options
            let options = MLPredictionOptions()
            // Prefetch to Neural Engine for better performance
            waveformBuffer.prefetchToNeuralEngine()
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

    /// Fill the waveform buffer with loop (repeat) padding
    private func fillWaveformBuffer<C>(
        audio: C,
        buffer: MLMultiArray
    ) where C: RandomAccessCollection, C.Element == Float, C.Index == Int {
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        var sampleCount = audio.count
        let requiredCount = 160_000

        // Load the original audio into the buffer
        memoryOptimizer.optimizedCopy(
            from: audio,
            to: buffer,
            offset: 0  // first speaker slot
        )

        // If sampleCount is zero then we'll get stuck in an infinite loop
        guard sampleCount > 0 else {
            return
        }

        // Repeat-pad the buffer by doubling it until it's full
        while sampleCount < requiredCount {
            let copyCount = min(sampleCount, requiredCount - sampleCount)
            vDSP_mmov(
                ptr,
                ptr.advanced(by: sampleCount),
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
            sampleCount += copyCount
        }
    }

    /// Fill the mask buffer with loop (repeat) padding
    private func fillMaskBufferOptimized(
        masks: [[Float]],
        numMasksInChunk: Int,
        speakerIndex: Int,
        buffer: MLMultiArray
    ) {
        // Clear buffer using vDSP for speed
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        let totalElements = buffer.count
        var zero: Float = 0
        vDSP_vfill(&zero, ptr, 1, vDSP_Length(totalElements))

        // Copy speaker mask to first slot using optimized memory copy
        let requiredCount = masks[speakerIndex].count
        var currentCount = numMasksInChunk

        masks[speakerIndex].withUnsafeBufferPointer { maskPtr in
            vDSP_mmov(
                maskPtr.baseAddress!,
                ptr,
                vDSP_Length(currentCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(currentCount)
            )
        }

        // If maskCount is zero then we'll get stuck in an infinite loop
        guard currentCount > 0 else {
            return
        }

        // Repeat-pad the buffer by doubling it until it's full
        while currentCount < requiredCount {
            let copyCount = min(currentCount, requiredCount - currentCount)
            vDSP_mmov(
                ptr,
                ptr.advanced(by: currentCount),
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
            currentCount += copyCount
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
}
