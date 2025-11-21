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

        let waveformSampleCount = 160_000
        let waveformShape = [3, waveformSampleCount] as [NSNumber]
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

        let audioSampleCount = audio.distance(from: audio.startIndex, to: audio.endIndex)
        let effectiveSampleCount = min(audioSampleCount, waveformSampleCount)

        let maskRepeatFrameCount: Int
        if effectiveSampleCount == 0 {
            maskRepeatFrameCount = 0
        } else if effectiveSampleCount < waveformSampleCount {
            let scaledFrames = Int(
                (Double(firstMask.count) * Double(effectiveSampleCount)) / Double(waveformSampleCount)
            )
            maskRepeatFrameCount = max(1, min(firstMask.count, scaledFrames))
        } else {
            maskRepeatFrameCount = firstMask.count
        }

        // Repeat shorter chunks to 10s so the embedding model sees consistent input length
        fillWaveformBuffer(
            audio: audio,
            buffer: waveformBuffer,
            targetSampleCount: waveformSampleCount,
            availableSampleCount: audioSampleCount
        )

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
                buffer: maskBuffer,
                repeatFrameCount: maskRepeatFrameCount
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

    private func fillWaveformBuffer<C>(
        audio: C,
        buffer: MLMultiArray,
        targetSampleCount: Int,
        availableSampleCount: Int
    ) where C: RandomAccessCollection, C.Element == Float, C.Index == Int {
        let copyCount = min(availableSampleCount, targetSampleCount)

        guard copyCount > 0 else { return }

        if copyCount == targetSampleCount {
            memoryOptimizer.optimizedCopy(from: audio, to: buffer, offset: 0)
            return
        }

        // Copy the available audio once
        memoryOptimizer.optimizedCopy(from: audio.prefix(copyCount), to: buffer, offset: 0)

        // Repeat in-place to reach target length with minimal additional copies
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        var filled = copyCount
        while filled < targetSampleCount {
            let repeatCount = min(filled, targetSampleCount - filled)
            let destination = ptr.advanced(by: filled)
            memmove(destination, ptr, repeatCount * MemoryLayout<Float>.size)
            filled += repeatCount
        }
    }

    private func fillMaskBufferOptimized(
        masks: [[Float]],
        speakerIndex: Int,
        buffer: MLMultiArray,
        repeatFrameCount: Int
    ) {
        let ptr = buffer.dataPointer.assumingMemoryBound(to: Float.self)
        let maskCount = masks[speakerIndex].count
        let framesToCopy = min(max(repeatFrameCount, 0), maskCount)
        guard framesToCopy > 0 else { return }

        // Clear just the active row (maskCount elements) before refilling
        vDSP_vclr(ptr, 1, vDSP_Length(maskCount))

        // Copy speaker mask to first slot using optimized memory copy
        masks[speakerIndex].withUnsafeBufferPointer { maskPtr in
            memcpy(ptr, maskPtr.baseAddress!, framesToCopy * MemoryLayout<Float>.size)
        }

        var filledFrames = framesToCopy
        while filledFrames < maskCount {
            let copyCount = min(filledFrames, maskCount - filledFrames)
            let destination = ptr.advanced(by: filledFrames)
            memmove(destination, ptr, copyCount * MemoryLayout<Float>.size)
            filledFrames += copyCount
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
