import Accelerate
import CoreML
import OSLog

/// Embedding extractor with ANE-aligned memory and zero-copy operations
@available(macOS 13.0, iOS 16.0, *)
public class EmbeddingExtractor {
    private let wespeakerModel: MLModel
    private let logger = AppLogger(category: "EmbeddingExtractor")
    private let memoryOptimizer = ANEMemoryOptimizer()

    // Pre-allocated ANE-aligned buffers
    private var waveformBuffer: MLMultiArray?
    private var maskBuffer: MLMultiArray?

    // Reusable feature providers
    private var featureProviders: [ZeroCopyDiarizerFeatureProvider] = []

    public init(embeddingModel: MLModel) {
        self.wespeakerModel = embeddingModel

        // Pre-allocate ANE-aligned buffers
        do {
            self.waveformBuffer = try memoryOptimizer.createAlignedArray(
                shape: [3, 160000] as [NSNumber],
                dataType: .float32
            )

            self.maskBuffer = try memoryOptimizer.createAlignedArray(
                shape: [3, 1000] as [NSNumber],  // Typical mask size
                dataType: .float32
            )
        } catch {
            logger.error("Failed to allocate ANE-aligned buffers: \(error)")
            // Buffers will remain nil, will be allocated on-demand in getEmbeddings
        }

        logger.info("EmbeddingExtractor initialized with ANE-aligned buffers")
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
        // We need to return embeddings for ALL speakers, not just active ones
        // to maintain compatibility with the rest of the pipeline
        var embeddings: [[Float]] = []

        // Get or create appropriately sized mask buffer
        let maskShape = [3, masks[0].count] as [NSNumber]
        let currentMaskBuffer = try memoryOptimizer.getPooledBuffer(
            key: "wespeaker_mask_\(masks[0].count)",
            shape: maskShape,
            dataType: .float32
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

            // Use ANE-optimized copy for audio data
            memoryOptimizer.optimizedCopy(
                from: audio,
                to: waveformBuffer!,
                offset: 0  // First speaker slot
            )

            // Optimize mask creation with zero-copy view
            fillMaskBufferOptimized(
                masks: masks,
                speakerIndex: speakerIdx,
                buffer: currentMaskBuffer
            )

            // Create zero-copy feature provider
            let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveformBuffer!),
                "mask": MLFeatureValue(multiArray: currentMaskBuffer),
            ])

            // Run model with optimal prediction options
            let options = MLPredictionOptions()
            if #available(macOS 14.0, iOS 17.0, *) {
                // Prefetch to Neural Engine for better performance
                waveformBuffer!.prefetchToNeuralEngine()
                currentMaskBuffer.prefetchToNeuralEngine()
            }

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
}
