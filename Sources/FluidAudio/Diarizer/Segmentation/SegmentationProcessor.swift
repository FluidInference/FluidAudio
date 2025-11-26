import Accelerate
import CoreML
import Foundation
import OSLog

/// Segmentation processor with ANE-aligned memory and zero-copy operations
public struct SegmentationProcessor {

    private let logger = AppLogger(category: "Segmentation")
    private let memoryOptimizer = ANEMemoryOptimizer()

    public init() {}

    /// Detect speaker segments using the CoreML segmentation model.
    ///
    /// This is the main model inference method that runs the pyannote segmentation model
    /// to detect speech activity and separate overlapping speakers.
    ///
    /// - Parameters:
    ///   - audioChunk: 10-second audio chunk (16kHz)
    ///   - segmentationModel: Pre-loaded CoreML segmentation model
    ///   - chunkSize: Expected audio size (default 160k samples = 10s)
    /// - Returns: Tuple of (binarized segments, raw feature provider)
    public func getSegments(
        audioChunk: ArraySlice<Float>,
        segmentationModel: MLModel,
        chunkSize: Int = 160_000
    ) throws -> (segments: [[[Float]]], featureProvider: MLFeatureProvider) {

        // Create ANE-aligned audio array
        let audioArray = try memoryOptimizer.createAlignedArray(
            shape: [1, 1, NSNumber(value: chunkSize)],
            dataType: .float32
        )

        // Use optimized memory copy
        let ptr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        let copyCount = min(audioChunk.count, chunkSize)

        audioChunk.prefix(chunkSize).withUnsafeBufferPointer { buffer in
            // Use vDSP for optimized copy
            vDSP_mmov(
                buffer.baseAddress!,
                ptr,
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
        }

        // Zero-fill remaining if needed
        if copyCount < chunkSize {
            var zero: Float = 0
            vDSP_vfill(&zero, ptr.advanced(by: copyCount), 1, vDSP_Length(chunkSize - copyCount))
        }

        // Create zero-copy feature provider
        let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
            "audio": MLFeatureValue(multiArray: audioArray)
        ])

        // Configure optimal prediction options
        let options = MLPredictionOptions()

        // Prefetch to Neural Engine for better performance
        audioArray.prefetchToNeuralEngine()

        let output = try segmentationModel.prediction(from: featureProvider, options: options)

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            throw DiarizerError.processingFailed("Missing segments output from segmentation model")
        }

        // Process segments with optimized memory access
        let segments = processSegmentsOptimized(segmentOutput)

        return (segments, output)
    }

    private func processSegmentsOptimized(_ segmentOutput: MLMultiArray) -> [[[Float]]] {
        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        // Pre-allocate result array
        var segments = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: combinations),
                count: frames),
            count: 1)

        // Use direct memory access for better performance
        let ptr = segmentOutput.dataPointer.assumingMemoryBound(to: Float.self)

        // Copy data in a cache-friendly manner
        for f in 0..<frames {
            segments[0][f].withUnsafeMutableBufferPointer { buffer in
                // Use vDSP_mmov for consistency with other vDSP operations
                vDSP_mmov(
                    ptr.advanced(by: f * combinations),
                    buffer.baseAddress!,
                    vDSP_Length(combinations),
                    1,
                    vDSP_Length(combinations),
                    1
                )
            }
        }

        return powersetConversionOptimized(segments)
    }

    private func powersetConversionOptimized(_ segments: [[[Float]]]) -> [[[Float]]] {
        let powerset: [[Int]] = [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        // Use ANE-aligned array for result
        let binarizedArray = try? memoryOptimizer.createAlignedArray(
            shape: [batchSize, numFrames, numSpeakers] as [NSNumber],
            dataType: .float32
        )

        guard let binarizedArray = binarizedArray else {
            // Fallback to regular array
            return powersetConversionFallback(segments)
        }

        // Direct memory access
        let ptr = binarizedArray.dataPointer.assumingMemoryBound(to: Float.self)

        // Process all frames
        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                // Find max using vDSP
                var maxValue: Float = 0
                var maxIndex: vDSP_Length = 0
                frame.withUnsafeBufferPointer { buffer in
                    vDSP_maxvi(buffer.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(frame.count))
                }

                // Set speakers based on powerset
                let baseIdx = (b * numFrames + f) * numSpeakers
                for speaker in powerset[Int(maxIndex)] {
                    ptr[baseIdx + speaker] = 1.0
                }
            }
        }

        // Convert back to nested array format
        var result = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers),
                count: numFrames
            ),
            count: batchSize
        )

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                for s in 0..<numSpeakers {
                    let idx = (b * numFrames + f) * numSpeakers + s
                    result[b][f][s] = ptr[idx]
                }
            }
        }

        return result
    }

    private func powersetConversionFallback(_ segments: [[[Float]]]) -> [[[Float]]] {
        // Original implementation as fallback
        let powerset: [[Int]] = [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        var binarized = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers),
                count: numFrames
            ),
            count: batchSize
        )

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                guard let bestIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) else {
                    continue
                }

                for speaker in powerset[bestIdx] {
                    binarized[b][f][speaker] = 1.0
                }
            }
        }

        return binarized
    }

    func createSlidingWindowFeature(
        binarizedSegments: [[[Float]]], chunkOffset: Double = 0.0
    ) -> SlidingWindowFeature {
        // These values come from the pyannote/speaker-diarization-3.1 model configuration
        let slidingWindow = SlidingWindow(
            start: chunkOffset,
            duration: 0.0619375,  // 991 samples at 16kHz (model's sliding window duration)
            step: 0.016875  // 270 samples at 16kHz (model's sliding window step)
        )
        // Model output frame rate: 1 / 0.016875 = ~59.26 fps (not 50 fps)

        return SlidingWindowFeature(
            data: binarizedSegments,
            slidingWindow: slidingWindow
        )
    }
}
