import AVFoundation
import CoreML
import CryptoKit
import Foundation

private final class LSEENDModelState {
    var encRetKv: MLMultiArray
    var encRetScale: MLMultiArray
    var encConvCache: MLMultiArray
    var decRetKv: MLMultiArray
    var decRetScale: MLMultiArray
    var topBuffer: MLMultiArray

    init(
        encRetKv: MLMultiArray,
        encRetScale: MLMultiArray,
        encConvCache: MLMultiArray,
        decRetKv: MLMultiArray,
        decRetScale: MLMultiArray,
        topBuffer: MLMultiArray
    ) {
        self.encRetKv = encRetKv
        self.encRetScale = encRetScale
        self.encConvCache = encConvCache
        self.decRetKv = decRetKv
        self.decRetScale = decRetScale
        self.topBuffer = topBuffer
    }

    func copy() throws -> LSEENDModelState {
        try LSEENDModelState(
            encRetKv: cloneAlignedMultiArray(encRetKv),
            encRetScale: cloneAlignedMultiArray(encRetScale),
            encConvCache: cloneAlignedMultiArray(encConvCache),
            decRetKv: cloneAlignedMultiArray(decRetKv),
            decRetScale: cloneAlignedMultiArray(decRetScale),
            topBuffer: cloneAlignedMultiArray(topBuffer)
        )
    }
}

private struct LSEENDStepOutput {
    let fullLogits: [Float]
    let nextState: LSEENDModelState
}

private final class LSEENDInferenceSharedResources {
    let descriptor: LSEENDModelDescriptor
    let computeUnits: MLComputeUnits
    let metadata: LSEENDModelMetadata
    let featureConfig: LSEENDFeatureConfig
    let model: MLModel
    let targetSampleRate: Int
    let modelFrameHz: Double
    let streamingLatencySeconds: Double
    let decodeMaxSpeakers: Int
    let melSpectrogram: AudioMelSpectrogram
    let offlineFeatureExtractor: LSEENDOfflineFeatureExtractor

    // Preallocated ANE-aligned input arrays reused across predictStep calls
    let memoryOptimizer: ANEMemoryOptimizer
    let frameArray: MLMultiArray  // [1, 1, inputDim]
    let ingestArray: MLMultiArray  // [1]
    let decodeArray: MLMultiArray  // [1]

    init(
        descriptor: LSEENDModelDescriptor,
        computeUnits: MLComputeUnits
    ) throws {
        self.descriptor = descriptor
        self.computeUnits = computeUnits

        let metadataData = try Data(contentsOf: descriptor.metadataURL)
        metadata = try JSONDecoder().decode(LSEENDModelMetadata.self, from: metadataData)
        featureConfig = LSEENDFeatureConfig(metadata: metadata)
        targetSampleRate = metadata.resolvedSampleRate
        modelFrameHz = metadata.frameHz
        streamingLatencySeconds = metadata.streamingLatencySeconds
        decodeMaxSpeakers = metadata.maxNspks
        melSpectrogram = AudioMelSpectrogram(
            sampleRate: featureConfig.sampleRate,
            nMels: featureConfig.nMels,
            nFFT: featureConfig.nFFT,
            hopLength: featureConfig.hopLength,
            winLength: featureConfig.winLength,
            preemph: 0,
            padTo: 1,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true
        model = try MLModel(
            contentsOf: try LSEENDInferenceHelper.compiledModelURL(for: descriptor.modelURL),
            configuration: configuration
        )
        offlineFeatureExtractor = LSEENDOfflineFeatureExtractor(metadata: metadata, spectrogram: melSpectrogram)

        // Preallocate ANE-aligned input arrays
        memoryOptimizer = ANEMemoryOptimizer()
        frameArray = try memoryOptimizer.createAlignedArray(
            shape: [1, 1, NSNumber(value: metadata.inputDim)],
            dataType: .float32
        )
        ingestArray = try memoryOptimizer.createAlignedArray(
            shape: [1],
            dataType: .float32
        )
        decodeArray = try memoryOptimizer.createAlignedArray(
            shape: [1],
            dataType: .float32
        )
    }
}

/// Low level CoreML inference engine for LS-EEND speaker diarization.
///
/// Each engine instance owns its own compiled model, mel spectrogram, and feature extractor.
/// There are no shared singletons — multiple engines can run concurrently without interference.
///
/// The engine supports three usage modes:
/// - **Offline**: ``infer(samples:sampleRate:)`` or ``infer(audioFileURL:)`` for batch processing.
/// - **Streaming**: ``createSession(inputSampleRate:)`` to get an ``LSEENDStreamingSession`` for
///   incremental audio processing.
/// - **Simulation**: ``simulateStreaming(audioFileURL:chunkSeconds:)`` to replay a file through the
///   streaming pipeline with fixed-size chunks.
public final class LSEENDInferenceHelper {
    private let logger = AppLogger(category: "LSEENDInference")
    private let sharedResources: LSEENDInferenceSharedResources

    /// The descriptor used to create this engine.
    public var descriptor: LSEENDModelDescriptor { sharedResources.descriptor }
    /// The CoreML compute units this engine was configured with.
    public var computeUnits: MLComputeUnits { sharedResources.computeUnits }
    /// Model metadata decoded from the JSON configuration file.
    public var metadata: LSEENDModelMetadata { sharedResources.metadata }
    /// Derived feature extraction parameters.
    public var featureConfig: LSEENDFeatureConfig { sharedResources.featureConfig }
    /// The loaded CoreML model.
    public var model: MLModel { sharedResources.model }
    /// Audio sample rate the model expects (e.g. 8000 Hz).
    public var targetSampleRate: Int { sharedResources.targetSampleRate }
    /// Output frame rate in Hz (e.g. 10.0).
    public var modelFrameHz: Double { sharedResources.modelFrameHz }
    /// Minimum latency in seconds before the first output frame can be produced.
    public var streamingLatencySeconds: Double { sharedResources.streamingLatencySeconds }
    /// Maximum number of speaker slots in the model output (including boundary tracks).
    public var decodeMaxSpeakers: Int { sharedResources.decodeMaxSpeakers }

    fileprivate var melSpectrogram: AudioMelSpectrogram { sharedResources.melSpectrogram }
    private var offlineFeatureExtractor: LSEENDOfflineFeatureExtractor { sharedResources.offlineFeatureExtractor }

    private let lock = NSLock()

    /// Creates an inference engine by loading and compiling the CoreML model.
    ///
    /// - Parameters:
    ///   - descriptor: Locates the model and metadata files.
    ///   - computeUnits: CoreML compute units to use (default: `.cpuOnly`,
    ///     which is typically fastest for this model's architecture).
    /// - Throws: ``LSEENDError`` if the model or metadata cannot be loaded.
    public init(
        descriptor: LSEENDModelDescriptor,
        computeUnits: MLComputeUnits = .cpuOnly
    ) throws {
        sharedResources = try LSEENDInferenceSharedResources(
            descriptor: descriptor,
            computeUnits: computeUnits
        )
        logger.info("Loaded LS-EEND variant \(descriptor.variant.rawValue) @ \(descriptor.modelURL.path)")
    }

    /// Creates a new streaming session for incremental audio processing.
    ///
    /// - Parameter inputSampleRate: Must match ``targetSampleRate``.
    /// - Returns: A session that accepts audio via ``LSEENDStreamingSession/pushAudio(_:)``.
    /// - Throws: ``LSEENDError/unsupportedAudio(_:)`` if the sample rate doesn't match.
    public func createSession(inputSampleRate: Int) throws -> LSEENDStreamingSession {
        try LSEENDStreamingSession(engine: self, inputSampleRate: inputSampleRate)
    }

    /// Creates a streaming session with a caller-owned mel spectrogram instance.
    ///
    /// Use this overload when thread-safety requires the session to have its own
    /// isolated spectrogram rather than sharing the engine's instance.
    ///
    /// - Parameters:
    ///   - inputSampleRate: Must match ``targetSampleRate``.
    ///   - melSpectrogram: A mel spectrogram instance owned by the caller.
    /// - Returns: A session that accepts audio via ``LSEENDStreamingSession/pushAudio(_:)``.
    public func createSession(
        inputSampleRate: Int, melSpectrogram: AudioMelSpectrogram
    ) throws -> LSEENDStreamingSession {
        try LSEENDStreamingSession(engine: self, inputSampleRate: inputSampleRate, melSpectrogram: melSpectrogram)
    }

    /// Runs offline inference on raw audio samples.
    ///
    /// Resamples to ``targetSampleRate`` if needed, extracts features, runs the full
    /// model, and returns speaker probabilities for every frame.
    ///
    /// - Parameters:
    ///   - samples: Mono audio samples.
    ///   - sampleRate: Sample rate of the input audio.
    /// - Returns: Complete inference result with logits and probabilities.
    public func infer(samples: [Float], sampleRate: Int) throws -> LSEENDInferenceResult {
        let normalizedAudio = try resampleIfNeeded(samples: samples, sampleRate: sampleRate)
        let session = try createSession(inputSampleRate: targetSampleRate)
        if !normalizedAudio.isEmpty {
            _ = try session.pushAudio(normalizedAudio)
        }
        _ = try session.finalize()
        return session.snapshot()
    }

    /// Runs offline inference on an audio file.
    ///
    /// Reads the file, resamples to ``targetSampleRate``, and runs full inference.
    ///
    /// - Parameter audioFileURL: Path to a WAV, CAF, or other audio file.
    /// - Returns: Complete inference result with logits and probabilities.
    public func infer(audioFileURL: URL) throws -> LSEENDInferenceResult {
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        let audio = try converter.resampleAudioFile(audioFileURL)
        return try infer(samples: audio, sampleRate: targetSampleRate)
    }

    /// Simulates streaming inference by processing an audio file in fixed-size chunks.
    ///
    /// Useful for testing and benchmarking the streaming pipeline against offline results.
    ///
    /// - Parameters:
    ///   - audioFileURL: Path to the audio file.
    ///   - chunkSeconds: Duration of each simulated audio chunk in seconds.
    /// - Returns: The final inference result along with per-chunk progress entries.
    public func simulateStreaming(audioFileURL: URL, chunkSeconds: Double) throws -> LSEENDStreamingSimulationResult {
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        let audio = try converter.resampleAudioFile(audioFileURL)
        let chunkSize = max(1, Int(round(chunkSeconds * Double(targetSampleRate))))
        let session = try createSession(inputSampleRate: targetSampleRate)
        var updates: [LSEENDStreamingProgress] = []
        var chunkIndex = 1
        var start = 0
        while start < audio.count {
            let stop = min(audio.count, start + chunkSize)
            let update = try session.pushAudio(Array(audio[start..<stop]))
            updates.append(
                LSEENDStreamingProgress(
                    chunkIndex: chunkIndex,
                    bufferSeconds: roundedMillis(Double(stop) / Double(targetSampleRate)),
                    numFramesEmitted: update?.probabilities.rows ?? 0,
                    totalFramesEmitted: session.emittedFrames,
                    flush: false
                )
            )
            start = stop
            chunkIndex += 1
        }
        let finalUpdate = try session.finalize()
        if let finalUpdate {
            updates.append(
                LSEENDStreamingProgress(
                    chunkIndex: chunkIndex,
                    bufferSeconds: roundedMillis(Double(audio.count) / Double(targetSampleRate)),
                    numFramesEmitted: finalUpdate.probabilities.rows,
                    totalFramesEmitted: session.emittedFrames,
                    flush: true
                )
            )
        }
        return LSEENDStreamingSimulationResult(result: session.snapshot(), updates: updates)
    }

    fileprivate func predictStep(
        frame: [Float],
        state: LSEENDModelState,
        ingest: Float,
        decode: Float
    ) throws -> LSEENDStepOutput {
        try lock.withLock {
            // Write into preallocated ANE-aligned arrays instead of allocating new ones
            sharedResources.memoryOptimizer.optimizedCopy(from: frame, to: sharedResources.frameArray)
            sharedResources.ingestArray[0] = NSNumber(value: ingest)
            sharedResources.decodeArray[0] = NSNumber(value: decode)

            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "frame": MLFeatureValue(multiArray: sharedResources.frameArray),
                "enc_ret_kv": MLFeatureValue(multiArray: state.encRetKv),
                "enc_ret_scale": MLFeatureValue(multiArray: state.encRetScale),
                "enc_conv_cache": MLFeatureValue(multiArray: state.encConvCache),
                "dec_ret_kv": MLFeatureValue(multiArray: state.decRetKv),
                "dec_ret_scale": MLFeatureValue(multiArray: state.decRetScale),
                "top_buffer": MLFeatureValue(multiArray: state.topBuffer),
                "ingest": MLFeatureValue(multiArray: sharedResources.ingestArray),
                "decode": MLFeatureValue(multiArray: sharedResources.decodeArray),
            ])
            let prediction = try model.prediction(from: provider)
            let fullLogitsArray = try feature(named: "full_logits", from: prediction)
            let nextState = LSEENDModelState(
                encRetKv: try cloneAligned(feature(named: "enc_ret_kv_out", from: prediction)),
                encRetScale: try cloneAligned(feature(named: "enc_ret_scale_out", from: prediction)),
                encConvCache: try cloneAligned(feature(named: "enc_conv_cache_out", from: prediction)),
                decRetKv: try cloneAligned(feature(named: "dec_ret_kv_out", from: prediction)),
                decRetScale: try cloneAligned(feature(named: "dec_ret_scale_out", from: prediction)),
                topBuffer: try cloneAligned(feature(named: "top_buffer_out", from: prediction))
            )
            return LSEENDStepOutput(
                fullLogits: floatValues(from: fullLogitsArray, count: metadata.fullOutputDim),
                nextState: nextState
            )
        }
    }

    fileprivate func initialState() throws -> LSEENDModelState {
        let optimizer = sharedResources.memoryOptimizer

        return try LSEENDModelState(
            encRetKv: optimizer.createAlignedArray(
                shape: metadata.stateShapes.encRetKv.map(NSNumber.init(value:)), dataType: .float32),
            encRetScale: optimizer.createAlignedArray(
                shape: metadata.stateShapes.encRetScale.map(NSNumber.init(value:)), dataType: .float32),
            encConvCache: optimizer.createAlignedArray(
                shape: metadata.stateShapes.encConvCache.map(NSNumber.init(value:)), dataType: .float32),
            decRetKv: optimizer.createAlignedArray(
                shape: metadata.stateShapes.decRetKv.map(NSNumber.init(value:)), dataType: .float32),
            decRetScale: optimizer.createAlignedArray(
                shape: metadata.stateShapes.decRetScale.map(NSNumber.init(value:)), dataType: .float32),
            topBuffer: optimizer.createAlignedArray(
                shape: metadata.stateShapes.topBuffer.map(NSNumber.init(value:)), dataType: .float32)
        )
    }

    private func resampleIfNeeded(samples: [Float], sampleRate: Int) throws -> [Float] {
        guard sampleRate > 0 else {
            throw LSEENDError.unsupportedAudio("Invalid sample rate \(sampleRate).")
        }
        guard sampleRate != targetSampleRate else {
            return samples
        }
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        return try converter.resample(samples, from: Double(sampleRate))
    }

    fileprivate static func compiledModelURL(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }
        let caches =
            FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let compiledRoot = caches.appendingPathComponent("LS-EENDCompiledModels", isDirectory: true)
        try FileManager.default.createDirectory(at: compiledRoot, withIntermediateDirectories: true)
        let fingerprint = try cacheFingerprint(for: modelURL)
        let compiledName =
            modelURL.deletingPathExtension().lastPathComponent
            + "-"
            + fingerprint
            + ".mlmodelc"
        let destination = compiledRoot.appendingPathComponent(compiledName, isDirectory: true)
        if FileManager.default.fileExists(atPath: destination.path) {
            return destination
        }
        let compiled = try MLModel.compileModel(at: modelURL)
        try FileManager.default.copyItem(at: compiled, to: destination)
        return destination
    }

    private func feature(named name: String, from provider: MLFeatureProvider) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw LSEENDError.missingFeature(name)
        }
        return value
    }

    /// Clone an MLMultiArray into a new ANE-aligned allocation using stride-aware copy.
    private func cloneAligned(_ source: MLMultiArray) throws -> MLMultiArray {
        let copy = try sharedResources.memoryOptimizer.createAlignedArray(
            shape: source.shape,
            dataType: .float32
        )
        ANEMemoryUtils.strideAwareCopy(from: source, to: copy)
        return copy
    }

    private static func cacheFingerprint(for url: URL) throws -> String {
        let fileManager = FileManager.default
        let standardizedURL = url.standardizedFileURL
        var hasher = SHA256()

        func update(for fileURL: URL, relativePath: String) throws {
            let attributes = try fileManager.attributesOfItem(atPath: fileURL.path)
            let fileType = attributes[.type] as? FileAttributeType ?? .typeUnknown
            let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
            let modification = (attributes[.modificationDate] as? Date)?.timeIntervalSinceReferenceDate ?? 0
            let record = "\(relativePath)|\(fileType.rawValue)|\(size)|\(modification)\n"
            hasher.update(data: Data(record.utf8))
        }

        let rootAttributes = try fileManager.attributesOfItem(atPath: standardizedURL.path)
        let rootType = rootAttributes[.type] as? FileAttributeType ?? .typeUnknown

        if rootType == .typeDirectory {
            let enumerator = fileManager.enumerator(
                at: standardizedURL,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
            var paths: [String] = []
            while let childURL = enumerator?.nextObject() as? URL {
                let relativePath = childURL.path.replacingOccurrences(of: standardizedURL.path + "/", with: "")
                paths.append(relativePath)
            }
            for relativePath in paths.sorted() {
                try update(for: standardizedURL.appendingPathComponent(relativePath), relativePath: relativePath)
            }
        } else {
            try update(for: standardizedURL, relativePath: standardizedURL.lastPathComponent)
        }

        let digest = hasher.finalize()
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}

/// A stateful streaming session that incrementally processes audio and emits diarization frames.
///
/// Created via ``LSEENDInferenceEngine/createSession(inputSampleRate:)``.
/// The session maintains internal RNN state across calls to ``pushAudio(_:)``.
///
/// - Important: This class is **not** thread-safe. All calls must be serialized externally.
public final class LSEENDStreamingSession {
    fileprivate let engine: LSEENDInferenceHelper
    /// The sample rate of audio being fed to this session.
    public let inputSampleRate: Int
    fileprivate let featureExtractor: LSEENDStreamingFeatureExtractor
    fileprivate var state: LSEENDModelState
    fileprivate let zeroFrame: [Float]
    fileprivate var fullLogitChunks: [LSEENDMatrix] = []
    fileprivate var finalized = false

    fileprivate var totalInputSamples: Int = 0
    fileprivate var totalFeatureFrames = 0
    fileprivate var emittedFrames = 0

    fileprivate init(
        engine: LSEENDInferenceHelper, inputSampleRate: Int, melSpectrogram: AudioMelSpectrogram? = nil
    ) throws {
        guard inputSampleRate == engine.targetSampleRate else {
            throw LSEENDError.unsupportedAudio(
                "Stateful LS-EEND streaming expects \(engine.targetSampleRate) Hz audio, received \(inputSampleRate) Hz."
            )
        }
        self.engine = engine
        self.inputSampleRate = inputSampleRate
        featureExtractor = LSEENDStreamingFeatureExtractor(
            metadata: engine.metadata, spectrogram: melSpectrogram ?? engine.melSpectrogram)
        state = try engine.initialState()
        zeroFrame = [Float](repeating: 0, count: engine.metadata.inputDim)
    }

    /// Feeds audio samples into the session and returns any newly committed frames.
    ///
    /// The returned update contains both committed (final) frames and a speculative preview
    /// of pending frames decoded by zero-padding the remaining state.
    ///
    /// - Parameter chunk: Mono audio samples at ``inputSampleRate``.
    /// - Returns: An update with new committed and preview frames, or `nil` if no frames were produced.
    /// - Throws: ``LSEENDError/unsupportedAudio(_:)`` if the session has already been finalized.
    public func pushAudio(_ chunk: [Float]) throws -> LSEENDStreamingUpdate? {
        guard !finalized else {
            throw LSEENDError.unsupportedAudio("Streaming session already finalized.")
        }
        guard !chunk.isEmpty else {
            return nil
        }
        totalInputSamples += chunk.count
        let features = try featureExtractor.pushAudio(chunk)
        let committed = try ingestFeatures(features)
        return try buildUpdate(committedFullLogits: committed, includePreview: true)
    }

    /// Flushes remaining buffered features and marks the session as complete.
    ///
    /// After finalization, ``pushAudio(_:)`` will throw. Calling `finalize()` again returns `nil`.
    ///
    /// - Returns: A final update with any remaining frames, or `nil` if no frames were pending.
    public func finalize() throws -> LSEENDStreamingUpdate? {
        guard !finalized else {
            return nil
        }

        var committedFullLogits = LSEENDMatrix.empty(columns: engine.decodeMaxSpeakers)
        let targetEndFrame = Int(
            round(Double(totalInputSamples) / Double(max(inputSampleRate, 1)) * engine.modelFrameHz))
        let exactPaddingSamples = try exactFinalizationPaddingSamples(targetEndFrame: targetEndFrame)
        if exactPaddingSamples > 0 {
            let features = try featureExtractor.pushAudio([Float](repeating: 0, count: exactPaddingSamples))
            let committed = try ingestFeatures(features)
            if committed.rows > 0 {
                committedFullLogits = committedFullLogits.appendingRows(committed)
            }
        }

        let finalFeatures = try featureExtractor.finalize()
        let finalCommitted = try ingestFeatures(finalFeatures)
        if finalCommitted.rows > 0 {
            committedFullLogits = committedFullLogits.appendingRows(finalCommitted)
        }

        let pending = totalFeatureFrames - emittedFrames
        let tail =
            try pending > 0 ? flushTail(from: state, pendingFrames: pending) : .empty(columns: engine.decodeMaxSpeakers)
        emittedFrames += tail.rows
        finalized = true
        return try buildUpdate(committedFullLogits: committedFullLogits.appendingRows(tail), includePreview: false)
    }

    private func exactFinalizationPaddingSamples(targetEndFrame: Int) throws -> Int {
        guard targetEndFrame > 0 else {
            return 0
        }
        let stableBlockSize = engine.metadata.resolvedHopLength * engine.metadata.resolvedSubsampling
        let (requiredTotalSamples, overflow) = targetEndFrame.multipliedReportingOverflow(by: stableBlockSize)
        guard !overflow else {
            throw LSEENDError.unsupportedAudio(
                "Finalization padding overflowed for \(targetEndFrame) frames at block size \(stableBlockSize)."
            )
        }
        return max(0, requiredTotalSamples - totalInputSamples)
    }

    /// Assembles the full inference result from all committed frames emitted so far.
    ///
    /// Can be called at any time (before or after finalization) to get a complete
    /// ``LSEENDInferenceResult`` covering all frames produced up to this point.
    public func snapshot() -> LSEENDInferenceResult {
        let fullLogits = fullLogitChunks.reduce(LSEENDMatrix.empty(columns: engine.decodeMaxSpeakers)) {
            partial, matrix in
            partial.appendingRows(matrix)
        }
        let fullProbabilities = fullLogits.applyingSigmoid()
        let logits = cropRealTracks(from: fullLogits)
        let probabilities = cropRealTracks(from: fullProbabilities)
        return LSEENDInferenceResult(
            logits: logits,
            probabilities: probabilities,
            fullLogits: fullLogits,
            fullProbabilities: fullProbabilities,
            frameHz: engine.modelFrameHz,
            durationSeconds: Double(totalInputSamples) / Double(max(inputSampleRate, 1))
        )
    }

    fileprivate func ingestFeatures(_ features: LSEENDMatrix) throws -> LSEENDMatrix {
        guard features.rows > 0 else {
            return .empty(columns: engine.decodeMaxSpeakers)
        }
        var output: [Float] = []
        for rowIndex in 0..<features.rows {
            let decode = totalFeatureFrames >= engine.metadata.convDelay ? Float(1) : Float(0)
            let step = try autoreleasepool {
                try engine.predictStep(
                    frame: Array(features.row(rowIndex)),
                    state: state,
                    ingest: 1,
                    decode: decode
                )
            }
            state = step.nextState
            totalFeatureFrames += 1
            if decode > 0 {
                output.append(contentsOf: step.fullLogits)
                emittedFrames += 1
            }
        }
        return LSEENDMatrix(
            validatingRows: output.isEmpty ? 0 : output.count / engine.decodeMaxSpeakers,
            columns: engine.decodeMaxSpeakers,
            values: output
        )
    }

    fileprivate func flushTail(from state: LSEENDModelState, pendingFrames: Int) throws -> LSEENDMatrix {
        guard pendingFrames > 0 else {
            return .empty(columns: engine.decodeMaxSpeakers)
        }
        var previewState = state
        var output: [Float] = []
        for _ in 0..<pendingFrames {
            let step = try autoreleasepool {
                try engine.predictStep(
                    frame: zeroFrame,
                    state: previewState,
                    ingest: 0,
                    decode: 1
                )
            }
            previewState = step.nextState
            output.append(contentsOf: step.fullLogits)
        }
        return LSEENDMatrix(
            validatingRows: output.count / engine.decodeMaxSpeakers,
            columns: engine.decodeMaxSpeakers,
            values: output
        )
    }

    private func buildUpdate(
        committedFullLogits: LSEENDMatrix,
        includePreview: Bool
    ) throws -> LSEENDStreamingUpdate? {
        let startFrame = emittedFrames - committedFullLogits.rows
        let committedProbabilities = committedFullLogits.applyingSigmoid()
        let committedLogits = cropRealTracks(from: committedFullLogits)
        let committedRealProbabilities = cropRealTracks(from: committedProbabilities)

        if committedFullLogits.rows > 0 {
            fullLogitChunks.append(committedFullLogits)
        }

        let previewFullLogits: LSEENDMatrix
        if includePreview {
            let previewState = try state.copy()
            let pending = totalFeatureFrames - emittedFrames
            previewFullLogits = try flushTail(from: previewState, pendingFrames: pending)
        } else {
            previewFullLogits = .empty(columns: engine.decodeMaxSpeakers)
        }
        let previewLogits = cropRealTracks(from: previewFullLogits)
        let previewProbabilities = cropRealTracks(from: previewFullLogits.applyingSigmoid())

        if committedLogits.isEmpty && previewLogits.isEmpty {
            return nil
        }

        return LSEENDStreamingUpdate(
            startFrame: startFrame,
            logits: committedLogits,
            probabilities: committedRealProbabilities,
            previewStartFrame: emittedFrames,
            previewLogits: previewLogits,
            previewProbabilities: previewProbabilities,
            frameHz: engine.modelFrameHz,
            durationSeconds: Double(totalInputSamples) / Double(max(inputSampleRate, 1)),
            totalEmittedFrames: emittedFrames
        )
    }

    private func cropRealTracks(from matrix: LSEENDMatrix) -> LSEENDMatrix {
        guard matrix.columns > 2 else {
            return .empty(columns: 0)
        }
        let realColumns = matrix.columns - 2
        var output = [Float](repeating: 0, count: matrix.rows * realColumns)
        for rowIndex in 0..<matrix.rows {
            let sourceBase = rowIndex * matrix.columns + 1
            let destinationBase = rowIndex * realColumns
            for columnIndex in 0..<realColumns {
                output[destinationBase + columnIndex] = matrix.values[sourceBase + columnIndex]
            }
        }
        return LSEENDMatrix(validatingRows: matrix.rows, columns: realColumns, values: output)
    }
}

private func roundedMillis(_ value: Double) -> Double {
    (value * 1000).rounded() / 1000
}

/// Clone an MLMultiArray into a new ANE-aligned allocation.
private func cloneAlignedMultiArray(_ source: MLMultiArray) throws -> MLMultiArray {
    let copy = try ANEMemoryUtils.createAlignedArray(
        shape: source.shape,
        dataType: .float32,
        zeroClear: false
    )
    ANEMemoryUtils.strideAwareCopy(from: source, to: copy)
    return copy
}

private func floatValues(from array: MLMultiArray, count: Int) -> [Float] {
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: max(count, array.count))
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}
