import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Streaming encoder configuration for different chunk sizes.
///
/// The Parakeet EOU model (`nvidia/parakeet_realtime_eou_120m-v1`) supports 160ms, 320ms, and 1600ms
/// streaming chunk sizes. Each requires a separately exported CoreML encoder model with the
/// correct NeMo streaming configuration baked in.
///
/// **160ms (default)**: Uses `chunk_size=[9, 16]`, `valid_out_len=2`
/// **320ms**: Uses `setup_streaming_params(chunk_size=8, shift_size=4)` → `chunk_size=[57, 64]`, `valid_out_len=4`
/// **1600ms**: Uses `setup_streaming_params(chunk_size=40, shift_size=20)` → `chunk_size=[313, 320]`, `valid_out_len=20`
///
/// Larger chunk sizes provide better throughput but higher latency.
public enum StreamingChunkSize: Sendable {
    /// 160ms chunk (16 audio frames → 17 mel frames)
    /// Encoder steps: 4, Shift: 2 (80ms overlap)
    /// Default configuration, well-tested with ~8-9% WER on LibriSpeech test-clean.
    case ms160

    /// 320ms mode - higher throughput, 320ms latency between outputs
    ///
    /// NeMo config: `encoder.setup_streaming_params(chunk_size=8, shift_size=4)`
    /// This reconfigures the encoder's internal streaming state:
    /// - `chunk_size: [57, 64]` mel frames → use 64 (larger input for more context)
    /// - `shift_size: [25, 32]` mel frames → use 32 (320ms = 32 * 10ms per frame)
    /// - `pre_encode_cache_size: [0, 9]` → use 9 (smaller than 160ms because chunk has more self-contained context)
    /// - `valid_out_len: 4` (produces 4 encoder output frames per chunk)
    ///
    /// Why bigger chunks need smaller pre_cache:
    /// - 160ms: 17 mel frames input, needs 16 frames of lookahead context
    /// - 320ms: 64 mel frames input, already contains more context, only needs 9 frames
    ///
    /// Performance: ~5.73% WER, 14x RTFx on LibriSpeech test-clean
    case ms320

    /// 1600ms mode (latency 1600ms, chunk ~3200ms audio → 320 mel frames)
    /// Encoder steps: 40, Shift: 20 (1600ms latency)
    /// Requires separately exported 1600ms model with NeMo's `setup_streaming_params(chunk_size=40)`.
    /// Note: The "1600ms" refers to the latency (shift), not the chunk duration.
    /// - Chunk: 320 mel frames (~3183ms of audio)
    /// - Shift: 160 mel frames (1600ms latency between outputs)
    /// - Output: 20 valid encoder frames per chunk
    case ms1600

    /// Number of audio samples per chunk
    /// Calculated from mel frames: (mel_frames - 1) * hop_length for center-padded mel spectrogram
    public var chunkSamples: Int {
        switch self {
        case .ms160: return 2560  // (17-1) * 160 = 2560 samples (160ms)
        case .ms320:
            // 320ms mode: 64 mel frames from NeMo's streaming_cfg.chunk_size[1]
            // Formula: (mel_frames - 1) * hop_length = (64-1) * 160 = 10080 samples
            // This is ~630ms of audio per chunk (but 320ms latency due to shift)
            return 10080
        case .ms1600: return 50928  // ~3183ms: (320-1)*160 + 400 - 512 = 50928 samples
        }
    }

    /// Number of mel spectrogram frames (from NeMo's chunk_size config)
    /// For 160ms: 17 mel frames → 2 valid encoder outputs
    /// For 320ms: 64 mel frames → 4 valid encoder outputs
    /// For 1600ms: 320 mel frames → 20 valid encoder outputs
    public var melFrames: Int {
        switch self {
        case .ms160: return 17
        case .ms320:
            // 320ms: NeMo's streaming_cfg.chunk_size = [57, 64] after setup_streaming_params(8, 4)
            // Use index [1] = 64 mel frames (the larger/padded size)
            return 64
        case .ms1600: return 320  // From NeMo cfg.chunk_size[1]
        }
    }

    /// Chunk duration in milliseconds
    public var durationMs: Int {
        switch self {
        case .ms160: return 160
        case .ms320: return 630  // 10080 samples / 16 = 630ms audio
        case .ms1600: return 3183  // ~3.2s of audio per chunk (latency is 1600ms)
        }
    }

    /// Default model subdirectory name
    public var modelSubdirectory: String {
        switch self {
        case .ms160: return "160ms"
        case .ms320: return "320ms"
        case .ms1600: return "1600ms"
        }
    }

    /// Number of valid encoder output frames per chunk
    public var validOutputLen: Int {
        switch self {
        case .ms160: return 2
        case .ms320: return 4
        case .ms1600: return 20
        }
    }

    /// Pre-cache size in mel frames (for mel-level context in loopback encoder)
    /// This is from NeMo's pre_encode_cache_size configuration.
    public var preCacheSize: Int {
        switch self {
        case .ms160: return 16  // Legacy value for 160ms
        case .ms320:
            // 320ms: NeMo's streaming_cfg.pre_encode_cache_size = [0, 9]
            // Smaller than 160ms (16) because 64 mel frames already contain more context
            // Bigger chunks are more self-contained, need less external lookahead
            return 9
        case .ms1600: return 9  // From NeMo cfg.pre_encode_cache_size[1]
        }
    }

    /// Audio samples to shift between chunks.
    ///
    /// For the loopback encoder architecture:
    /// - The encoder receives `melFrames` mel frames as input
    /// - Pre_cache is concatenated internally, providing mel-level context
    /// - The shift should match NeMo's shift_size in mel frames
    ///
    /// NeMo shift_size (mel frames) * hopLength = audio samples shift
    /// - 160ms: shift_size=16 mel frames → 16*160 = 2560 samples (but use 1280 for 50% overlap)
    /// - 320ms: shift_size=32 mel frames → 32*160 = 5120 samples (but use custom)
    /// - 1600ms: shift_size=160 mel frames → 160*160 = 25600 samples (1600ms latency)
    ///
    /// For 160ms, we use 50% overlap (1280 samples) because the model was trained that way.
    /// For 1600ms, use NeMo's shift_size directly (160 mel frames = 1600ms).
    public var shiftSamples: Int {
        switch self {
        case .ms160:
            // 160ms uses 50% audio overlap (matches NeMo's default behavior)
            return chunkSamples / 2  // 1280 samples
        case .ms320:
            // 320ms: NeMo's streaming_cfg.shift_size = [25, 32] mel frames
            // Use index [1] = 32 mel frames × 10ms/frame = 320ms latency
            // shift_samples = 32 * hop_length = 32 * 160 = 5120 samples
            return 5120
        case .ms1600:
            // NeMo shift_size=160 mel frames (1600ms latency)
            return 160 * 160  // 25600 samples
        }
    }
}

/// Callback invoked when End-of-Utterance is detected during streaming.
/// - Parameter transcript: The accumulated transcript up to the EOU point
public typealias EouCallback = @Sendable (String) -> Void

/// Callback invoked when new tokens are decoded (for ghost text).
/// - Parameter transcript: The current accumulated partial transcript
public typealias PartialCallback = @Sendable (String) -> Void

/// High-level manager for the Parakeet EOU streaming pipeline.
/// Uses native Swift mel spectrogram for exact NeMo parity.
public actor StreamingEouAsrManager {
    private let logger = AppLogger(category: "StreamingEOU")

    private var processedChunks = 0

    // Models
    private var streamingEncoder: MLModel?  // Single Loopback Model
    private var decoder: MLModel?
    private var joint: MLModel?

    // Components
    private var rnntDecoder: RnntDecoder?
    private let audioConverter = AudioConverter()
    private var tokenizer: Tokenizer?
    private let melProcessor = NeMoMelSpectrogram()  // Native Swift mel spectrogram

    // Configuration - now based on chunkSize
    public let chunkSize: StreamingChunkSize
    private let hopLength = 160
    private var chunkSamples: Int { chunkSize.chunkSamples }
    // Shift based on valid encoder output frames (see StreamingChunkSize.shiftSamples)
    private var shiftSamples: Int { chunkSize.shiftSamples }

    // Audio Buffer
    private var audioBuffer: [Float] = []

    // Accumulated token IDs from incremental decoding (NeMo-style)
    private var accumulatedTokenIds: [Int] = []

    // EOU Detection
    /// Whether End-of-Utterance was detected in the last chunk processed
    public private(set) var eouDetected: Bool = false
    /// Optional callback invoked when EOU is detected
    private var eouCallback: EouCallback?
    /// Optional callback invoked after each chunk with partial transcript
    private var partialCallback: PartialCallback?

    // EOU Debouncing - requires sustained silence before triggering
    /// Minimum duration of silence (in ms) before EOU is confirmed
    public var eouDebounceMs: Int = 1280
    /// Timestamp when EOU was first detected (for debouncing)
    private var eouFirstDetectedAt: Int?  // in processed samples
    /// Total samples processed (for timing)
    private var totalSamplesProcessed: Int = 0

    public private(set) var configuration: MLModelConfiguration
    public let debugFeatures: Bool
    private var debugFeatureBuffer: [Float] = []

    // --- Loopback States ---
    // 1. Pre-Cache (Audio Context) [1, 128, 16]
    private var preCache: MLMultiArray?

    // 2. Conformer Caches
    // cache_last_channel: [17, 1, 70, 512]
    // cache_last_time: [17, 1, 512, 8]
    // cache_last_channel_len: [1]
    private var cacheLastChannel: MLMultiArray?
    private var cacheLastTime: MLMultiArray?
    private var cacheLastChannelLen: MLMultiArray?

    public init(
        configuration: MLModelConfiguration = MLModelConfiguration(),
        chunkSize: StreamingChunkSize = .ms160,
        eouDebounceMs: Int = 1280,
        debugFeatures: Bool = false
    ) {
        self.configuration = configuration
        self.chunkSize = chunkSize
        self.eouDebounceMs = eouDebounceMs
        self.debugFeatures = debugFeatures
        logger.info("Initialized with chunk size: \(chunkSize.durationMs)ms, EOU debounce: \(eouDebounceMs)ms")
    }

    /// Set a callback to be invoked when End-of-Utterance is detected.
    /// The callback receives the transcript accumulated up to the EOU point.
    public func setEouCallback(_ callback: @escaping EouCallback) {
        self.eouCallback = callback
    }

    /// Set a callback to be invoked when new tokens are decoded.
    /// Useful for displaying "ghost text" during speech.
    public func setPartialCallback(_ callback: @escaping PartialCallback) {
        self.partialCallback = callback
    }

    public func loadModels(modelDir: URL) async throws {
        logger.info("Loading CoreML models from \(modelDir.path)...")

        // No longer loading preprocessor - using native Swift NeMoMelSpectrogram instead
        self.streamingEncoder = try await MLModel.load(
            contentsOf: modelDir.appendingPathComponent("streaming_encoder.mlmodelc"), configuration: self.configuration
        )
        self.decoder = try await MLModel.load(
            contentsOf: modelDir.appendingPathComponent("decoder.mlmodelc"), configuration: self.configuration)
        self.joint = try await MLModel.load(
            contentsOf: modelDir.appendingPathComponent("joint_decision.mlmodelc"), configuration: self.configuration)

        // Load Tokenizer
        let vocabUrl = modelDir.appendingPathComponent("vocab.json")
        self.tokenizer = try Tokenizer(vocabPath: vocabUrl)

        self.rnntDecoder = RnntDecoder(decoderModel: self.decoder!, jointModel: self.joint!)

        // Initialize States
        try self.resetStates()

        self.audioBuffer.removeAll()

        logger.info("Models loaded successfully.")
    }

    private func resetStates() throws {
        // Initialize with Zeros
        // pre_cache: [1, 128, preCacheSize] - size varies by chunk size
        let preCacheSize = chunkSize.preCacheSize
        self.preCache = try MLMultiArray(
            shape: [1, 128, NSNumber(value: preCacheSize)], dataType: .float32)
        self.preCache?.reset(to: 0)

        // cache_last_channel: [17, 1, 70, 512]
        self.cacheLastChannel = try MLMultiArray(shape: [17, 1, 70, 512], dataType: .float32)
        self.cacheLastChannel?.reset(to: 0)

        // cache_last_time: [17, 1, 512, 8]
        self.cacheLastTime = try MLMultiArray(shape: [17, 1, 512, 8], dataType: .float32)
        self.cacheLastTime?.reset(to: 0)

        // cache_last_channel_len: [1]
        self.cacheLastChannelLen = try MLMultiArray(shape: [1], dataType: .int32)
        self.cacheLastChannelLen?.reset(to: 0)
    }

    /// Append audio to buffer without processing (for Simulated Streaming and VAD)
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        let samples = try audioConverter.resampleBuffer(buffer)
        self.audioBuffer.append(contentsOf: samples)
    }

    public func process(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // 1. Convert to 16kHz Mono Float32
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        self.audioBuffer.append(contentsOf: samples)

        // 2. Process chunks with 50% overlap (NeMo-style)
        // We accumulate encoder outputs and decode at the end
        while self.audioBuffer.count >= chunkSamples {
            // Extract chunk (160ms)
            let chunk = Array(self.audioBuffer.prefix(chunkSamples))

            // 3. Run encoder and decode incrementally (NeMo-style)
            try await processChunkAndDecode(chunk)

            // 4. Shift buffer by 80ms (50% overlap)
            self.audioBuffer.removeFirst(shiftSamples)
        }

        // Return empty - actual transcription happens in finish()
        return ""
    }

    public func finish() async throws -> String {
        // 1. Process remaining audio (padded) if any
        if !audioBuffer.isEmpty {
            let remaining = audioBuffer.count
            let paddingNeeded = chunkSamples - remaining

            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            // Process final chunk with decoding
            let chunk = Array(audioBuffer.prefix(chunkSamples))
            try await processChunkAndDecode(chunk)

            // Clear buffer
            audioBuffer.removeAll()
        }

        // 2. Return accumulated transcript from incremental decoding
        guard let tokenizer = tokenizer else {
            return ""
        }

        let transcript = tokenizer.decode(ids: accumulatedTokenIds)

        // Clear accumulated tokens
        accumulatedTokenIds.removeAll()

        return transcript
    }

    public func reset() async {
        audioBuffer.removeAll()
        debugFeatureBuffer.removeAll()
        accumulatedTokenIds.removeAll()
        eouDetected = false
        eouFirstDetectedAt = nil
        totalSamplesProcessed = 0
        try? resetStates()
        rnntDecoder?.resetState()
        processedChunks = 0
    }

    public func injectSilence(_ seconds: Double) {
        let silenceSamples = Int(seconds * 16000)
        audioBuffer.append(contentsOf: Array(repeating: 0.0, count: silenceSamples))
    }

    /// Process a chunk through native mel spectrogram, encoder, and RNNT decoder (NeMo-style incremental)
    private func processChunkAndDecode(_ samples: [Float]) async throws {
        guard let streamingEncoder = streamingEncoder,
            let preCache = preCache,
            let cacheLastChannel = cacheLastChannel,
            let cacheLastTime = cacheLastTime,
            let cacheLastChannelLen = cacheLastChannelLen,
            let rnntDecoder = rnntDecoder
        else {
            throw ASRError.notInitialized
        }

        // A. Compute mel spectrogram with native Swift implementation (NeMo-matching)
        let (melFlat, melLength, numFrames) = melProcessor.computeFlat(audio: samples)

        // Create MLMultiArray for mel: [1, 128, numFrames]
        let mel = try MLMultiArray(shape: [1, 128, NSNumber(value: numFrames)], dataType: .float32)
        let melPtr = mel.dataPointer.bindMemory(to: Float.self, capacity: mel.count)

        // NeMoMelSpectrogram returns [nMels, T] row-major (mel bin, then time)
        // CoreML expects [1, 128, T] which is the same layout
        melPtr.assign(from: melFlat, count: melFlat.count)

        // Create mel_length: [1] with valid frame count
        let melLen = try MLMultiArray(shape: [1], dataType: .int32)
        melLen[0] = NSNumber(value: melLength)

        if debugFeatures {
            debugFeatureBuffer.append(contentsOf: melFlat)
        }

        // B. Streaming Encoder
        let encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: mel),
            "audio_length": MLFeatureValue(multiArray: melLen),
            "pre_cache": MLFeatureValue(multiArray: preCache),
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen),
        ])

        let encoderOutput = try await streamingEncoder.prediction(from: encoderInput)

        // C. Update States (Loopback)
        if let newPreCache = encoderOutput.featureValue(for: "new_pre_cache")?.multiArrayValue {
            self.preCache = newPreCache
        }
        if let newChannel = encoderOutput.featureValue(for: "new_cache_last_channel")?.multiArrayValue {
            self.cacheLastChannel = newChannel
        }
        if let newTime = encoderOutput.featureValue(for: "new_cache_last_time")?.multiArrayValue {
            self.cacheLastTime = newTime
        }
        if let newChannelLen = encoderOutput.featureValue(for: "new_cache_last_channel_len")?.multiArrayValue {
            self.cacheLastChannelLen = newChannelLen
        }

        // D. Decode this chunk's encoder output incrementally (NeMo-style)
        guard let encoded = encoderOutput.featureValue(for: "encoded_output")?.multiArrayValue else {
            throw ASRError.processingFailed("Missing encoder output")
        }

        // Decode this chunk - RNNT decoder state (h, c, lastToken) carries across chunks
        // NeMo truncates encoder output to valid_out_len frames before decoding
        let decodeResult = try rnntDecoder.decodeWithEOU(
            encoderOutput: encoded, timeOffset: processedChunks, skipFrames: 0,
            validOutLen: chunkSize.validOutputLen)
        accumulatedTokenIds.append(contentsOf: decodeResult.tokenIds)

        // Invoke partial callback for ghost text (only when new tokens decoded)
        if let callback = partialCallback, let tokenizer = tokenizer, !decodeResult.tokenIds.isEmpty {
            let partial = tokenizer.decode(ids: accumulatedTokenIds)
            callback(partial)
        }

        // Track total samples for timing
        totalSamplesProcessed += shiftSamples

        // Handle EOU detection with debouncing
        // EOU requires sustained silence for eouDebounceMs before triggering
        if decodeResult.eouDetected {
            // If new tokens were produced, speech is ongoing - reset debounce timer
            if !decodeResult.tokenIds.isEmpty {
                eouFirstDetectedAt = nil
            } else if eouFirstDetectedAt == nil {
                // First EOU detection - start debounce timer
                eouFirstDetectedAt = totalSamplesProcessed
                logger.debug("EOU candidate at chunk \(processedChunks), starting debounce timer")
            }

            // Check if debounce period has elapsed
            if let firstDetected = eouFirstDetectedAt {
                let elapsedSamples = totalSamplesProcessed - firstDetected
                let elapsedMs = (elapsedSamples * 1000) / 16000  // Convert samples to ms at 16kHz

                if elapsedMs >= eouDebounceMs && !eouDetected {
                    eouDetected = true
                    logger.info("EOU confirmed at chunk \(processedChunks) after \(elapsedMs)ms silence")

                    // Invoke callback with current transcript
                    if let callback = eouCallback, let tokenizer = tokenizer {
                        let transcript = tokenizer.decode(ids: accumulatedTokenIds)
                        callback(transcript)
                    }
                }
            }
        } else {
            // Model did not predict EOU - speech is ongoing, reset debounce timer
            eouFirstDetectedAt = nil
        }

        processedChunks += 1
    }

    public func saveDebugFeatures(to url: URL) throws {
        let outputData: [String: Any] = [
            "mel_features": debugFeatureBuffer,
            "count": debugFeatureBuffer.count,
        ]

        let data = try JSONSerialization.data(withJSONObject: outputData, options: .prettyPrinted)
        try data.write(to: url)
        logger.info("Dumped \(debugFeatureBuffer.count) features to \(url.path)")
    }
}

extension MLMultiArray {
    func reset(to value: NSNumber) {
        let count = self.count
        let ptr = self.dataPointer.bindMemory(to: Float.self, capacity: count)
        // Assuming Float32 for simplicity, but should check dataType
        if self.dataType == .float32 {
            ptr.assign(repeating: value.floatValue, count: count)
        } else if self.dataType == .int32 {
            let intPtr = self.dataPointer.bindMemory(to: Int32.self, capacity: count)
            intPtr.assign(repeating: value.int32Value, count: count)
        }
    }
}
