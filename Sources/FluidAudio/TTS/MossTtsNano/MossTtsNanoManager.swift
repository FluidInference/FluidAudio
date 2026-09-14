@preconcurrency import CoreML
import Foundation

/// Public API for MOSS-TTS-Nano: 0.1B multilingual streaming TTS with zero-shot
/// voice cloning, native 48 kHz stereo output.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
///
/// ```swift
/// let manager = try await MossTtsNanoManager.downloadAndCreate()
/// let voice = try await manager.loadVoice(.en2)
/// let audio = try await manager.synthesize(text: "Hello from the neural engine.", voice: voice)
/// // audio.left / audio.right are 48 kHz Float32; audio.mono for single-channel sinks
///
/// for try await frame in try await manager.synthesizeStreaming(text: "…", voice: voice) {
///     player.schedule(left: frame.left, right: frame.right)   // 80 ms per frame
/// }
/// ```
///
/// Voices are codec-token sequences of a reference clip: `loadVoice` fetches the
/// two published presets, `cloneVoice(audioURL:)` encodes any clip (≤ ~25 s) with
/// the fp32 codec encoder. Long text is chunked at sentence boundaries with the
/// upstream 50-token budget and joined with short pauses.
public actor MossTtsNanoManager {

    private let logger = AppLogger(category: "MossTtsNanoManager")

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private var store: MossTtsNanoModelStore?
    private var synthesizer: MossTtsNanoSynthesizer?

    /// - Parameter computeUnits: unit preference for the Frame and CodecStep graphs
    ///   (Prefill/Step always run on CPU+GPU). `.cpuAndGPU` is the measured best.
    public init(directory: URL? = nil, computeUnits: MLComputeUnits = .cpuAndGPU) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    public var isAvailable: Bool { synthesizer != nil }

    public static func downloadAndCreate(
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> MossTtsNanoManager {
        let manager = MossTtsNanoManager(directory: cacheDirectory, computeUnits: computeUnits)
        try await manager.initialize()
        return manager
    }

    /// Download (if missing) and load the streaming-path models.
    public func initialize() async throws {
        if synthesizer != nil { return }
        let store = MossTtsNanoModelStore(directory: directory, computeUnits: computeUnits)
        try await store.loadIfNeeded()
        synthesizer = MossTtsNanoSynthesizer(
            prefill: try await store.prefill(),
            step: try await store.step(),
            frame: try await store.frame(),
            codecStep: try await store.codecStep(),
            config: try await store.config(),
            tokenizer: try await store.tokenizer())
        self.store = store
        logger.info("MOSS-TTS-Nano ready")
    }

    public func cleanup() async {
        if let store { await store.unload() }
        store = nil
        synthesizer = nil
    }

    // MARK: - Voices

    /// Fetch and decode a published preset voice.
    public func loadVoice(_ voice: MossTtsNanoBuiltInVoice) async throws -> MossTtsNanoVoice {
        try await MossTtsNanoResourceDownloader.loadVoice(voice, directory: directory)
    }

    /// Encode a reference clip into a reusable voice. Any AVFoundation-readable file;
    /// resampled to 48 kHz stereo internally. Downloads the fp32 codec encoder on first use.
    public func cloneVoice(audioURL: URL, name: String? = nil) async throws -> MossTtsNanoVoice {
        guard let store, let synthesizer else { throw MossTtsNanoError.notInitialized }
        let encoder = try await store.encoder()
        let (left, right) = try MossTtsNanoAudioLoader.loadStereo48k(url: audioURL)
        let voice = try MossTtsNanoSynthesizer.encodeVoice(
            encoder: encoder, left: left, right: right,
            name: name ?? audioURL.deletingPathExtension().lastPathComponent,
            nVq: synthesizer.config.model.nVq,
            samplesPerFrame: synthesizer.config.model.samplesPerFramePerChannel)
        let overhead = synthesizer.builder.voiceCloneOverheadRows(voice: voice)
        let capacity = synthesizer.config.coreml.prefillRows
        guard overhead < capacity - 8 else {
            throw MossTtsNanoError.invalidVoice(
                "reference clip is \(voice.frames) frames (\(Double(voice.frames) * MossTtsNanoConstants.frameDuration) s); "
                    + "it leaves no room for text in the \(capacity)-row prompt. Use a clip under ~25 s.")
        }
        return voice
    }

    // MARK: - Synthesis

    /// Tokenize text exactly as the model prompt does (SentencePiece BPE ids).
    public func tokenize(_ text: String) throws -> [Int] {
        guard let synthesizer else { throw MossTtsNanoError.notInitialized }
        return synthesizer.tokenizer.encode(text)
    }

    /// Synthesize a full utterance (48 kHz stereo).
    public func synthesize(
        text: String,
        voice: MossTtsNanoVoice,
        options: MossTtsNanoSamplingOptions = .default
    ) async throws -> MossTtsNanoAudio {
        var left: [Float] = []
        var right: [Float] = []
        let stream = try synthesizeStreaming(text: text, voice: voice, options: options)
        for try await frame in stream {
            left.append(contentsOf: frame.left)
            right.append(contentsOf: frame.right)
        }
        return MossTtsNanoAudio(left: left, right: right)
    }

    /// Synthesize as a stream of 80 ms stereo frames, yielded as soon as each is decoded.
    /// Pause frames (`isPause == true`) separate text chunks.
    public func synthesizeStreaming(
        text: String,
        voice: MossTtsNanoVoice,
        options: MossTtsNanoSamplingOptions = .default
    ) throws -> AsyncThrowingStream<MossTtsNanoAudioFrame, Error> {
        guard let synthesizer else { throw MossTtsNanoError.notInitialized }
        try voice.validate()
        let chunks = try synthesizer.planChunks(text: text, voice: voice, options: options)
        logger.info("MOSS-TTS-Nano synthesizing \(chunks.count) chunk(s), voice=\(voice.name)")
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.run(chunks: chunks, voice: voice, options: options, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(
        chunks: [String],
        voice: MossTtsNanoVoice,
        options: MossTtsNanoSamplingOptions,
        continuation: AsyncThrowingStream<MossTtsNanoAudioFrame, Error>.Continuation
    ) async throws {
        guard let synthesizer else { throw MossTtsNanoError.notInitialized }
        var rng = options.seed.map { MossTtsNanoRandom(seed: $0) } ?? MossTtsNanoRandom()
        let start = Date()
        var totalFrames = 0
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let produced = try synthesizer.generateChunk(
                text: chunk, voice: voice, options: options, rng: &rng,
                chunkIndex: index, chunkCount: chunks.count
            ) { frame in
                try Task.checkCancellation()
                continuation.yield(frame)
            }
            totalFrames += produced
            if index < chunks.count - 1 {
                let pause = MossTtsNanoTextChunker.pauseSeconds(after: chunk)
                let samples = Int((pause * Float(MossTtsNanoConstants.sampleRate)).rounded())
                if samples > 0 {
                    let silence = [Float](repeating: 0, count: samples)
                    continuation.yield(
                        MossTtsNanoAudioFrame(
                            left: silence, right: silence, frameIndex: produced,
                            chunkIndex: index, chunkCount: chunks.count, isPause: true))
                }
            }
        }
        let elapsed = Date().timeIntervalSince(start)
        let audioSeconds = Double(totalFrames) * MossTtsNanoConstants.frameDuration
        logger.info(
            "MOSS-TTS-Nano generated \(totalFrames) frames (\(String(format: "%.2f", audioSeconds)) s) in "
                + "\(String(format: "%.2f", elapsed)) s (RTFx \(String(format: "%.2f", elapsed > 0 ? audioSeconds / elapsed : 0)))"
        )
    }
}
