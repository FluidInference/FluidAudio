#if arch(arm64)
@preconcurrency import CoreML
import Foundation

/// Result of an offline LFM3-Audio transcription.
public struct LFM3AudioTranscription: Sendable {
    public let text: String
    /// Seconds of input audio.
    public let audioDuration: TimeInterval
    /// Wall-clock seconds spent transcribing (mel + encoder + LM).
    public let processingTime: TimeInterval
    /// Generated text tokens (excluding the end-of-turn token).
    public let tokenCount: Int

    /// Real-time factor: audio seconds per second of processing.
    public var rtfx: Double { processingTime > 0 ? audioDuration / processingTime : 0 }
}

/// LFM3-Audio-2B (Liquid AI) on device: offline ASR (long-form included), LiveASR (0.96 s chunks) and, with the
/// TTS trait, text-to-speech — all from one 2B audio language model.
///
/// ```swift
/// let manager = try await LFM3AudioManager.downloadAndLoad()
/// let result = try await manager.transcribe(samples)  // 16 kHz mono
/// ```
///
/// The manager owns one LM context, so calls are serialized: offline transcription and TTS throw
/// `LFM3AudioError.liveSessionActive` while a live session is open.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
public actor LFM3AudioManager {

    struct LiveState {
        var melStream: LFM3AudioMelStream
        var planner = LFM3AudioEncoderPlanner(tokensPerCall: LFM3AudioConfig.liveEncoderTokens)
        var melBuffer = LFM3AudioMelBuffer()
        /// Ids fed before the next chunk's audio rows: the prompt after a (re)start, else `<|text_end|>`.
        var pendingIds: [Int32] = []
        var fresh = true
        var chunks = 0
        var tokens: [Int32] = []
    }

    private static let logger = AppLogger(category: "LFM3AudioManager")

    public nonisolated let backend: LFM3AudioBackend
    public nonisolated let includesSpeech: Bool
    let tokenizer: LFM3AudioTokenizer
    let languageModel: any LFM3AudioLanguageModel
    let encoder: LFM3AudioCoreMLEncoder
    let mel = LFM3AudioLogMel()
    let livePrompt: [Int32]
    var live: LiveState?
    /// FIFO gate over the LM context: the actor is reentrant at every `await`, so two calls must not interleave.
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    #if TTS
    let speech: LFM3AudioSpeechRuntime?
    #endif

    public init(models: LFM3AudioModels) async throws {
        backend = models.backend
        includesSpeech = models.includesSpeech
        let tokenizer = try LFM3AudioTokenizer(url: models.tokenizerURL)
        self.tokenizer = tokenizer
        switch models.languageModel {
        case .coreML(let stages, let head):
            languageModel = try LFM3AudioCoreMLBackbone(
                config: models.config, embedding: models.embedding, stageModels: stages, head: head)
        case .coreAI(let bundle):
            languageModel = try await Self.makeCoreAIBackbone(bundle: bundle, embedding: models.embedding)
        }
        encoder = try LFM3AudioCoreMLEncoder(models: models.encoder)
        livePrompt = tokenizer.encode(LFM3AudioConfig.livePrompt)
        #if TTS
        if let speechModels = models.speech {
            var coreAIDepth: (any LFM3AudioDepthModel)?
            if case .coreAI(let bundle) = speechModels.depth {
                coreAIDepth = try await Self.makeCoreAIDepth(bundle: bundle)
            }
            speech = try LFM3AudioSpeechRuntime(models: speechModels, coreAIDepth: coreAIDepth)
        } else {
            speech = nil
        }
        #endif
    }

    /// Download (if needed), load and wrap the models.
    public static func downloadAndLoad(
        backend: LFM3AudioBackend = .coreML,
        includeSpeech: Bool = true,
        computeUnits: LFM3AudioComputeUnits = .default,
        progressHandler: ProgressHandler? = nil
    ) async throws -> LFM3AudioManager {
        let models = try await LFM3AudioModels.downloadAndLoad(
            backend: backend, includeSpeech: includeSpeech, computeUnits: computeUnits,
            progressHandler: progressHandler)
        return try await LFM3AudioManager(models: models)
    }

    // MARK: - Offline ASR

    /// Transcribe 16 kHz mono samples. Clips up to 40 s run in one pass; longer audio is split into <= 30 s
    /// segments at quiet points.
    ///
    /// - Parameter language: English name of the spoken language (`"English"`, `"French"`, ...), or `nil` to let
    ///   the model infer it.
    public func transcribe(_ samples: [Float], language: String? = "English") async throws -> LFM3AudioTranscription {
        guard live == nil else { throw LFM3AudioError.liveSessionActive }
        await acquire()
        defer { release() }
        let start = Date()
        var parts: [String] = []
        var tokens = 0
        for segment in LFM3AudioConfig.splitLongAudio(samples) {
            let ids = try await transcribeSegment(segment, language: language)
            tokens += ids.count
            let text = tokenizer.decode(ids).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { parts.append(text) }
        }
        return LFM3AudioTranscription(
            text: parts.joined(separator: " "),
            audioDuration: Double(samples.count) / Double(LFM3AudioConfig.inputSampleRate),
            processingTime: Date().timeIntervalSince(start), tokenCount: tokens)
    }

    /// Transcribe an audio file (any format AVFoundation reads; resampled to 16 kHz mono).
    public func transcribe(_ url: URL, language: String? = "English") async throws -> LFM3AudioTranscription {
        let samples = try AudioConverter().resampleAudioFile(url)
        return try await transcribe(samples, language: language)
    }

    /// One <= 40 s segment -> generated ids (greedy; placeholders, `<|mixed_start|>`, `<|audio_start|>` suppressed).
    private func transcribeSegment(_ samples: ArraySlice<Float>, language: String?) async throws -> [Int32] {
        let rows = try encoder.encode(mel(samples))
        let audioRows = rows.count / LFM3AudioConfig.hiddenSize
        let prompt = LFM3AudioConfig.asrPrompt(language: language)
        let ids =
            tokenizer.encode(prompt.prefix) + [Int32](repeating: LFM3AudioConfig.Token.audio, count: audioRows)
            + tokenizer.encode(prompt.suffix)
        let suppress = LFM3AudioConfig.Token.placeholders.union([
            LFM3AudioConfig.Token.mixedStart, LFM3AudioConfig.Token.audioStart,
        ])
        try await languageModel.reset()
        var token = try await languageModel.feed(
            .mixed(ids: ids, rows: rows, placeholder: LFM3AudioConfig.Token.audio), want: .token(suppress: suppress)
        ).token
        var generated: [Int32] = []
        while !LFM3AudioConfig.Token.eos.contains(token) && generated.count < LFM3AudioConfig.maxTranscriptTokens {
            generated.append(token)
            token = try await languageModel.feed(.ids([token]), want: .token(suppress: suppress)).token
        }
        return generated
    }

    // MARK: - LiveASR

    /// Start a live (streaming) English transcription. Audio is processed in 0.96 s chunks as it arrives.
    public func startLiveTranscription() async throws {
        guard live == nil else { throw LFM3AudioError.liveSessionActive }
        encoder.resetState()
        live = LiveState(melStream: LFM3AudioMelStream(mel))
        // The live chunk's prefill bucket loads on first use; load it here rather than stalling the first chunk.
        do {
            try await languageModel.prepare(rows: 1 + LFM3AudioConfig.liveEncoderTokens)
        } catch {
            live = nil
            throw error
        }
    }

    /// Append 16 kHz mono samples to the live session; returns the transcript so far.
    @discardableResult
    public func appendLiveAudio(_ samples: [Float]) async throws -> String {
        await acquire()
        defer { release() }
        guard var state = live else { throw LFM3AudioError.noLiveSession }
        state.melBuffer.append(state.melStream.push(samples))
        while state.melBuffer.frames >= LFM3AudioConfig.liveChunkFrames {
            let chunk = state.melBuffer.columns(0, LFM3AudioConfig.liveChunkFrames)
            state.melBuffer.dropFirst(LFM3AudioConfig.liveChunkFrames)
            try await transcribeLiveChunk(chunk, isFinal: false, state: &state)
        }
        live = state
        return tokenizer.decode(state.tokens)
    }

    /// Flush the remaining audio, end the session and return the final transcript.
    public func finishLiveTranscription() async throws -> String {
        await acquire()
        defer { release() }
        guard var state = live else { throw LFM3AudioError.noLiveSession }
        live = nil
        state.melBuffer.append(state.melStream.push([], isFinal: true))
        while state.melBuffer.frames > LFM3AudioConfig.liveChunkFrames {
            let chunk = state.melBuffer.columns(0, LFM3AudioConfig.liveChunkFrames)
            state.melBuffer.dropFirst(LFM3AudioConfig.liveChunkFrames)
            try await transcribeLiveChunk(chunk, isFinal: false, state: &state)
        }
        try await transcribeLiveChunk(state.melBuffer, isFinal: true, state: &state)
        return tokenizer.decode(state.tokens).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Transcript of the open live session so far (empty when none is open).
    public var liveTranscript: String { live.map { tokenizer.decode($0.tokens) } ?? "" }

    /// Abandon the open live session, if any.
    public func cancelLiveTranscription() { live = nil }

    /// One chunk of HF `transcribe_live`: feed the pending ids + the chunk's encoder rows, then decode up to 48
    /// tokens (96 on the final chunk) until `<|text_end|>`; the LM restarts from the prompt every 31 chunks.
    private func transcribeLiveChunk(_ frames: LFM3AudioMelBuffer, isFinal: Bool, state: inout LiveState) async throws {
        typealias T = LFM3AudioConfig.Token
        var rows: [Float16] = []
        try encoder.run(
            state.planner.push(frames, isFinal: isFinal), tokensPerCall: LFM3AudioConfig.liveEncoderTokens,
            rows: &rows)
        if state.fresh || (state.chunks > 0 && state.chunks % LFM3AudioConfig.liveRestartChunks == 0) {
            try await languageModel.reset()
            state.pendingIds = livePrompt
            state.fresh = false
        }
        state.chunks += 1
        let audioRows = rows.count / LFM3AudioConfig.hiddenSize
        guard audioRows > 0 || isFinal, !state.pendingIds.isEmpty || audioRows > 0 else { return }
        let ids = state.pendingIds + [Int32](repeating: T.audio, count: audioRows)
        state.pendingIds = []
        let suppress =
            isFinal
            ? T.placeholders
            : T.placeholders.union([T.audioStart, T.mixedStart, T.imEnd, T.mixedEnd, T.endOfText])
        let stop: Set<Int32> = [T.imEnd, T.endOfText, T.imStart]
        var token = try await languageModel.feed(
            .mixed(ids: ids, rows: rows, placeholder: T.audio), want: .token(suppress: suppress)
        ).token
        let budget = isFinal ? 96 : 48
        for step in 0..<budget {
            if token == T.textEnd || (isFinal && stop.contains(token)) { break }
            state.tokens.append(token)
            // fed even when the budget ends (as upstream)
            token = try await languageModel.feed(.ids([token]), want: .token(suppress: suppress)).token
            if step == budget - 1 { break }
        }
        if !isFinal { state.pendingIds = [T.textEnd] }
    }

    // MARK: - Serialization

    func acquire() async {
        guard busy else {
            busy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }  // resumed with the gate handed over
    }

    func release() {
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
    }

    // MARK: - Backends

    #if TTS
    private static func makeCoreAIDepth(bundle: URL) async throws -> any LFM3AudioDepthModel {
        #if canImport(CoreAI)
        if #available(macOS 27, iOS 27, tvOS 27, visionOS 27, *) {
            return try await LFM3AudioCoreAIDepthDecoder(bundle: bundle, sampling: LFM3AudioSpeechSampling())
        }
        #endif
        throw LFM3AudioError.unsupportedPlatform(
            "The LFM3-Audio Core AI backend needs macOS 27 / iOS 27 (Core AI); use .coreML")
    }
    #endif

    private static func makeCoreAIBackbone(
        bundle: URL, embedding: LFM3AudioEmbeddingTable
    ) async throws -> any LFM3AudioLanguageModel {
        #if canImport(CoreAI)
        if #available(macOS 27, iOS 27, tvOS 27, visionOS 27, *) {
            return try await LFM3AudioCoreAIBackbone(bundle: bundle, embedding: embedding)
        }
        #endif
        throw LFM3AudioError.unsupportedPlatform(
            "The LFM3-Audio Core AI backend needs macOS 27 / iOS 27 (Core AI); use .coreML")
    }
}
#endif
