@preconcurrency import CoreML
import Foundation

/// `.aneState` (mobius Trial 23) entry points. Both public synthesis APIs
/// route through a stateful `PocketTtsSession`: the MLState pipeline is
/// session-shaped by construction (one shared KV state, reset per chunk by
/// the voice-snapshot write), so the streaming API simply runs a one-shot
/// session instead of duplicating the frame loop a fourth time.
extension PocketTtsSynthesizer {

    /// Whether the current OS can run the `.aneState` placement.
    static var isStatePlacementAvailable: Bool {
        if #available(macOS 15.0, iOS 18.0, *) {
            return true
        }
        return false
    }

    /// Stateful counterpart of `synthesizeStreaming(text:voiceData:...)`.
    ///
    /// Frame metadata matches the session API (`utteranceIndex` is `0`
    /// rather than `nil`); samples, chunking, and EOS behavior are otherwise
    /// identical to the IO streaming path.
    static func synthesizeStreamingStateful(
        text: String,
        voiceData: PocketTtsVoiceData,
        temperature: Float,
        seed: UInt64?,
        maxTokensPerChunk: Int,
        language: PocketTtsLanguage
    ) async throws -> AsyncThrowingStream<AudioFrame, Error> {
        let session = try await makeStateSession(
            voiceData: voiceData,
            temperature: temperature,
            seed: seed,
            language: language,
            maxTokensPerChunk: maxTokensPerChunk
        )
        session.enqueue(text)
        session.finish()
        return session.frames
    }

    /// Build a `.aneState` session: only the multifunction state models,
    /// the Mimi decoder, and the constants bundle are needed — no IO
    /// models, layer keys, or host-side KV cache.
    ///
    /// The voice prefill/injection happens lazily on the first chunk (the
    /// fp16 snapshot conversion is a few ms; cloned voices pay one stateful
    /// prefill and are then captured from the state).
    static func makeStateSession(
        voiceData: PocketTtsVoiceData,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        language: PocketTtsLanguage = .english,
        maxTokensPerChunk: Int = PocketTtsConstants.maxTokensPerChunk
    ) async throws -> PocketTtsSession {
        let store = try currentModelStore()
        guard isStatePlacementAvailable else {
            throw PocketTTSError.processingFailed(
                "PocketTTS `.aneState` placement requires macOS 15+/iOS 18+ "
                    + "(MLState and multifunction CoreML models)."
            )
        }

        let constants = try await store.constants()
        let stateModels = try await store.stateModels()
        let mimiModel = try await store.mimiDecoder()
        let mimiKeys = try await store.mimiDecoderKeys()
        let repoDir = try await store.repoDir()
        let mimiState = try loadMimiInitialState(from: repoDir, mimiKeys: mimiKeys)
        let bosEmb = try createBosEmbedding(constants.bosEmbedding)
        let seedValue = seed ?? UInt64.random(in: 0...UInt64.max)

        let session = PocketTtsSession(
            stateModels: stateModels,
            voiceData: voiceData,
            mimiState: mimiState,
            constants: constants,
            mimiModel: mimiModel,
            mimiKeys: mimiKeys,
            bosEmb: bosEmb,
            temperature: temperature,
            seed: seedValue,
            language: language,
            maxTokensPerChunk: maxTokensPerChunk
        )
        await session.start()
        return session
    }
}
