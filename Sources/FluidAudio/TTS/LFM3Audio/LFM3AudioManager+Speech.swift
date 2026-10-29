#if TTS && arch(arm64)
@preconcurrency import CoreML
import Foundation

/// TTS voice of LFM3-Audio-2B.
public enum LFM3AudioVoice: String, Sendable, CaseIterable {
    case female
    case male
}

/// Result of LFM3-Audio speech synthesis (24 kHz mono).
public struct LFM3AudioSynthesis: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    /// Speech frames generated (12.5 Hz, 12 codes each).
    public let frames: Int
    /// `false` when generation stopped at `maxFrames` before the model ended the utterance.
    public let finished: Bool
    /// Wall-clock seconds from the call to the last sample.
    public let processingTime: TimeInterval
    /// Seconds from the call to the first decoded audio.
    public let firstAudioLatency: TimeInterval?

    public var duration: TimeInterval { Double(samples.count) / Double(sampleRate) }
    /// Real-time factor: audio seconds per second of processing.
    public var rtfx: Double { processingTime > 0 ? duration / processingTime : 0 }
}

/// Speech components owned by the manager (one depth decoder; a codec worker per utterance).
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
final class LFM3AudioSpeechRuntime {
    let models: LFM3AudioSpeechModels
    let depth: LFM3AudioDepthDecoder
    let codebook: LFM3AudioCodebookEmbedding

    init(models: LFM3AudioSpeechModels) throws {
        self.models = models
        depth = try LFM3AudioDepthDecoder(models: models, sampling: LFM3AudioSpeechSampling())
        codebook = LFM3AudioCodebookEmbedding(table: models.codebook)
    }
}

@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
extension LFM3AudioManager {
    static let speechLogger = AppLogger(category: "LFM3AudioSpeech")

    /// Synthesize speech for `text`.
    ///
    /// Speech frames are decoded to audio while the LM keeps generating; `onAudio` receives each new chunk of
    /// 24 kHz samples as soon as it is decoded (streaming playback).
    ///
    /// - Parameters:
    ///   - language: a language / locale of the model card (`"American English"`, `"British English"`,
    ///     `"French"`, `"German"`, `"Japanese"`, ...).
    ///   - seed: sampling seed (`nil` = random).
    ///   - maxFrames: upper bound on speech frames (80 ms each).
    public func synthesize(
        _ text: String,
        voice: LFM3AudioVoice = .female,
        language: String = "American English",
        seed: UInt64? = nil,
        maxFrames: Int = 1500,
        onAudio: (@Sendable ([Float]) -> Void)? = nil
    ) async throws -> LFM3AudioSynthesis {
        guard live == nil else { throw LFM3AudioError.liveSessionActive }
        guard let speech else { throw LFM3AudioError.speechNotLoaded }
        await acquire()
        defer { release() }
        typealias T = LFM3AudioConfig.Token
        let start = Date()
        let prompt = tokenizer.encode(LFM3AudioConfig.ttsPrompt(text: text, language: language, voice: voice.rawValue))
        let worker = try LFM3AudioCodecWorker(models: speech.models.codec, head: speech.models.codecHead)
        let queue = LFM3AudioFrameQueue()
        let decoder = Task.detached { () async throws -> (samples: [Float], firstAudio: Date?) in
            var samples: [Float] = []
            var firstAudio: Date?
            while let frames = await queue.next() {
                let chunk = try await worker.push(frames, finalize: false)
                guard !chunk.isEmpty else { continue }
                if firstAudio == nil { firstAudio = Date() }
                onAudio?(chunk)
                samples += chunk
            }
            let tail = try await worker.push([], finalize: true)
            if !tail.isEmpty {
                onAudio?(tail)
                samples += tail
            }
            return (samples, firstAudio)
        }

        var random = LFM3AudioRandom(seed: seed ?? UInt64.random(in: 0...UInt64.max))
        var frames = 0
        var textTokens = 0
        var finished = false
        var lmSeconds = 0.0
        var depthSeconds = 0.0
        do {
            try await languageModel.reset()
            speech.depth.reset()
            var speaking = true  // the prompt ends with <|audio_start|>
            var result = try await languageModel.feed(.ids(prompt), want: .hidden)
            // HF mode machine for batch 1: audio frames until the EOS frame, then text until an end token.
            while true {
                let next: LFM3AudioRows
                if speaking {
                    let depthStart = Date()
                    var codes = try speech.depth.frame(result.hidden, random: &random)
                    depthSeconds += Date().timeIntervalSince(depthStart)
                    if LFM3AudioDepthDecoder.isEndOfSpeech(codes) {
                        codes = [Int32](
                            repeating: LFM3AudioSpeechConfig.eosCode, count: LFM3AudioSpeechConfig.codebooks)
                        speaking = false
                    } else {
                        speech.depth.record(codes)
                        frames += 1
                        await queue.push(codes)
                    }
                    next = .embeddings(speech.codebook.embed(codes))
                } else {
                    let token = result.token
                    if token == T.textEnd || T.eos.contains(token) {
                        finished = true
                        break
                    }
                    if token == T.audioStart { speaking = true }
                    textTokens += 1
                    next = .ids([token])
                }
                if frames >= maxFrames || textTokens > 256 { break }
                let lmStart = Date()
                result = try await languageModel.feed(
                    next, want: speaking ? .hidden : .token(suppress: T.placeholders))
                lmSeconds += Date().timeIntervalSince(lmStart)
            }
        } catch {
            await queue.finish()
            decoder.cancel()
            throw error
        }
        await queue.finish()
        let decoded = try await decoder.value
        Self.speechLogger.debug(
            "TTS \(frames) frames: LM \(String(format: "%.2f", 1000 * lmSeconds / Double(max(frames, 1)))) ms/frame, "
                + "depth \(String(format: "%.2f", 1000 * depthSeconds / Double(max(frames, 1)))) ms/frame")
        return LFM3AudioSynthesis(
            samples: decoded.samples, sampleRate: LFM3AudioConfig.outputSampleRate, frames: frames,
            finished: finished, processingTime: Date().timeIntervalSince(start),
            firstAudioLatency: decoded.firstAudio.map { $0.timeIntervalSince(start) })
    }
}
#endif
