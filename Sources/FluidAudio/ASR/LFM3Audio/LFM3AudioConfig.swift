// LFM3-Audio needs Apple silicon (Float16 host buffers; Swift has no Float16 on macOS x86_64).
#if arch(arm64)
import Foundation

/// Which Apple runtime executes LFM3-Audio-2B.
public enum LFM3AudioBackend: String, Sendable, CaseIterable {
    /// Core ML (macOS 15 / iOS 18+): int8 LM pack on the GPU, fp16 audio encoder / depth decoder / codec.
    case coreML = "coreml"
    /// Core AI (macOS 27 / iOS 27+): the LM runs as a Core AI `.aimodel` bundle.
    case coreAI = "coreai"
}

/// Errors thrown by LFM3-Audio.
public enum LFM3AudioError: Error, LocalizedError, Sendable {
    case unsupportedPlatform(String)
    case modelNotFound(String)
    case invalidModel(String)
    case contextFull(needed: Int, available: Int)
    case speechNotLoaded
    case liveSessionActive
    case noLiveSession
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedPlatform(let message): return message
        case .modelNotFound(let name): return "LFM3-Audio model not found: \(name)"
        case .invalidModel(let message): return "Invalid LFM3-Audio model: \(message)"
        case .contextFull(let needed, let available):
            return "LFM3-Audio context full: \(needed) positions needed, \(available) available"
        case .speechNotLoaded: return "LFM3-Audio speech models are not loaded (load with includeSpeech: true)"
        case .liveSessionActive: return "A LFM3-Audio live transcription session is active; finish it first"
        case .noLiveSession: return "No LFM3-Audio live transcription session; call startLiveTranscription() first"
        case .processingFailed(let message): return "LFM3-Audio processing failed: \(message)"
        }
    }
}

/// Fixed contract of LFM3-Audio-2B: token ids, prompts, audio rates and limits
/// (`configuration_lfm3_audio.py`, `tokenizer.json`, `chat_template.jinja` of LiquidAI/LFM3-Audio-2B).
enum LFM3AudioConfig {
    static let inputSampleRate = 16_000
    static let outputSampleRate = 24_000
    static let hiddenSize = 2048

    /// `AudioEncoder-fp16` functions (tokens per call at 12.5 tokens/s); n12 is one 0.96 s LiveASR chunk.
    static let liveEncoderTokens = 12
    static let offlineEncoderTokens = [96, 375]

    /// Offline ASR transcribes clips up to 40 s in one pass (as upstream) and cuts longer audio into
    /// <= 30 s segments at the quietest 20 ms window in each segment's last 5 s.
    static let singlePassSeconds = 40.0
    static let segmentSeconds = 30.0
    static let maxTranscriptTokens = 1024

    /// LiveASR: 96 mel frames (0.96 s) per chunk; the LM context restarts every 31 chunks (upstream default).
    static let liveChunkFrames = 96
    static let liveRestartChunks = 31

    enum Token {
        static let endOfText: Int32 = 124_895
        static let imStart: Int32 = 124_899
        static let imEnd: Int32 = 124_900
        static let audioStart: Int32 = 125_012
        static let textEnd: Int32 = 125_014
        static let mixedStart: Int32 = 125_015
        static let mixedEnd: Int32 = 125_016
        static let audio: Int32 = 125_031
        static let audioOut: Int32 = 125_032
        static let placeholders: Set<Int32> = [audio, audioOut]
        static let eos: Set<Int32> = [imEnd, textEnd]
    }

    /// Offline ASR prompt around the audio rows (`Perform ASR in <language>.`, or `Perform ASR.` to infer it).
    static func asrPrompt(language: String?) -> (prefix: String, suffix: String) {
        let system = language.map { "Perform ASR in \($0)." } ?? "Perform ASR."
        return (
            "<|startoftext|><|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n"
        )
    }

    static let livePrompt =
        "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n"

    /// TTS prompt; the assistant turn is prefilled with `<|audio_start|>`.
    static func ttsPrompt(text: String, language: String, voice: String) -> String {
        "<|startoftext|><|im_start|>system\nPerform TTS in \(language) using a \(voice) voice.<|im_end|>\n"
            + "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n<|audio_start|>"
    }

    /// Offline-ASR segments: one pass up to `singlePass` seconds, otherwise <= `maxSegment` s pieces cut at the
    /// lowest-energy 20 ms window within the last `search` seconds of each piece.
    static func splitLongAudio(
        _ samples: [Float], maxSegment: Double = segmentSeconds, singlePass: Double = singlePassSeconds,
        search: Double = 5, sampleRate: Int = inputSampleRate
    ) -> [ArraySlice<Float>] {
        let sr = Double(sampleRate)
        guard samples.count > Int(singlePass * sr) else { return [samples[...]] }
        let length = Int(maxSegment * sr)
        let window = Int(0.02 * sr)
        var segments: [ArraySlice<Float>] = []
        var start = 0
        while samples.count - start > length {
            let lo = start + length - Int(search * sr)
            let hi = start + length
            var energy = 0.0
            for i in lo..<(lo + window) { energy += Double(samples[i] * samples[i]) }
            var best = energy
            var bestIndex = lo
            for i in (lo + 1)...(hi - window) {
                let added = samples[i + window - 1]
                let removed = samples[i - 1]
                energy += Double(added * added) - Double(removed * removed)
                if energy < best {
                    best = energy
                    bestIndex = i
                }
            }
            let cut = bestIndex + window / 2
            segments.append(samples[start..<cut])
            start = cut
        }
        segments.append(samples[start...])
        return segments
    }
}
#endif
