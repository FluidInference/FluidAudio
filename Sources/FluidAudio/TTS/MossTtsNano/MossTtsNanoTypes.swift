import Foundation

/// Mirror of `config.json` in `FluidInference/moss-tts-nano-coreml`.
///
/// Carries everything the Swift host needs that is derived from the upstream
/// tokenizer / model config rather than baked into the CoreML graphs: special
/// token ids, pre-tokenized prompt template segments, sampling defaults.
public struct MossTtsNanoConfig: Codable, Sendable {

    public struct Model: Codable, Sendable {
        public let nVq: Int
        public let rowWidth: Int
        public let hiddenSize: Int
        public let audioCodebookSize: Int
        public let sampleRate: Int
        public let channels: Int
        public let samplesPerFramePerChannel: Int
    }

    public struct CoreML: Codable, Sendable {
        public let prefillRows: Int
        public let maxLen: Int
    }

    public struct Tokens: Codable, Sendable {
        public let pad: Int
        public let imStart: Int
        public let imEnd: Int
        public let audioStart: Int
        public let audioEnd: Int
        public let audioUserSlot: Int
        public let audioAssistantSlot: Int
        public let audioPad: Int
    }

    /// Pre-tokenized template segments (upstream `prompting.py`).
    public struct Prompt: Codable, Sendable {
        /// `<im_start> user … Reference(s): <audio_start>`
        public let voiceClonePrefix: [Int]
        /// `<audio_end> … Text:` — sits between the reference rows and the text ids.
        public let voiceCloneAfterReference: [Int]
        /// `</user_inst> <im_end> <im_start> assistant <audio_start>`
        public let assistantSuffix: [Int]
        /// Plain (no reference) prompt prefix; kept for completeness.
        public let plainPrefix: [Int]
    }

    public struct SamplingDefaults: Codable, Sendable {
        public let textTemperature: Float
        public let textTopK: Int
        public let textTopP: Float
        public let audioTemperature: Float
        public let audioTopK: Int
        public let audioTopP: Float
        public let audioRepetitionPenalty: Float
        public let maxNewFrames: Int
    }

    public struct Text: Codable, Sendable {
        public let voiceCloneMaxTextTokens: Int
    }

    public let model: Model
    public let coreml: CoreML
    public let tokens: Tokens
    public let prompt: Prompt
    public let samplingDefaults: SamplingDefaults
    public let text: Text

    public static func load(from url: URL) throws -> MossTtsNanoConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(MossTtsNanoConfig.self, from: data)
    }
}

/// A voice-clone reference: the codec tokens of a short speech clip.
///
/// `codes` is `[frames][16]` (12.5 Hz frames × RVQ codebooks). Built-in voices
/// ship as `voices/<name>.json` in the HF repo; custom voices come from
/// `MossTtsNanoManager.cloneVoice(audioURL:)`.
public struct MossTtsNanoVoice: Codable, Sendable, Equatable {
    public let name: String
    public let sampleRate: Int
    public let frames: Int
    public let codes: [[Int32]]

    public init(name: String, codes: [[Int32]], sampleRate: Int = MossTtsNanoConstants.sampleRate) {
        self.name = name
        self.sampleRate = sampleRate
        self.frames = codes.count
        self.codes = codes
    }

    public static func load(from url: URL) throws -> MossTtsNanoVoice {
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let voice = try decoder.decode(MossTtsNanoVoice.self, from: data)
            try voice.validate()
            return voice
        } catch let error as MossTtsNanoError {
            throw error
        } catch {
            throw MossTtsNanoError.voiceLoadFailed(path: url.path, underlying: "\(error)")
        }
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        try encoder.encode(self).write(to: url)
    }

    func validate() throws {
        guard !codes.isEmpty else { throw MossTtsNanoError.invalidVoice("no frames") }
        for (i, row) in codes.enumerated() {
            guard row.count == MossTtsNanoConstants.numCodebooks else {
                throw MossTtsNanoError.invalidVoice(
                    "frame \(i) has \(row.count) codes, expected \(MossTtsNanoConstants.numCodebooks)")
            }
            guard row.allSatisfy({ $0 >= 0 && $0 < Int32(MossTtsNanoConstants.audioCodebookSize) }) else {
                throw MossTtsNanoError.invalidVoice("frame \(i) has a code outside 0..<1024")
            }
        }
    }
}

/// Voices published alongside the models (`voices/<name>.json`).
public enum MossTtsNanoBuiltInVoice: String, CaseIterable, Sendable {
    /// English female speaker (upstream demo clip `en_2.wav`).
    case en2 = "en_2"
    /// Mandarin speaker (upstream demo clip `zh_1.wav`).
    case zh1 = "zh_1"

    public static let `default`: MossTtsNanoBuiltInVoice = .en2

    public var fileName: String { "\(MossTtsNanoBuiltInVoiceFiles.subdirectory)/\(rawValue).json" }

    public init?(name: String) {
        self.init(rawValue: name.lowercased().replacingOccurrences(of: "-", with: "_"))
    }
}

enum MossTtsNanoBuiltInVoiceFiles {
    static let subdirectory = "voices"
}

/// Sampling controls. Defaults mirror upstream `inference()`.
public struct MossTtsNanoSamplingOptions: Sendable, Equatable {
    public var textTemperature: Float
    public var audioTemperature: Float
    public var audioTopP: Float
    public var repetitionPenalty: Float
    /// Argmax decoding. Deterministic, but upstream greedy decoding never emits the
    /// stop token and runs to `maxNewFrames`; it exists for parity testing only.
    public var greedy: Bool
    public var maxNewFrames: Int
    /// Seed for the host-side uniform draws that feed the in-graph sampler. `nil`
    /// draws from the system RNG.
    public var seed: UInt64?
    /// Token budget per text chunk; `0` disables chunking.
    public var maxTextTokens: Int

    public init(
        textTemperature: Float = MossTtsNanoConstants.defaultTextTemperature,
        audioTemperature: Float = MossTtsNanoConstants.defaultAudioTemperature,
        audioTopP: Float = MossTtsNanoConstants.defaultAudioTopP,
        repetitionPenalty: Float = MossTtsNanoConstants.defaultRepetitionPenalty,
        greedy: Bool = false,
        maxNewFrames: Int = MossTtsNanoConstants.defaultMaxNewFrames,
        seed: UInt64? = nil,
        maxTextTokens: Int = MossTtsNanoConstants.defaultMaxTextTokens
    ) {
        self.textTemperature = textTemperature
        self.audioTemperature = audioTemperature
        self.audioTopP = audioTopP
        self.repetitionPenalty = repetitionPenalty
        self.greedy = greedy
        self.maxNewFrames = maxNewFrames
        self.seed = seed
        self.maxTextTokens = maxTextTokens
    }

    public static let `default` = MossTtsNanoSamplingOptions()
}

/// One decoded 80 ms frame (3840 samples per channel at 48 kHz).
public struct MossTtsNanoAudioFrame: Sendable {
    public let left: [Float]
    public let right: [Float]
    /// Zero-based frame index within the current text chunk.
    public let frameIndex: Int
    /// Zero-based text chunk index and total chunk count for the utterance.
    public let chunkIndex: Int
    public let chunkCount: Int
    /// Silence frame inserted between chunks (not model output).
    public let isPause: Bool
    public var sampleRate: Int { MossTtsNanoConstants.sampleRate }
}

/// A complete stereo utterance.
public struct MossTtsNanoAudio: Sendable {
    public let left: [Float]
    public let right: [Float]
    public var sampleRate: Int { MossTtsNanoConstants.sampleRate }
    public var duration: Double { Double(left.count) / Double(sampleRate) }

    /// Mid channel `(L + R) / 2`.
    public var mono: [Float] {
        var out = [Float](repeating: 0, count: left.count)
        for i in 0..<left.count {
            out[i] = 0.5 * (left[i] + right[i])
        }
        return out
    }

    /// Interleaved `L R L R …` for stereo sinks.
    public var interleaved: [Float] {
        var out = [Float](repeating: 0, count: left.count * 2)
        for i in 0..<left.count {
            out[2 * i] = left[i]
            out[2 * i + 1] = right[i]
        }
        return out
    }
}
