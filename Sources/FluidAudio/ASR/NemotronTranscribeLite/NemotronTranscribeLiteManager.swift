@preconcurrency import CoreML
import Accelerate
import Foundation

/// A transcribed word.
public struct NemotronTranscribeLiteWord: Sendable, Codable, Equatable {
    public let text: String
    /// Start time in seconds: the 40 ms CTC frame of the word's first piece.
    public let start: Double
    /// Diarizer speaker slot (0...7, in order of first appearance in the diarizer), or nil without
    /// speaker attribution.
    public let speaker: Int?

    public init(text: String, start: Double, speaker: Int? = nil) {
        self.text = text
        self.start = start
        self.speaker = speaker
    }
}

/// Result of a Nemotron 3.5 Transcribe Lite run.
public struct NemotronTranscribeLiteResult: Sendable {
    /// One stretch of consecutive words by the same speaker.
    public struct Turn: Sendable, Equatable {
        public let speaker: Int?
        public let start: Double
        public let words: [NemotronTranscribeLiteWord]
        public var text: String { words.map(\.text).joined(separator: " ") }
    }

    public struct Timings: Sendable {
        /// Mel, normalization and encoder windows.
        public let encoderSeconds: Double
        public let ctcSeconds: Double
        /// Nemotron-3-Diarization, run concurrently with the encoder (nil without speakers).
        public let diarizerSeconds: Double?
        public let totalSeconds: Double
    }

    public let words: [NemotronTranscribeLiteWord]
    public let durationSeconds: Double
    public let timings: Timings

    public var text: String { words.map(\.text).joined(separator: " ") }

    public var turns: [Turn] {
        var turns: [Turn] = []
        var current: [NemotronTranscribeLiteWord] = []
        for word in words {
            if let first = current.first, first.speaker != word.speaker {
                turns.append(Turn(speaker: first.speaker, start: first.start, words: current))
                current = []
            }
            current.append(word)
        }
        if let first = current.first {
            turns.append(Turn(speaker: first.speaker, start: first.start, words: current))
        }
        return turns
    }

    /// Real-time factor: audio duration / processing time.
    public var rtfx: Double { timings.totalSeconds > 0 ? durationSeconds / timings.totalSeconds : 0 }
}

/// Nemotron 3.5 Transcribe Lite: NVIDIA Nemotron 3.5 Transcribe's ASR encoder + CTC timestamp head,
/// without the LLM, as a greedy CTC recognizer with 40 ms word start times.
///
/// Pipeline: NeMo log-mel with the checkpoint's window + filterbank → per-feature normalization over
/// the whole recording → encoder over 48 s windows (40 s cores, 4 s context) → CTC head over 600-frame
/// windows (500 core) → greedy CTC → words. The CTC head was trained by NVIDIA as a timestamp aligner
/// for the LLM's output; used directly it is much less accurate than the full model.
public actor NemotronTranscribeLiteManager {

    private let models: NemotronTranscribeLiteModels
    private let mel: AudioMelSpectrogram
    private static let logger = AppLogger(category: "NemotronTranscribeLite")

    public init(models: NemotronTranscribeLiteModels) {
        self.models = models
        let c = NemotronTranscribeLiteConfig.self
        self.mel = AudioMelSpectrogram(
            sampleRate: c.sampleRate, nMels: c.nMels, nFFT: c.nFFT, hopLength: c.hopLength,
            winLength: c.winLength, window: models.melWindow, filterbank: models.melFilterbank)
    }

    /// Download (if needed) and load the models, then build a manager.
    public static func load(
        computeUnits: MLComputeUnits = .cpuAndGPU,
        progressHandler: ProgressHandler? = nil
    ) async throws -> NemotronTranscribeLiteManager {
        let models = try await NemotronTranscribeLiteModels.downloadAndLoad(
            computeUnits: computeUnits, progressHandler: progressHandler)
        return NemotronTranscribeLiteManager(models: models)
    }

    /// One prediction per model on zeros. The first call builds the GPU graphs (~1 s); do it before
    /// timing anything.
    public func warmup() throws {
        let c = NemotronTranscribeLiteConfig.self
        let mel = try MLMultiArray(shape: [1, c.nMels as NSNumber, c.windowMelFrames as NSNumber], dataType: .float32)
        memset(mel.dataPointer, 0, mel.count * MemoryLayout<Float>.size)
        _ = try predict(models.encoder, ["mel": mel, "enc_len": try Self.int32Array(c.windowEncoderFrames)])
        let states = try MLMultiArray(
            shape: [1, c.encoderChannels as NSNumber, c.windowEncoderFrames as NSNumber], dataType: .float32)
        memset(states.dataPointer, 0, states.count * MemoryLayout<Float>.size)
        _ = try predict(models.ctcHead, ["states": states, "enc_len": try Self.int32Array(c.windowEncoderFrames)])
    }

    /// Transcribe an audio file (any format AVFoundation reads; resampled to 16 kHz mono).
    public func transcribe(audioURL: URL) throws -> NemotronTranscribeLiteResult {
        let samples = try AudioConverter().resampleAudioFile(audioURL)
        return try transcribe(audio: samples)
    }

    /// Transcribe 16 kHz mono samples into words with start times (no speakers).
    public func transcribe(audio: [Float]) throws -> NemotronTranscribeLiteResult {
        let t0 = Date()
        let (states, frames) = try encode(audio: audio)
        let tEncoder = Date().timeIntervalSince(t0)
        let t1 = Date()
        let ids = try runCtcHead(states: states, frames: frames)
        let words = Self.greedyWords(tokenIds: ids, vocabulary: models.vocabulary).map {
            NemotronTranscribeLiteWord(text: $0.text, start: $0.start)
        }
        let tCtc = Date().timeIntervalSince(t1)
        return NemotronTranscribeLiteResult(
            words: words,
            durationSeconds: Double(audio.count) / Double(NemotronTranscribeLiteConfig.sampleRate),
            timings: .init(
                encoderSeconds: tEncoder, ctcSeconds: tCtc, diarizerSeconds: nil,
                totalSeconds: Date().timeIntervalSince(t0)))
    }

    // MARK: - Encoder

    /// Encoder states for the whole recording, channel-major `[1280 * frames]` with 80 ms frames.
    func encode(audio: [Float]) throws -> (states: [Float], frames: Int) {
        let c = NemotronTranscribeLiteConfig.self
        let valid = audio.count / c.hopLength
        guard valid > 1 else { throw ASRError.invalidAudioData }
        let numFrames = valid + 1  // NeMo: 1 + N / hop with centered constant padding
        let (melFlat, _, _) = mel.computeFlatTransposed(
            audio: audio, paddingMode: .center, expectedFrameCount: numFrames)
        let normalized = Self.normalizePerFeature(
            frameMajor: melFlat, frames: numFrames, valid: valid, nMels: c.nMels)

        let total = Self.ceilDiv(valid, c.stack)
        var states = [Float](repeating: 0, count: c.encoderChannels * total)
        let input = try MLMultiArray(
            shape: [1, c.nMels as NSNumber, c.windowMelFrames as NSNumber], dataType: .float32)
        let inPtr = input.dataPointer.assumingMemoryBound(to: Float.self)
        let inRow = input.strides[1].intValue

        for w in Self.windows(length: valid, core: c.coreMelFrames, context: c.contextMelFrames) {
            memset(input.dataPointer, 0, input.count * MemoryLayout<Float>.size)
            let n = w.contextEnd - w.contextStart
            normalized.withUnsafeBufferPointer { src in
                for m in 0..<c.nMels {
                    memcpy(
                        inPtr + m * inRow, src.baseAddress! + m * numFrames + w.contextStart,
                        n * MemoryLayout<Float>.size)
                }
            }
            let out = try predict(
                models.encoder, ["mel": input, "enc_len": try Self.int32Array(Self.ceilDiv(n, c.stack))])
            guard let windowStates = out.featureValue(for: "states")?.multiArrayValue else {
                throw ASRError.processingFailed("Encoder returned no states")
            }
            let left = (w.start - w.contextStart) / c.stack
            let count = Self.ceilDiv(w.end, c.stack) - Self.ceilDiv(w.start, c.stack)
            Self.copyStates(
                windowStates, fromFrame: left, count: count, into: &states,
                frameOffset: Self.ceilDiv(w.start, c.stack), totalFrames: total)
        }
        return (states, total)
    }

    // MARK: - CTC head

    /// Argmax token ids per 40 ms frame, `[2 * frames]`.
    func runCtcHead(states: [Float], frames: Int) throws -> [Int32] {
        let c = NemotronTranscribeLiteConfig.self
        var ids = [Int32](repeating: Int32(c.blankId), count: 2 * frames)
        let input = try MLMultiArray(
            shape: [1, c.encoderChannels as NSNumber, c.windowEncoderFrames as NSNumber], dataType: .float32)
        let inPtr = input.dataPointer.assumingMemoryBound(to: Float.self)
        let inRow = input.strides[1].intValue

        for w in Self.windows(length: frames, core: c.coreEncoderFrames, context: c.contextEncoderFrames) {
            memset(input.dataPointer, 0, input.count * MemoryLayout<Float>.size)
            let n = w.contextEnd - w.contextStart
            states.withUnsafeBufferPointer { src in
                for ch in 0..<c.encoderChannels {
                    memcpy(
                        inPtr + ch * inRow, src.baseAddress! + ch * frames + w.contextStart,
                        n * MemoryLayout<Float>.size)
                }
            }
            let out = try predict(models.ctcHead, ["states": input, "enc_len": try Self.int32Array(n)])
            guard let tokens = out.featureValue(for: "token_ids")?.multiArrayValue else {
                throw ASRError.processingFailed("CTC head returned no token_ids")
            }
            let stride = tokens.strides[tokens.strides.count - 1].intValue
            let left = 2 * (w.start - w.contextStart)
            let tokPtr = tokens.dataPointer.assumingMemoryBound(to: Int32.self)
            for k in 0..<(2 * (w.end - w.start)) {
                ids[2 * w.start + k] = tokPtr[(left + k) * stride]
            }
        }
        return ids
    }

    private func predict(_ model: MLModel, _ inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        try autoreleasepool {
            try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs.mapValues { $0 as Any }))
        }
    }

    // MARK: - Host-side math (internal for tests)

    struct Window: Equatable {
        let start: Int
        let end: Int
        let contextStart: Int
        let contextEnd: Int
    }

    /// NeMo long-form windows over `[0, length)`: cores of `core` frames with up to `context` frames of
    /// context on each side, clipped to the recording.
    static func windows(length: Int, core: Int, context: Int) -> [Window] {
        stride(from: 0, to: length, by: core).map { start in
            let end = min(start + core, length)
            return Window(
                start: start, end: end, contextStart: max(start - context, 0),
                contextEnd: min(end + context, length))
        }
    }

    static func ceilDiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }

    /// NeMo `normalize_batch(per_feature)`: per mel bin, mean and unbiased std over the valid frames,
    /// std + 1e-5; frames past `valid` are zero. Input `[frames * nMels]` frame-major, output
    /// `[nMels * frames]` mel-major.
    static func normalizePerFeature(frameMajor: [Float], frames: Int, valid: Int, nMels: Int) -> [Float] {
        var out = [Float](repeating: 0, count: nMels * frames)
        for m in 0..<nMels {
            var sum = 0.0
            for t in 0..<valid { sum += Double(frameMajor[t * nMels + m]) }
            let mean = sum / Double(valid)
            var sq = 0.0
            for t in 0..<valid {
                let d = Double(frameMajor[t * nMels + m]) - mean
                sq += d * d
            }
            let std = (sq / Double(valid - 1)).squareRoot() + 1e-5
            for t in 0..<valid {
                out[m * frames + t] = Float((Double(frameMajor[t * nMels + m]) - mean) / std)
            }
        }
        return out
    }

    /// Greedy CTC → words. Repeated ids collapse, blanks drop, `<...>` tags (`<spk:N>`, `<ev:*>`, `<unk>`)
    /// are skipped. A piece starting with ▁ starts a new word, stamped with that piece's 40 ms frame.
    static func greedyWords(tokenIds: [Int32], vocabulary: [String]) -> [(start: Double, text: String)] {
        let c = NemotronTranscribeLiteConfig.self
        var words: [(start: Double, text: String)] = []
        var prev: Int32 = -1
        for (frame, id) in tokenIds.enumerated() {
            defer { prev = id }
            guard id != prev, Int(id) != c.blankId, Int(id) >= 0, Int(id) < vocabulary.count else { continue }
            let piece = vocabulary[Int(id)]
            if piece.hasPrefix("<") { continue }
            if piece.first == c.wordStart || words.isEmpty {
                words.append((Double(frame) * c.ctcFrameSeconds, String(piece.drop { $0 == c.wordStart })))
            } else {
                words[words.count - 1].text += piece
            }
        }
        return words.filter { !$0.text.isEmpty }
    }

    private static func int32Array(_ value: Int) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1], dataType: .int32)
        a[0] = NSNumber(value: Int32(value))
        return a
    }

    /// Copy `count` frames starting at `fromFrame` of a (1, 1280, 600) states output (fp16 or fp32,
    /// any strides) into channel-major `states` at `frameOffset`.
    private static func copyStates(
        _ src: MLMultiArray, fromFrame: Int, count: Int, into states: inout [Float], frameOffset: Int,
        totalFrames: Int
    ) {
        let channelStride = src.strides[1].intValue
        let frameStride = src.strides[2].intValue
        let channels = NemotronTranscribeLiteConfig.encoderChannels
        states.withUnsafeMutableBufferPointer { dst in
            switch src.dataType {
            case .float16:
                let p = src.dataPointer.assumingMemoryBound(to: UInt16.self)
                var row = [UInt16](repeating: 0, count: count)
                for ch in 0..<channels {
                    let base = ch * channelStride + fromFrame * frameStride
                    for k in 0..<count { row[k] = p[base + k * frameStride] }
                    row.withUnsafeBufferPointer {
                        Float16Conversion.toFloat32(
                            src: $0.baseAddress!, dst: dst.baseAddress! + ch * totalFrames + frameOffset, count: count)
                    }
                }
            default:
                let p = src.dataPointer.assumingMemoryBound(to: Float.self)
                for ch in 0..<channels {
                    let base = ch * channelStride + fromFrame * frameStride
                    for k in 0..<count { dst[ch * totalFrames + frameOffset + k] = p[base + k * frameStride] }
                }
            }
        }
    }
}
