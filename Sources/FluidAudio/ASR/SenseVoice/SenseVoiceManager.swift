@preconcurrency import CoreML
import Foundation

/// Manager for SenseVoiceSmall transcription.
///
/// Pipeline: waveform → [Preprocessor fp32/CPU] → 560-d features → pad to the
/// smallest enumerated encoder bucket → [encoder+CTC fp16/ANE] → greedy CTC
/// decode (drop blank 0, collapse) → SentencePiece detokenize → strip the
/// leading `<|lang|><|emo|><|event|><|itn|>` tags.
public actor SenseVoiceManager {

    private let models: SenseVoiceModels
    private let language: Int32
    private let textNorm: Int32
    private static let logger = AppLogger(category: "SenseVoiceManager")

    public init(
        models: SenseVoiceModels,
        language: Int32 = SenseVoiceConfig.defaultLanguage,
        textNorm: Int32 = SenseVoiceConfig.defaultTextNorm
    ) {
        self.models = models
        self.language = language
        self.textNorm = textNorm
    }

    /// Load models from the default cache (downloading if needed), then build a manager.
    public static func load(
        precision: SenseVoiceEncoderPrecision = .fp16,
        progressHandler: ProgressHandler? = nil
    ) async throws -> SenseVoiceManager {
        let models = try await SenseVoiceModels.downloadAndLoad(
            precision: precision, progressHandler: progressHandler)
        return SenseVoiceManager(models: models)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL) throws -> String {
        let converter = AudioConverter(sampleRate: Double(SenseVoiceConfig.sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples)
    }

    /// Transcribe 16 kHz mono float samples (in [-1, 1]).
    public func transcribe(audio: [Float]) throws -> String {
        let features = try runPreprocessor(audio: audio)
        let (logits, validFrames) = try runEncoder(features: features)
        return decode(logits: logits, validFrames: validFrames)
    }

    // MARK: - Pipeline

    /// waveform [1, N] (scaled to int16 range) → features [1, T, 560].
    private func runPreprocessor(audio: [Float]) throws -> MLMultiArray {
        let n = audio.count
        let waveform = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let scale = SenseVoiceConfig.waveformScale
        let wptr = waveform.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<n { wptr[i] = audio[i] * scale }

        let input = try MLDictionaryFeatureProvider(
            dictionary: ["waveform": MLFeatureValue(multiArray: waveform)])
        let out = try models.preprocessor.prediction(from: input)
        guard let features = out.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice preprocessor produced no `features`")
        }
        return features
    }

    /// features [1, T, 560] → (ctc_logits [1, bucket+4, V], validFrames = 4 + T).
    private func runEncoder(features: MLMultiArray) throws -> (MLMultiArray, Int) {
        let dim = SenseVoiceConfig.featureDim
        var t = features.shape[1].intValue
        if t > SenseVoiceConfig.maxFrames {
            Self.logger.warning("Audio exceeds max length; truncating \(t) → \(SenseVoiceConfig.maxFrames) frames")
            t = SenseVoiceConfig.maxFrames
        }
        let bucket = SenseVoiceConfig.pickBucket(forFrames: t)

        // Zero-padded [1, bucket, 560] with the first T feature frames copied in.
        let speech = try MLMultiArray(shape: [1, bucket as NSNumber, dim as NSNumber], dataType: .float32)
        let sptr = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sptr, 0, bucket * dim * MemoryLayout<Float32>.size)
        let count = t * dim
        if features.dataType == .float32 {
            memcpy(sptr, features.dataPointer, count * MemoryLayout<Float32>.size)
        } else {
            for i in 0..<count { sptr[i] = features[i].floatValue }
        }

        let lengths = try MLMultiArray(shape: [1], dataType: .int32)
        lengths[0] = NSNumber(value: t)
        let lang = try MLMultiArray(shape: [1], dataType: .int32)
        lang[0] = NSNumber(value: language)
        let tn = try MLMultiArray(shape: [1], dataType: .int32)
        tn[0] = NSNumber(value: textNorm)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "speech": MLFeatureValue(multiArray: speech),
            "speech_lengths": MLFeatureValue(multiArray: lengths),
            "language": MLFeatureValue(multiArray: lang),
            "textnorm": MLFeatureValue(multiArray: tn),
        ])
        let out = try models.encoder.prediction(from: input)
        guard let logits = out.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice encoder produced no `ctc_logits`")
        }
        return (logits, SenseVoiceConfig.numQueryTokens + t)
    }

    /// Greedy CTC over the first `validFrames` (drop blank 0, collapse repeats),
    /// detokenize, then strip the `<|...|>` meta tags.
    private func decode(logits: MLMultiArray, validFrames: Int) -> String {
        let vocab = logits.shape[2].intValue
        let frames = min(validFrames, logits.shape[1].intValue)
        var ids: [Int] = []
        var prev = -1

        func appendArgmax(frameBase: (Int) -> Float) {
            var best = 0
            var bestVal = frameBase(0)
            for v in 1..<vocab {
                let x = frameBase(v)
                if x > bestVal {
                    bestVal = x
                    best = v
                }
            }
            if best != SenseVoiceConfig.blankId && best != prev { ids.append(best) }
            prev = best
        }

        if logits.dataType == .float32 {
            let p = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<frames {
                let base = t * vocab
                appendArgmax { p[base + $0] }
            }
        } else {
            for t in 0..<frames {
                appendArgmax { logits[[0, t as NSNumber, $0 as NSNumber]].floatValue }
            }
        }

        let raw = decodeCtcTokenIds(ids, vocabulary: models.vocabulary)
        return
            raw
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }
}
