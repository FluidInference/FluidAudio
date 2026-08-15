@preconcurrency import CoreML
import Foundation

/// Manager for Paraformer-large (zh) transcription.
///
/// Pipeline: waveform -> [Preprocessor fp32/CPU] -> features
///   -> [Encoder fp16/ANE] -> enc_out
///   -> [CifAlphas fp16/ANE] -> alphas -> [host integrate-and-fire] -> acoustic_embeds, L
///   -> [Decoder fp16/ANE] -> logits -> argmax -> drop sos/eos/blank -> CharTokenizer.
public actor ParaformerManager {

    private let models: ParaformerModels
    private static let logger = AppLogger(category: "ParaformerManager")

    public init(models: ParaformerModels) {
        self.models = models
    }

    public static func load(
        precision: ParaformerPrecision = .fp16, progressHandler: ProgressHandler? = nil
    ) async throws -> ParaformerManager {
        ParaformerManager(
            models: try await ParaformerModels.downloadAndLoad(precision: precision, progressHandler: progressHandler))
    }

    public func transcribe(audioURL: URL) throws -> String {
        let converter = AudioConverter(sampleRate: Double(ParaformerConfig.sampleRate))
        return try transcribe(audio: try converter.resampleAudioFile(audioURL))
    }

    public func transcribe(audio: [Float]) throws -> String {
        let dim = ParaformerConfig.encoderDim
        // 1) preprocessor: waveform -> features [1, T, 560]
        let features = try runPreprocessor(audio: audio)
        var T = features.shape[1].intValue
        if T > ParaformerConfig.decoderEncFrames {
            Self.logger.warning("audio too long (\(T) frames); truncating to \(ParaformerConfig.decoderEncFrames)")
            T = ParaformerConfig.decoderEncFrames
        }

        // 2) encoder (padded to bucket) -> enc_out
        let bucket = ParaformerConfig.pickEncoderBucket(forFrames: T)
        let encOut = try runEncoder(features: features, validLen: T, bucket: bucket)

        // 3) CIF alphas (CoreML) + host integrate-and-fire
        let alphas = try runCifAlphas(encOut: encOut, validLen: T)
        let encRows = rows(of: encOut, count: T, dim: dim)
        let embeds = ParaformerCif.integrateAndFire(encRows: encRows, alphas: alphas)
        let L = min(embeds.count, ParaformerConfig.decoderMaxTokens)
        if L == 0 { return "" }

        // 4) decoder -> logits, then greedy decode
        let logits = try runDecoder(encRows: encRows, validLen: T, embeds: embeds, tokenCount: L)
        return decode(logits: logits, tokenCount: L, dim: dim)
    }

    // MARK: - Stages

    private func runPreprocessor(audio: [Float]) throws -> MLMultiArray {
        let n = audio.count
        let wav = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let p = wav.dataPointer.assumingMemoryBound(to: Float32.self)
        let scale = ParaformerConfig.waveformScale
        for i in 0..<n { p[i] = audio[i] * scale }
        let out = try models.preprocessor.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["waveform": MLFeatureValue(multiArray: wav)]))
        guard let f = out.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("Paraformer preprocessor produced no `features`")
        }
        return f
    }

    private func runEncoder(features: MLMultiArray, validLen: Int, bucket: Int) throws -> MLMultiArray {
        let dim = ParaformerConfig.featureDim
        let speech = try MLMultiArray(shape: [1, bucket as NSNumber, dim as NSNumber], dataType: .float32)
        let sp = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sp, 0, bucket * dim * MemoryLayout<Float32>.size)
        let count = validLen * dim
        if features.dataType == .float32 {
            memcpy(sp, features.dataPointer, count * MemoryLayout<Float32>.size)
        } else {
            for i in 0..<count { sp[i] = features[i].floatValue }
        }
        let len = try MLMultiArray(shape: [1], dataType: .int32)
        len[0] = NSNumber(value: validLen)
        let out = try models.encoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "speech": MLFeatureValue(multiArray: speech),
                "speech_lengths": MLFeatureValue(multiArray: len),
            ]))
        guard let e = out.featureValue(for: "enc_out")?.multiArrayValue else {
            throw ASRError.processingFailed("Paraformer encoder produced no `enc_out`")
        }
        return e
    }

    private func runCifAlphas(encOut: MLMultiArray, validLen: Int) throws -> [Float] {
        let out = try models.cifAlphas.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["enc_out": MLFeatureValue(multiArray: encOut)]))
        guard let a = out.featureValue(for: "alphas")?.multiArrayValue else {
            throw ASRError.processingFailed("Paraformer CifAlphas produced no `alphas`")
        }
        var alphas = [Float](repeating: 0, count: validLen)
        if a.dataType == .float32 {
            let p = a.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<validLen { alphas[t] = p[t] }
        } else {
            for t in 0..<validLen { alphas[t] = a[[0, t as NSNumber]].floatValue }
        }
        return alphas
    }

    private func runDecoder(
        encRows: [[Float]], validLen: Int, embeds: [[Float]], tokenCount: Int
    ) throws -> MLMultiArray {
        let dim = ParaformerConfig.encoderDim
        let Tb = ParaformerConfig.decoderEncFrames
        let Lb = ParaformerConfig.decoderMaxTokens

        let enc = try MLMultiArray(shape: [1, Tb as NSNumber, dim as NSNumber], dataType: .float32)
        let ep = enc.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(ep, 0, Tb * dim * MemoryLayout<Float32>.size)
        for t in 0..<validLen { for d in 0..<dim { ep[t * dim + d] = encRows[t][d] } }

        let ac = try MLMultiArray(shape: [1, Lb as NSNumber, dim as NSNumber], dataType: .float32)
        let ap = ac.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(ap, 0, Lb * dim * MemoryLayout<Float32>.size)
        for l in 0..<tokenCount { for d in 0..<dim { ap[l * dim + d] = embeds[l][d] } }

        let elen = try MLMultiArray(shape: [1], dataType: .int32)
        elen[0] = NSNumber(value: validLen)
        let tn = try MLMultiArray(shape: [1], dataType: .int32)
        tn[0] = NSNumber(value: tokenCount)
        let out = try models.decoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "enc": MLFeatureValue(multiArray: enc),
                "elen": MLFeatureValue(multiArray: elen),
                "ac": MLFeatureValue(multiArray: ac),
                "tn": MLFeatureValue(multiArray: tn),
            ]))
        guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Paraformer decoder produced no `logits`")
        }
        return logits
    }

    private func decode(logits: MLMultiArray, tokenCount: Int, dim: Int) -> String {
        let vocab = logits.shape[2].intValue
        var pieces: [String] = []
        let useFast = logits.dataType == .float32
        let p = useFast ? logits.dataPointer.assumingMemoryBound(to: Float32.self) : nil
        for t in 0..<tokenCount {
            var best = 0
            var bestVal: Float = -.infinity
            for v in 0..<vocab {
                let x = useFast ? p![t * vocab + v] : logits[[0, t as NSNumber, v as NSNumber]].floatValue
                if x > bestVal {
                    bestVal = x
                    best = v
                }
            }
            if best == ParaformerConfig.blankId || best == ParaformerConfig.sosId || best == ParaformerConfig.eosId {
                continue
            }
            if let tok = models.vocabulary[best] { pieces.append(tok) }
        }
        // CharTokenizer: join chars; SentencePiece word-boundary -> space if present.
        return pieces.joined()
            .replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private func rows(of arr: MLMultiArray, count: Int, dim: Int) -> [[Float]] {
        var out: [[Float]] = []
        out.reserveCapacity(count)
        if arr.dataType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<count {
                out.append(Array(UnsafeBufferPointer(start: p + t * dim, count: dim)))
            }
        } else {
            for t in 0..<count {
                var r = [Float](repeating: 0, count: dim)
                for d in 0..<dim { r[d] = arr[[0, t as NSNumber, d as NSNumber]].floatValue }
                out.append(r)
            }
        }
        return out
    }
}
