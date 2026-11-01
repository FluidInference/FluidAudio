@preconcurrency import CoreML
import Foundation

/// CAM++ speaker-embedding extractor: audio -> 192-d L2-normalized embedding.
///
/// Use cosine similarity between embeddings for speaker verification / diarization
/// clustering. Pipeline: waveform -> [Preprocessor fp32/CPU] -> fbank [1,T,80]
/// -> per-utterance mean subtraction -> [CAM++ fp32] -> [1,192] -> L2 normalize.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor CampPlusEmbedder {

    public static let embeddingDim = 192
    private static let waveformScale: Float = 32_768.0  // kaldi int16 range

    private let models: CampPlusModels
    private static let logger = AppLogger(category: "CampPlusEmbedder")

    public init(models: CampPlusModels) {
        self.models = models
    }

    public static func load(progressHandler: ProgressHandler? = nil) async throws -> CampPlusEmbedder {
        CampPlusEmbedder(models: try await CampPlusModels.downloadAndLoad(progressHandler: progressHandler))
    }

    /// 16 kHz mono file -> 192-d L2-normalized embedding.
    public func embed(audioURL: URL) throws -> [Float] {
        let converter = AudioConverter(sampleRate: 16_000)
        return try embed(audio: try converter.resampleAudioFile(audioURL))
    }

    /// 16 kHz mono samples ([-1, 1]) -> 192-d L2-normalized embedding.
    public func embed(audio: [Float]) throws -> [Float] {
        let n = audio.count
        let wav = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let p = wav.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<n { p[i] = audio[i] * Self.waveformScale }
        let feats = try models.preprocessor.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["waveform": MLFeatureValue(multiArray: wav)]))
        guard let fbank = feats.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("CAM++ preprocessor produced no `features`")
        }
        try Self.subtractUtteranceMean(fbank)
        let out = try models.model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["feats": MLFeatureValue(multiArray: fbank)]))
        guard let emb = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw ASRError.processingFailed("CAM++ produced no `embedding`")
        }
        var v = [Float](repeating: 0, count: emb.count)
        if emb.dataType == .float32 {
            let ep = emb.dataPointer.assumingMemoryBound(to: Float32.self)
            for i in 0..<emb.count { v[i] = ep[i] }
        } else {
            for i in 0..<emb.count { v[i] = emb[i].floatValue }
        }
        let norm = max(sqrt(v.reduce(0) { $0 + $1 * $1 }), 1e-9)
        return v.map { $0 / norm }
    }

    /// Subtracts each mel band's mean over the utterance from a `[1, T, 80]` fbank, in place.
    ///
    /// CAM++ is trained on mean-normalized fbank (3D-Speaker `FBank(mean_nor: true)`, FunASR
    /// `extract_feature`), and neither CoreML graph normalizes: the preprocessor ends at `log` and
    /// CAM++ feeds `feats` straight into its first conv. Without it the embedding of an AMI segment
    /// sits at a median cosine of 0.74 to the PyTorch reference, and different speakers score
    /// closer together (#985).
    static func subtractUtteranceMean(_ fbank: MLMultiArray) throws {
        guard fbank.dataType == .float32, fbank.shape.count == 3 else {
            throw ASRError.processingFailed(
                "CAM++ preprocessor features: expected float32 [1, T, 80], got \(fbank.dataType) \(fbank.shape)")
        }
        let frames = fbank.shape[1].intValue
        let bands = fbank.shape[2].intValue
        guard frames > 0 else { return }
        fbank.withUnsafeMutableBytes { raw, strides in
            guard let base = raw.bindMemory(to: Float32.self).baseAddress else { return }
            let frameStride = strides[1]
            let bandStride = strides[2]
            for band in 0..<bands {
                let column = base + band * bandStride
                var sum: Double = 0
                for frame in 0..<frames { sum += Double(column[frame * frameStride]) }
                let mean = Float32(sum / Double(frames))
                for frame in 0..<frames { column[frame * frameStride] -= mean }
            }
        }
    }

    /// Cosine similarity of two L2-normalized embeddings.
    public nonisolated static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
