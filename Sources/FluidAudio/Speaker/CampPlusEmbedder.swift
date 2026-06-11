@preconcurrency import CoreML
import Foundation

/// CAM++ speaker-embedding extractor: audio -> 192-d L2-normalized embedding.
///
/// Use cosine similarity between embeddings for speaker verification / diarization
/// clustering. Pipeline: waveform -> [Preprocessor fp32/CPU] -> fbank [1,T,80]
/// -> [CAM++ fp16/ANE] -> [1,192] -> L2 normalize.
public actor CampPlusEmbedder {

    public static let embeddingDim = 192
    private static let waveformScale: Float = 32_768.0  // kaldi int16 range

    private let models: CampPlusModels
    private static let logger = AppLogger(category: "CampPlusEmbedder")

    public init(models: CampPlusModels) {
        self.models = models
    }

    public static func load(progressHandler: DownloadUtils.ProgressHandler? = nil) async throws -> CampPlusEmbedder {
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

    /// Cosine similarity of two L2-normalized embeddings.
    public nonisolated static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }
}
