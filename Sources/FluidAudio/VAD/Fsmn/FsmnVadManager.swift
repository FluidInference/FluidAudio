@preconcurrency import CoreML
import Foundation

/// A detected speech segment, in milliseconds.
public struct FsmnVadSegment: Sendable, Equatable {
    public let startMs: Int
    public let endMs: Int
}

/// FSMN-VAD voice activity detection: audio -> speech segments.
///
/// Pipeline: waveform -> [Preprocessor fp32/CPU] -> 400-d features
///   -> [FSMN fp16/ANE, enumerated buckets] -> per-frame scores (col 0 = silence prob)
///   -> host decision (window-detector hysteresis + silence->endpoint) -> [start_ms, end_ms].
///
/// Audio longer than the largest bucket is processed in ~30 s chunks; the per-frame
/// silence probabilities are concatenated and the decision runs once over all frames.
public actor FsmnVadManager {

    // Enumerated scorer buckets (post-LFR frames; matches the converted model).
    private static let buckets = [512, 1024, 2048, 3072]
    private static let featureDim = 400
    private static let waveformScale: Float = 32_768.0

    // Decision params (derived from FunASR vad_opts; 10 ms frames).
    private static let silenceThreshold: Float = 0.2  // GetFrameState: speech if silence_prob <= 0.2
    private static let windowFrames = 20  // window_size_ms 200 / 10
    private static let silToSpeech = 15  // sil_to_speech_time 150 / 10
    private static let speechToSil = 15  // speech_to_sil_time 150 / 10
    private static let maxEndSilenceFrames = 80  // max_end_silence_time 800 / 10
    private static let lookbackFrames = 20  // lookback_time_start_point 200 / 10
    private static let lookaheadFrames = 10  // lookahead_time_end_point 100 / 10
    private static let maxSegmentFrames = 6000  // max_single_segment_time 60000 / 10
    private static let frameMs = 10

    private let models: FsmnVadModels
    private static let logger = AppLogger(category: "FsmnVadManager")

    public init(models: FsmnVadModels) {
        self.models = models
    }

    public static func load(progressHandler: DownloadUtils.ProgressHandler? = nil) async throws -> FsmnVadManager {
        FsmnVadManager(models: try await FsmnVadModels.downloadAndLoad(progressHandler: progressHandler))
    }

    public func detect(audioURL: URL) throws -> [FsmnVadSegment] {
        let converter = AudioConverter(sampleRate: 16_000)
        return try detect(audio: try converter.resampleAudioFile(audioURL))
    }

    public func detect(audio: [Float]) throws -> [FsmnVadSegment] {
        let silence = try silenceProbabilities(audio: audio)
        return decide(silence: silence)
    }

    // MARK: - Scoring (chunked)

    /// Per-frame silence probability over the whole audio (concatenated across chunks).
    private func silenceProbabilities(audio: [Float]) throws -> [Float] {
        // ~30 s chunks (largest bucket); samples ≈ frames * 160.
        let chunkSamples = (Self.buckets.last! - Self.windowFrames) * 160
        var sil: [Float] = []
        var offset = 0
        while offset < audio.count {
            let end = min(offset + chunkSamples, audio.count)
            let chunk = Array(audio[offset..<end])
            sil.append(contentsOf: try chunkSilence(chunk))
            offset = end
        }
        return sil
    }

    private func chunkSilence(_ audio: [Float]) throws -> [Float] {
        let n = audio.count
        let wav = try MLMultiArray(shape: [1, n as NSNumber], dataType: .float32)
        let wp = wav.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<n { wp[i] = audio[i] * Self.waveformScale }
        let feats = try models.preprocessor.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["waveform": MLFeatureValue(multiArray: wav)]))
        guard let f = feats.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("FSMN-VAD preprocessor produced no `features`")
        }
        let t = f.shape[1].intValue
        if t == 0 { return [] }
        let bucket = Self.buckets.first(where: { $0 >= t }) ?? Self.buckets.last!
        let speech = try MLMultiArray(shape: [1, bucket as NSNumber, Self.featureDim as NSNumber], dataType: .float32)
        let sp = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sp, 0, bucket * Self.featureDim * MemoryLayout<Float32>.size)
        let count = t * Self.featureDim
        if f.dataType == .float32 {
            memcpy(sp, f.dataPointer, count * MemoryLayout<Float32>.size)
        } else {
            for i in 0..<count { sp[i] = f[i].floatValue }
        }
        let out = try models.scorer.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["feats": MLFeatureValue(multiArray: speech)]))
        guard let scores = out.featureValue(for: "scores")?.multiArrayValue else {
            throw ASRError.processingFailed("FSMN-VAD scorer produced no `scores`")
        }
        let vocab = scores.shape[2].intValue
        var sil = [Float](repeating: 0, count: t)
        if scores.dataType == .float32 {
            let p = scores.dataPointer.assumingMemoryBound(to: Float32.self)
            for frame in 0..<t { sil[frame] = p[frame * vocab] }  // col 0 = silence prob
        } else {
            for frame in 0..<t { sil[frame] = scores[[0, frame as NSNumber, 0]].floatValue }
        }
        return sil
    }

    // MARK: - Decision (port of FunASR FsmnVADStreaming)

    private func decide(silence: [Float]) -> [FsmnVadSegment] {
        let T = silence.count
        var win = [Int](repeating: 0, count: Self.windowFrames)
        var pos = 0
        var winSum = 0
        var preSpeech = false
        var inSeg = false
        var segStart = 0
        var contSil = 0
        var segs: [FsmnVadSegment] = []

        func close(at frame: Int) {
            segs.append(FsmnVadSegment(startMs: segStart * Self.frameMs, endMs: frame * Self.frameMs))
            inSeg = false
        }

        for t in 0..<T {
            let cur = silence[t] <= Self.silenceThreshold ? 1 : 0
            winSum -= win[pos]
            winSum += cur
            win[pos] = cur
            pos = (pos + 1) % Self.windowFrames
            if !preSpeech && winSum >= Self.silToSpeech {
                preSpeech = true
                if !inSeg {
                    inSeg = true
                    segStart = max(0, t - Self.silToSpeech - Self.lookbackFrames)
                    contSil = 0
                }
            } else if preSpeech && winSum <= Self.speechToSil {
                preSpeech = false
            }
            if inSeg && !preSpeech { contSil += 1 } else { contSil = 0 }
            if inSeg && contSil >= Self.maxEndSilenceFrames {
                close(at: t - Self.maxEndSilenceFrames + Self.lookaheadFrames)
            } else if inSeg && (t - segStart) >= Self.maxSegmentFrames {
                close(at: t)
                preSpeech = false
            }
        }
        if inSeg { close(at: T) }
        return segs
    }
}
