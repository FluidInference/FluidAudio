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
/// Audio longer than the largest bucket is processed in ~30 s chunks overlapped by the
/// frontend context (fbank window + LFR padding); each chunk's pad-degraded leading frames
/// are trimmed so the concatenated probabilities stay on the absolute 10 ms grid, and the
/// decision runs once over all frames.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor FsmnVadManager {

    // Enumerated scorer buckets (post-LFR frames; matches the converted model).
    private static let buckets = [512, 1024, 2048, 3072]
    private static let featureDim = 400
    private static let waveformScale: Float = 32_768.0

    // Preprocessor geometry (16 kHz): valid 400-sample fbank window with 160-sample hop,
    // then 2 replicate-padded frames + valid width-5 LFR -> one output frame per 10 ms hop.
    static let hopSamples = 160
    static let fbankWindowSamples = 400
    static let lfrPadFrames = 2
    private static let lfrWidth = 5

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

    public static func load(progressHandler: ProgressHandler? = nil) async throws -> FsmnVadManager {
        FsmnVadManager(models: try await FsmnVadModels.downloadAndLoad(progressHandler: progressHandler))
    }

    public func detect(audioURL: URL) throws -> [FsmnVadSegment] {
        let audio = try autoreleasepool { () -> [Float] in
            let converter = AudioConverter(sampleRate: 16_000)
            return try converter.resampleAudioFile(audioURL)
        }
        return try detect(audio: audio)
    }

    public func detect(audio: [Float]) throws -> [FsmnVadSegment] {
        let silence = try silenceProbabilities(audio: audio)
        return decide(silence: silence)
    }

    // MARK: - Scoring (chunked)

    /// Per-frame silence probability over the whole audio (concatenated across chunks).
    private func silenceProbabilities(audio: [Float]) throws -> [Float] {
        try Self.concatenateChunks(sampleCount: audio.count) { range in
            // Drain CoreML's autoreleased MLMultiArrays per chunk so memory stays
            // bounded on long audio; otherwise they accumulate across the whole file.
            try autoreleasepool { try chunkSilence(Array(audio[range])) }
        }
    }

    /// Chunk schedule that keeps concatenated output frames on the absolute 10 ms grid.
    ///
    /// The valid frontend convolutions make a chunk yield fewer frames than the audio it
    /// spans (a full chunk consumes 30,520 ms but yields 30,480 ms), so back-to-back
    /// chunking drifts 40 ms per boundary. Instead, chunk k starts at sample
    /// (framesSoFar - lfrPadFrames) * hopSamples: its first `lfrPadFrames` outputs are
    /// computed against replicate padding and duplicate frames the previous chunk already
    /// produced with real context, so they are dropped and output frame i of the chunk
    /// lands exactly at absolute frame start/hopSamples + i.
    static func concatenateChunks(
        sampleCount: Int, score: (Range<Int>) throws -> [Float]
    ) rethrows -> [Float] {
        // ~30 s chunks (largest bucket); samples ≈ frames * 160.
        let chunkSamples = (buckets.last! - windowFrames) * hopSamples
        var sil: [Float] = []
        var start = 0
        while start < sampleCount {
            let end = min(start + chunkSamples, sampleCount)
            let chunkSil = try score(start..<end)
            let keepFrom = start == 0 ? 0 : lfrPadFrames
            // Only the final (short) chunk can come up empty or pad-only.
            guard chunkSil.count > keepFrom else { break }
            sil.append(contentsOf: chunkSil[keepFrom...])
            if end == sampleCount { break }
            start = (sil.count - lfrPadFrames) * hopSamples
        }
        return sil
    }

    /// Post-LFR frame count the preprocessor yields for a chunk of `samples` samples
    /// (valid fbank conv, then +lfrPadFrames replicate pad and valid width-5 LFR).
    static func lfrFrameCount(samples: Int) -> Int {
        guard samples >= fbankWindowSamples else { return 0 }
        let fbankFrames = (samples - fbankWindowSamples) / hopSamples + 1
        return max(0, fbankFrames + lfrPadFrames - lfrWidth + 1)
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
