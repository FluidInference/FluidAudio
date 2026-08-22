@preconcurrency import CoreML
import Accelerate
import Foundation

/// A recognized token together with its time span (seconds) in the source audio.
///
/// Mirrors the per-character word-level timestamp JSON produced by FunASR's
/// Paraformer (`ts_prediction_lfr6_standard`), e.g. `m4ai/audio/x-cut.words.json`:
/// `startTime`, `endTime`, `text`. (Silence `<sil>` and the SentencePiece
/// word-boundary token `▁` are *not* emitted — segments with empty `text` are
/// skipped during decoding.)
public struct TimestampedSegment: Sendable {
    public let startTime: Double
    public let endTime: Double
    public let text: String

    public init(startTime: Double, endTime: Double, text: String) {
        self.startTime = startTime
        self.endTime = endTime
        self.text = text
    }
}

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
        return decode(logits: logits, tokenCount: L)
    }

    // MARK: - Timestamps

    /// Transcribe with per-token timestamps (seconds). CIF integrate-and-fire
    /// gives each token's acoustic centroid; an energy-based refinement then
    /// snaps the spans to the true waveform onset/offset (see
    /// `decodeWithTimestamps`).
    public func transcribeWithTimestamps(audioURL: URL) throws -> [TimestampedSegment] {
        let converter = AudioConverter(sampleRate: Double(ParaformerConfig.sampleRate))
        return try transcribeWithTimestamps(audio: try converter.resampleAudioFile(audioURL))
    }

    public func transcribeWithTimestamps(audio: [Float]) throws -> [TimestampedSegment] {
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

        // 3) CIF alphas (CoreML) + host integrate-and-fire (decoder embeddings)
        let alphas = try runCifAlphas(encOut: encOut, validLen: T)
        let encRows = rows(of: encOut, count: T, dim: dim)
        let embeds = ParaformerCif.integrateAndFire(encRows: encRows, alphas: alphas)
        let tokenCount = min(embeds.count, ParaformerConfig.decoderMaxTokens)
        if tokenCount == 0 { return [] }

        // 4) decoder -> logits, then decode tokens + timestamps
        let logits = try runDecoder(
            encRows: encRows, validLen: T,
            embeds: Array(embeds.prefix(tokenCount)), tokenCount: tokenCount)
        return decodeWithTimestamps(
            logits: logits, tokenCount: tokenCount, alphas: alphas, audio: audio)
    }

    /// Greedy-decode the logits into tokens, then assign each a `[start, end]` time
    /// span by faithfully porting FunASR's `ts_prediction_lfr6_standard`, with an
    /// additional **energy-based boundary refinement** so the emitted spans line up
    /// with the visible waveform (e.g. CapCut/剪映) instead of the CIF acoustic
    /// centroids:
    ///
    ///   * `alphas` are **upsampled 3×** (FunASR `repeat_interleave`), then
    ///     `cif_wo_hidden` integrates them at the 20 ms resolution that yields
    ///     `pre_peak_index` — these are the per-token acoustic *centroids*.
    ///   * CIF gives the *relative* token order/spacing (centroids); the 16 kHz
    ///     RMS energy envelope (smoothed) gives the *absolute* onset/offset. We
    ///     walk tokens left to right, each starting where the previous ended (this
    ///     absorbs the CIF drift), and snap its span to the energy "on" run whose
    ///     centre is nearest the centroid. This removes the systematic half-token
    ///     shift of CIF and the heuristic `force_time_shift` used by FunASR.
    ///   * BPE continuations (`cu@@`+`t` → `cut`) are merged; the `▁` boundary
    ///     and inter-token silence are NOT emitted.
    private func decodeWithTimestamps(
        logits: MLMultiArray, tokenCount: Int, alphas: [Float], audio: [Float]
    ) -> [TimestampedSegment] {
        let upsampleRate = 3
        let timeRate = 10.0 * 6.0 / 1000.0 / Double(upsampleRate)  // 0.02 s
        let cifThreshold: Float = 1.0 - 1e-4

        // 1) Greedy decode every decoder position; ids stay 1:1 with the acoustic
        //    fire frames.
        let tokenIds = LogitsArgmax.argmaxPerFrame(logits: logits, frames: tokenCount)

        // 2) char_list: drop <blank>/<s>/</s> (keep ▁ if present).
        var charList: [String] = []
        for id in tokenIds {
            if id == ParaformerConfig.blankId
                || id == ParaformerConfig.sosId
                || id == ParaformerConfig.eosId
            {
                continue
            }
            guard let tok = models.vocabulary[id], !tok.isEmpty else { continue }
            charList.append(tok)
        }
        guard charList.count >= 1 else { return [] }

        // 3) Upsample alphas 3× and run `cif_wo_hidden` to obtain the 20 ms-
        //    resolution fire frames (FunASR `pre_peak_index`). A CIF tail alpha is
        //    appended so the final boundary fire is produced.
        var usAlphas: [Float] = []
        usAlphas.reserveCapacity(alphas.count * upsampleRate + 1)
        for a in alphas { usAlphas.append(contentsOf: Array(repeating: a, count: upsampleRate)) }
        usAlphas.append(ParaformerConfig.cifTailThreshold)
        var fireIndices = Self.cifWoHiddenFireIndices(alphas: usAlphas, threshold: cifThreshold)

        // 4) Fallback (mirrors FunASR): if the fire count doesn't line up with the
        //    token count, renormalise the alphas so their sum equals
        //    len(charList)+1 and recompute. Otherwise keep the true acoustic fires
        //    (best match for the leading word, which carries no ▁ boundary).
        if fireIndices.count != charList.count + 1 {
            let target = Float(charList.count + 1)
            let sum = usAlphas.reduce(0, +)
            let scale = target / max(sum, 1e-6)
            usAlphas = usAlphas.map { $0 * scale }
            fireIndices = Self.cifWoHiddenFireIndices(alphas: usAlphas, threshold: cifThreshold)
        }
        guard fireIndices.count >= 2 else { return [] }

        let audioEnd = Double(audio.count) / Double(ParaformerConfig.sampleRate)
        let timeOf: (Double) -> Double = { $0 * timeRate }
        // Acoustic centroids in seconds (no force_time_shift — energy refinement
        // replaces it).
        let centroids: [Double] = fireIndices.map { timeOf(Double($0)) }

        // 5) Build the token layout via sequential energy forced-alignment.
        //    CIF gives the *relative* token order and spacing (centroids); the 16 kHz
        //    RMS envelope (smoothed) gives the *absolute* onset/offset. We walk
        //    tokens left to right, each starting where the previous ended (this
        //    absorbs the CIF drift), and snap its span to the energy "on" run whose
        //    centre is nearest the centroid. No silence entries are emitted.
        let rawEnv = Self.energyEnvelope(audio: audio, sampleRate: Double(ParaformerConfig.sampleRate), hopSec: 0.01)
        let env = Self.smooth(rawEnv, window: 3)
        let floor = env.isEmpty ? 0 : Self.percentile(env, q: 0.1)
        let energyThreshold = max(floor * 2.5, 1e-4)
        let minRun = 3  // frames (~30 ms) — reject single-frame noise spikes
        let n = min(charList.count, centroids.count - 1)
        // Typical inter-token spacing; bounds the final token so trailing silence
        // (audioEnd - centroid) can't inflate its span to the end of the file.
        let spacings = (1..<max(n, 1)).map { Float(centroids[$0] - centroids[$0 - 1]) }
        let typicalDur = Double(spacings.isEmpty ? 0.3 : Self.percentile(spacings, q: 0.5))
        var raw: [(text: String, start: Double, end: Double)] = []
        var cursor: Double = 0.0
        for i in 0..<n {
            let dur =
                (i < n - 1)
                ? (centroids[i + 1] - centroids[i])
                : min(audioEnd - centroids[i], max(typicalDur * 2, 0.4))
            let searchEnd = min(audioEnd, centroids[i] + dur * 1.5 + 0.15)
            let span = Self.energySpan(
                from: cursor, to: searchEnd, centroid: centroids[i],
                env: env, hopSec: 0.01, threshold: energyThreshold, minRun: minRun)
            let (s, e): (Double, Double)
            if let span {
                s = span.0
                e = span.1
            } else {
                // No energy run found (very quiet token): advance by the expected
                // duration so we don't get stuck, and keep a plausible span.
                s = cursor
                e = min(audioEnd, cursor + max(dur, 0.1))
            }
            cursor = e
            raw.append((text: charList[i], start: s, end: e))
        }

        // 6) Emit, merging BPE continuations and stripping the `▁` boundary.
        //    Empty-text (`▁` / silence) segments are skipped entirely.
        var segments: [TimestampedSegment] = []
        var i = 0
        while i < raw.count {
            let item = raw[i]
            var text = item.text
            var end = item.end
            while text.hasSuffix("@@") {
                text = String(text.dropLast(2))
                i += 1
                if i < raw.count {
                    let piece =
                        raw[i].text.hasPrefix(ASRConstants.sentencePieceWordBoundary)
                        ? String(raw[i].text.dropFirst()) : raw[i].text
                    text += piece
                    end = raw[i].end
                }
            }
            if text.hasPrefix(ASRConstants.sentencePieceWordBoundary) {
                text = String(text.dropFirst())
            }
            if !text.isEmpty {
                segments.append(
                    TimestampedSegment(startTime: max(0, item.start), endTime: end, text: text))
            }
            i += 1
        }
        return segments
    }

    /// CIF integrate-and-fire *without* a hidden state (FunASR `cif_wo_hidden`):
    /// returns every frame index at which the running sum of `alphas` crosses
    /// `threshold`. Operates on the upsampled (20 ms) alphas.
    private static func cifWoHiddenFireIndices(alphas: [Float], threshold: Float) -> [Int] {
        var integrate: Float = 0
        var fires: [Int] = []
        for t in 0..<alphas.count {
            integrate += alphas[t]
            if integrate >= threshold {
                fires.append(t)
                integrate -= 1.0
            }
        }
        return fires
    }

    /// RMS energy envelope of `audio` at `hopSec` resolution (seconds). Used to
    /// snap token boundaries to the visible waveform onset/offset.
    private static func energyEnvelope(audio: [Float], sampleRate: Double, hopSec: Double) -> [Float] {
        let hop = max(1, Int(hopSec * sampleRate))
        guard audio.count > hop else { return [] }
        var env: [Float] = []
        env.reserveCapacity(audio.count / hop + 1)
        var idx = 0
        while idx + hop <= audio.count {
            var sum: Float = 0
            for j in 0..<hop {
                let s = audio[idx + j]
                sum += s * s
            }
            env.append(sqrt(sum / Float(hop)))
            idx += hop
        }
        return env
    }

    /// `q`-th percentile (0..1) of `values`, used to estimate a noise floor.
    private static func percentile(_ values: [Float], q: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let pos = max(0, min(sorted.count - 1, Int(Float(sorted.count - 1) * q)))
        return sorted[pos]
    }

    /// Moving-average smoothing of `x` (window frames, centred).
    private static func smooth(_ x: [Float], window: Int) -> [Float] {
        guard window > 1, x.count > window else { return x }
        var out = x
        let half = window / 2
        for i in 0..<x.count {
            let lo = max(0, i - half)
            let hi = min(x.count - 1, i + half)
            var sum: Float = 0
            for k in lo...hi { sum += x[k] }
            out[i] = sum / Float(hi - lo + 1)
        }
        return out
    }

    /// Within `[from, to]`, find every RMS-energy "on" run (consecutive frames
    /// above `threshold`, length >= `minRun`) and return the `(start, end)` of the
    /// run whose centre is nearest `centroid`. Returns `nil` if no run qualifies.
    /// Used for sequential forced-alignment of token boundaries to the waveform.
    private static func energySpan(
        from: Double, to: Double, centroid: Double, env: [Float],
        hopSec: Double, threshold: Float, minRun: Int
    ) -> (Double, Double)? {
        guard !env.isEmpty, hopSec > 0, to > from, minRun > 0 else { return nil }
        let i0 = max(0, Int(from / hopSec))
        let i1 = min(env.count - 1, max(i0, Int(to / hopSec)))
        guard i1 >= i0 else { return nil }

        // Collect qualifying runs.
        var runs: [(Int, Int)] = []
        var j = i0
        while j <= i1 {
            if env[j] > threshold {
                var k = j
                while k <= i1, env[k] > threshold { k += 1 }
                if k - j >= minRun { runs.append((j, k - 1)) }
                j = k
            } else {
                j += 1
            }
        }
        guard !runs.isEmpty else { return nil }

        // Pick the run whose centre is closest to the centroid.
        let ci = Int(centroid / hopSec)
        var best = runs[0]
        var bestD = abs((runs[0].0 + runs[0].1) - 2 * ci)
        for r in runs[1...] {
            let d = abs((r.0 + r.1) - 2 * ci)
            if d < bestD {
                bestD = d
                best = r
            }
        }
        return (Double(best.0) * hopSec, Double(best.1) * hopSec)
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

    private func decode(logits: MLMultiArray, tokenCount: Int) -> String {
        let ids = LogitsArgmax.argmaxPerFrame(logits: logits, frames: tokenCount)
        var pieces: [String] = []
        for best in ids {
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
        // Use the real row stride: CoreML pads rows for ANE alignment, so the
        // stride between consecutive frames may exceed `dim`.
        let frameStride = arr.strides[1].intValue
        if arr.dataType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float32.self)
            for t in 0..<count {
                out.append(Array(UnsafeBufferPointer(start: p + t * frameStride, count: dim)))
            }
        } else {
            // Address fp16 storage as raw bit patterns: Swift's `Float16` is
            // unavailable on macOS x86_64.
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for t in 0..<count {
                var r = [Float](repeating: 0, count: dim)
                r.withUnsafeMutableBufferPointer { dst in
                    Float16Conversion.toFloat32(
                        src: p + t * frameStride, dst: dst.baseAddress!, count: dim)
                }
                out.append(r)
            }
        }
        return out
    }
}
