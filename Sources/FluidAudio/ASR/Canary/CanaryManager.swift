@preconcurrency import CoreML
import Foundation

/// Manager for NVIDIA Canary-1B-v2 transcription (attention encoder-decoder).
///
/// Pipeline: waveform → [Preprocessor fp32/CPU] mel → [Encoder int4/ANE] →
/// transpose to [1, T, D] → greedy autoregressive loop ([Decoder] → last hidden
/// → [Projection] → argmax until EOS) → SentencePiece detokenize.
///
/// The decoder carries no KV cache: each step re-runs the full `[1, 256]` token
/// sequence (matches the converted CoreML model). The 15 s window is fixed; audio
/// longer than 15 s is truncated (chunking is a future addition).
public actor CanaryManager {

    private let models: CanaryModels
    private let prompt: [Int32]
    private static let logger = AppLogger(category: "CanaryManager")

    public init(models: CanaryModels, prompt: [Int32] = CanaryConfig.promptEnTranscribePnc) {
        self.models = models
        self.prompt = prompt
    }

    /// Load models from the default cache (downloading if needed), then build a manager.
    public static func load(
        precision: CanaryPrecision = .int4,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CanaryManager {
        let models = try await CanaryModels.downloadAndLoad(precision: precision, progressHandler: progressHandler)
        return CanaryManager(models: models)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL) throws -> String {
        let converter = AudioConverter(sampleRate: Double(CanaryConfig.sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples)
    }

    /// Transcribe 16 kHz mono float samples (in [-1, 1]).
    ///
    /// Audio within the 15 s window is decoded in one pass. Longer audio is split
    /// into overlapping 15 s windows (hop = 15 s − `chunkOverlapSeconds`), decoded
    /// independently, and stitched at the seams via token-level
    /// longest-common-substring (`mergeTokenStreams`). No model change — each
    /// window still sees the fixed 15 s contract and the decoder is reset per window.
    public func transcribe(audio: [Float]) throws -> String {
        let maxN = CanaryConfig.maxSamples
        if audio.count <= maxN {
            return detokenize(try transcribeWindow(audio: audio))
        }

        let hop = maxN - CanaryConfig.chunkOverlapSamples
        var merged: [Int] = []
        var start = 0
        var chunkIndex = 0
        while start < audio.count {
            let end = min(start + maxN, audio.count)
            // Don't decode a final tail that is pure overlap — the previous window
            // already covered it.
            if chunkIndex > 0, (end - start) <= (maxN - hop) { break }

            let tokens = try transcribeWindow(audio: Array(audio[start..<end]))
            merged = Self.mergeTokenStreams(prefix: merged, suffix: tokens)

            chunkIndex += 1
            if end >= audio.count { break }
            start += hop
        }
        return detokenize(merged)
    }

    /// Run the 4-stage pipeline over a single ≤15 s window; returns generated
    /// token ids (prompt stripped, EOS excluded).
    private func transcribeWindow(audio: [Float]) throws -> [Int] {
        let (mel, melLength) = try runPreprocessor(audio: audio)
        let (encoder, encoderLength) = try runEncoder(mel: mel, melLength: melLength)
        let (embeddings, encoderMask) = try makeDecoderContext(encoder: encoder, encoderLength: encoderLength)
        return try greedyDecode(embeddings: embeddings, encoderMask: encoderMask)
    }

    /// Merge two adjacent window token streams using longest-common-substring.
    ///
    /// Both windows transcribe `chunkOverlapSeconds` of identical audio at their
    /// seam, so their token ids share a common substring near the prefix's tail /
    /// the suffix's head. Search a bounded window (`windowTokens` at the boundary)
    /// for the longest common substring of length ≥ `minMatch`. On a hit, drop the
    /// suffix's matched head so the seam is not duplicated; on a miss, concatenate
    /// plainly — better to duplicate a few tokens than to lose content.
    static func mergeTokenStreams(
        prefix: [Int],
        suffix: [Int],
        windowTokens: Int = 32,
        minMatch: Int = 4
    ) -> [Int] {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }

        let pTail = Array(prefix.suffix(windowTokens))
        let sHead = Array(suffix.prefix(windowTokens))
        let m = pTail.count
        let n = sHead.count
        if m == 0 || n == 0 { return prefix + suffix }

        // Classic LCS-substring DP (O(m·n), m,n ≤ windowTokens).
        var dp = [Int](repeating: 0, count: n + 1)
        var bestLen = 0
        var bestSEnd = 0  // index in sHead (exclusive) where the match ends
        for i in 1...m {
            var prev = 0
            for j in 1...n {
                let temp = dp[j]
                if pTail[i - 1] == sHead[j - 1] {
                    dp[j] = prev + 1
                    if dp[j] > bestLen {
                        bestLen = dp[j]
                        bestSEnd = j
                    }
                } else {
                    dp[j] = 0
                }
                prev = temp
            }
        }

        guard bestLen >= minMatch else { return prefix + suffix }
        return prefix + Array(suffix.dropFirst(bestSEnd))
    }

    // MARK: - Pipeline

    /// waveform → mel [1, 128, 1501]. Audio is padded/truncated to the fixed 15 s window.
    private func runPreprocessor(audio: [Float]) throws -> (MLMultiArray, MLMultiArray) {
        let maxN = CanaryConfig.maxSamples
        let validN = min(audio.count, maxN)
        if audio.count > maxN {
            Self.logger.warning("Audio \(audio.count) samples > 15 s window; truncating to \(maxN)")
        }

        let signal = try MLMultiArray(shape: [1, maxN as NSNumber], dataType: .float32)
        let sptr = signal.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(sptr, 0, maxN * MemoryLayout<Float32>.size)
        audio.prefix(validN).withUnsafeBufferPointer { src in
            sptr.update(from: src.baseAddress!, count: validN)
        }

        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: validN)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: signal),
            "audio_length": MLFeatureValue(multiArray: length),
        ])
        let out = try models.preprocessor.prediction(from: input)
        guard let mel = out.featureValue(for: "processed")?.multiArrayValue,
            let melLen = out.featureValue(for: "processed_length")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Canary preprocessor produced no `processed`")
        }
        return (mel, melLen)
    }

    /// mel → encoder [1, 1024, 188].
    private func runEncoder(mel: MLMultiArray, melLength: MLMultiArray) throws -> (MLMultiArray, Int) {
        let featLen = try MLMultiArray(shape: [1], dataType: .int32)
        featLen[0] = NSNumber(value: melLength[0].intValue)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features": MLFeatureValue(multiArray: mel),
            "features_length": MLFeatureValue(multiArray: featLen),
        ])
        let out = try models.encoder.prediction(from: input)
        guard let enc = out.featureValue(for: "encoder")?.multiArrayValue else {
            throw ASRError.processingFailed("Canary encoder produced no `encoder`")
        }
        let encLen = out.featureValue(for: "encoder_length")?.multiArrayValue?[0].intValue ?? CanaryConfig.encoderFrames
        return (enc, encLen)
    }

    /// encoder [1, D, T] → encoder_embeddings [1, T, D] + encoder_mask [1, T].
    ///
    /// CoreML pads the encoder's last dim to a 64-element boundary (T=188 →
    /// stride 192), so the transpose must use the array's real strides, not a
    /// dense linear read.
    private func makeDecoderContext(encoder: MLMultiArray, encoderLength: Int) throws -> (MLMultiArray, MLMultiArray) {
        let d = CanaryConfig.encoderHidden
        let t = CanaryConfig.encoderFrames
        let emb = try MLMultiArray(shape: [1, t as NSNumber, d as NSNumber], dataType: .float32)
        let eptr = emb.dataPointer.assumingMemoryBound(to: Float32.self)
        let strides = encoder.strides.map { $0.intValue }
        let sD = strides[1]
        let sT = strides[2]
        let read = floatReader(encoder)
        for ti in 0..<t {
            let dst = ti * d
            let tBase = ti * sT
            for di in 0..<d {
                eptr[dst + di] = read(di * sD + tBase)
            }
        }

        let mask = try MLMultiArray(shape: [1, t as NSNumber], dataType: .float32)
        let mptr = mask.dataPointer.assumingMemoryBound(to: Float32.self)
        let valid = min(max(encoderLength, 1), t)
        for i in 0..<t { mptr[i] = i < valid ? 1.0 : 0.0 }
        return (emb, mask)
    }

    /// Greedy autoregressive decode: returns generated token ids (prompt stripped).
    private func greedyDecode(embeddings: MLMultiArray, encoderMask: MLMultiArray) throws -> [Int] {
        // Use the decoder's actual sequence length (the exported `[1, S]` shape),
        // so a shorter decoder export (e.g. S=128) is picked up automatically.
        let s =
            models.decoder.modelDescription.inputDescriptionsByName["input_ids"]?
            .multiArrayConstraint?.shape.last?.intValue ?? CanaryConfig.maxDecoderSteps

        let inputIds = try MLMultiArray(shape: [1, s as NSNumber], dataType: .int32)
        let decoderMask = try MLMultiArray(shape: [1, s as NSNumber], dataType: .float32)
        let idptr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let mkptr = decoderMask.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<s {
            idptr[i] = 0
            mkptr[i] = 0
        }
        let promptLen = min(prompt.count, s)
        for i in 0..<promptLen {
            idptr[i] = prompt[i]
            mkptr[i] = 1
        }
        var pos = promptLen

        let hidden = try MLMultiArray(shape: [1, CanaryConfig.encoderHidden as NSNumber], dataType: .float32)
        let hptr = hidden.dataPointer.assumingMemoryBound(to: Float32.self)
        let d = CanaryConfig.encoderHidden

        var generated: [Int] = []
        while pos < s {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: inputIds),
                "decoder_mask": MLFeatureValue(multiArray: decoderMask),
                "encoder_embeddings": MLFeatureValue(multiArray: embeddings),
                "encoder_mask": MLFeatureValue(multiArray: encoderMask),
            ])
            let out = try models.decoder.prediction(from: input)
            guard let dec = out.featureValue(for: "decoder")?.multiArrayValue else {
                throw ASRError.processingFailed("Canary decoder produced no `decoder`")
            }

            // hidden state at the last valid position (decoder output may be stride-padded)
            let decStrides = dec.strides.map { $0.intValue }
            let rowBase = (pos - 1) * decStrides[1]
            let elemStride = decStrides[2]
            let readDec = floatReader(dec)
            for h in 0..<d { hptr[h] = readDec(rowBase + h * elemStride) }

            let projInput = try MLDictionaryFeatureProvider(dictionary: [
                "hidden": MLFeatureValue(multiArray: hidden)
            ])
            let projOut = try models.projection.prediction(from: projInput)
            guard let logits = projOut.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("Canary projection produced no `logits`")
            }

            let next = argmax(logits)
            if next == CanaryConfig.eosId { break }

            generated.append(next)
            idptr[pos] = Int32(next)
            mkptr[pos] = 1
            pos += 1
        }
        return generated
    }

    private func detokenize(_ tokens: [Int]) -> String {
        models.tokenizer.decode(ids: tokens)
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    // MARK: - MLMultiArray helpers

    /// Returns a dtype-aware element reader for `arr` indexed by flat offset.
    /// The closure captures a pointer derived from `arr.dataPointer`; it is only
    /// valid while `arr` is alive (which it is for the duration of each use here).
    private func floatReader(_ arr: MLMultiArray) -> (Int) -> Float {
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float32.self)
            return { p[$0] }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            return { float16BitsToFloat(p[$0]) }
        default:
            return { arr[$0].floatValue }
        }
    }

    private func argmax(_ logits: MLMultiArray) -> Int {
        let n = logits.count
        var best = 0
        var bestVal = -Float.greatestFiniteMagnitude
        switch logits.dataType {
        case .float32:
            let p = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for i in 0..<n where p[i] > bestVal {
                bestVal = p[i]
                best = i
            }
        case .float16:
            let p = logits.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<n {
                let v = float16BitsToFloat(p[i])
                if v > bestVal {
                    bestVal = v
                    best = i
                }
            }
        default:
            for i in 0..<n {
                let v = logits[i].floatValue
                if v > bestVal {
                    bestVal = v
                    best = i
                }
            }
        }
        return best
    }
}

/// Decode an IEEE-754 half-precision bit pattern to Float (avoids a hard Float16 dependency).
@inline(__always)
private func float16BitsToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp = UInt32(h & 0x7C00) >> 10
    let mant = UInt32(h & 0x03FF)
    if exp == 0 {
        if mant == 0 { return Float(bitPattern: sign) }
        // subnormal
        var e: UInt32 = 127 - 15 + 1
        var m = mant
        while (m & 0x0400) == 0 {
            m <<= 1
            e -= 1
        }
        m &= 0x03FF
        return Float(bitPattern: sign | (e << 23) | (m << 13))
    }
    if exp == 0x1F {
        return Float(bitPattern: sign | 0x7F80_0000 | (mant << 13))
    }
    let e = exp - 15 + 127
    return Float(bitPattern: sign | (e << 23) | (mant << 13))
}
