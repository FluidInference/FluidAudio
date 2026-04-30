import CoreML
import FluidAudio
import Foundation

/// Phase 1 parity harness CLI for the CosyVoice3 Swift port.
///
/// Usage:
/// ```
/// fluidaudio tts --backend cosyvoice3-parity \
///   --fixture    .../build/frontend/shipping.safetensors \
///   --models-dir .../coreml/build \
///   --reference  .../build/wavs/e2e_shipping.wav \
///   --output     .../build/swift_e2e.wav \
///   --seed 42
/// ```
@available(macOS 15, iOS 18, *)
enum CosyVoice3ParityCLI {

    private static let logger = AppLogger(category: "CosyVoice3ParityCLI")

    static func run(
        fixturePath: String,
        modelsDir: String,
        referencePath: String?,
        outputPath: String,
        seed: UInt64,
        cpuOnly: Bool,
        replayTokens: Bool
    ) async {
        let fixtureURL = URL(fileURLWithPath: (fixturePath as NSString).expandingTildeInPath)
        let modelsURL = URL(
            fileURLWithPath: (modelsDir as NSString).expandingTildeInPath, isDirectory: true)
        let outputURL = URL(fileURLWithPath: (outputPath as NSString).expandingTildeInPath)

        let computeUnits: MLComputeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
        let manager = CosyVoice3TtsManager(directory: modelsURL, computeUnits: computeUnits)

        do {
            let tLoad = Date()
            try await manager.initialize()
            logger.info(
                "Loaded CosyVoice3 models in \(String(format: "%.2f", Date().timeIntervalSince(tLoad)))s"
            )

            let options = CosyVoice3ParityOptions(
                maxNewTokens: nil, seed: seed, replayDecodedTokens: replayTokens)

            let tSynth = Date()
            let result = try await manager.synthesizeFromFixture(
                fixtureURL: fixtureURL, options: options)
            let synthElapsed = Date().timeIntervalSince(tSynth)
            let audioSec = Double(result.samples.count) / Double(result.sampleRate)
            let rtfx = audioSec / synthElapsed
            logger.info(
                "Synthesized \(result.samples.count) samples (\(String(format: "%.2fs", audioSec))) in \(String(format: "%.2fs", synthElapsed))"
            )
            print(
                String(
                    format:
                        "RTFX audio=%.3fs synth=%.3fs RTFx=%.3fx tokens=%d",
                    audioSec, synthElapsed, rtfx, result.generatedTokenCount))

            try writeWAV(samples: result.samples, sampleRate: result.sampleRate, to: outputURL)
            logger.info("Wrote WAV: \(outputURL.path)")

            if let refPath = referencePath {
                let refURL = URL(
                    fileURLWithPath: (refPath as NSString).expandingTildeInPath)
                let refSamples = try readWAVMono(url: refURL)
                let metrics = compareWaveforms(
                    swift: result.samples, reference: refSamples)
                print("")
                print(
                    "  reference samples : \(refSamples.count)  swift samples : \(result.samples.count)"
                )
                print(
                    "  MAE               : \(String(format: "%.6f", metrics.mae))")
                print(
                    "  max|Δ|            : \(String(format: "%.6f", metrics.maxAbsDiff))")
                print("  SNR               : \(String(format: "%.2f dB", metrics.snrDb))")
                if metrics.maxAbsDiff > 1e-3 {
                    logger.warning(
                        "Parity tolerance exceeded: max|Δ|=\(metrics.maxAbsDiff) > 1e-3")
                    exit(1)
                }
            }
        } catch {
            logger.error("CosyVoice3 parity harness failed: \(error)")
            exit(2)
        }
    }

    // MARK: - WAV IO (un-normalized)

    private static func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        // Clamp to [-1, 1] to avoid int16 overflow; do NOT rescale to max=1.
        let numSamples = samples.count
        let byteRate = sampleRate * 2
        let dataSize = numSamples * 2
        var header = Data()
        header.append("RIFF".data(using: .ascii)!)
        header.appendUInt32LE(UInt32(36 + dataSize))
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        header.appendUInt32LE(16)
        header.appendUInt16LE(1)  // PCM
        header.appendUInt16LE(1)  // mono
        header.appendUInt32LE(UInt32(sampleRate))
        header.appendUInt32LE(UInt32(byteRate))
        header.appendUInt16LE(2)  // block align
        header.appendUInt16LE(16)  // bits/sample
        header.append("data".data(using: .ascii)!)
        header.appendUInt32LE(UInt32(dataSize))

        var pcm = Data(capacity: dataSize)
        for s in samples {
            let clipped = max(-1.0, min(1.0, s))
            let i16 = Int16(clipped * 32_767.0)
            var le = i16.littleEndian
            Swift.withUnsafeBytes(of: &le) { pcm.append(contentsOf: $0) }
        }
        try (header + pcm).write(to: url)
    }

    private static func readWAVMono(url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 44 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        // Find 'data' chunk.
        var offset = 12
        var dataStart = -1
        var dataSize = 0
        while offset + 8 <= data.count {
            let id = data.subdata(in: offset..<offset + 4)
            let size = data.subdata(in: offset + 4..<offset + 8).readUInt32LE()
            if id == "data".data(using: .ascii) {
                dataStart = offset + 8
                dataSize = Int(size)
                break
            }
            offset += 8 + Int(size)
        }
        guard dataStart > 0 else { throw CocoaError(.fileReadCorruptFile) }
        let pcm = data.subdata(in: dataStart..<min(dataStart + dataSize, data.count))
        let count = pcm.count / 2
        var out = [Float](repeating: 0, count: count)
        pcm.withUnsafeBytes { buf in
            let ptr = buf.bindMemory(to: Int16.self)
            for i in 0..<count {
                out[i] = Float(ptr[i]) / 32_768.0
            }
        }
        return out
    }

    // MARK: - Metrics

    struct WaveformMetrics {
        let mae: Double
        let maxAbsDiff: Double
        let snrDb: Double
    }

    private static func compareWaveforms(swift: [Float], reference: [Float]) -> WaveformMetrics {
        let n = min(swift.count, reference.count)
        guard n > 0 else { return WaveformMetrics(mae: .infinity, maxAbsDiff: .infinity, snrDb: -.infinity) }
        var sumAbs: Double = 0
        var maxAbs: Double = 0
        var sumSigSq: Double = 0
        var sumErrSq: Double = 0
        for i in 0..<n {
            let diff = Double(swift[i]) - Double(reference[i])
            let a = abs(diff)
            sumAbs += a
            if a > maxAbs { maxAbs = a }
            sumSigSq += Double(reference[i]) * Double(reference[i])
            sumErrSq += diff * diff
        }
        let snr = sumErrSq > 0 ? 10 * log10(sumSigSq / sumErrSq) : .infinity
        return WaveformMetrics(mae: sumAbs / Double(n), maxAbsDiff: maxAbs, snrDb: snr)
    }
}

// MARK: - Data helpers

extension Data {
    fileprivate mutating func appendUInt32LE(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
    fileprivate mutating func appendUInt16LE(_ v: UInt16) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
    fileprivate func readUInt32LE() -> UInt32 {
        self.withUnsafeBytes { buf -> UInt32 in
            var v: UInt32 = 0
            memcpy(&v, buf.baseAddress!, 4)
            return UInt32(littleEndian: v)
        }
    }
}
