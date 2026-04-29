import CoreML
import FluidAudio
import Foundation

/// Phase 2 text-driven synthesis CLI for the CosyVoice3 Swift port.
///
/// Drives `CosyVoice3TtsManager.synthesize(text:promptAssets:options:)` end
/// to end: tokenizer + frontend + LLM + Flow + HiFT, writing a 24 kHz WAV.
///
/// Usage:
/// ```
/// fluidaudio tts --backend cosyvoice3-text \
///   --text "希望你以后能够做的比我还好用" \
///   --models-dir          .../coreml/build \
///   --tokenizer-dir       .../cosyvoice3_dl/CosyVoice-BlankEN \
///   --embeddings-file     .../build/embeddings/embeddings-runtime-fp32.safetensors \
///   --special-tokens-file .../build/frontend/tokenizer_fixture.json \
///   --prompt-assets       .../build/frontend/shipping.safetensors \
///   --output              .../build/swift_cv3_text.wav \
///   --seed 42
/// ```
enum CosyVoice3TextCLI {

    private static let logger = AppLogger(category: "CosyVoice3TextCLI")

    static func run(
        text: String,
        modelsDir: String,
        tokenizerDir: String,
        embeddingsFile: String,
        specialTokensFile: String,
        promptAssetsPath: String,
        outputPath: String,
        seed: UInt64,
        maxNewTokens: Int?,
        cpuOnly: Bool
    ) async {
        let modelsURL = URL(
            fileURLWithPath: (modelsDir as NSString).expandingTildeInPath, isDirectory: true)
        let tokURL = URL(
            fileURLWithPath: (tokenizerDir as NSString).expandingTildeInPath, isDirectory: true)
        let embURL = URL(fileURLWithPath: (embeddingsFile as NSString).expandingTildeInPath)
        let specURL = URL(fileURLWithPath: (specialTokensFile as NSString).expandingTildeInPath)
        let promptURL = URL(fileURLWithPath: (promptAssetsPath as NSString).expandingTildeInPath)
        let outputURL = URL(fileURLWithPath: (outputPath as NSString).expandingTildeInPath)

        let computeUnits: MLComputeUnits = cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
        let manager = CosyVoice3TtsManager(
            modelsDirectory: modelsURL,
            tokenizerDirectory: tokURL,
            textEmbeddingsFile: embURL,
            specialTokensFile: specURL,
            computeUnits: computeUnits)

        do {
            let tLoad = Date()
            try await manager.initialize()
            logger.info(
                "Loaded CosyVoice3 models + frontend in \(String(format: "%.2f", Date().timeIntervalSince(tLoad)))s"
            )

            let tPrompt = Date()
            let promptAssets = try CosyVoice3PromptAssets.load(from: promptURL)
            logger.info(
                "Loaded prompt assets in \(String(format: "%.2f", Date().timeIntervalSince(tPrompt)))s — N_speech=\(promptAssets.promptSpeechIds.count), mel_frames=\(promptAssets.promptMelFrames)"
            )

            let options = CosyVoice3SynthesisOptions(
                maxNewTokens: maxNewTokens, seed: seed)

            let tSynth = Date()
            let result = try await manager.synthesize(
                text: text, promptAssets: promptAssets, options: options)
            let synthElapsed = Date().timeIntervalSince(tSynth)
            let audioSecs = Double(result.samples.count) / Double(result.sampleRate)
            let rtfx = synthElapsed > 0 ? audioSecs / synthElapsed : 0
            logger.info(
                "Synthesized \(result.samples.count) samples (\(String(format: "%.2fs", audioSecs))) in \(String(format: "%.2fs", synthElapsed)) — RTFx \(String(format: "%.2fx", rtfx))"
            )
            logger.info("Generated \(result.generatedTokenCount) speech tokens")

            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try writeWAV(samples: result.samples, sampleRate: result.sampleRate, to: outputURL)
            logger.info("Wrote WAV: \(outputURL.path)")
        } catch {
            logger.error("CosyVoice3 text synthesis failed: \(error)")
            exit(2)
        }
    }

    private static func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
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
}

// MARK: - Data helpers (file-scoped duplicate of the helpers in
// CosyVoice3ParityCommand.swift; kept here so this file compiles on its own).

extension Data {
    fileprivate mutating func appendUInt32LE(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
    fileprivate mutating func appendUInt16LE(_ v: UInt16) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { self.append(contentsOf: $0) }
    }
}
