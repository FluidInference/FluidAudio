import CoreML
import FluidAudio
import Foundation

/// Phase 2 text-frontend parity harness.
///
/// Loads `shipping.safetensors` (expected `lm_input_embeds`, `llm_prompt_speech_ids`)
/// plus its JSON sidecar (`prompt_text`, `tts_text`), tokenizes the text via
/// `Qwen2BpeTokenizer`, assembles via `CosyVoice3TextFrontend`, and compares
/// element-wise against the fixture.
///
/// Usage:
/// ```
/// fluidaudio tts --backend cosyvoice3-frontend-parity \
///   --tokenizer-dir   .../cosyvoice3_dl/CosyVoice-BlankEN \
///   --embeddings-file .../build/embeddings/embeddings-fp32.safetensors \
///   --fixture         .../build/frontend/shipping.safetensors \
///   --tok-fixture     .../build/frontend/tokenizer_fixture.json
/// ```
enum CosyVoice3FrontendParityCLI {

    private static let logger = AppLogger(category: "CosyVoice3FrontendParityCLI")

    static func run(
        tokenizerDir: String,
        embeddingsFile: String,
        fixturePath: String,
        tokFixturePath: String
    ) async {
        let tokURL = URL(
            fileURLWithPath: (tokenizerDir as NSString).expandingTildeInPath,
            isDirectory: true)
        let embURL = URL(fileURLWithPath: (embeddingsFile as NSString).expandingTildeInPath)
        let fixURL = URL(fileURLWithPath: (fixturePath as NSString).expandingTildeInPath)
        let tokFixURL = URL(fileURLWithPath: (tokFixturePath as NSString).expandingTildeInPath)
        let sidecarURL = fixURL.deletingPathExtension().appendingPathExtension("json")

        struct TokFix: Decodable {
            let special_tokens: [String: Int32]
        }
        struct Sidecar: Decodable {
            let prompt_text: String
            let tts_text: String
        }

        do {
            let tokFix = try JSONDecoder().decode(
                TokFix.self, from: try Data(contentsOf: tokFixURL))
            let sidecar = try JSONDecoder().decode(
                Sidecar.self, from: try Data(contentsOf: sidecarURL))

            let tStart = Date()
            let tokenizer = try Qwen2BpeTokenizer.load(
                directory: tokURL, specialTokens: tokFix.special_tokens)
            let embeddings = try CosyVoice3TextEmbeddings(url: embURL)
            logger.info(
                "Loaded tokenizer + text_embedding table in \(String(format: "%.2fs", Date().timeIntervalSince(tStart)))"
            )

            let fixture = try CosyVoice3FrontendFixture.load(from: fixURL)
            logger.info("Fixture: T_pre=\(fixture.tPre) N_prompt_speech=\(fixture.promptSpeechIds.count)")

            let frontend = CosyVoice3TextFrontend(tokenizer: tokenizer, embeddings: embeddings)
            let assembled = try frontend.assemble(
                promptText: sidecar.prompt_text,
                ttsText: sidecar.tts_text,
                promptSpeechIds: fixture.promptSpeechIds)

            print("")
            print("  swift T_pre     : \(assembled.tPre)")
            print("  fixture T_pre   : \(fixture.tPre)")

            guard assembled.tPre == fixture.tPre else {
                print("T_pre mismatch — tokenization diverged.")
                exit(1)
            }

            // Element-wise comparison: fixture is compact fp32, swift array
            // may have padded strides.
            let dim = CosyVoice3Constants.embedDim
            let strides = assembled.lmInputEmbeds.strides.map { $0.intValue }
            let ptr = assembled.lmInputEmbeds.dataPointer.bindMemory(
                to: Float.self, capacity: assembled.lmInputEmbeds.count)
            var maxAbs: Double = 0
            var maxAt: (t: Int, d: Int) = (0, 0)
            var sumAbs: Double = 0
            var rowMax = [Double](repeating: 0, count: assembled.tPre)
            let n = assembled.tPre * dim
            for t in 0..<assembled.tPre {
                for d in 0..<dim {
                    let got = ptr[t * strides[1] + d * strides[2]]
                    let exp = fixture.lmInputEmbeds[t * dim + d]
                    let diff = Double(got) - Double(exp)
                    let a = abs(diff)
                    sumAbs += a
                    if a > rowMax[t] { rowMax[t] = a }
                    if a > maxAbs {
                        maxAbs = a
                        maxAt = (t, d)
                    }
                }
            }
            let mae = sumAbs / Double(n)
            print("  MAE             : \(String(format: "%.6e", mae))")
            print("  max|Δ|          : \(String(format: "%.6e", maxAbs)) at (t=\(maxAt.t), d=\(maxAt.d))")

            // Show the top-5 worst rows to see if divergence is concentrated
            // at sos (t=0), task_id (t=1+nText), or specific text/speech rows.
            let N_speech = fixture.promptSpeechIds.count
            let nText = assembled.tPre - 2 - N_speech
            print(
                "  layout          : sos@0  text@1..\(nText)  task@\(1 + nText)  speech@\(2 + nText)..\(assembled.tPre - 1)"
            )
            let ranked = rowMax.enumerated().sorted { $0.element > $1.element }.prefix(5)
            print("  top rows:")
            for (t, m) in ranked {
                let slot: String
                if t == 0 {
                    slot = "sos"
                } else if t == 1 + nText {
                    slot = "task_id"
                } else if t < 1 + nText {
                    slot = "text[\(t - 1)]"
                } else {
                    slot = "speech[\(t - 2 - nText)]"
                }
                print(
                    "    t=\(t)  \(slot.padding(toLength: 12, withPad: " ", startingAt: 0))  max|Δ|=\(String(format: "%.6e", m))"
                )
            }

            // Compare Swift's reconstructed token ids for sanity.
            print("  swift textToken ids (first 10): \(assembled.textTokenIds.prefix(10).map { $0 })")
            print("  swift textToken ids (last 5) : \(assembled.textTokenIds.suffix(5).map { $0 })")

            if maxAbs > 1e-4 {
                print("parity tolerance exceeded (max|Δ| > 1e-4)")
                exit(1)
            }
            print("frontend parity OK")
        } catch {
            logger.error("Frontend parity failed: \(error)")
            exit(2)
        }
    }
}
