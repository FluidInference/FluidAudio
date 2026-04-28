import FluidAudio
import Foundation

/// Phase 2 tokenizer parity harness.
///
/// Loads the Python-exported tokenizer_fixture.json (special token map + test
/// cases) and asserts the Swift Qwen2BpeTokenizer produces the same ID stream
/// for every case.
///
/// Usage:
/// ```
/// fluidaudio tts --backend cosyvoice3-tokenizer-parity \
///   --tokenizer-dir .../cosyvoice3_dl/CosyVoice-BlankEN \
///   --fixture       .../build/frontend/tokenizer_fixture.json
/// ```
enum CosyVoice3TokenizerParityCLI {

    private static let logger = AppLogger(category: "CosyVoice3TokenizerParityCLI")

    static func run(tokenizerDir: String, fixturePath: String) async {
        let tokURL = URL(
            fileURLWithPath: (tokenizerDir as NSString).expandingTildeInPath,
            isDirectory: true)
        let fixURL = URL(fileURLWithPath: (fixturePath as NSString).expandingTildeInPath)

        struct Fixture: Decodable {
            let special_tokens: [String: Int32]
            let cases: [Case]
            struct Case: Decodable {
                let text: String
                let ids: [Int32]
            }
        }

        do {
            let data = try Data(contentsOf: fixURL)
            let fixture = try JSONDecoder().decode(Fixture.self, from: data)
            let tokenizer = try Qwen2BpeTokenizer.load(
                directory: tokURL, specialTokens: fixture.special_tokens)

            var passed = 0
            var failed = 0
            var firstFail: (String, [Int32], [Int32])? = nil
            for tc in fixture.cases {
                let got = tokenizer.encode(tc.text)
                if got == tc.ids {
                    passed += 1
                } else {
                    failed += 1
                    if firstFail == nil {
                        firstFail = (tc.text, tc.ids, got)
                    }
                }
            }

            print("cases: \(passed + failed)  passed: \(passed)  failed: \(failed)")
            if let (text, expected, got) = firstFail {
                print("")
                print("first mismatch:")
                print("  text     : \(text.debugDescription)")
                print("  expected : \(expected)")
                print("  got      : \(got)")
            }
            if failed > 0 { exit(1) }
        } catch {
            logger.error("Tokenizer parity failed: \(error)")
            exit(2)
        }
    }
}
