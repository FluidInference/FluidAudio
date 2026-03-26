import Foundation
import XCTest

@testable import FluidAudio

/// Demo tests showing practical CTC decoder usage with language models
final class CtcDecoderDemoTests: XCTestCase {

    // MARK: - Demo: Greedy vs Beam Search

    func testDemoGreedyVsBeamSearch() throws {
        // Simulate CTC log-probabilities where acoustic model is ambiguous
        // between "patient has diabetes" and "patient has die beetus"

        let vocab: [Int: String] = [
            0: "▁patient",
            1: "▁has",
            2: "▁diabetes",
            3: "▁die",
            4: "▁beetus",
            5: "▁high",
            6: "▁blood",
            7: "▁pressure",
        ]
        let blankId = 8

        // Acoustically ambiguous: "diabetes" vs "die beetus"
        // Frame sequence: patient, has, die/diabetes (ambiguous), beetus (only if die)
        let logProbs: [[Float]] = [
            // Frame 0-1: "patient" is clear
            [-1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            [-1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],

            // Frame 2: blank
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -1.0],

            // Frame 3-4: "has" is clear
            [-10.0, -1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            [-10.0, -1.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],

            // Frame 5: blank
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -1.0],

            // Frame 6-7: Ambiguous! "diabetes" (-1.5) vs "die" (-1.4)
            // Acoustically "die" is slightly better
            [-10.0, -10.0, -1.5, -1.4, -10.0, -10.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, -1.5, -1.4, -10.0, -10.0, -10.0, -10.0, -10.0],

            // Frame 8: blank
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -1.0],

            // Frame 9-10: "beetus" if we chose "die" path
            [-10.0, -10.0, -10.0, -10.0, -1.2, -10.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0, -10.0, -1.2, -10.0, -10.0, -10.0, -10.0],
        ]

        // Greedy picks "die" (acoustically better at frame 6-7)
        let greedy = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        print("Greedy (no LM):   \(greedy)")
        XCTAssertEqual(greedy, "patient has die beetus")  // Wrong!

        // Beam search without LM should also pick "die" (better acoustic score)
        let beamNoLM = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: nil,
            beamWidth: 10, blankId: blankId
        )
        print("Beam (no LM):     \(beamNoLM)")
        XCTAssertEqual(beamNoLM, "patient has die beetus")  // Still wrong!

        // But with a medical LM, "diabetes" is a real word with high probability
        let lm = createMedicalLM()
        let beamWithLM = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: lm,
            beamWidth: 10, lmWeight: 5.0,  // Strong LM to override acoustics
            blankId: blankId
        )
        print("Beam (with LM):   \(beamWithLM)")
        XCTAssertEqual(beamWithLM, "patient has diabetes")  // Correct!

        print("\n✅ Demo: Language model successfully corrected misrecognition!")
        print("   Acoustic model preferred: 'die beetus' (-1.4 + -1.2 = -2.6)")
        print("   LM model preferred:       'diabetes' (real medical term)")
    }

    // MARK: - Demo: Language Model Scoring

    func testDemoLanguageModelScoring() throws {
        // Demonstrate how LM scores affect decoding
        let vocab: [Int: String] = [
            0: "▁the",
            1: "▁cat",
            2: "▁sat",
            3: "▁dog",
        ]
        let blankId = 4

        // Both "cat" and "dog" are acoustically similar
        let logProbs: [[Float]] = [
            // "the" is clear
            [-1.0, -10.0, -10.0, -10.0, -10.0],
            [-1.0, -10.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0, -10.0, -1.0],  // blank

            // "cat" vs "dog" - dog slightly better acoustically
            [-10.0, -1.5, -10.0, -1.4, -10.0],
            [-10.0, -1.5, -10.0, -1.4, -10.0],
            [-10.0, -10.0, -10.0, -10.0, -1.0],  // blank

            // "sat"
            [-10.0, -10.0, -1.0, -10.0, -10.0],
            [-10.0, -10.0, -1.0, -10.0, -10.0],
        ]

        let greedy = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        print("Greedy: \(greedy)")
        // Should pick "dog" (better acoustic score: -1.4 vs -1.5)
        XCTAssertEqual(greedy, "the dog sat")

        // LM with strong "the cat" bigram
        var lm = ARPALanguageModel()
        lm.unigrams["the"] = ARPALanguageModel.Entry(logProb: -1.0, backoff: -0.3)
        lm.unigrams["cat"] = ARPALanguageModel.Entry(logProb: -1.5, backoff: 0.0)
        lm.unigrams["dog"] = ARPALanguageModel.Entry(logProb: -1.5, backoff: 0.0)
        lm.unigrams["sat"] = ARPALanguageModel.Entry(logProb: -1.5, backoff: 0.0)
        lm.bigrams["the"] = [
            "cat": ARPALanguageModel.Entry(logProb: -0.3, backoff: 0.0),  // Strong bigram
            "dog": ARPALanguageModel.Entry(logProb: -2.0, backoff: 0.0),  // Weak bigram
        ]

        let withLM = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: lm,
            beamWidth: 10, lmWeight: 2.0, blankId: blankId
        )
        print("With LM: \(withLM)")
        // Should pick "cat" due to strong "the cat" bigram overriding acoustics
        XCTAssertEqual(withLM, "the cat sat")

        print("\n✅ Demo: LM bigram score (-0.3) overrode acoustic preference for 'dog'")
    }

    // MARK: - Demo: Windows Line Endings Support

    func testDemoWindowsLineEndings() throws {
        // Create ARPA file with Windows line endings (\r\n)
        let tempDir = FileManager.default.temporaryDirectory
        let arpaFile = tempDir.appendingPathComponent("windows_\(UUID().uuidString).arpa")

        let content =
            "\\data\\\r\nngram 1=2\r\n\r\n\\1-grams:\r\n-1.0\thello\t0.0\r\n-1.0\tworld\t0.0\r\n\r\n\\end\\\r\n"
        try content.write(to: arpaFile, atomically: true, encoding: .utf8)
        addTeardownBlock { try? FileManager.default.removeItem(at: arpaFile) }

        // Should load successfully with .whitespacesAndNewlines trimming
        let lm = try ARPALanguageModel.load(from: arpaFile)

        XCTAssertEqual(lm.unigrams.count, 2, "Should parse Windows ARPA file correctly")
        XCTAssertNotNil(lm.unigrams["hello"])
        XCTAssertNotNil(lm.unigrams["world"])

        print("✅ Demo: Successfully loaded ARPA file with Windows (\\r\\n) line endings")
    }

    // MARK: - Helpers

    /// Creates a simple medical language model for testing
    private func createMedicalLM() -> ARPALanguageModel {
        var lm = ARPALanguageModel()

        // Medical unigrams with realistic probabilities
        let medicalTerms: [(word: String, logProb: Float, backoff: Float)] = [
            ("patient", -1.5, -0.3),
            ("has", -1.8, -0.2),
            ("diabetes", -2.2, -0.1),  // Real medical term
            ("hypertension", -2.5, -0.1),  // Real medical term
            ("die", -4.0, -0.5),  // Unlikely in medical context
            ("beetus", -5.0, -0.5),  // Not a word
            ("hyper", -4.5, -0.4),  // Unlikely without "tension"
            ("tension", -4.3, -0.4),  // Unlikely without "hyper"
        ]

        for term in medicalTerms {
            lm.unigrams[term.word] = ARPALanguageModel.Entry(
                logProb: term.logProb,
                backoff: term.backoff
            )
        }

        // Medical bigrams - "patient has diabetes" is common
        lm.bigrams["patient"] = [
            "has": ARPALanguageModel.Entry(logProb: -0.3, backoff: 0.0)
        ]
        lm.bigrams["has"] = [
            "diabetes": ARPALanguageModel.Entry(logProb: -0.5, backoff: 0.0),
            "hypertension": ARPALanguageModel.Entry(logProb: -0.6, backoff: 0.0),
        ]

        return lm
    }
}
