import XCTest

@testable import FluidAudio

/// Parity tests for the NeuTTS-2E byte-level BPE tokenizer against the
/// HuggingFace reference implementation. Fixture ids were generated with
/// `transformers.AutoTokenizer.from_pretrained("neuphonic/neutts-2e")`
/// encoding the NFKC+quote-normalized text with `add_special_tokens=False`
/// (see mobius `models/tts/neutts-2e/coreml`).
final class NeuTtsTokenizerTests: XCTestCase {

    /// (input text, expected ids for the normalized text)
    private static let fixtures: [(String, [Int])] = [
        (
            "I can't believe it's finally here! The whole team worked so hard on this.",
            [40, 646, 944, 4411, 432, 594, 5499, 1588, 0, 576, 4361, 2083, 6439, 773, 2588, 389, 419, 13]
        ),
        ("The launch is finally here.", [785, 7050, 374, 5499, 1588, 13]),
        (
            "Hello, world! 123 tests — with em-dash, ellipsis… and “curly quotes”.",
            [
                9707, 11, 1879, 0, 220, 16, 17, 18, 7032, 1959, 448, 976, 1737, 988, 11, 25607, 47402, 1112, 323, 330,
                2352, 398, 17194, 3263,
            ]
        ),
        ("  leading spaces and\nnewlines\t tabs  ", [220, 6388, 12621, 323, 198, 931, 7969, 197, 22398, 256]),
        (
            "Mixed CASE with numbers 42345 and email test@example.com",
            [86433, 38319, 448, 5109, 220, 19, 17, 18, 19, 20, 323, 2551, 1273, 35487, 905]
        ),
        (
            "So we've got 'quotes', don't we? It'll work — I'm sure.",
            [4416, 582, 3003, 2684, 364, 53282, 516, 1513, 944, 582, 30, 1084, 3278, 975, 1959, 358, 2776, 2704, 13]
        ),
    ]

    private static let specialIds: [(String, Int)] = [
        ("<|TEXT_PROMPT_START|>", 151671),
        ("<|TEXT_PROMPT_END|>", 151672),
        ("<|SPEECH_GENERATION_START|>", 151673),
        ("<|SPEECH_GENERATION_END|>", 151674),
        ("<|speech_0|>", 151684),
        ("<|HAPPY|>", 217221),
        ("<|ANGRY|>", 217220),
        ("<|SAD|>", 217223),
        ("<|DISGUSTED|>", 217222),
        ("<|FEARFUL|>", 217224),
        ("<|SURPRISED|>", 217225),
    ]

    /// Parsed at most once: `static let` closure initialization is
    /// concurrency-safe (unlike the previous mutable `static var` cache) and
    /// NeuTtsBpeTokenizer is Sendable. nil when the asset is missing/unreadable.
    private static let tokenizer: NeuTtsBpeTokenizer? = {
        guard let root = try? TtsCacheDirectory.ensure() else { return nil }
        let url =
            root
            .appendingPathComponent("Models")
            .appendingPathComponent(Repo.neuTts.folderName)
            .appendingPathComponent(ModelNames.NeuTts.tokenizerFile)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try? NeuTtsBpeTokenizer(tokenizerJsonURL: url)
    }()

    /// Loads tokenizer.json from the local model cache; skips (does not
    /// fail) when the asset has not been downloaded on this machine/CI.
    private func loadTokenizer() throws -> NeuTtsBpeTokenizer {
        guard let tokenizer = Self.tokenizer else {
            throw XCTSkip("neutts tokenizer.json not cached locally")
        }
        return tokenizer
    }

    func testEncodeMatchesHuggingFaceReference() throws {
        let tokenizer = try loadTokenizer()
        for (text, expected) in Self.fixtures {
            let normalized = NeuTtsPrompt.normalize(text)
            XCTAssertEqual(
                tokenizer.encode(normalized), expected,
                "tokenization mismatch for: \(text)")
        }
    }

    func testSpecialTokenIds() throws {
        let tokenizer = try loadTokenizer()
        for (token, id) in Self.specialIds {
            XCTAssertEqual(tokenizer.tokenId(token), id, "id mismatch for \(token)")
        }
    }

    func testSpeechTokenRangeContiguity() throws {
        let tokenizer = try loadTokenizer()
        let base = try XCTUnwrap(tokenizer.tokenId("<|speech_0|>"))
        XCTAssertEqual(tokenizer.tokenId("<|speech_1|>"), base + 1)
        XCTAssertEqual(tokenizer.tokenId("<|speech_65535|>"), base + 65_535)
    }
}
