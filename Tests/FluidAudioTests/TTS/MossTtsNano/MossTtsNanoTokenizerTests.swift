import XCTest

@testable import FluidAudio

/// Normalization rules of the MOSS-TTS-Nano SentencePiece frontend (`nmt_nfkc` +
/// dummy prefix). Expected strings were produced with the upstream `sentencepiece`
/// processor on `OpenMOSS-Team/MOSS-TTS-Nano-100M/tokenizer.model`.
final class MossTtsNanoTokenizerTests: XCTestCase {

    func testNormalizeAddsDummyPrefixAndEscapesSpaces() {
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("The quick fox"), "▁The▁quick▁fox")
    }

    func testNormalizeCollapsesAndStripsWhitespace() {
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("  leading and trailing  "), "▁leading▁and▁trailing")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("a  b"), "▁a▁b")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("a\tb"), "▁a▁b")
        // Newlines are whitespace; a trailing newline disappears entirely.
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("user\n"), "▁user")
    }

    func testNormalizeTreatsUnicodeSpacesAndFormatCharsAsSpace() {
        // NBSP + ideographic space, and a zero-width space between letters.
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("\u{00A0}x\u{3000}y"), "▁x▁y")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("x\u{200B}y"), "▁x▁y")
    }

    func testNormalizeAppliesNFKC() {
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("ﬁne"), "▁fine")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("①"), "▁1")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize("ｈｅｌｌｏ"), "▁hello")
    }

    func testNormalizeEmptyInput() {
        XCTAssertEqual(MossTtsNanoTokenizer.normalize(""), "")
        XCTAssertEqual(MossTtsNanoTokenizer.normalize(" \n\t"), "")
    }

    /// Full BPE parity against the upstream tokenizer; runs only when the model
    /// file is already in the local cache (no network in unit tests).
    func testEncodeMatchesUpstreamWhenModelCached() throws {
        let root = try TtsCacheDirectory.ensure()
            .appendingPathComponent("Models")
            .appendingPathComponent(Repo.mossTtsNano.folderName)
            .appendingPathComponent(ModelNames.MossTtsNano.tokenizerFile)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: root.path), "tokenizer.model not cached")
        let tokenizer = try MossTtsNanoTokenizer(modelURL: root)
        XCTAssertEqual(tokenizer.vocabularySize, 16384)
        XCTAssertEqual(
            tokenizer.encode("The quick brown fox jumps over the lazy dog near the riverbank."),
            [
                453, 2234, 824, 732, 860, 10403, 4301, 10363, 722, 280, 765, 2440, 4538, 5301, 280, 1276, 356, 10379,
                2013, 10380,
            ])
        XCTAssertEqual(
            tokenizer.encode("欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"),
            [
                8651, 2691, 11099, 10670, 7669, 10508, 4627, 11074, 11315, 6439, 10617, 10859, 11643, 2957, 1531, 4139,
                2305, 4146, 11255, 10382,
            ])
        // Byte fallback: the emoji has no piece and is emitted as four <0xNN> ids.
        XCTAssertEqual(tokenizer.encode("emoji 🎉 test"), [1393, 10360, 766, 10356, 255, 174, 157, 152, 2422])
        XCTAssertEqual(tokenizer.encode("user\n"), [600, 289])
        XCTAssertEqual(tokenizer.encode("  leading and trailing  "), [6864, 311, 468, 1012, 287])
    }
}
