import Foundation
import XCTest

@testable import FluidAudio

final class TokenizerTests: XCTestCase {

    // MARK: - Helpers

    private func createTempVocabFile(_ vocab: [String: String]) throws -> URL {
        let data = try JSONSerialization.data(withJSONObject: vocab, options: [])
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
        try data.write(to: file)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: file)
        }
        return file
    }

    // MARK: - Decode Known Token IDs

    func testDecodeKnownTokenIds() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello",
            "1": "\u{2581}world",
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 1])
        XCTAssertEqual(result, "Hello world")
    }

    func testDecodeUnknownTokenIdIsSkipped() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello"
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 999])
        XCTAssertEqual(result, "Hello")
    }

    func testDecodeEmptyIdsReturnsEmpty() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}Hello"
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [])
        XCTAssertEqual(result, "")
    }

    func testSentencePieceBoundaryReplacement() throws {
        let vocab: [String: String] = [
            "0": "\u{2581}The",
            "1": "\u{2581}quick",
            "2": "\u{2581}brown",
        ]
        let file = try createTempVocabFile(vocab)
        let tokenizer = try Tokenizer(vocabPath: file)

        let result = tokenizer.decode(ids: [0, 1, 2])
        XCTAssertEqual(result, "The quick brown")
    }

    func testInvalidJsonThrows() {
        let tempDir = FileManager.default.temporaryDirectory
        let file = tempDir.appendingPathComponent("bad_vocab_\(UUID().uuidString).json")

        do {
            try "not json at all".write(to: file, atomically: true, encoding: .utf8)
            addTeardownBlock {
                try? FileManager.default.removeItem(at: file)
            }
        } catch {
            XCTFail("Failed to write temp file: \(error)")
            return
        }

        XCTAssertThrowsError(try Tokenizer(vocabPath: file))
    }

    func testNonExistentFileThrows() {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(
            "nonexistent_\(UUID().uuidString).json")
        XCTAssertThrowsError(try Tokenizer(vocabPath: file))
    }
}
