import Foundation
import XCTest

@testable import FluidAudio

/// Opt-in checks against the converted checkpoint and fixtures from the Mobius laya toolkit
/// (`models/computer-use/laya/coreml`): set `FLUIDAUDIO_LAYA_MODEL_DIR` to a directory holding the
/// bucket bundles plus `tokenizer.json`, and `FLUIDAUDIO_LAYA_FIXTURES_DIR` to its `fixtures/`.
final class LayaIntegrationTests: XCTestCase {
    private struct TokenizerCase: Decodable {
        let text: String
        let ids: [Int]
    }

    private struct SequenceCase: Decodable {
        let name: String
        let question: String
        let length: Int
        let state: String
        let type: String
        let instructions: String
        let options: [[String?]]
        let ids: [Int]
        let markers: [Int]

        enum CodingKeys: String, CodingKey {
            case name = "case"
            case question, length, state, type, instructions, options, ids, markers
        }

        var layaQuestion: LayaQuestion {
            switch type {
            case "choice":
                return .choice(
                    instructions, options: options.map { .init($0[0] ?? "", description: $0.count > 1 ? $0[1] : nil) })
            case "score":
                return .score(instructions, levels: options.map { $0[0] ?? "" })
            default:
                return .noul(instructions)
            }
        }
    }

    private func directory(_ name: String) throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[name], !path.isEmpty else {
            throw XCTSkip("Set \(name) to enable real laya integration tests")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    func testTokenizerMatchesHuggingFaceFixtures() throws {
        let tokenizer = try LayaTokenizer(
            tokenizerJsonURL: try directory("FLUIDAUDIO_LAYA_MODEL_DIR").appendingPathComponent("tokenizer.json"))
        let fixtures = try directory("FLUIDAUDIO_LAYA_FIXTURES_DIR").appendingPathComponent("tokenizer-cases.json")
        let cases = try JSONDecoder().decode([TokenizerCase].self, from: Data(contentsOf: fixtures))
        XCTAssertGreaterThan(cases.count, 20)
        for item in cases {
            XCTAssertEqual(tokenizer.encode(item.text), item.ids, item.text.debugDescription)
        }
    }

    func testPromptsMatchUpstreamBuildSequence() async throws {
        let manager = try await LayaManager.load(
            from: try directory("FLUIDAUDIO_LAYA_MODEL_DIR"), configuration: .init(lengths: [128]))
        let fixtures = try directory("FLUIDAUDIO_LAYA_FIXTURES_DIR").appendingPathComponent("sequence-cases.json")
        let cases = try JSONDecoder().decode([SequenceCase].self, from: Data(contentsOf: fixtures))
        XCTAssertGreaterThan(cases.count, 20)
        for item in cases {
            let sequence = try await manager.tokenSequence(
                state: item.state, question: item.layaQuestion, length: item.length)
            XCTAssertEqual(sequence.ids, item.ids, "\(item.name)/\(item.question) L\(item.length)")
            XCTAssertEqual(sequence.markers, item.markers, "\(item.name)/\(item.question) L\(item.length)")
        }
    }

    func testTetrisPlacementDecisionIsStableAcrossBuckets() async throws {
        let manager = try await LayaManager.load(
            from: try directory("FLUIDAUDIO_LAYA_MODEL_DIR"), configuration: .init(lengths: [128, 512]))
        let hole =
            "The T piece dropped at column 3 leaves two holes under it, makes the surface bumpier, "
            + "and makes the stack taller."
        let flat =
            "The I piece dropped at column 0 leaves no holes, keeps the surface flat, keeps the stack low, "
            + "and clears one line."
        // Same wording as the CLI Tetris demo.
        let question = LayaQuestion.noul("Is this a clean placement?")
        let bad = try await manager.answer(state: hole, question: question)
        let good = try await manager.answer(state: flat, question: question)
        XCTAssertEqual(bad.bucketLength, 128)
        XCTAssertLessThan(bad.noul!, good.noul!)
        XCTAssertEqual(bad.probabilities.reduce(0, +), 1, accuracy: 1e-5)

        // Force the 512 bucket with a long state; the truncation flag must be honest.
        let long = String(repeating: flat + " ", count: 30)
        let truncated = try await manager.answer(state: long, question: question)
        XCTAssertEqual(truncated.bucketLength, 512)
        XCTAssertTrue(truncated.stateWasTruncated)
    }
}
