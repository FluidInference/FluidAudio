import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class CtcDecoderTests: XCTestCase {

    // MARK: - logAddExp

    func testLogAddExpEqualValues() {
        let result = logAddExp(0.0, 0.0)
        XCTAssertEqual(result, Float(log(2.0)), accuracy: 0.001)
    }

    func testLogAddExpWithNegInfinity() {
        XCTAssertEqual(logAddExp(-.infinity, 5.0), 5.0)
        XCTAssertEqual(logAddExp(5.0, -.infinity), 5.0)
    }

    func testLogAddExpBothNegInfinity() {
        XCTAssertEqual(logAddExp(-.infinity, -.infinity), -.infinity)
    }

    func testLogAddExpLargeDifference() {
        let result = logAddExp(100.0, 0.0)
        XCTAssertEqual(result, 100.0, accuracy: 0.001)
    }

    func testLogAddExpIsCommutative() {
        let a = logAddExp(1.0, 2.0)
        let b = logAddExp(2.0, 1.0)
        XCTAssertEqual(a, b, accuracy: 1e-6)
    }

    // MARK: - decodeCtcTokenIds

    func testDecodeTokenIdsBasic() {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let result = decodeCtcTokenIds([0, 1], vocabulary: vocab)
        XCTAssertEqual(result, "hello world")
    }

    func testDecodeTokenIdsEmpty() {
        let vocab: [Int: String] = [0: "▁hello"]
        let result = decodeCtcTokenIds([], vocabulary: vocab)
        XCTAssertEqual(result, "")
    }

    func testDecodeTokenIdsSkipsUnknown() {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let result = decodeCtcTokenIds([0, 999, 1], vocabulary: vocab)
        XCTAssertEqual(result, "hello world")
    }

    func testDecodeTokenIdsSubwordJoin() {
        let vocab: [Int: String] = [0: "he", 1: "llo", 2: "▁world"]
        let result = decodeCtcTokenIds([0, 1, 2], vocabulary: vocab)
        XCTAssertEqual(result, "hello world")
    }

    // MARK: - CTC Greedy Decode ([[Float]])

    func testGreedyDecodeSimple() {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let blankId = 2
        // Frame 0: token 0 dominant, Frame 1: blank, Frame 2: token 1 dominant
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0],
            [-100.0, -100.0, 0.0],
            [-100.0, 0.0, -100.0],
        ]
        let result = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        XCTAssertEqual(result, "hello world")
    }

    func testGreedyDecodeCollapsesRepeats() {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let blankId = 2
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0],
            [0.0, -100.0, -100.0],
            [-100.0, 0.0, -100.0],
        ]
        let result = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        XCTAssertEqual(result, "hello world")
    }

    func testGreedyDecodeBlankAllowsRepeats() {
        let vocab: [Int: String] = [0: "▁hello"]
        let blankId = 1
        let logProbs: [[Float]] = [
            [0.0, -100.0],
            [-100.0, 0.0],
            [0.0, -100.0],
        ]
        let result = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        XCTAssertEqual(result, "hello hello")
    }

    func testGreedyDecodeAllBlanks() {
        let vocab: [Int: String] = [0: "▁hello"]
        let blankId = 1
        let logProbs: [[Float]] = [
            [-100.0, 0.0],
            [-100.0, 0.0],
            [-100.0, 0.0],
        ]
        let result = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        XCTAssertEqual(result, "")
    }

    func testGreedyDecodeEmptyInput() {
        let vocab: [Int: String] = [0: "▁hello"]
        let result = ctcGreedyDecode(logProbs: [], vocabulary: vocab, blankId: 1)
        XCTAssertEqual(result, "")
    }

    // MARK: - CTC Greedy Decode (MLMultiArray)

    func testGreedyDecodeMLMultiArray() throws {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let blankId = 2
        let vocabSize = 3
        let timeSteps = 3

        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: timeSteps), NSNumber(value: vocabSize)],
            dataType: .float32
        )
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        // Fill with -100
        for i in 0..<(timeSteps * vocabSize) { ptr[i] = -100.0 }
        // Frame 0: token 0, Frame 1: blank, Frame 2: token 1
        ptr[0 * vocabSize + 0] = 0.0
        ptr[1 * vocabSize + blankId] = 0.0
        ptr[2 * vocabSize + 1] = 0.0

        let result = ctcGreedyDecode(logProbs: arr, vocabulary: vocab, blankId: blankId)
        XCTAssertEqual(result, "hello world")
    }

    // MARK: - CTC Beam Search

    func testBeamSearchNoLMMatchesGreedy() {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let blankId = 2
        let logProbs: [[Float]] = [
            [0.0, -100.0, -100.0],
            [-100.0, -100.0, 0.0],
            [-100.0, 0.0, -100.0],
        ]
        let greedy = ctcGreedyDecode(logProbs: logProbs, vocabulary: vocab, blankId: blankId)
        let beam = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: nil,
            beamWidth: 5, lmWeight: 0.0, blankId: blankId
        )
        XCTAssertEqual(greedy, beam)
    }

    func testBeamSearchAllBlanks() {
        let vocab: [Int: String] = [0: "▁hello"]
        let blankId = 1
        let logProbs: [[Float]] = [
            [-100.0, 0.0],
            [-100.0, 0.0],
            [-100.0, 0.0],
        ]
        let result = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: nil,
            beamWidth: 5, lmWeight: 0.0, blankId: blankId
        )
        XCTAssertEqual(result, "")
    }

    func testBeamSearchEmptyInput() {
        let vocab: [Int: String] = [0: "▁hello"]
        let result = ctcBeamSearch(
            logProbs: [], vocabulary: vocab, lm: nil,
            beamWidth: 5, lmWeight: 0.0, blankId: 1
        )
        XCTAssertEqual(result, "")
    }

    func testBeamSearchSingleToken() {
        let vocab: [Int: String] = [0: "▁hello"]
        let blankId = 1
        let logProbs: [[Float]] = [
            [0.0, -100.0]
        ]
        let result = ctcBeamSearch(
            logProbs: logProbs, vocabulary: vocab, lm: nil,
            beamWidth: 5, lmWeight: 0.0, blankId: blankId
        )
        XCTAssertEqual(result, "hello")
    }

    // MARK: - CtcBeam

    func testCtcBeamTotalAcoustic() {
        let beam = CtcBeam(
            prefix: [1], pBlank: -1.0, pNonBlank: -2.0,
            lmScore: 0.0, wordPieces: [], prevWord: nil
        )
        let expected = logAddExp(-1.0, -2.0)
        XCTAssertEqual(beam.totalAcoustic, expected, accuracy: 1e-6)
    }

    func testCtcBeamTotalIncludesLM() {
        let beam = CtcBeam(
            prefix: [1], pBlank: 0.0, pNonBlank: -.infinity,
            lmScore: 5.0, wordPieces: [], prevWord: nil
        )
        XCTAssertEqual(beam.total, 5.0, accuracy: 1e-6)
    }

    func testCtcBeamLastToken() {
        let beam = CtcBeam(
            prefix: [1, 2, 3], pBlank: 0.0, pNonBlank: 0.0,
            lmScore: 0.0, wordPieces: [], prevWord: nil
        )
        XCTAssertEqual(beam.lastToken, 3)
    }

    func testCtcBeamLastTokenEmpty() {
        let beam = CtcBeam(
            prefix: [], pBlank: 0.0, pNonBlank: 0.0,
            lmScore: 0.0, wordPieces: [], prevWord: nil
        )
        XCTAssertNil(beam.lastToken)
    }

    // MARK: - CTC Beam Search (MLMultiArray)

    func testBeamSearchMLMultiArrayMatchesGreedy() throws {
        let vocab: [Int: String] = [0: "▁hello", 1: "▁world"]
        let blankId = 2
        let vocabSize = 3
        let timeSteps = 3

        let arr = try MLMultiArray(
            shape: [1, NSNumber(value: timeSteps), NSNumber(value: vocabSize)],
            dataType: .float32
        )
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        for i in 0..<(timeSteps * vocabSize) { ptr[i] = -100.0 }
        ptr[0 * vocabSize + 0] = 0.0
        ptr[1 * vocabSize + blankId] = 0.0
        ptr[2 * vocabSize + 1] = 0.0

        let greedy = ctcGreedyDecode(logProbs: arr, vocabulary: vocab, blankId: blankId)
        let beam = ctcBeamSearch(
            logProbs: arr, vocabulary: vocab, lm: nil,
            beamWidth: 5, lmWeight: 0.0, blankId: blankId
        )
        XCTAssertEqual(greedy, beam)
    }
}
