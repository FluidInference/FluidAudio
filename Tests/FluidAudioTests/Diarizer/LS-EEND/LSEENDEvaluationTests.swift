import Foundation
import XCTest

@testable import FluidAudio

final class LSEENDEvaluationTests: XCTestCase {

    func testRTTMRoundTripPreservesFrameActivity() throws {
        let reference = LSEENDMatrix(
            validatingRows: 8,
            columns: 2,
            values: [
                1, 0,
                1, 0,
                1, 1,
                0, 1,
                0, 1,
                0, 0,
                1, 0,
                1, 0,
            ]
        )
        let outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("rttm")
        defer { try? FileManager.default.removeItem(at: outputURL) }

        try LSEENDEvaluation.writeRTTM(
            recordingID: "fixture",
            binaryPrediction: reference,
            outputURL: outputURL,
            frameRate: 2.0,
            speakerLabels: ["spkA", "spkB"]
        )

        let parsed = try LSEENDEvaluation.parseRTTM(url: outputURL)
        XCTAssertEqual(parsed.speakers, ["spkA", "spkB"])

        let reconstructed = LSEENDEvaluation.rttmToFrameMatrix(
            entries: parsed.entries,
            speakers: parsed.speakers,
            numFrames: reference.rows,
            frameRate: 2.0
        )
        XCTAssertEqual(reconstructed, reference)
    }

    func testComputeDERHandlesSpeakerPermutation() {
        let reference = LSEENDMatrix(
            validatingRows: 4,
            columns: 2,
            values: [
                1, 0,
                1, 0,
                0, 1,
                0, 1,
            ]
        )
        let probabilities = LSEENDMatrix(
            validatingRows: 4,
            columns: 2,
            values: [
                0.1, 0.9,
                0.2, 0.8,
                0.8, 0.1,
                0.9, 0.2,
            ]
        )

        let result = LSEENDEvaluation.computeDER(
            probabilities: probabilities,
            referenceBinary: reference,
            settings: .init(threshold: 0.5, medianWidth: 1, collarSeconds: 0, frameRate: 10)
        )

        XCTAssertEqual(result.der, 0, accuracy: 1e-7)
        XCTAssertEqual(result.assignment, [0: 1, 1: 0])
        XCTAssertTrue(result.unmatchedPredictionIndices.isEmpty)
        XCTAssertEqual(result.mappedBinary, reference)
    }

    func testComputeDERCountsUnmatchedPredictionAsFalseAlarm() {
        let reference = LSEENDMatrix(
            validatingRows: 3,
            columns: 1,
            values: [
                1,
                1,
                0,
            ]
        )
        let probabilities = LSEENDMatrix(
            validatingRows: 3,
            columns: 2,
            values: [
                0.9, 0.8,
                0.9, 0.8,
                0.1, 0.8,
            ]
        )

        let result = LSEENDEvaluation.computeDER(
            probabilities: probabilities,
            referenceBinary: reference,
            settings: .init(threshold: 0.5, medianWidth: 1, collarSeconds: 0, frameRate: 10)
        )

        XCTAssertEqual(result.speakerScored, 2, accuracy: 1e-7)
        XCTAssertEqual(result.speakerMiss, 0, accuracy: 1e-7)
        XCTAssertEqual(result.speakerError, 0, accuracy: 1e-7)
        XCTAssertEqual(result.speakerFalseAlarm, 3, accuracy: 1e-7)
        XCTAssertEqual(result.der, 1.5, accuracy: 1e-7)
        XCTAssertEqual(result.unmatchedPredictionIndices, [1])
    }

    func testCollarMaskExcludesFramesAroundTransitions() {
        let reference = LSEENDMatrix(
            validatingRows: 6,
            columns: 1,
            values: [
                0,
                0,
                1,
                1,
                0,
                0,
            ]
        )

        let mask = LSEENDEvaluation.collarMask(reference: reference, collarFrames: 1)
        XCTAssertEqual(mask, [true, false, false, false, false, true])
    }

    func testThresholdAndMedianFilterBehaveDeterministically() {
        let probabilities = LSEENDMatrix(
            validatingRows: 5,
            columns: 1,
            values: [
                0.1,
                0.9,
                0.2,
                0.9,
                0.1,
            ]
        )

        let binary = LSEENDEvaluation.threshold(probabilities: probabilities, value: 0.5)
        XCTAssertEqual(binary.values, [0, 1, 0, 1, 0])

        let filtered = LSEENDEvaluation.medianFilter(binary: binary, width: 3)
        XCTAssertEqual(filtered.values, [1, 0, 1, 0, 1])
    }
}
