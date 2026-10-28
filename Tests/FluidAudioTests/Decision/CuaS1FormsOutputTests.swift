import XCTest

@testable import FluidAudio

final class CuaS1FormsOutputTests: XCTestCase {
    // Actual FP16/ANE output for published synthetic row 19270, label 2.
    // Dataset SHA-256: d63a7e0db195d4d20154a40b2f8dd09ce3bb65487a158c638da5c609d4475e7c
    // Model SHA-256: 70485fc18cbb21785df833cbddddc0b5b59acb00d22394b76e55307e2c135dd0
    // Full fixture/provenance: Mobius tests/fixtures/synthetic-normalization-output.json.
    private let logits: [Float] = [
        20.078125, -2.6953125, 26.53125, -1.556640625,
        -8.265625, -14.171875, 1.1025390625, 3.796875,
        -0.475341796875, -8.8203125, 19.21875, 8.078125,
        -0.100830078125, -9.2578125, 4.2734375, -7.91015625,
        0.382568359375, -3.333984375, -5.40625, 20.890625,
        -10000.0, -10000.0, -10000.0, -10000.0,
        -10000.0, -10000.0, -10000.0, -10000.0,
        -10000.0, -10000.0, -10000.0, -10000.0,
    ]
    private let raw: [Float] = [
        0.0015707015991210938, 0.0, 0.9931640625, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.00066375732421875, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0035381317138671875,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]

    func testRecordedFP16SumFailureUsesStableSoftmaxAndRetainsRawOutput() throws {
        let output = try CuaS1FormsOutput(logits: logits, rawProbabilities: raw, optionCount: 20)
        XCTAssertEqual(output.selectedIndex, 2)
        XCTAssertEqual(output.logits, Array(logits.prefix(20)))
        XCTAssertEqual(output.rawProbabilities, Array(raw.prefix(20)))
        XCTAssertEqual(output.rawProbabilities.reduce(0, +), 0.99893665, accuracy: 0.0000001)
        XCTAssertEqual(output.probabilities.reduce(0, +), 1, accuracy: 0.0000001)
        // Independent double-precision reference from the recorded logits.
        XCTAssertEqual(output.probabilities[2], 0.9942399736656553, accuracy: 0.0000001)
        XCTAssertTrue(output.probabilities.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
    }

    func testSoftmaxRemainsFiniteWithLargeLogits() throws {
        // Rescale the recorded logits to exercise exponent overflow protection.
        let scaled = logits.map { $0 * 1e30 }
        let output = try CuaS1FormsOutput(logits: scaled, rawProbabilities: raw, optionCount: 20)
        XCTAssertEqual(output.selectedIndex, 2)
        XCTAssertEqual(output.probabilities[2], 1)
        XCTAssertEqual(output.probabilities.reduce(0, +), 1)
        XCTAssertTrue(output.probabilities.allSatisfy(\.isFinite))
    }

    func testCorruptTensorsAreStillRejected() {
        for value in [Float.nan, Float.infinity, -Float.infinity] {
            var changed = logits
            changed[0] = value
            XCTAssertThrowsError(try CuaS1FormsOutput(logits: changed, rawProbabilities: raw, optionCount: 20))
        }
        for value in [Float.nan, Float.infinity, -0.01, 1.01] {
            var changed = raw
            changed[0] = value
            XCTAssertThrowsError(try CuaS1FormsOutput(logits: logits, rawProbabilities: changed, optionCount: 20))
        }
        var padding = raw
        padding[31] = 0.01
        XCTAssertThrowsError(try CuaS1FormsOutput(logits: logits, rawProbabilities: padding, optionCount: 20))
        XCTAssertThrowsError(
            try CuaS1FormsOutput(logits: Array(logits.dropLast()), rawProbabilities: raw, optionCount: 20))
        XCTAssertThrowsError(
            try CuaS1FormsOutput(logits: logits, rawProbabilities: Array(raw.dropLast()), optionCount: 20))
        for count in [0, 1, 33] {
            XCTAssertThrowsError(try CuaS1FormsOutput(logits: logits, rawProbabilities: raw, optionCount: count))
        }
    }
}
