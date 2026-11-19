import Foundation
import XCTest

@testable import FluidAudio

/// Network-free tests for the parts of `MandarinG2pwModel` that don't
/// need the CoreML bundle: the feature names it binds and the
/// `phoneme_mask` row it builds for a target char.
final class MandarinG2pwModelTests: XCTestCase {

    func testFeatureNamesMatchShippedBundle() {
        // Signature of kokoro-82m-coreml/ANE-zh/g2pw/g2pw.mlmodelc.
        XCTAssertEqual(
            Set(MandarinG2pwModel.Feature.inputs.map { $0.rawValue }),
            ["input_ids", "token_type_ids", "attention_mask", "phoneme_mask", "char_ids", "position_ids"])
        XCTAssertEqual(MandarinG2pwModel.Feature.probs.rawValue, "probs")
    }

    func testPhonemeMaskMarksCandidatesOnly() {
        let mask = MandarinG2pwModel.phonemeMask(candidates: [1, 3], labelCount: 5)
        XCTAssertEqual(mask, [0, 1, 0, 1, 0])
    }

    func testPhonemeMaskIgnoresOutOfRange() {
        let mask = MandarinG2pwModel.phonemeMask(candidates: [-1, 2, 7], labelCount: 3)
        XCTAssertEqual(mask, [0, 0, 1])
    }
}
