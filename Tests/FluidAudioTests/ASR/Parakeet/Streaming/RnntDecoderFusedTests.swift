import CoreML
import XCTest

@testable import FluidAudio

/// Parity tests for the opt-in fused decoder+joint_decision path.
///
/// These tests require the real Parakeet EOU 160ms models plus the locally compiled
/// `decoder_joint_decision_fused.mlmodelc` in the model cache; they skip when absent
/// (e.g. on CI). No dummy models are used.
final class RnntDecoderFusedTests: XCTestCase {

    private static func modelDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models/parakeet-eou-streaming/160ms", isDirectory: true)
    }

    func testFusedDecodePathMatchesReferenceOnNeutralState() async throws {
        let dir = Self.modelDirectory()
        let decoderUrl = dir.appendingPathComponent("decoder.mlmodelc")
        let jointUrl = dir.appendingPathComponent("joint_decision.mlmodelc")
        let fusedUrl = dir.appendingPathComponent("decoder_joint_decision_fused.mlmodelc")

        try XCTSkipIf(
            !FileManager.default.fileExists(atPath: decoderUrl.path)
                || !FileManager.default.fileExists(atPath: jointUrl.path)
                || !FileManager.default.fileExists(atPath: fusedUrl.path),
            "Parakeet EOU 160ms models (incl. fused mlmodelc) not present in local cache"
        )

        let configuration = MLModelConfiguration()
        let decoderModel = try await MLModel.load(contentsOf: decoderUrl, configuration: configuration)
        let jointModel = try await MLModel.load(contentsOf: jointUrl, configuration: configuration)
        let fusedModel = try await MLModel.load(contentsOf: fusedUrl, configuration: configuration)

        let reference = RnntDecoder(decoderModel: decoderModel, jointModel: jointModel)
        let fused = RnntDecoder(decoderModel: decoderModel, jointModel: jointModel, fusedModel: fusedModel)

        // Neutral (zero) encoder frames with zero LSTM state: both paths must agree
        // (the model emits blank on silence-like input, so this exercises the fused IO
        // contract — input/output names, shapes, and state plumbing — without audio).
        let encoderOutput = try MLMultiArray(shape: [1, 512, 2], dataType: .float32)
        encoderOutput.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.update(repeating: 0, count: ptr.count)
        }

        let refResult = try reference.decodeWithEOU(encoderOutput: encoderOutput, validOutLen: 2)
        let fusedResult = try fused.decodeWithEOU(encoderOutput: encoderOutput, validOutLen: 2)

        XCTAssertEqual(fusedResult.tokenIds, refResult.tokenIds)
        XCTAssertEqual(fusedResult.tokenFrames, refResult.tokenFrames)
        XCTAssertEqual(fusedResult.eouDetected, refResult.eouDetected)

        // State reset after fused predictions (fp16 h/c outputs) must not trap and the
        // decoder must remain usable.
        fused.resetState()
        let afterReset = try fused.decodeWithEOU(encoderOutput: encoderOutput, validOutLen: 2)
        XCTAssertEqual(afterReset.tokenIds, refResult.tokenIds)
    }
}
