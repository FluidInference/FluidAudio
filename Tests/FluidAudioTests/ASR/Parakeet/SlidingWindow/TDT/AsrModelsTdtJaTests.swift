#if os(macOS)
import XCTest
@testable import FluidAudio
import AVFoundation

final class AsrModelsTdtJaTests: XCTestCase {

    func testTdtJaWithAsrModels() async throws {
        // Skip in CI environment - HuggingFace downloads are unreliable
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model download tests in CI environment"
        )

        // Load TDT Japanese models via AsrModels
        print("Loading TDT Japanese models via AsrModels...")
        let models = try await AsrModels.load(from: AsrModels.defaultCacheDirectory(for: .tdtJa), version: .tdtJa)
        print("✅ Models loaded via AsrModels")

        // Verify correct models were loaded
        XCTAssertNotNil(models.encoder, "Encoder should be loaded")
        XCTAssertNotNil(models.preprocessor, "Preprocessor should be loaded")
        XCTAssertNotNil(models.decoder, "Decoder should be loaded")
        XCTAssertNotNil(models.joint, "Joint should be loaded")
        XCTAssertEqual(models.version, .tdtJa, "Version should be .tdtJa")
        XCTAssertEqual(models.vocabulary.count, 3072, "Vocabulary should have 3072 tokens")

        print("✅ TDT Japanese models work with AsrModels!")
    }

    func testTdtJaWithAsrManager() async throws {
        // Skip in CI environment - HuggingFace downloads are unreliable
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["CI"] != nil,
            "Skipping model download tests in CI environment"
        )

        // Load TDT Japanese models
        print("Loading TDT Japanese models...")
        let models = try await AsrModels.load(from: AsrModels.defaultCacheDirectory(for: .tdtJa), version: .tdtJa)
        print("✅ Models loaded")

        // Create AsrManager with the loaded models
        print("Creating AsrManager with TDT Japanese models...")
        let manager = AsrManager(models: models)
        print("✅ AsrManager created with .tdtJa")

        // Verify manager is properly initialized with TDT Japanese models
        let isAvailable = await manager.isAvailable
        XCTAssertTrue(isAvailable, "Manager should be available")

        print("✅ TDT Japanese models successfully work with AsrManager!")
        print("   This allows users to get timing information via AsrManager")
        print("   instead of using TdtJaManager which doesn't provide timing info")
    }
}
#endif
