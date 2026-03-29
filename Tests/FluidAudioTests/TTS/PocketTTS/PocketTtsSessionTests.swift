import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsSessionTests: XCTestCase {

    // MARK: - Manager Guard Tests

    func testMakeSessionFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.makeSession()
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testMakeSessionWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.makeSession(voiceData: fakeVoiceData)
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - KV Cache Clone Tests

    func testCloneKVCacheStateProducesIndependentCopy() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState()

        // Write a known value into the original
        let ptr = original.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        ptr[0] = 42.0
        original.positions[0][0] = NSNumber(value: Float(7.0))

        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        // Clone should have the same values
        let clonePtr = clone.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(clonePtr[0], 42.0)
        XCTAssertEqual(clone.positions[0][0].floatValue, 7.0)

        // Mutating clone should not affect original
        clonePtr[0] = 99.0
        clone.positions[0][0] = NSNumber(value: Float(15.0))
        XCTAssertEqual(ptr[0], 42.0, "Original cache should be unaffected by clone mutation")
        XCTAssertEqual(
            original.positions[0][0].floatValue, 7.0,
            "Original position should be unaffected by clone mutation"
        )
    }

    func testCloneKVCacheStatePreservesShape() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState()
        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        XCTAssertEqual(clone.caches.count, original.caches.count)
        XCTAssertEqual(clone.positions.count, original.positions.count)

        for i in 0..<clone.caches.count {
            XCTAssertEqual(clone.caches[i].shape, original.caches[i].shape)
            XCTAssertEqual(clone.caches[i].dataType, original.caches[i].dataType)
        }
    }

}
