import Foundation
import XCTest

@testable import FluidAudio

final class ProgressEmitterTests: XCTestCase {

    // MARK: - Session Lifecycle

    func testEnsureSessionYieldsInitialZero() async throws {
        let emitter = ProgressEmitter()
        let stream = await emitter.ensureSession()

        var values: [Double] = []
        for try await value in stream {
            values.append(value)
            if values.count >= 1 { break }
        }

        XCTAssertEqual(values.first, 0.0, "First yielded value should be 0.0")
    }

    func testFinishSessionYieldsOne() async throws {
        let emitter = ProgressEmitter()
        let stream = await emitter.ensureSession()

        // Report some progress, then finish
        await emitter.report(progress: 0.5)
        await emitter.finishSession()

        var values: [Double] = []
        for try await value in stream {
            values.append(value)
        }

        XCTAssertTrue(values.contains(1.0), "finishSession should yield 1.0")
        XCTAssertEqual(values.last, 1.0, "Last value should be 1.0")
    }

    // MARK: - Progress Clamping

    func testReportClampsToZeroOne() async throws {
        let emitter = ProgressEmitter()
        let stream = await emitter.ensureSession()

        await emitter.report(progress: -0.5)
        await emitter.report(progress: 1.5)
        await emitter.finishSession()

        var values: [Double] = []
        for try await value in stream {
            values.append(value)
        }

        // Initial 0.0, clamped -0.5 -> 0.0, clamped 1.5 -> 1.0, finish 1.0
        for value in values {
            XCTAssertGreaterThanOrEqual(value, 0.0, "All values should be >= 0")
            XCTAssertLessThanOrEqual(value, 1.0, "All values should be <= 1")
        }
    }

    // MARK: - Report Without Session

    func testReportWithoutSessionDoesNotCrash() async {
        let emitter = ProgressEmitter()
        // Should silently ignore since no session is active
        await emitter.report(progress: 0.5)
    }

    // MARK: - Fail Session

    func testFailSessionThrowsError() async {
        let emitter = ProgressEmitter()
        let stream = await emitter.ensureSession()

        struct TestError: Error {}
        await emitter.failSession(TestError())

        var threwError = false
        do {
            for try await _ in stream {
                // consume
            }
        } catch {
            threwError = true
        }

        XCTAssertTrue(threwError, "Stream should throw after failSession")
    }
}
