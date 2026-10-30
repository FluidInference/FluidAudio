import XCTest

@testable import FluidAudio

/// `finish()` on a cancelled task throws `CancellationError` before any model work (#1033). The check runs ahead
/// of the models-loaded guard, so unloaded managers show it: without the check they throw `notInitialized`.
final class StreamingCancellationTests: XCTestCase {

    /// The error `finish()` throws when called from a task that is already cancelled, nil if it returned.
    private func finishError(_ finish: @escaping @Sendable () async throws -> String) async -> Error? {
        let task = Task { () async throws -> String in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await finish()
        }
        do {
            _ = try await task.value
            return nil
        } catch {
            return error
        }
    }

    func testEouFinishOnACancelledTaskThrowsCancellation() async {
        let manager = StreamingEouAsrManager()
        let error = await finishError { try await manager.finish() }
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
    }

    func testNemotronFinishOnACancelledTaskThrowsCancellation() async {
        let manager = StreamingNemotronAsrManager()
        let error = await finishError { try await manager.finish() }
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
    }

    func testNemotronMultilingualFinishOnACancelledTaskThrowsCancellation() async {
        let manager = StreamingNemotronMultilingualAsrManager()
        let error = await finishError { try await manager.finish() }
        XCTAssertTrue(error is CancellationError, "got \(String(describing: error))")
    }
}
