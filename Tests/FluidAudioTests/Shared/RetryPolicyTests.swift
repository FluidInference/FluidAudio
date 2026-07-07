import Foundation
import XCTest
import os

@testable import FluidAudio

/// Unit tests for the extracted retry primitive (#765 Wave 2). Classification
/// itself is characterized by `DownloadClassifierTests` (which now runs
/// through the `DownloadUtils.isRetryableDownloadError` forward); these tests
/// pin the loop mechanics: attempt counting, fail-fast on permanent errors,
/// and error propagation.
final class RetryPolicyTests: XCTestCase {

    private final class Counter: Sendable {
        private let value = OSAllocatedUnfairLock<Int>(initialState: 0)
        func increment() -> Int {
            value.withLock { (v: inout Int) -> Int in
                v += 1
                return v
            }
        }
        var count: Int { value.withLock { $0 } }
    }

    func testTransientFailuresConsumeAllAttemptsThenThrowLastError() async {
        let counter = Counter()
        do {
            let _: Never = try await RetryPolicy.withRetry(
                label: "transient", maxAttempts: 3, minBackoff: 0.01
            ) { _ in
                _ = counter.increment()
                throw URLError(.timedOut)
            }
            XCTFail("expected error")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .timedOut)
        } catch {
            XCTFail("expected URLError.timedOut, got \(error)")
        }
        XCTAssertEqual(counter.count, 3, "transient errors must use the whole attempt budget")
    }

    func testPermanentFailureFailsFastOnFirstAttempt() async {
        let counter = Counter()
        do {
            let _: Never = try await RetryPolicy.withRetry(
                label: "permanent", maxAttempts: 4, minBackoff: 0.01
            ) { _ in
                _ = counter.increment()
                throw DownloadUtils.HuggingFaceDownloadError.downloadFailed(
                    path: "missing.bin", underlying: NSError(domain: "HTTP", code: 404))
            }
            XCTFail("expected error")
        } catch {
            // expected
        }
        XCTAssertEqual(counter.count, 1, "a permanent 404 must not consume the backoff budget")
    }

    func testSucceedsAfterTransientFailuresAndReportsAttemptNumber() async throws {
        let counter = Counter()
        let result = try await RetryPolicy.withRetry(
            label: "flaky", maxAttempts: 3, minBackoff: 0.01
        ) { attempt -> Int in
            XCTAssertEqual(attempt, counter.increment(), "closure must receive the 1-based attempt")
            if attempt < 3 {
                throw DownloadUtils.HuggingFaceDownloadError.rateLimited(
                    statusCode: 503, message: "busy")
            }
            return attempt
        }
        XCTAssertEqual(result, 3)
        XCTAssertEqual(counter.count, 3)
    }

    func testCancellationPropagatesWithoutRetry() async {
        let counter = Counter()
        do {
            let _: Never = try await RetryPolicy.withRetry(
                label: "cancelled", maxAttempts: 4, minBackoff: 0.01
            ) { _ in
                _ = counter.increment()
                throw URLError(.cancelled)
            }
            XCTFail("expected error")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .cancelled)
        } catch {
            XCTFail("expected URLError.cancelled, got \(error)")
        }
        XCTAssertEqual(counter.count, 1, "cancellation is not transient; it must not retry")
    }
}
