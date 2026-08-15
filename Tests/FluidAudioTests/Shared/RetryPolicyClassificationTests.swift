import Foundation
import XCTest

@testable import FluidAudio

/// Verdict-table coverage for `RetryPolicy.isRetryable` — which errors the
/// retry loop treats as transient (retry) vs permanent (fail fast). The
/// complementary `isCancellation` classifier is covered by
/// `DownloadCancellationTests`; the loop mechanics by `RetryPolicyTests`.
final class RetryPolicyClassificationTests: XCTestCase {

    private typealias HFError = DownloadError

    // MARK: - Transient (retry)

    func testTransientURLErrorsAreRetryable() {
        let transient: [URLError.Code] = [
            .timedOut, .cannotConnectToHost, .cannotFindHost, .networkConnectionLost,
            .notConnectedToInternet, .dnsLookupFailed, .secureConnectionFailed, .resourceUnavailable,
        ]
        for code in transient {
            XCTAssertTrue(
                RetryPolicy.isRetryable(URLError(code)),
                "URLError.\(code) should be retryable")
        }
    }

    func testRateLimitedIsRetryable() {
        XCTAssertTrue(RetryPolicy.isRetryable(HFError.rateLimited(statusCode: 429, message: "x")))
        XCTAssertTrue(RetryPolicy.isRetryable(HFError.rateLimited(statusCode: 503, message: "x")))
    }

    func testInvalidArtifactIsRetryable() {
        // An HTML error page / truncated body is usually a transient bad network path.
        XCTAssertTrue(RetryPolicy.isRetryable(HFError.invalidArtifact(path: "p", reason: "html")))
    }

    func testDownloadFailedWith5xxIsRetryable() {
        for code in [500, 502, 503, 599] {
            let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "HTTP", code: code))
            XCTAssertTrue(RetryPolicy.isRetryable(err), "HTTP \(code) should be retryable")
        }
    }

    // MARK: - Permanent (fail fast)

    func testPermanentURLErrorsAreNotRetryable() {
        for code in [URLError.Code.badURL, .unsupportedURL, .badServerResponse, .cancelled, .fileDoesNotExist] {
            XCTAssertFalse(
                RetryPolicy.isRetryable(URLError(code)),
                "URLError.\(code) should not be retryable")
        }
    }

    func testDownloadFailedWith4xxIsNotRetryable() {
        for code in [400, 401, 403, 404, 410] {
            let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "HTTP", code: code))
            XCTAssertFalse(RetryPolicy.isRetryable(err), "HTTP \(code) should not retry")
        }
    }

    func testDownloadFailedWithNonHTTPDomainIsNotRetryable() {
        // Only the synthetic "HTTP" domain drives the 5xx rule; a 5xx code in some
        // other domain must not be mistaken for a retryable status.
        let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "Other", code: 500))
        XCTAssertFalse(RetryPolicy.isRetryable(err))
    }

    func testOtherErrorsAreNotRetryable() {
        XCTAssertFalse(RetryPolicy.isRetryable(HFError.invalidResponse))
        XCTAssertFalse(RetryPolicy.isRetryable(HFError.modelNotFound(path: "p")))
        XCTAssertFalse(RetryPolicy.isRetryable(HFError.htmlErrorResponse(path: "p", snippet: "s")))
        XCTAssertFalse(RetryPolicy.isRetryable(CancellationError()))
        XCTAssertFalse(RetryPolicy.isRetryable(NSError(domain: "x", code: 1)))
    }
}
