import Foundation
import XCTest

@testable import FluidAudio

/// Characterization tests pinning the two classifiers the #765 refactor will
/// consolidate: the transient/permanent retry policy (`isRetryableDownloadError`)
/// and cancellation-vs-corruption detection (`isCancellationError`). Unifying the
/// three divergent download paths onto one retry policy must not change these
/// verdicts, so they are locked in against the current behavior first.
final class DownloadClassifierTests: XCTestCase {

    typealias HFError = DownloadError

    // MARK: - isRetryableDownloadError: transient (retry)

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
        XCTAssertTrue(
            RetryPolicy.isRetryable(HFError.rateLimited(statusCode: 429, message: "x")))
        XCTAssertTrue(
            RetryPolicy.isRetryable(HFError.rateLimited(statusCode: 503, message: "x")))
    }

    func testInvalidArtifactIsRetryable() {
        // An HTML error page / truncated body is usually a transient bad network path.
        XCTAssertTrue(
            RetryPolicy.isRetryable(HFError.invalidArtifact(path: "p", reason: "html")))
    }

    func testDownloadFailedWith5xxIsRetryable() {
        for code in [500, 502, 503, 599] {
            let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "HTTP", code: code))
            XCTAssertTrue(RetryPolicy.isRetryable(err), "HTTP \(code) should be retryable")
        }
    }

    // MARK: - isRetryableDownloadError: permanent (fail fast)

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
        XCTAssertFalse(
            RetryPolicy.isRetryable(HFError.htmlErrorResponse(path: "p", snippet: "s")))
        XCTAssertFalse(RetryPolicy.isRetryable(CancellationError()))
        XCTAssertFalse(RetryPolicy.isRetryable(NSError(domain: "x", code: 1)))
    }

    // MARK: - isCancellationError

    func testRecognizesCancellation() {
        XCTAssertTrue(RetryPolicy.isCancellation(CancellationError()))
        XCTAssertTrue(RetryPolicy.isCancellation(URLError(.cancelled)))
        XCTAssertTrue(
            RetryPolicy.isCancellation(NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)))
        XCTAssertTrue(
            RetryPolicy.isCancellation(NSError(domain: NSCocoaErrorDomain, code: NSUserCancelledError)))
    }

    func testRecognizesCancellationNestedInUnderlyingError() {
        let wrapped = NSError(
            domain: "Wrapper", code: 1,
            userInfo: [NSUnderlyingErrorKey: URLError(.cancelled) as NSError])
        XCTAssertTrue(RetryPolicy.isCancellation(wrapped))
    }

    func testNonCancellationErrorsAreNotCancellation() {
        XCTAssertFalse(RetryPolicy.isCancellation(URLError(.timedOut)))
        XCTAssertFalse(RetryPolicy.isCancellation(NSError(domain: "x", code: 1)))
        XCTAssertFalse(
            RetryPolicy.isCancellation(HFError.downloadFailed(path: "p", underlying: URLError(.timedOut))))
    }

    // MARK: - Cross-check: a cancelled download is neither retryable nor treated as corruption

    func testCancellationIsNotRetryableButIsCancellation() {
        let cancelled = URLError(.cancelled)
        XCTAssertFalse(RetryPolicy.isRetryable(cancelled))
        XCTAssertTrue(RetryPolicy.isCancellation(cancelled))
    }
}
