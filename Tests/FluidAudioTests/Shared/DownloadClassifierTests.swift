import Foundation
import XCTest

@testable import FluidAudio

/// Characterization tests pinning the two classifiers the #765 refactor will
/// consolidate: the transient/permanent retry policy (`isRetryableDownloadError`)
/// and cancellation-vs-corruption detection (`isCancellationError`). Unifying the
/// three divergent download paths onto one retry policy must not change these
/// verdicts, so they are locked in against the current behavior first.
final class DownloadClassifierTests: XCTestCase {

    typealias HFError = DownloadUtils.HuggingFaceDownloadError

    // MARK: - isRetryableDownloadError: transient (retry)

    func testTransientURLErrorsAreRetryable() {
        let transient: [URLError.Code] = [
            .timedOut, .cannotConnectToHost, .cannotFindHost, .networkConnectionLost,
            .notConnectedToInternet, .dnsLookupFailed, .secureConnectionFailed, .resourceUnavailable,
        ]
        for code in transient {
            XCTAssertTrue(
                DownloadUtils.isRetryableDownloadError(URLError(code)),
                "URLError.\(code) should be retryable")
        }
    }

    func testRateLimitedIsRetryable() {
        XCTAssertTrue(
            DownloadUtils.isRetryableDownloadError(HFError.rateLimited(statusCode: 429, message: "x")))
        XCTAssertTrue(
            DownloadUtils.isRetryableDownloadError(HFError.rateLimited(statusCode: 503, message: "x")))
    }

    func testInvalidArtifactIsRetryable() {
        // An HTML error page / truncated body is usually a transient bad network path.
        XCTAssertTrue(
            DownloadUtils.isRetryableDownloadError(HFError.invalidArtifact(path: "p", reason: "html")))
    }

    func testDownloadFailedWith5xxIsRetryable() {
        for code in [500, 502, 503, 599] {
            let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "HTTP", code: code))
            XCTAssertTrue(DownloadUtils.isRetryableDownloadError(err), "HTTP \(code) should be retryable")
        }
    }

    // MARK: - isRetryableDownloadError: permanent (fail fast)

    func testPermanentURLErrorsAreNotRetryable() {
        for code in [URLError.Code.badURL, .unsupportedURL, .badServerResponse, .cancelled, .fileDoesNotExist] {
            XCTAssertFalse(
                DownloadUtils.isRetryableDownloadError(URLError(code)),
                "URLError.\(code) should not be retryable")
        }
    }

    func testDownloadFailedWith4xxIsNotRetryable() {
        for code in [400, 401, 403, 404, 410] {
            let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "HTTP", code: code))
            XCTAssertFalse(DownloadUtils.isRetryableDownloadError(err), "HTTP \(code) should not retry")
        }
    }

    func testDownloadFailedWithNonHTTPDomainIsNotRetryable() {
        // Only the synthetic "HTTP" domain drives the 5xx rule; a 5xx code in some
        // other domain must not be mistaken for a retryable status.
        let err = HFError.downloadFailed(path: "p", underlying: NSError(domain: "Other", code: 500))
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(err))
    }

    func testOtherErrorsAreNotRetryable() {
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(HFError.invalidResponse))
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(HFError.modelNotFound(path: "p")))
        XCTAssertFalse(
            DownloadUtils.isRetryableDownloadError(HFError.htmlErrorResponse(path: "p", snippet: "s")))
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(CancellationError()))
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(NSError(domain: "x", code: 1)))
    }

    // MARK: - isCancellationError

    func testRecognizesCancellation() {
        XCTAssertTrue(DownloadUtils.isCancellationError(CancellationError()))
        XCTAssertTrue(DownloadUtils.isCancellationError(URLError(.cancelled)))
        XCTAssertTrue(
            DownloadUtils.isCancellationError(NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)))
        XCTAssertTrue(
            DownloadUtils.isCancellationError(NSError(domain: NSCocoaErrorDomain, code: NSUserCancelledError)))
    }

    func testRecognizesCancellationNestedInUnderlyingError() {
        let wrapped = NSError(
            domain: "Wrapper", code: 1,
            userInfo: [NSUnderlyingErrorKey: URLError(.cancelled) as NSError])
        XCTAssertTrue(DownloadUtils.isCancellationError(wrapped))
    }

    func testNonCancellationErrorsAreNotCancellation() {
        XCTAssertFalse(DownloadUtils.isCancellationError(URLError(.timedOut)))
        XCTAssertFalse(DownloadUtils.isCancellationError(NSError(domain: "x", code: 1)))
        XCTAssertFalse(
            DownloadUtils.isCancellationError(HFError.downloadFailed(path: "p", underlying: URLError(.timedOut))))
    }

    // MARK: - Cross-check: a cancelled download is neither retryable nor treated as corruption

    func testCancellationIsNotRetryableButIsCancellation() {
        let cancelled = URLError(.cancelled)
        XCTAssertFalse(DownloadUtils.isRetryableDownloadError(cancelled))
        XCTAssertTrue(DownloadUtils.isCancellationError(cancelled))
    }
}
