import XCTest

@testable import FluidAudio

/// `ModelHub.loadModels` treats a failed first load as a corrupted
/// cache and deletes the repo folder before re-downloading. Cancellation
/// (task cancelled during app teardown, user abort mid-load) must NOT be
/// classified as corruption — a cancelled load once wiped a fully
/// downloaded multi-hundred-MB model repo.
///
/// These tests validate the `isCancellationError` classifier that gates
/// the delete-and-redownload fallback.
final class DownloadCancellationTests: XCTestCase {

    // MARK: - cancellation shapes that must be detected

    func testSwiftCancellationErrorDetected() {
        XCTAssertTrue(RetryPolicy.isCancellation(CancellationError()))
    }

    func testURLErrorCancelledDetected() {
        XCTAssertTrue(RetryPolicy.isCancellation(URLError(.cancelled)))
    }

    func testRawNSURLErrorCancelledDetected() {
        let error = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)
        XCTAssertTrue(RetryPolicy.isCancellation(error))
    }

    func testCocoaUserCancelledDetected() {
        let error = NSError(domain: NSCocoaErrorDomain, code: NSUserCancelledError)
        XCTAssertTrue(RetryPolicy.isCancellation(error))
    }

    func testUnderlyingCancellationDetected() {
        let underlying = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)
        let wrapper = NSError(
            domain: "com.example.wrapper", code: 1,
            userInfo: [NSUnderlyingErrorKey: underlying])
        XCTAssertTrue(RetryPolicy.isCancellation(wrapper))
    }

    func testDeeplyNestedCancellationDetected() {
        let leaf = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)
        let mid = NSError(
            domain: "com.example.mid", code: 2,
            userInfo: [NSUnderlyingErrorKey: leaf])
        let top = NSError(
            domain: "com.example.top", code: 3,
            userInfo: [NSUnderlyingErrorKey: mid])
        XCTAssertTrue(RetryPolicy.isCancellation(top))
    }

    // MARK: - genuine failures must NOT be classified as cancellation

    func testTimeoutNotCancellation() {
        XCTAssertFalse(RetryPolicy.isCancellation(URLError(.timedOut)))
    }

    func testGenericCocoaErrorNotCancellation() {
        let error = NSError(domain: NSCocoaErrorDomain, code: NSFileReadCorruptFileError)
        XCTAssertFalse(RetryPolicy.isCancellation(error))
    }

    func testMatchingCodeWrongDomainNotCancellation() {
        // -999 only means "cancelled" inside NSURLErrorDomain.
        let error = NSError(domain: "com.example.custom", code: NSURLErrorCancelled)
        XCTAssertFalse(RetryPolicy.isCancellation(error))
    }

    func testNonCancellationUnderlyingChainNotClassified() {
        // A wrapped chain with no cancellation anywhere stays classified
        // as a genuine failure.
        let inner = NSError(domain: "com.example.a", code: 1)
        let outer = NSError(
            domain: "com.example.b", code: 2,
            userInfo: [NSUnderlyingErrorKey: inner])
        XCTAssertFalse(RetryPolicy.isCancellation(outer))
    }

    func testDeeplyBuriedCancellationDetected() {
        // Cancellation wrapped far deeper than any realistic chain is still
        // found: the walk has no depth cap, and detecting cancellation
        // anywhere is the safe direction (never wipe a good cache).
        var error = NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)
        for i in 0..<12 {
            error = NSError(
                domain: "com.example.wrap\(i)", code: i,
                userInfo: [NSUnderlyingErrorKey: error])
        }
        XCTAssertTrue(RetryPolicy.isCancellation(error))
    }

    func testSelfReferentialChainTerminates() {
        // A genuinely cyclic chain must terminate rather than loop forever.
        // Plain `NSError` can't form a cycle (its `userInfo` is fixed at
        // init), so a subclass whose underlying error is itself is the only
        // honest way to construct one. No cancellation on the cycle → false.
        XCTAssertFalse(RetryPolicy.isCancellation(SelfReferentialError()))
    }
}

/// `NSError` whose `NSUnderlyingErrorKey` points back at itself, used to
/// prove the identity-tracked walk terminates on a cycle.
private final class SelfReferentialError: NSError, @unchecked Sendable {
    init() { super.init(domain: "com.example.cycle", code: 1, userInfo: nil) }
    required init?(coder: NSCoder) { super.init(coder: coder) }
    override var userInfo: [String: Any] { [NSUnderlyingErrorKey: self] }
}
