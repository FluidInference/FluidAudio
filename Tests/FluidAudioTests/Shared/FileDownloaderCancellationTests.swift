import Foundation
import XCTest

@testable import FluidAudio

/// Regression tests for the pre-cancelled download race (#863 follow-up): a
/// Swift task that is already cancelled runs `withTaskCancellationHandler`'s
/// handler BEFORE the operation body, so `StreamingDownloadDelegate.cancel()`
/// fired with no URLSession task registered — and the request then started
/// anyway. The delegate now latches the cancel and `attach` fails the
/// continuation without ever resuming the URLSession task.
final class FileDownloaderCancellationTests: XCTestCase {

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("FileDownloaderCancel-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
        ResumeStubURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        ResumeStubURLProtocol.reset()
        try? FileManager.default.removeItem(at: workDir)
    }

    private var stubConfiguration: URLSessionConfiguration {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ResumeStubURLProtocol.self]
        return config
    }

    func testPreCancelledDownloadThrowsCancellationWithoutStartingARequest() async throws {
        // A scripted 200 the download would consume if it (wrongly) ran.
        ResumeStubURLProtocol.enqueue(
            .init(
                status: 200, headers: ["ETag": "\"v1\""],
                chunks: [Data("BINARYCONTENT-0123456789".utf8)]))

        let request = URLRequest(url: URL(string: "https://stub.test/repo/resolve/main/model.bin")!)
        let partial = workDir.appendingPathComponent("model.bin.partial")
        let configuration = stubConfiguration

        let task = Task<URL, Error> {
            // Enter the download only once cancellation has definitely landed,
            // so streamDownload begins inside an already-cancelled task and
            // its onCancel handler runs before the operation body.
            while !Task.isCancelled { await Task.yield() }
            return try await FileDownloader.download(
                request: request,
                path: "model.bin",
                expectedSize: 24,
                partialFileURL: partial,
                onProgress: nil,
                maxAttempts: 3,
                minBackoff: 0.01,
                configuration: configuration
            )
        }
        task.cancel()
        let result = await task.result

        guard case .failure(let error) = result else {
            return XCTFail("pre-cancelled download must throw, got a result")
        }
        XCTAssertTrue(
            RetryPolicy.isCancellation(error),
            "pre-cancelled download must surface a cancellation (not retry or succeed), got \(error)")
        XCTAssertTrue(
            ResumeStubURLProtocol.recordedRequests().isEmpty,
            "a pre-cancelled download must not start a network request")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: partial.path),
            "a pre-cancelled download must not leave a partial file")
    }
}
