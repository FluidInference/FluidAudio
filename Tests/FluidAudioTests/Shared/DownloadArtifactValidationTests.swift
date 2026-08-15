import XCTest

@testable import FluidAudio

/// `FileDownloader.validateDownloadedArtifact` rejects HTML error pages and
/// truncated bodies before they reach the cache (issue #740).
final class DownloadArtifactValidationTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fluidaudio-artifact-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    }

    // MARK: - helpers

    private func writeTemp(_ data: Data, name: String = UUID().uuidString) throws -> URL {
        let url = tempDir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private func response(
        contentType: String? = "application/octet-stream"
    ) -> HTTPURLResponse {
        var headers: [String: String] = [:]
        if let contentType { headers["Content-Type"] = contentType }
        return HTTPURLResponse(
            url: URL(string: "https://huggingface.co/test/file")!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: headers
        )!
    }

    private func assertInvalid(
        _ body: @autoclosure () throws -> Void,
        reasonContains: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        do {
            try body()
            XCTFail("expected invalidArtifact to be thrown", file: file, line: line)
        } catch let DownloadError.invalidArtifact(_, reason) {
            XCTAssertTrue(
                reason.lowercased().contains(reasonContains.lowercased()),
                "reason \"\(reason)\" should mention \"\(reasonContains)\"",
                file: file, line: line
            )
        } catch {
            XCTFail("expected invalidArtifact, got: \(error)", file: file, line: line)
        }
    }

    // MARK: - looksLikeHTML

    func testLooksLikeHTMLDetectsDoctype() {
        XCTAssertTrue(HFClient.looksLikeHTML(Data("<!DOCTYPE html><html></html>".utf8)))
    }

    func testLooksLikeHTMLDetectsLeadingWhitespaceAndCasing() {
        XCTAssertTrue(HFClient.looksLikeHTML(Data("\n\n   <HTML lang=\"en\">".utf8)))
    }

    func testLooksLikeHTMLDetectsXMLProxyEnvelope() {
        XCTAssertTrue(HFClient.looksLikeHTML(Data("<?xml version=\"1.0\"?><error/>".utf8)))
    }

    func testLooksLikeHTMLAllowsBinaryWeights() {
        // Markup-like bytes mid-stream ('<h') must not trip the leading-byte check.
        let binary = Data([0x00, 0x01, 0x02, 0xFF, 0xFE, 0x3C, 0x68])
        XCTAssertFalse(HFClient.looksLikeHTML(binary))
    }

    func testLooksLikeHTMLAllowsJSON() {
        XCTAssertFalse(HFClient.looksLikeHTML(Data("{\"vocab\": 1}".utf8)))
    }

    // MARK: - validateDownloadedArtifact

    func testValidArtifactPasses() throws {
        let payload = Data(repeating: 0xAB, count: 1024)
        let url = try writeTemp(payload)
        XCTAssertNoThrow(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(), path: "Model.mlmodelc/weights/weight.bin",
                expectedSize: 1024))
    }

    func testValidArtifactWithUnknownSizeSkipsSizeCheck() throws {
        let url = try writeTemp(Data(repeating: 0x01, count: 50))
        XCTAssertNoThrow(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(), path: "file.json", expectedSize: -1))
    }

    func testRejectsHTMLContentType() throws {
        let url = try writeTemp(Data(repeating: 0xAB, count: 1024))
        assertInvalid(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(contentType: "text/html; charset=utf-8"),
                path: "Model.mlmodelc/coremldata.bin", expectedSize: 1024),
            reasonContains: "content-type")
    }

    func testRejectsEmptyBody() throws {
        let url = try writeTemp(Data())
        assertInvalid(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(), path: "file.bin", expectedSize: 0),
            reasonContains: "empty")
    }

    func testRejectsHTMLBodyServedAsBinaryContentType() throws {
        let html = Data("<!DOCTYPE html>\n<html><body>Proxy error</body></html>".utf8)
        let url = try writeTemp(html)
        assertInvalid(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(contentType: "application/octet-stream"),
                path: "Model.mlmodelc/weights/weight.bin", expectedSize: -1),
            reasonContains: "html")
    }

    func testRejectsTruncatedBody() throws {
        let url = try writeTemp(Data(repeating: 0x7F, count: 500))
        assertInvalid(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(), path: "Model.mlmodelc/weights/weight.bin",
                expectedSize: 1000),
            reasonContains: "size mismatch")
    }

    func testRejectsOversizedBody() throws {
        let url = try writeTemp(Data(repeating: 0x7F, count: 2000))
        assertInvalid(
            try FileDownloader.validateDownloadedArtifact(
                at: url, response: response(), path: "file.bin", expectedSize: 1000),
            reasonContains: "size mismatch")
    }

    // MARK: - error description

    func testInvalidArtifactErrorDescription() {
        let err = DownloadError.invalidArtifact(
            path: "Encoder.mlmodelc/weights/weight.bin", reason: "empty file")
        XCTAssertEqual(
            err.errorDescription,
            "Downloaded artifact for Encoder.mlmodelc/weights/weight.bin is invalid (empty file); refusing to cache it."
        )
    }
}
