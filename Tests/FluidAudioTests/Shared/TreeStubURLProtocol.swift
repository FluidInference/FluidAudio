import Foundation

// MARK: - Tree-serving URLProtocol stub

/// Serves canned HF tree JSON per path and a fixed body for every
/// resolve file request, including pinned revisions. Thread-safe via a lock; keyed on URL shape.
///
/// Shared HTTP test double for the `Shared/Download` suite (used by
/// `ProgressSequenceTests` and others that drive `ModelHub`'s listing/download
/// pipeline through the `configuration:` seam).
final class TreeStubURLProtocol: URLProtocol {

    private static let lock = NSLock()
    nonisolated(unsafe) private static var _trees: [String: [[String: Any]]] = [:]
    nonisolated(unsafe) private static var _fileBody = Data()
    nonisolated(unsafe) private static var _fileRequestCount = 0

    static var fileRequestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _fileRequestCount
    }

    static var trees: [String: [[String: Any]]] {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _trees
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _trees = newValue
        }
    }

    static var fileBody: Data {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _fileBody
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _fileBody = newValue
        }
    }

    static func reset() {
        trees = [:]
        fileBody = Data()
        lock.lock()
        _fileRequestCount = 0
        lock.unlock()
    }

    override static func canInit(with request: URLRequest) -> Bool { true }
    override static func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.badURL))
            return
        }
        let path = url.path

        let payload: Data
        if let treeRange = path.range(of: "/tree/") {
            // Listing request: drop the revision component ("" for root).
            let components = path[treeRange.upperBound...].split(separator: "/")
            let key = components.dropFirst().joined(separator: "/")
            guard let items = Self.trees[key],
                let json = try? JSONSerialization.data(withJSONObject: items)
            else {
                respond(status: 404, data: Data("[]".utf8))
                return
            }
            payload = json
        } else if path.contains("/resolve/") {
            Self.lock.lock()
            Self._fileRequestCount += 1
            Self.lock.unlock()
            payload = Self.fileBody
        } else {
            respond(status: 404, data: Data())
            return
        }
        respond(status: 200, data: payload)
    }

    override func stopLoading() {}

    private func respond(status: Int, data: Data) {
        let response = HTTPURLResponse(
            url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": String(data.count)])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }
}
