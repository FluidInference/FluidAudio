import Foundation

/// Core ML compilation relocates package files, which can break relative HF cache symlinks.
struct CuaS1FormsPackage: Sendable {
    let url: URL
    private let temporaryDirectory: URL?

    static func prepare(_ modelURL: URL) throws -> Self {
        let source = modelURL.resolvingSymlinksInPath()
        let manager = FileManager.default
        let entries = manager.enumerator(at: source, includingPropertiesForKeys: [.isSymbolicLinkKey])
        var containsLinks = false
        while let entry = entries?.nextObject() as? URL {
            if try entry.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true {
                containsLinks = true
                break
            }
        }
        guard containsLinks else { return Self(url: source, temporaryDirectory: nil) }

        let temporary = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let destination = temporary.appendingPathComponent(modelURL.lastPathComponent, isDirectory: true)
        try manager.createDirectory(at: temporary, withIntermediateDirectories: true)
        do {
            try copyResolvingLinks(from: source, to: destination, ancestors: [])
            return Self(url: destination, temporaryDirectory: temporary)
        } catch {
            try? manager.removeItem(at: temporary)
            throw error
        }
    }

    func cleanup() {
        guard let temporaryDirectory else { return }
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    private static func copyResolvingLinks(from source: URL, to destination: URL, ancestors: Set<String>) throws {
        let resolved = source.resolvingSymlinksInPath()
        let manager = FileManager.default
        guard try resolved.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true else {
            try manager.copyItem(at: resolved, to: destination)
            return
        }
        guard !ancestors.contains(resolved.path) else {
            throw CuaS1FormsError.invalidModel("Package contains a symbolic-link cycle")
        }
        var nextAncestors = ancestors
        nextAncestors.insert(resolved.path)
        try manager.createDirectory(at: destination, withIntermediateDirectories: false)
        for child in try manager.contentsOfDirectory(at: resolved, includingPropertiesForKeys: nil) {
            try copyResolvingLinks(
                from: child, to: destination.appendingPathComponent(child.lastPathComponent), ancestors: nextAncestors)
        }
    }
}
