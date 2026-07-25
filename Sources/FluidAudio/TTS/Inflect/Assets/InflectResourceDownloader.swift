import Foundation

/// Downloads Inflect v2 CoreML bundles from
/// `FluidInference/inflect-v2-coreml`. Each variant lives under its own
/// subdirectory (`micro/`, `nano/`) holding `encoder.mlmodelc` plus the eight
/// `synthesizer_f<N>.mlmodelc` frame buckets.
public enum InflectResourceDownloader {

    private static let logger = AppLogger(category: "InflectResourceDownloader")

    /// Ensure the variant's encoder + all synthesizer buckets are cached
    /// locally. Returns the directory that holds the `.mlmodelc` bundles.
    @discardableResult
    public static func ensureModels(
        variant: InflectVariant,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let repo = variant.repo
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(repo.folderName)

        let allPresent = ModelNames.Inflect.requiredModels.allSatisfy { entry in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(entry).path)
        }

        if allPresent {
            logger.info("Inflect \(variant.rawValue) models found in cache at \(repoDir.path)")
            return repoDir
        }

        logger.info("Downloading Inflect \(variant.rawValue) CoreML models from HuggingFace…")
        do {
            try await ModelHub.download(repo, to: modelsRoot, progressHandler: progressHandler)
        } catch {
            throw InflectError.downloadFailed("\(error)")
        }
        return repoDir
    }

    private static func defaultCacheRoot() throws -> URL {
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}

extension InflectVariant {
    /// HuggingFace repo case for this variant's subdirectory.
    var repo: Repo {
        switch self {
        case .micro: return .inflectMicro
        case .nano: return .inflectNano
        }
    }
}
