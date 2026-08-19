import Foundation

/// Validates and transactionally repairs legacy Kokoro ANE model caches.
///
/// Early copies of the seven-stage bundles could contain dynamic MIL inputs
/// without the `FlexibleShapeInformation` function attribute. OS 26 tolerated
/// those bundles, but the OS 27 E5 runtime can execute them with unknown shapes
/// and return invalid values (issue #738). The current Hugging Face bundles
/// contain the attribute; this migrator makes sure existence-only caching does
/// not keep the older artifacts forever.
enum KokoroAneModelCompatibility {
    private static let flexibleShapeAttribute = Data("[FlexibleShapeInformation =".utf8)
    private static let backupSuffix = ".pre-flexible-shape-migration"

    static func milProgramHasFlexibleShapeInformation(_ data: Data) -> Bool {
        data.range(of: flexibleShapeAttribute) != nil
    }

    static func bundleHasFlexibleShapeInformation(at bundleURL: URL) -> Bool {
        let milURL = bundleURL.appendingPathComponent("model.mil")
        guard let mil = try? Data(contentsOf: milURL, options: .mappedIfSafe) else {
            return false
        }
        return milProgramHasFlexibleShapeInformation(mil)
    }

    static func existingBundlesRequiringMigration(
        in repoDirectory: URL,
        modelNames: Set<String>,
        fileManager: FileManager = .default
    ) -> [String] {
        modelNames.filter { name in
            let bundleURL = repoDirectory.appendingPathComponent(name)
            return fileManager.fileExists(atPath: bundleURL.path)
                && !bundleHasFlexibleShapeInformation(at: bundleURL)
        }.sorted()
    }

    static func backupURL(for modelName: String, in repoDirectory: URL) -> URL {
        repoDirectory.appendingPathComponent(modelName + backupSuffix)
    }
}

/// Serializes cache migration within the process. Multiple managers may be
/// initialized concurrently, but only one may move or replace a bundle.
actor KokoroAneModelCacheMigrationCoordinator {
    static let shared = KokoroAneModelCacheMigrationCoordinator()

    private let logger = AppLogger(category: "KokoroAneModelCacheMigration")

    func repairIfNeeded(
        repo: Repo,
        modelsDirectory: URL,
        repoDirectory: URL,
        progressHandler: ProgressHandler?
    ) async throws {
        let modelNames = ModelNames.KokoroAne.requiredCoreMLModels
        try recoverInterruptedMigration(
            in: repoDirectory, modelNames: modelNames)

        let incompatible = KokoroAneModelCompatibility.existingBundlesRequiringMigration(
            in: repoDirectory, modelNames: modelNames)
        guard !incompatible.isEmpty else { return }

        logger.warning(
            "Replacing legacy Kokoro ANE bundles without OS 27 flexible-shape metadata: "
                + incompatible.joined(separator: ", "))

        var backedUp: [String] = []
        do {
            for name in incompatible {
                let currentURL = repoDirectory.appendingPathComponent(name)
                let backupURL = KokoroAneModelCompatibility.backupURL(
                    for: name, in: repoDirectory)
                try FileManager.default.moveItem(at: currentURL, to: backupURL)
                backedUp.append(name)
            }

            try await ModelHub.download(
                repo,
                to: modelsDirectory,
                progressHandler: progressHandler
            )

            let invalidReplacements = backedUp.filter { name in
                let bundleURL = repoDirectory.appendingPathComponent(name)
                return !KokoroAneModelCompatibility.bundleHasFlexibleShapeInformation(
                    at: bundleURL)
            }
            guard invalidReplacements.isEmpty else {
                throw KokoroAneError.downloadFailed(
                    "Replacement bundles still lack FlexibleShapeInformation: "
                        + invalidReplacements.joined(separator: ", "))
            }

            for name in backedUp {
                let backupURL = KokoroAneModelCompatibility.backupURL(
                    for: name, in: repoDirectory)
                do {
                    try FileManager.default.removeItem(at: backupURL)
                } catch {
                    logger.warning(
                        "Could not remove Kokoro ANE migration backup at \(backupURL.path): "
                            + error.localizedDescription)
                }
            }
            logger.info("Kokoro ANE flexible-shape cache migration completed")
        } catch {
            rollback(backedUp, in: repoDirectory)
            throw error
        }
    }

    /// Restore a backup left by process termination during migration. A valid
    /// replacement wins; otherwise the previous bundle is restored before a
    /// fresh transactional attempt starts.
    private func recoverInterruptedMigration(
        in repoDirectory: URL,
        modelNames: Set<String>
    ) throws {
        for name in modelNames.sorted() {
            let currentURL = repoDirectory.appendingPathComponent(name)
            let backupURL = KokoroAneModelCompatibility.backupURL(
                for: name, in: repoDirectory)
            guard FileManager.default.fileExists(atPath: backupURL.path) else {
                continue
            }

            if KokoroAneModelCompatibility.bundleHasFlexibleShapeInformation(
                at: currentURL)
            {
                do {
                    try FileManager.default.removeItem(at: backupURL)
                } catch {
                    logger.warning(
                        "Could not remove stale Kokoro ANE migration backup at \(backupURL.path): "
                            + error.localizedDescription)
                }
                continue
            }

            if FileManager.default.fileExists(atPath: currentURL.path) {
                try FileManager.default.removeItem(at: currentURL)
            }
            try FileManager.default.moveItem(at: backupURL, to: currentURL)
            logger.info("Recovered interrupted Kokoro ANE cache migration for \(name)")
        }
    }

    private func rollback(_ modelNames: [String], in repoDirectory: URL) {
        for name in modelNames.reversed() {
            let currentURL = repoDirectory.appendingPathComponent(name)
            let backupURL = KokoroAneModelCompatibility.backupURL(
                for: name, in: repoDirectory)
            do {
                if FileManager.default.fileExists(atPath: currentURL.path) {
                    try FileManager.default.removeItem(at: currentURL)
                }
                if FileManager.default.fileExists(atPath: backupURL.path) {
                    try FileManager.default.moveItem(at: backupURL, to: currentURL)
                }
            } catch {
                logger.error(
                    "Failed to roll back Kokoro ANE cache migration for \(name): "
                        + error.localizedDescription)
            }
        }
    }
}
