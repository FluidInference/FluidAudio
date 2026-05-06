import Foundation

/// Actor-based store for the StyleTTS2 *shared assets* (vocab, bundle
/// config, voice presets) that live at the root of the upstream
/// HuggingFace repo (`FluidInference/StyleTTS-2-coreml`).
///
/// The 7 `.mlmodelc` bundles themselves are owned by `StyleTTS2ModelStore`
/// and live under the `ANE/` subdirectory of the same repo; this store
/// only handles the language/voice metadata that's shared across them.
///
/// Actor isolation guarantees the lazy bring-up + caches are written by at
/// most one task at a time, and `ensureAssetsAvailable` dedupes concurrent
/// first-callers via a shared `Task` so a parallel bring-up never triggers
/// duplicate network / FS work.
public actor StyleTTS2AssetStore {

    private var repoRootDirectory: URL?
    private var loadTask: Task<URL, Error>?
    private var cachedVocab: StyleTTS2Vocab?
    private var cachedBundleConfig: StyleTTS2BundleConfig?
    private let cacheBaseOverride: URL?

    public init(directory: URL? = nil) {
        self.cacheBaseOverride = directory
    }

    // MARK: - Bring-up

    /// Ensure the shared assets are downloaded and resolve the repo root.
    ///
    /// Concurrent first-callers join an in-flight `Task` instead of each
    /// kicking off their own download. Only the first call's
    /// `progressHandler` is observed; later callers receive the resolved URL
    /// when the shared task finishes.
    public func ensureAssetsAvailable(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        if let dir = repoRootDirectory {
            return dir
        }
        if let task = loadTask {
            return try await task.value
        }
        let baseOverride = cacheBaseOverride
        let task = Task<URL, Error> {
            try await StyleTTS2AssetDownloader.ensureModels(
                directory: baseOverride,
                progressHandler: progressHandler
            )
        }
        loadTask = task
        do {
            let dir = try await task.value
            repoRootDirectory = dir
            loadTask = nil
            return dir
        } catch {
            loadTask = nil
            throw error
        }
    }

    // MARK: - Bundle paths

    /// Path to the espeak-ng IPA vocabulary JSON. Requires
    /// `ensureAssetsAvailable()` to have been called first.
    public func vocabularyURL() throws -> URL {
        let root = try repoRoot()
        return root.appendingPathComponent(ModelNames.StyleTTS2.vocabularyFile)
    }

    /// Path to the bundle `config.json`. Requires
    /// `ensureAssetsAvailable()` to have been called first.
    public func configURL() throws -> URL {
        let root = try repoRoot()
        return root.appendingPathComponent(ModelNames.StyleTTS2.configFile)
    }

    /// Lazily-loaded 178-token IPA vocabulary. Requires
    /// `ensureAssetsAvailable()` to have been called first.
    ///
    /// `StyleTTS2Vocab.load(from:)` is synchronous, so the read-or-fill
    /// sequence below has no `await` between the cache check and the
    /// assignment — actor isolation alone is enough to keep two concurrent
    /// callers from racing to load twice.
    public func vocabulary() throws -> StyleTTS2Vocab {
        if let cached = cachedVocab {
            return cached
        }
        let url = try vocabularyURL()
        let vocab = try StyleTTS2Vocab.load(from: url)
        cachedVocab = vocab
        return vocab
    }

    /// Lazily-loaded `config.json`. Decoded once and cached for the lifetime
    /// of the store. Requires `ensureAssetsAvailable()` to have been called
    /// first. Same synchronous-loader invariant as `vocabulary()`.
    public func bundleConfig() throws -> StyleTTS2BundleConfig {
        if let cached = cachedBundleConfig {
            return cached
        }
        let url = try configURL()
        let config = try StyleTTS2BundleConfig.load(from: url)
        cachedBundleConfig = config
        return config
    }

    /// Resolved repo root URL. Requires `ensureAssetsAvailable()` to have
    /// been called first.
    public func repoRoot() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotLoaded("StyleTTS2 shared assets")
        }
        return root
    }
}
