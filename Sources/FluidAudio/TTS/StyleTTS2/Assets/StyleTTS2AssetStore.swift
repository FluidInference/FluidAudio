import Foundation

/// Actor-based store for the StyleTTS2 *shared assets* (vocab, bundle
/// config, voice presets) that live at the root of the upstream
/// HuggingFace repo (`FluidInference/StyleTTS-2-coreml`).
///
/// The 7 `.mlmodelc` bundles themselves are owned by `StyleTTS2ModelStore`
/// and live under the `ANE/` subdirectory of the same repo; this store
/// only handles the language/voice metadata that's shared across them.
public actor StyleTTS2AssetStore {

    private var repoRootDirectory: URL?
    private var cachedVocab: StyleTTS2Vocab?
    private var cachedBundleConfig: StyleTTS2BundleConfig?
    private let directory: URL?

    public init(directory: URL? = nil) {
        self.directory = directory
    }

    // MARK: - Bring-up

    /// Ensure the shared assets are downloaded and resolve the repo root.
    public func ensureAssetsAvailable(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        if let dir = repoRootDirectory {
            return dir
        }
        let dir = try await StyleTTS2AssetDownloader.ensureModels(
            directory: directory,
            progressHandler: progressHandler
        )
        repoRootDirectory = dir
        return dir
    }

    // MARK: - Bundle paths

    /// Path to the espeak-ng IPA vocabulary JSON.
    public func vocabularyURL() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root.appendingPathComponent(ModelNames.StyleTTS2.vocabularyFile)
    }

    /// Path to the bundle `config.json`.
    public func configURL() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root.appendingPathComponent(ModelNames.StyleTTS2.configFile)
    }

    /// Lazily-loaded 178-token IPA vocabulary.
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
    /// of the store.
    public func bundleConfig() throws -> StyleTTS2BundleConfig {
        if let cached = cachedBundleConfig {
            return cached
        }
        let url = try configURL()
        let config = try StyleTTS2BundleConfig.load(from: url)
        cachedBundleConfig = config
        return config
    }

    public func repoRoot() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root
    }
}
