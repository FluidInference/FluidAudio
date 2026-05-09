import Foundation

/// Downloads StyleTTS2 LibriTTS (iteration_3) CoreML models from HuggingFace.
///
/// The HF tree at `FluidInference/StyleTTS-2-coreml/iteration_3/compiled/` ships
/// 14 `.mlmodelc` directories: 8 default-path stages + 6 bucketed variants of the
/// two stages that can't accept `RangeDim` on the token axis (`bert`,
/// `fused_diffusion_sampler`).
///
/// The phonemizer side reuses Kokoro's preprocessed Misaki lexicon
/// (`us_lexicon_cache.json`) and the BART G2P CoreML model
/// (`G2PEncoder.mlmodelc` / `G2PDecoder.mlmodelc` + `g2p_vocab.json`),
/// both fetched from the kokoro HF repo and cached under
/// `~/.cache/fluidaudio/Models/kokoro/` — exactly where Kokoro itself
/// looks for them, so a single download serves both backends.
public enum StyleTTS2ResourceDownloader {

    private static let logger = AppLogger(category: "StyleTTS2ResourceDownloader")

    /// Ensure the 8 default-path mlmodelc bundles are present locally.
    /// Bucketed variants are fetched lazily by `ensureBucket(forT:in:)` when
    /// the synthesizer needs one.
    ///
    /// - Returns: the resolved repo directory (i.e. the directory that holds
    ///   the `.mlmodelc` bundles after `subPath` stripping).
    @discardableResult
    public static func ensureDefaultModels(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.styletts2.folderName)

        let allDefaultsPresent = ModelNames.StyleTTS2.requiredModels.allSatisfy { entry in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(entry).path)
        }

        if !allDefaultsPresent {
            logger.info("Downloading StyleTTS2 LibriTTS models (iteration_3) from HuggingFace…")
            do {
                try await DownloadUtils.downloadRepo(
                    .styletts2, to: modelsRoot, progressHandler: progressHandler)
            } catch {
                throw StyleTTS2Error.downloadFailed("\(error)")
            }
        } else {
            logger.info("StyleTTS2 default models found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    /// Ensure Kokoro's preprocessed Misaki lexicon cache
    /// (`us_lexicon_cache.json`) is present locally, then return the kokoro
    /// cache directory that holds it. The same payload is consumed by
    /// `KokoroSynthesizer.LexiconCache`, so the StyleTTS2 backend is using
    /// the exact same word→phoneme map Kokoro ships.
    @discardableResult
    public static func ensureLexiconCache() async throws -> URL {
        do {
            let cacheURL = try await TtsResourceDownloader.ensureLexiconFile(
                named: "us_lexicon_cache.json")
            return cacheURL.deletingLastPathComponent()
        } catch {
            throw StyleTTS2Error.downloadFailed("us_lexicon_cache.json: \(error)")
        }
    }

    /// Ensure Kokoro's BART grapheme-to-phoneme CoreML assets
    /// (`G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`, `g2p_vocab.json`)
    /// are present in the kokoro cache. `G2PModel.shared` only loads from
    /// disk — without this call, a first-time StyleTTS2 user (who hasn't
    /// already downloaded the kokoro repo) would fail with
    /// `G2PModelError.vocabLoadFailed` the first time the OOV path is hit.
    public static func ensureG2PAssets(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let kokoroDir = modelsRoot.appendingPathComponent(Repo.kokoro.folderName)
        let allPresent = ModelNames.G2P.requiredModels.allSatisfy { name in
            FileManager.default.fileExists(atPath: kokoroDir.appendingPathComponent(name).path)
        }
        if allPresent {
            return
        }
        logger.info("Downloading kokoro G2P CoreML assets (g2p-only variant) from HuggingFace…")
        do {
            try await DownloadUtils.downloadRepo(
                .kokoro,
                to: modelsRoot,
                variant: "g2p-only",
                progressHandler: progressHandler
            )
        } catch {
            throw StyleTTS2Error.downloadFailed("kokoro G2P assets: \(error)")
        }
    }

    /// Ensure the bucket-variant pair (`bert_fp16_t<T>` +
    /// `fused_diffusion_sampler_fp16_t<T>`) for token bucket `t` is present.
    /// No-op when both files already exist locally.
    public static func ensureBucket(
        forT t: Int,
        in repoDir: URL
    ) async throws {
        let needed = ModelNames.StyleTTS2.bucketModels(forT: t)
        guard !needed.isEmpty else {
            throw StyleTTS2Error.noBucketAvailable(tokenCount: t)
        }
        let missing = needed.filter { !FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path) }
        if missing.isEmpty {
            return
        }

        logger.info("Fetching StyleTTS2 bucket T=\(t) (\(missing.count) bundles)")
        for fileName in missing {
            do {
                try await DownloadUtils.downloadSubdirectory(
                    .styletts2,
                    subdirectory: fileName,
                    to: repoDir
                )
            } catch {
                throw StyleTTS2Error.downloadFailed(
                    "bucket T=\(t) bundle \(fileName) — \(error)")
            }
        }
    }

    private static func defaultCacheRoot() throws -> URL {
        let base: URL
        #if os(macOS)
        base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else {
            throw StyleTTS2Error.downloadFailed("failed to locate caches directory")
        }
        base = first
        #endif
        let root = base.appendingPathComponent("fluidaudio").appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}
