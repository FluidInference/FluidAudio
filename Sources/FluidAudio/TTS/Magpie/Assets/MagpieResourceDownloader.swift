import Foundation

/// Downloads Magpie TTS models, constants, and per-language tokenizer data from HuggingFace.
///
/// The HF repo (`FluidInference/magpie-tts-multilingual-357m-coreml`) ships:
/// - 3 required CoreML models + 1 optional prefill model at the repo root
/// - `constants/` with model config, speaker embeddings, audio codebook tables, and
///   the local-transformer weights (downloaded as one subtree)
/// - `tokenizer/` with per-language lookup data (lazy per language)
public enum MagpieResourceDownloader {

    private static let logger = AppLogger(category: "MagpieResourceDownloader")

    /// Ensure the CoreML models + `constants/` directory are present locally, and
    /// ensure tokenizer data for each requested language is present. Returns the
    /// resolved repo directory (i.e. the root containing the `.mlmodelc` files).
    public static func ensureAssets(
        languages: Set<MagpieLanguage> = [.english],
        directory: URL? = nil,
        includePrefill: Bool = true,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.magpieTts.folderName)

        let rootModelsPresent = ModelNames.Magpie.requiredModels.allSatisfy { entry in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(entry).path)
        }

        if !rootModelsPresent {
            logger.info("Downloading Magpie TTS models from HuggingFace…")
            try await DownloadUtils.downloadRepo(
                .magpieTts, to: modelsRoot, progressHandler: progressHandler)
        } else {
            logger.info("Magpie TTS models found in cache")
        }

        if includePrefill {
            let prefillURL = repoDir.appendingPathComponent(ModelNames.Magpie.decoderPrefillFile)
            if !FileManager.default.fileExists(atPath: prefillURL.path) {
                logger.info("Fetching optional decoder_prefill model")
                do {
                    try await DownloadUtils.downloadSubdirectory(
                        .magpieTts,
                        subdirectory: ModelNames.Magpie.decoderPrefillFile,
                        to: repoDir
                    )
                } catch {
                    logger.warning(
                        "decoder_prefill unavailable; falling back to step-by-step prefill: \(error)"
                    )
                }
            }
        }

        for language in languages {
            try await ensureTokenizer(for: language, repoDirectory: repoDir)
        }

        return repoDir
    }

    /// Ensure tokenizer data for `language` exists. No-op for ByT5-only languages
    /// (French, Italian, Vietnamese) since those use pure byte-level encoding.
    public static func ensureTokenizer(
        for language: MagpieLanguage, repoDirectory: URL
    ) async throws {
        let files = MagpieTokenizerFiles.files(for: language)
        if files.isEmpty { return }

        let tokenizerDir = repoDirectory.appendingPathComponent(ModelNames.Magpie.tokenizerDir)
        if !FileManager.default.fileExists(atPath: tokenizerDir.path) {
            try FileManager.default.createDirectory(
                at: tokenizerDir, withIntermediateDirectories: true)
        }

        for file in files {
            let localURL = tokenizerDir.appendingPathComponent(file)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }

            let remotePath = "\(ModelNames.Magpie.tokenizerDir)/\(file)"
            logger.info("Downloading Magpie tokenizer file: \(remotePath)")
            let remoteURL: URL
            do {
                remoteURL = try ModelRegistry.resolveModel(Repo.magpieTts.remotePath, remotePath)
            } catch {
                throw MagpieError.downloadFailed(
                    "failed to resolve HF URL for \(remotePath): \(error)")
            }

            do {
                let data = try await AssetDownloader.fetchData(
                    from: remoteURL,
                    description: "magpie tokenizer \(file)",
                    logger: logger
                )
                try data.write(to: localURL, options: [.atomic])
            } catch {
                throw MagpieError.tokenizerDataMissing(
                    language: language.rawValue, file: file)
            }
        }
    }

    /// Return the directory that holds constants (JSON + npy + local_transformer/).
    public static func constantsDirectory(in repoDirectory: URL) -> URL {
        repoDirectory.appendingPathComponent(ModelNames.Magpie.constantsDir)
    }

    /// Return the directory that holds per-language tokenizer lookups.
    public static func tokenizerDirectory(in repoDirectory: URL) -> URL {
        repoDirectory.appendingPathComponent(ModelNames.Magpie.tokenizerDir)
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
            throw MagpieError.downloadFailed("failed to locate caches directory")
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

/// Authoritative list of per-language tokenizer files. The emitters in
/// `mobius/models/tts/magpie/export_tokenizers.py` produce these names; the Swift
/// tokenizers consume them.
public enum MagpieTokenizerFiles {
    /// Tokenizer filenames emitted by
    /// `mobius/models/tts/magpie/coreml/export_tokenizers.py`. The naming convention
    /// is `{tokenizer_name}_{suffix}.json` where `tokenizer_name` follows the NeMo
    /// AggregatedTTSTokenizer names (e.g. `english_phoneme`, `french_chartokenizer`).
    public static func files(for language: MagpieLanguage) -> [String] {
        let base = tokenizerName(for: language)
        switch language {
        case .english, .spanish, .italian, .vietnamese:
            // IPA G2P: token2id + phoneme_dict.
            return ["\(base)_token2id.json", "\(base)_phoneme_dict.json"]
        case .german:
            // IPA G2P with heteronym fallback.
            return [
                "\(base)_token2id.json",
                "\(base)_phoneme_dict.json",
                "\(base)_heteronyms.json",
            ]
        case .french, .hindi:
            // Char-based tokenizers: only token2id lookup.
            return ["\(base)_token2id.json"]
        case .mandarin:
            // pypinyin (phrase + char) + tone / letter / token2id maps.
            return [
                "\(base)_token2id.json",
                "\(base)_pinyin_dict.json",
                "\(base)_tone_dict.json",
                "\(base)_ascii_letter_dict.json",
                "mandarin_pypinyin_char_dict.json",
                "mandarin_pypinyin_phrase_dict.json",
                "mandarin_jieba_dict.json",
            ]
        }
    }

    /// NeMo tokenizer name for the given language (matches the Python map in
    /// `generate_coreml._tokenize_text`).
    public static func tokenizerName(for language: MagpieLanguage) -> String {
        switch language {
        case .english: return "english_phoneme"
        case .spanish: return "spanish_phoneme"
        case .german: return "german_phoneme"
        case .italian: return "italian_phoneme"
        case .vietnamese: return "vietnamese_phoneme"
        case .mandarin: return "mandarin_phoneme"
        case .french: return "french_chartokenizer"
        case .hindi: return "hindi_chartokenizer"
        }
    }
}
