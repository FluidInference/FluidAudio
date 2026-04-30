#if os(macOS)
import FluidAudio
import Foundation

/// Swift port of `Scripts/fetch_minimax_tts_corpus.py`.
///
/// Fetches the MiniMax Multilingual TTS Test Set per-language `.txt` files
/// from HuggingFace and converts them to the FluidAudio TTS-benchmark
/// corpus format (strip `<cloning_audio_filename>|` prefix, prepend a
/// header documenting source + revision + license).
///
/// Reuses `DownloadUtils.fetchHuggingFaceFile` so we get the same auth
/// (HF_TOKEN env), retry, and backoff treatment as every other HF asset
/// pull in the project — no hardcoded URLs, no swift-transformers
/// dependency added just for one corpus fetch.
///
/// Source dataset:  https://huggingface.co/datasets/MiniMaxAI/TTS-Multilingual-Test-Set
/// License:         CC-BY-SA-4.0
public enum MinimaxCorpusCommand {

    private static let logger = AppLogger(category: "MinimaxCorpusCommand")

    private static let repo = "MiniMaxAI/TTS-Multilingual-Test-Set"

    /// Pin to the initial public commit so re-runs reproduce the vendored
    /// files. Matches `DEFAULT_REVISION` in the Python script.
    private static let defaultRevision = "cb416f0ac3658da0577e97873065e19fe6488917"

    /// All 24 languages in the upstream `text/` directory. Keep in sync with
    /// `ALL_LANGUAGES` in `Scripts/fetch_minimax_tts_corpus.py`.
    private static let allLanguages: [String] = [
        "arabic", "cantonese", "chinese", "czech", "dutch", "english",
        "finnish", "french", "german", "greek", "hindi", "indonesian",
        "italian", "japanese", "korean", "polish", "portuguese", "romanian",
        "russian", "spanish", "thai", "turkish", "ukrainian", "vietnamese",
    ]

    public static func run(arguments: [String]) async {
        var languages = allLanguages
        var revision = defaultRevision
        var outDir: URL? = nil

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--languages", "-l":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1]
                        .split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    i += 1
                }
            case "--revision":
                if i + 1 < arguments.count {
                    revision = arguments[i + 1]
                    i += 1
                }
            case "--out-dir":
                if i + 1 < arguments.count {
                    outDir = URL(fileURLWithPath: arguments[i + 1])
                    i += 1
                }
            case "help", "--help", "-h":
                printUsage()
                return
            default:
                logger.error("Unknown argument: \(arg)")
                printUsage()
                exit(1)
            }
            i += 1
        }

        let unknown = Set(languages).subtracting(allLanguages).sorted()
        if !unknown.isEmpty {
            logger.error("Unknown language(s): \(unknown.joined(separator: ", "))")
            logger.error("Available: \(allLanguages.joined(separator: ", "))")
            exit(2)
        }

        let resolvedOutDir = outDir ?? defaultOutDir()

        do {
            try FileManager.default.createDirectory(
                at: resolvedOutDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create output directory: \(error.localizedDescription)")
            exit(1)
        }

        logger.info("Fetching MiniMax TTS Multilingual Test Set @ \(revision)")
        logger.info("  out_dir: \(resolvedOutDir.path)")
        logger.info("  langs:   \(languages.count)")

        var total = 0
        for lang in languages {
            guard let url = URL(string: hfURL(repo: repo, revision: revision, path: "text/\(lang).txt"))
            else {
                logger.error("[\(lang)] failed to construct URL")
                exit(1)
            }
            do {
                let data = try await DownloadUtils.fetchHuggingFaceFile(
                    from: url, description: "minimax TTS corpus (\(lang))")
                guard let raw = String(data: data, encoding: .utf8) else {
                    logger.error("[\(lang)] response was not valid UTF-8")
                    exit(1)
                }
                let phrases = convert(raw: raw)
                let outPath = try writeCorpus(
                    lang: lang, phrases: phrases, outDir: resolvedOutDir,
                    revision: revision)
                let countStr = String(format: "%3d", phrases.count)
                let relPath = relativePath(outPath, from: repoRoot())
                logger.info("  [\(lang)] \(countStr) phrases -> \(relPath)")
                total += phrases.count
            } catch {
                logger.error("[\(lang)] FAILED: \(error.localizedDescription)")
                exit(1)
            }
        }

        logger.info("OK — \(total) phrases across \(languages.count) language(s).")
    }

    // MARK: - Helpers

    private static func hfURL(repo: String, revision: String, path: String) -> String {
        "https://huggingface.co/datasets/\(repo)/resolve/\(revision)/\(path)"
    }

    /// Strip `<filename>|` prefix and return the list of trimmed phrases.
    /// Mirrors `convert()` in the Python script.
    private static func convert(raw: String) -> [String] {
        var out: [String] = []
        for rawLine in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            // Format: "<cloning_audio_filename>|<text>". Some lines may have
            // extra `|` inside the text — keep only the first split.
            let text: String
            if let sepIdx = line.firstIndex(of: "|") {
                text = String(line[line.index(after: sepIdx)...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                text = line
            }
            if !text.isEmpty {
                out.append(text)
            }
        }
        return out
    }

    private static func writeCorpus(
        lang: String,
        phrases: [String],
        outDir: URL,
        revision: String
    ) throws -> URL {
        let outPath = outDir.appendingPathComponent("\(lang).txt")
        let header: [String] = [
            "# MiniMax Multilingual TTS Test Set — \(lang)",
            "# Source:   https://huggingface.co/datasets/\(repo)",
            "# Revision: \(revision)",
            "# License:  CC-BY-SA-4.0 (Creative Commons Attribution-ShareAlike 4.0)",
            "# Phrases:  \(phrases.count)",
            "#",
            "# Cloning-audio filenames have been stripped — we only need the",
            "# text for the FluidAudio TTS benchmark harness. Voice selection",
            "# is per-backend (see Documentation/TTS/MinimaxCorpus.md).",
            "",
        ]
        let body = (header + phrases).joined(separator: "\n") + "\n"
        try body.write(to: outPath, atomically: true, encoding: .utf8)
        return outPath
    }

    /// `<repo>/Benchmarks/tts/corpus/minimax/`. Resolves relative to the
    /// current working directory (the standard place `swift run` is invoked
    /// from); falls back gracefully if the layout doesn't exist yet because
    /// we `createDirectory(withIntermediateDirectories: true)` before write.
    private static func defaultOutDir() -> URL {
        repoRoot()
            .appendingPathComponent("Benchmarks", isDirectory: true)
            .appendingPathComponent("tts", isDirectory: true)
            .appendingPathComponent("corpus", isDirectory: true)
            .appendingPathComponent("minimax", isDirectory: true)
    }

    private static func repoRoot() -> URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    }

    private static func relativePath(_ url: URL, from base: URL) -> String {
        let path = url.standardizedFileURL.path
        let basePath = base.standardizedFileURL.path
        if path.hasPrefix(basePath + "/") {
            return String(path.dropFirst(basePath.count + 1))
        }
        return path
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio minimax-corpus [options]

            Fetches the MiniMax Multilingual TTS Test Set text files from
            HuggingFace and converts them to the FluidAudio TTS-benchmark
            corpus format. Outputs one file per language.

            Options:
                --languages, -l <list>   Comma-separated subset of languages
                                         (default: all 24).
                --revision <sha>         HuggingFace dataset revision
                                         (default: \(defaultRevision)).
                --out-dir <path>         Output directory
                                         (default: Benchmarks/tts/corpus/minimax).
                --help, -h               Show this help.

            Available languages:
                \(allLanguages.joined(separator: ", "))

            Examples:
                fluidaudio minimax-corpus
                fluidaudio minimax-corpus --languages english,spanish,hindi
                fluidaudio minimax-corpus --revision <commit-sha>
            """)
    }
}
#endif
