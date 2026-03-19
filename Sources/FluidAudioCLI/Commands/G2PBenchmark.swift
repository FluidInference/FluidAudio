#if os(macOS)
import FluidAudio
import Foundation

/// Benchmark for the multilingual G2P (CharsiuG2P ByT5) CoreML model.
///
/// Evaluates Phoneme Error Rate (PER) and speed against reference IPA
/// transcriptions from the CharsiuG2P test set.
struct G2PBenchmark {
    private static let logger = AppLogger(category: "G2P")

    /// All Kokoro-mapped languages, used as the default benchmark set.
    private static let defaultLanguages: [(code: String, language: MultilingualG2PLanguage)] = [
        ("eng-us", .americanEnglish),
        ("eng-uk", .britishEnglish),
        ("spa", .spanish),
        ("fra", .french),
        ("hin", .hindi),
        ("ita", .italian),
        ("jpn", .japanese),
        ("por-bz", .brazilianPortuguese),
        ("zho-s", .mandarinChinese),
    ]

    /// Maps TSV filename codes to languages when they differ from the CharsiuG2P model rawValue.
    private static let tsvCodeToLanguage: [String: MultilingualG2PLanguage] = [
        "zho-s": .mandarinChinese
    ]

    /// Maps model rawValues to TSV filename codes when they differ.
    private static let languageToTsvCode: [String: String] = [
        "cmn": "zho-s"
    ]

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        do {
            try await runBenchmark(arguments: arguments)
        } catch {
            logger.error("G2P Benchmark failed: \(error)")
        }
    }

    // MARK: - Private

    private static func runBenchmark(arguments: [String]) async throws {
        // Parse arguments
        var languageCodes: [String]?
        var maxWords: Int?
        var dataDir = "data/CharsiuG2P/data/test"
        var outputFile = "g2p_benchmark_results.json"
        var stripStress = true

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--languages":
                if i + 1 < arguments.count {
                    languageCodes = arguments[i + 1].split(separator: ",").map(String.init)
                    i += 1
                }
            case "--max-words":
                if i + 1 < arguments.count {
                    maxWords = Int(arguments[i + 1])
                    i += 1
                }
            case "--data-dir":
                if i + 1 < arguments.count {
                    dataDir = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--keep-stress":
                stripStress = false
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Resolve language list
        let languages: [(code: String, language: MultilingualG2PLanguage)]
        if let codes = languageCodes {
            languages = codes.compactMap { code in
                // Accept both CharsiuG2P model codes (rawValue) and TSV filename codes
                if let lang = tsvCodeToLanguage[code] {
                    return (code, lang)
                }
                if let lang = MultilingualG2PLanguage(rawValue: code) {
                    let tsvCode = languageToTsvCode[code] ?? code
                    return (tsvCode, lang)
                }
                logger.warning("Unknown language code: \(code), skipping")
                return nil
            }
            guard !languages.isEmpty else {
                logger.error("No valid languages specified.")
                return
            }
        } else {
            languages = defaultLanguages
        }

        logger.info("Multilingual G2P Benchmark")
        logger.info("==========================")
        logger.info("Model: CharsiuG2P ByT5 (CoreML)")
        logger.info("Data: \(dataDir)")
        logger.info("Languages: \(languages.map(\.code).joined(separator: ", "))")
        if let maxWords { logger.info("Max words per language: \(maxWords)") }
        logger.info("Stress marks: \(stripStress ? "stripped" : "kept")")

        // Load model
        let model = MultilingualG2PModel.shared
        do {
            try await model.ensureModelsAvailable()
        } catch {
            logger.error("Failed to load multilingual G2P models: \(error)")
            logger.error(
                "Ensure models are downloaded to the TTS cache directory under Models/\(Repo.kokoro.folderName)/"
            )
            return
        }

        // Run benchmark per language
        var results: [LanguageResult] = []

        for (index, entry) in languages.enumerated() {
            let tsvPath = "\(dataDir)/\(entry.code).tsv"
            guard FileManager.default.fileExists(atPath: tsvPath) else {
                logger.warning(
                    "[\(index + 1)/\(languages.count)] \(entry.code) — TSV not found at \(tsvPath), skipping"
                )
                continue
            }

            let wordPairs = try loadTSV(path: tsvPath, maxWords: maxWords)
            guard !wordPairs.isEmpty else {
                logger.warning(
                    "[\(index + 1)/\(languages.count)] \(entry.code) — no words in TSV, skipping")
                continue
            }

            var totalEditDistance = 0
            var totalRefLength = 0
            var exactMatches = 0
            var errors = 0
            let langStart = Date()

            for (word, referenceIPA) in wordPairs {
                do {
                    guard let phonemes = try await model.phonemize(word: word, language: entry.language)
                    else {
                        errors += 1
                        continue
                    }

                    let predicted = phonemes.joined()
                    let refNorm = normalizeIPA(referenceIPA, stripStress: stripStress)
                    let predNorm = normalizeIPA(predicted, stripStress: stripStress)

                    let refChars = Array(refNorm)
                    let predChars = Array(predNorm)

                    let dist = levenshteinDistance(refChars, predChars)
                    totalEditDistance += dist
                    totalRefLength += refChars.count

                    if refNorm == predNorm {
                        exactMatches += 1
                    }
                } catch {
                    errors += 1
                }
            }

            let totalSeconds = Date().timeIntervalSince(langStart)
            let wordsProcessed = wordPairs.count - errors
            let per = totalRefLength > 0 ? Double(totalEditDistance) / Double(totalRefLength) : 0
            let wer =
                wordsProcessed > 0 ? 1.0 - Double(exactMatches) / Double(wordsProcessed) : 1.0
            let msPerWord = wordsProcessed > 0 ? totalSeconds / Double(wordsProcessed) * 1000 : 0

            let result = LanguageResult(
                code: entry.code,
                wordsTested: wordsProcessed,
                per: per,
                wer: wer,
                avgMsPerWord: msPerWord,
                totalSeconds: totalSeconds,
                errors: errors
            )
            results.append(result)

            logger.info(
                "[\(index + 1)/\(languages.count)] \(entry.code) — "
                    + "\(wordsProcessed) words, "
                    + "PER=\(String(format: "%.1f", per * 100))%, "
                    + "WER=\(String(format: "%.1f", wer * 100))%, "
                    + "\(String(format: "%.0f", msPerWord))ms/word "
                    + "(\(String(format: "%.1f", totalSeconds))s)"
            )
        }

        guard !results.isEmpty else {
            logger.error("No languages were benchmarked successfully.")
            _exit(1)
        }

        // Summary
        logSummary(results)

        // Write JSON
        try writeJSON(results: results, outputFile: outputFile)
        logger.info("Results written to: \(outputFile)")

        // Force immediate exit to avoid CoreML ANE cleanup segfault.
        // The ANE teardown on model deallocation is a known issue with these ByT5 CoreML models.
        Foundation.exit(0)
    }

    // MARK: - TSV Loading

    private static func loadTSV(path: String, maxWords: Int?) throws -> [(String, String)] {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        var pairs: [(String, String)] = []

        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let word = String(parts[0])
            let ipa = String(parts[1])
            pairs.append((word, ipa))

            if let maxWords, pairs.count >= maxWords { break }
        }

        return pairs
    }

    // MARK: - IPA Normalization

    private static func normalizeIPA(_ ipa: String, stripStress: Bool) -> String {
        var result = ipa.precomposedStringWithCanonicalMapping  // NFC

        if stripStress {
            result = result.replacingOccurrences(of: "\u{02C8}", with: "")  // ˈ primary stress
            result = result.replacingOccurrences(of: "\u{02CC}", with: "")  // ˌ secondary stress
        }

        // Collapse whitespace
        result =
            result
            .split(separator: " ", omittingEmptySubsequences: true)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)

        return result
    }

    // MARK: - Output

    private static func logSummary(_ results: [LanguageResult]) {
        logger.info("")
        logger.info("Summary")
        logger.info("-------")

        func pad(_ s: String, _ width: Int) -> String {
            s.padding(toLength: width, withPad: " ", startingAt: 0)
        }

        logger.info(
            "| \(pad("Language", 10)) | \(pad("Words", 5)) | \(pad("PER", 7)) | \(pad("WER", 7)) | \(pad("ms/word", 7)) |"
        )
        logger.info("|------------|-------|---------|---------|---------|")

        for r in results {
            let perStr = String(format: "%5.1f%%", r.per * 100)
            let werStr = String(format: "%5.1f%%", r.wer * 100)
            let msStr = String(format: "%7.1f", r.avgMsPerWord)
            logger.info(
                "| \(pad(r.code, 10)) | \(pad(String(r.wordsTested), 5)) | \(pad(perStr, 7)) | \(pad(werStr, 7)) | \(msStr) |"
            )
        }

        let totalWords = results.reduce(0) { $0 + $1.wordsTested }
        let avgPER = results.reduce(0.0) { $0 + $1.per } / Double(results.count)
        let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
        let avgMs = results.reduce(0.0) { $0 + $1.avgMsPerWord } / Double(results.count)

        let perStr = String(format: "%5.1f%%", avgPER * 100)
        let werStr = String(format: "%5.1f%%", avgWER * 100)
        let msStr = String(format: "%7.1f", avgMs)
        logger.info("|------------|-------|---------|---------|---------|")
        logger.info(
            "| \(pad("Overall", 10)) | \(pad(String(totalWords), 5)) | \(pad(perStr, 7)) | \(pad(werStr, 7)) | \(msStr) |"
        )
    }

    private static func writeJSON(results: [LanguageResult], outputFile: String) throws {
        let totalWords = results.reduce(0) { $0 + $1.wordsTested }
        let avgPER = results.reduce(0.0) { $0 + $1.per } / Double(results.count)
        let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
        let avgMs = results.reduce(0.0) { $0 + $1.avgMsPerWord } / Double(results.count)

        let formatter = ISO8601DateFormatter()
        let timestamp = formatter.string(from: Date())

        let output: [String: Any] = [
            "timestamp": timestamp,
            "languages": results.map { r -> [String: Any] in
                [
                    "code": r.code,
                    "words_tested": r.wordsTested,
                    "per": r.per,
                    "wer": r.wer,
                    "avg_ms_per_word": r.avgMsPerWord,
                    "total_seconds": r.totalSeconds,
                    "errors": r.errors,
                ]
            },
            "summary": [
                "total_languages": results.count,
                "total_words": totalWords,
                "avg_per": avgPER,
                "avg_wer": avgWER,
                "avg_ms_per_word": avgMs,
            ] as [String: Any],
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputFile))
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudiocli g2p-benchmark [options]

            Benchmark the multilingual G2P (CharsiuG2P ByT5) CoreML model against
            reference IPA transcriptions.

            Options:
                --languages <codes>   Comma-separated CharsiuG2P language codes
                                      (default: eng-us,eng-uk,spa,fra,hin,ita,jpn,por-bz,zho-s)
                --max-words <n>       Max words per language (default: all)
                --data-dir <path>     Path to CharsiuG2P test TSV directory
                                      (default: data/CharsiuG2P/data/test)
                --output <path>       JSON output file (default: g2p_benchmark_results.json)
                --keep-stress         Keep stress marks in IPA comparison (default: strip)
                --help                Show this help message

            Examples:
                fluidaudiocli g2p-benchmark --languages eng-us,fra --max-words 50
                fluidaudiocli g2p-benchmark --data-dir /path/to/test --output results.json
            """)
    }
}

// MARK: - Data Types

private struct LanguageResult {
    let code: String
    let wordsTested: Int
    let per: Double
    let wer: Double
    let avgMsPerWord: Double
    let totalSeconds: Double
    let errors: Int
}
#endif
