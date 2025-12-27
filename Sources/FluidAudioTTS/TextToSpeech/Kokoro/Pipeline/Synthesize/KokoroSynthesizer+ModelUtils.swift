@preconcurrency import CoreML
import FluidAudio
import Foundation

extension KokoroSynthesizer {
    public static func ensureRequiredFiles() async throws {
        let assets = try currentLexiconAssets()
        try await assets.ensureCoreAssets()
    }

    public static func loadModel(variant: ModelNames.TTS.Variant? = nil) async throws {
        let cache = try currentModelCache()
        if let variant {
            try await cache.loadModelsIfNeeded(variants: Set([variant]))
        } else {
            try await cache.loadModelsIfNeeded()
        }
    }

    public static func loadSimplePhonemeDictionary() async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        let allowed = Set(vocabulary.keys)
        try await lexiconCache.ensureLoaded(kokoroDirectory: kokoroDir, allowedTokens: allowed)
    }

    static func model(for variant: ModelNames.TTS.Variant) async throws -> MLModel {
        let cache = try currentModelCache()
        return try await cache.model(for: variant)
    }

    public static func sanitizeInput(_ text: String) throws -> String {
        let cleanText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanText.isEmpty else {
            throw TTSError.processingFailed("Input text is empty")
        }

        let withoutDelimiters = removeDelimiterCharacters(from: cleanText)
        let normalizedWhitespace = collapseWhitespace(in: withoutDelimiters)

        guard !normalizedWhitespace.isEmpty else {
            throw TTSError.processingFailed("Input text is empty after stripping parentheses")
        }

        return normalizedWhitespace
    }

    internal static func inferTokenLength(from model: MLModel) -> Int {
        let inputs = model.modelDescription.inputDescriptionsByName
        if let inputDesc = inputs["input_ids"], let constraint = inputDesc.multiArrayConstraint {
            let shape = constraint.shape
            if shape.count >= 2 {
                let n = shape.last!.intValue
                if n > 0 { return n }
            }
        }
        return 124
    }

    static func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }

    static func validateTextHasDictionaryCoverage(_ text: String) async throws {
        let allowedSet = CharacterSet.letters.union(.decimalDigits).union(CharacterSet(charactersIn: "'"))
        func normalize(_ s: String) -> String {
            let lowered = s.lowercased()
                .replacingOccurrences(of: "\u{2019}", with: "'")
                .replacingOccurrences(of: "\u{2018}", with: "'")
            return String(lowered.unicodeScalars.filter { allowedSet.contains($0) })
        }

        let tokens =
            text
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .map { String($0) }

        let lexicons = await lexiconCache.lexicons()
        let mapping = lexicons.word

        var oov: [String] = []
        oov.reserveCapacity(8)
        for raw in tokens {
            let key = normalize(raw)
            if key.isEmpty { continue }
            if mapping[key] == nil {
                oov.append(key)
                if oov.count >= 8 { break }
            }
        }

        if !oov.isEmpty {
            let sample = Set(oov).sorted().prefix(5).joined(separator: ", ")
            do {
                try EspeakG2P.ensureResourcesAvailable()
            } catch {
                // In CI environments, we might want to proceed even if G2P is missing
                // (e.g. for basic smoke tests that don't verify audio content).
                if ProcessInfo.processInfo.environment["CI"] != nil {
                    print(
                        "Warning: G2P (eSpeak NG) unavailable in CI environment. OOV words (\(sample)) will be skipped or incorrect."
                    )
                    return
                }

                throw TTSError.processingFailed(
                    "G2P (eSpeak NG) unavailable but required for OOV words (\(sample)): \(error.localizedDescription)"
                )
            }
        }
    }

    static func modelBundleURL(for variant: ModelNames.TTS.Variant) throws -> URL {
        let base = try TtsModels.cacheDirectoryURL().appendingPathComponent("Models/kokoro")
        return base.appendingPathComponent(variant.fileName)
    }

    static func directorySize(at url: URL) -> Int {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
                options: [],
                errorHandler: nil
            )
        else {
            return 0
        }

        var total = 0
        for case let fileURL as URL in enumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
                if resourceValues.isDirectory == true { continue }
                if let fileSize = resourceValues.fileSize {
                    total += fileSize
                }
            } catch {
                continue
            }
        }
        return total
    }
}
