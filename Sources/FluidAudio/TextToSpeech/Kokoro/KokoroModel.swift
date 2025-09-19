import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using unified CoreML model
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModel {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModel")

    public struct ChunkDebugInfo {
        public let index: Int
        public let text: String
        public let words: [String]
        public let pauseAfterMs: Int
        public let isForcedSplit: Bool
        public let phonemeCount: Int
        public let trailingPunctuation: String?
        public let stillInSentence: Bool
    }

    // Single model reference
    private static var kokoroModel: MLModel?
    private static var isModelLoaded = false

    internal static var isModelInitialized: Bool { isModelLoaded }
    internal static var currentModel: MLModel? { kokoroModel }

    // Legacy: Phoneme dictionary with frame counts (kept for backward compatibility)
    private static var phonemeDictionary: [String: (frameCount: Float, phonemes: [String])] = [:]
    private static var isDictionaryLoaded = false

    // Preferred: Simple word -> phonemes mapping from word_phonemes.json
    private static var wordToPhonemes: [String: [String]] = [:]
    private static var wordToPhonemesByLanguage: [String: [String: [String]]] = [:]
    private static var isSimpleDictLoaded = false

    // Model and data URLs
    private static let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"

    /// Download file from URL if needed (uses DownloadUtils for consistency)
    private static func downloadFileIfNeeded(filename: String, urlPath: String) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create directory if needed
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)

        guard !FileManager.default.fileExists(atPath: localURL.path) else {
            logger.info("File already exists: \(filename)")
            return
        }

        logger.info("Downloading \(filename)...")
        let downloadURL = URL(string: "\(baseURL)/\(urlPath)")!

        // Use DownloadUtils.sharedSession for consistent proxy and configuration handling
        let (data, response) = try await DownloadUtils.sharedSession.data(from: downloadURL)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TTSError.modelNotFound("Failed to download \(filename)")
        }

        try data.write(to: localURL)
        logger.info("Downloaded \(filename) (\(data.count) bytes)")
    }

    /// Ensure required dictionary files exist
    public static func ensureRequiredFiles() async throws {
        // Download dictionary files using our simplified helper (which uses DownloadUtils.sharedSession)
        try await downloadFileIfNeeded(filename: "word_phonemes.json", urlPath: "word_phonemes.json")
        try await downloadFileIfNeeded(filename: "word_frames_phonemes.json", urlPath: "word_frames_phonemes.json")

        // Ensure eSpeak NG data bundle exists (download from HuggingFace Resources if missing)
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelsDirectory = cacheDir.appendingPathComponent("Models")
        _ = try? await DownloadUtils.ensureEspeakDataBundle(in: modelsDirectory)
    }

    /// Load the Kokoro model
    public static func loadModel() async throws {
        guard !isModelLoaded else { return }

        // Delegate to TtsModels which uses DownloadUtils for all model downloads
        let models = try await TtsModels.download()
        kokoroModel = models.kokoro

        isModelLoaded = true
        logger.info("Kokoro model successfully loaded")
    }

    /// Use a pre-loaded Kokoro model instance (primarily for unit tests or embedding contexts).
    public static func useExternalModel(_ model: MLModel) {
        kokoroModel = model
        isModelLoaded = true
        logger.info("Using externally provided Kokoro model instance")
    }

    /// Load simple word->phonemes dictionary (preferred)
    /// Supports two formats:
    /// 1) { "word_to_phonemes": { word: [token, ...] } }
    /// 2) { word: "ipa string" | [token, ...] } (flat map)
    public static func loadSimplePhonemeDictionary() throws {
        guard !isSimpleDictLoaded else { return }

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let dictURL = kokoroDir.appendingPathComponent("word_phonemes.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TTSError.processingFailed("Invalid word_phonemes.json (not a JSON object)")
        }

        let vocabulary = KokoroVocabulary.getVocabulary()
        let allowed = Set(vocabulary.keys)

        wordToPhonemes = try parseWordPhonemeDictionary(
            json, allowed: allowed,
            preferredLanguage: nil)
        buildLanguageSpecificDictionaries(baseDirectory: kokoroDir, allowed: allowed)

        isSimpleDictLoaded = true
        logger.info("Loaded \(wordToPhonemes.count) words from word_phonemes.json")
    }

    private static func parseWordPhonemeDictionary(
        _ json: [String: Any],
        allowed: Set<String>,
        preferredLanguage: String?
    ) throws -> [String: [String]] {
        if let mapping = json["word_to_phonemes"] as? [String: Any] {
            return try parseFlatDictionary(mapping, allowed: allowed, preferredLanguage: preferredLanguage)
        }
        return try parseFlatDictionary(json, allowed: allowed, preferredLanguage: preferredLanguage)
    }

    private static func parseFlatDictionary(
        _ dictionary: [String: Any],
        allowed: Set<String>,
        preferredLanguage: String?
    ) throws -> [String: [String]] {
        var result: [String: [String]] = [:]
        var converted = 0
        for (word, value) in dictionary {
            if let tokens = normalizedPhonemeEntry(
                value,
                allowed: allowed,
                preferredLanguage: preferredLanguage
            ) {
                result[word.lowercased()] = tokens
                converted += 1
            }
        }

        guard !result.isEmpty else {
            throw TTSError.processingFailed("No valid phoneme entries parsed from dictionary")
        }
        return result
    }

    private static func normalizedPhonemeEntry(
        _ value: Any,
        allowed: Set<String>,
        preferredLanguage: String?
    ) -> [String]? {
        if let arr = value as? [String] {
            let filtered = arr.filter { allowed.contains($0) }
            return filtered.isEmpty ? nil : filtered
        }

        if let str = value as? String {
            let tokens = tokenizeIPAString(str).filter { allowed.contains($0) }
            return tokens.isEmpty ? nil : tokens
        }

        if let dict = value as? [String: Any] {
            if let preferredLanguage,
                let match = dict[preferredLanguage] ?? dict[preferredLanguage.lowercased()]
            {
                if let entry = normalizedPhonemeEntry(match, allowed: allowed, preferredLanguage: nil) {
                    return entry
                }
            }

            if let defaultEntry = dict["default"] ?? dict["DEFAULT"] {
                if let entry = normalizedPhonemeEntry(defaultEntry, allowed: allowed, preferredLanguage: nil) {
                    return entry
                }
            }

            for (_, nestedValue) in dict {
                if let entry = normalizedPhonemeEntry(
                    nestedValue, allowed: allowed, preferredLanguage: preferredLanguage)
                {
                    return entry
                }
            }
        }

        return nil
    }

    private static func buildLanguageSpecificDictionaries(baseDirectory: URL, allowed: Set<String>) {
        wordToPhonemesByLanguage.removeAll()

        let languages = Set(KokoroVoiceCatalog.supportedLanguageCodes.map { $0.lowercased() })
        guard !languages.isEmpty else { return }

        for language in languages {
            guard let overlay = loadOverlay(for: language, baseDirectory: baseDirectory, allowed: allowed),
                !overlay.isEmpty
            else { continue }

            var merged = wordToPhonemes
            for (word, tokens) in overlay { merged[word] = tokens }
            wordToPhonemesByLanguage[language] = merged
        }
    }

    private static func loadOverlay(
        for language: String,
        baseDirectory: URL,
        allowed: Set<String>
    ) -> [String: [String]]? {
        let candidates = overlayFileCandidates(for: language)
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)

        var urls: [URL] = []
        for name in candidates {
            urls.append(baseDirectory.appendingPathComponent(name))
            urls.append(cwd.appendingPathComponent(name))
        }

        for url in urls where fm.fileExists(atPath: url.path) {
            do {
                let data = try Data(contentsOf: url)
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    let mapping = try parseWordPhonemeDictionary(
                        json,
                        allowed: allowed,
                        preferredLanguage: language
                    )
                    logger.info(
                        "Loaded \(mapping.count) language-specific phoneme entries from \(url.lastPathComponent)")
                    return mapping
                }
            } catch {
                logger.warning("Failed to parse overlay dictionary at \(url.path): \(error.localizedDescription)")
            }
        }

        return nil
    }

    private static func overlayFileCandidates(for language: String) -> [String] {
        var variants: [String] = []
        let normalized = language
        let lower = language.lowercased()
        let hyphenToUnderscore = language.replacingOccurrences(of: "-", with: "_")
        let lowerHyphen = hyphenToUnderscore.lowercased()

        variants.append("word_phonemes_\(normalized).json")
        variants.append("word_phonemes_\(lower).json")
        if normalized != hyphenToUnderscore {
            variants.append("word_phonemes_\(hyphenToUnderscore).json")
        }
        if lower != lowerHyphen {
            variants.append("word_phonemes_\(lowerHyphen).json")
        }

        if let hyphenIndex = language.firstIndex(of: "-") {
            let prefix = String(language[..<hyphenIndex])
            let lowerPrefix = prefix.lowercased()
            variants.append("word_phonemes_\(prefix).json")
            if prefix != lowerPrefix {
                variants.append("word_phonemes_\(lowerPrefix).json")
            }
        }

        var seen: Set<String> = []
        var deduped: [String] = []
        for name in variants {
            if seen.insert(name).inserted {
                deduped.append(name)
            }
        }
        return deduped
    }

    internal static func phonemeDictionary(for languageCode: String?) throws -> [String: [String]] {
        try loadSimplePhonemeDictionary()

        guard let codeRaw = languageCode, !codeRaw.isEmpty else {
            return wordToPhonemes
        }

        let code = codeRaw.lowercased()
        if let dict = wordToPhonemesByLanguage[code] {
            return dict
        }

        if let hyphenIndex = code.firstIndex(of: "-") {
            let prefix = String(code[..<hyphenIndex])
            if let dict = wordToPhonemesByLanguage[prefix] {
                return dict
            }
        }

        return wordToPhonemes
    }

    internal static func resetPhonemeDictionariesForTesting() {
        wordToPhonemes = [:]
        wordToPhonemesByLanguage = [:]
        isSimpleDictLoaded = false
    }

    /// Normalize arbitrary voice identifiers to a snake_case form suitable for filenames.
    private static func sanitizeVoiceIdentifier(_ voice: String) -> String {
        let trimmed = voice.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        var working = trimmed.replacingOccurrences(of: "-", with: "_")
        working = working.replacingOccurrences(of: " ", with: "_")

        if !working.contains("_") {
            var output = ""
            output.reserveCapacity(working.count * 2)
            for scalar in working {
                if scalar.isUppercase {
                    if !output.isEmpty { output.append("_") }
                    output.append(Character(scalar.lowercased()))
                } else {
                    output.append(Character(scalar.lowercased()))
                }
            }
            return output
        }

        return working.lowercased()
    }

    /// Determine canonical and fallback identifiers used when resolving voice embeddings.
    private static func voiceSearchParameters(
        for voice: String
    ) -> (
        canonical: String,
        fileCandidates: [String],
        keyCandidates: [String]
    ) {
        let sanitizedInput = sanitizeVoiceIdentifier(voice)
        let canonical = KokoroVoiceCatalog.canonicalVoiceId(for: voice) ?? sanitizedInput

        func appendUnique(_ value: String, to array: inout [String]) {
            let normalized = sanitizeVoiceIdentifier(value)
            guard !normalized.isEmpty else { return }
            if !array.contains(normalized) {
                array.append(normalized)
            }
        }

        var fileCandidates: [String] = []
        appendUnique(canonical, to: &fileCandidates)
        appendUnique(voice, to: &fileCandidates)
        appendUnique(sanitizedInput, to: &fileCandidates)

        if fileCandidates.isEmpty {
            fileCandidates.append(KokoroVoiceCatalog.defaultVoiceId)
        }

        var keyCandidates = fileCandidates
        if !canonical.isEmpty && !keyCandidates.contains(canonical) {
            keyCandidates.insert(canonical, at: 0)
        }
        if !keyCandidates.contains(KokoroVoiceCatalog.defaultVoiceId) {
            keyCandidates.append(KokoroVoiceCatalog.defaultVoiceId)
        }

        let resolvedCanonical = canonical.isEmpty ? KokoroVoiceCatalog.defaultVoiceId : canonical
        return (resolvedCanonical, fileCandidates, keyCandidates)
    }

    /// Tokenize an IPA string into model tokens.
    /// If the string contains whitespace, split on whitespace; otherwise split into Unicode scalars.
    private static func tokenizeIPAString(_ s: String) -> [String] {
        if s.contains(where: { $0.isWhitespace }) {
            return
                s
                .components(separatedBy: .whitespacesAndNewlines)
                .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ",.;:()[]{}\"'")) }
                .filter { !$0.isEmpty }
        }
        // Split into Unicode scalars to preserve single-codepoint IPA tokens (e.g., ʧ, ʤ, ˈ)
        return s.unicodeScalars.map { String($0) }
    }

    /// Structure to hold a chunk of text that fits within 3.17 seconds
    // TextChunk is defined in KokoroChunker.swift

    /// Chunk text into segments under the model token budget, using punctuation-driven pauses.
    private static func chunkText(_ text: String, using dictionary: [String: [String]]) throws -> [TextChunk] {
        let target = targetTokenLength()
        let hasLang = false
        return KokoroChunker.chunk(
            text: text,
            wordToPhonemes: dictionary,
            targetTokens: target,
            hasLanguageToken: hasLang
        )
    }

    /// Convert phonemes to input IDs
    public static func phonemesToInputIds(_ phonemes: [String]) -> [Int32] {
        let vocabulary = KokoroVocabulary.getVocabulary()
        var ids: [Int32] = [0]  // BOS/EOS token per Python harness
        for phoneme in phonemes {
            if let id = vocabulary[phoneme] {
                ids.append(id)
            } else {
                logger.warning("Missing phoneme in vocab: '\(phoneme)'")
            }
        }
        ids.append(0)

        // Debug: validate id range
        #if DEBUG
        if !vocabulary.isEmpty {
            let maxId = vocabulary.values.max() ?? 0
            let minId = vocabulary.values.min() ?? 0
            let outOfRange = ids.filter { $0 != 0 && ($0 < minId || $0 > maxId) }
            if !outOfRange.isEmpty {
                print(
                    "Warning: Found \(outOfRange.count) token IDs out of range [\(minId), \(maxId)] (excluding BOS/EOS=0)"
                )
            }
            print("Tokenized \(ids.count) ids; first 32: \(ids.prefix(32))")
        }
        #endif

        return ids
    }

    /// Inspect model to determine the expected token length for input_ids
    internal static func targetTokenLength() -> Int {
        if let model = kokoroModel {
            let inputs = model.modelDescription.inputDescriptionsByName
            if let inputDesc = inputs["input_ids"], let constraint = inputDesc.multiArrayConstraint {
                let shape = constraint.shape
                if shape.count >= 2 {
                    let n = shape.last!.intValue
                    if n > 0 { return n }
                }
            }
        }
        // Fallback to a common unified length if not discoverable
        return 124
    }

    /// Load voice embedding (simplified for 3-second model)
    public static func loadVoiceEmbedding(
        voice: String = KokoroVoiceCatalog.defaultVoiceId,
        phonemeCount: Int
    ) throws -> MLMultiArray {
        let requestedVoice = voice
        let (canonicalVoice, fileCandidates, keyCandidates) = voiceSearchParameters(for: requestedVoice)

        if requestedVoice != canonicalVoice {
            logger.debug("Voice alias '\(requestedVoice)' normalized to '\(canonicalVoice)'")
        }

        // Try to load from cache: ~/.cache/fluidaudio/Models/kokoro/voices/<voice>.json
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let candidatePaths: [URL] = fileCandidates.flatMap { name in
            [
                cwd.appendingPathComponent("voices/\(name).json"),
                cwd.appendingPathComponent("\(name).json"),
                voicesDir.appendingPathComponent("\(name).json"),
            ]
        }

        guard let primaryVoiceName = fileCandidates.first else {
            throw TTSError.processingFailed("Unable to derive voice identifier from '\(requestedVoice)'")
        }

        let voiceJSON =
            candidatePaths.first { FileManager.default.fileExists(atPath: $0.path) }
            ?? voicesDir.appendingPathComponent("\(primaryVoiceName).json")

        var vector: [Float]?
        if FileManager.default.fileExists(atPath: voiceJSON.path) {
            do {
                let data = try Data(contentsOf: voiceJSON)
                let json = try JSONSerialization.jsonObject(with: data)

                func parseArray(_ any: Any) -> [Float]? {
                    if let ds = any as? [Double] { return ds.map { Float($0) } }
                    if let fs = any as? [Float] { return fs }
                    if let ns = any as? [NSNumber] { return ns.map { $0.floatValue } }
                    if let arr = any as? [Any] {
                        var out: [Float] = []
                        out.reserveCapacity(arr.count)
                        for v in arr {
                            if let n = v as? NSNumber {
                                out.append(n.floatValue)
                            } else if let d = v as? Double {
                                out.append(Float(d))
                            } else if let f = v as? Float {
                                out.append(f)
                            } else {
                                return nil
                            }
                        }
                        return out
                    }
                    return nil
                }

                if let arr = parseArray(json) {
                    vector = arr
                } else if let dict = json as? [String: Any] {
                    for key in keyCandidates {
                        if let embed = dict[key], let arr = parseArray(embed) {
                            vector = arr
                            break
                        }
                    }

                    if vector == nil, let embed = dict["embedding"], let arr = parseArray(embed) {
                        vector = arr
                    }

                    if vector == nil {
                        let numericKeys = dict.keys.compactMap { Int($0) }.sorted()
                        if let exact = dict["\(phonemeCount)"], let arr = parseArray(exact) {
                            vector = arr
                        } else if let key = numericKeys.last(where: { $0 <= phonemeCount }),
                            let candidate = dict["\(key)"], let arr = parseArray(candidate)
                        {
                            vector = arr
                        }
                    }

                    if vector == nil, let any = dict.values.first {
                        vector = parseArray(any)
                    }
                }
            } catch {
                logger.warning(
                    "Failed to parse voice embedding JSON at \(voiceJSON.path): \(error.localizedDescription)")
            }
        }

        // Require a valid voice embedding; fail if missing or invalid
        let dim = refDimFromModel()
        guard let vec = vector, vec.count == dim else {
            let descriptor =
                canonicalVoice == requestedVoice
                ? canonicalVoice
                : "\(requestedVoice) → \(canonicalVoice)"
            throw TTSError.modelNotFound(
                "Voice embedding for \(descriptor) not found or invalid at \(voiceJSON.path)")
        }
        let embedding = try MLMultiArray(shape: [1, NSNumber(value: dim)] as [NSNumber], dataType: .float32)
        let destPointer = embedding.dataPointer.bindMemory(to: Float.self, capacity: dim)
        vec.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            destPointer.initialize(from: base, count: dim)
        }
        let varsum = vec.reduce(into: Float.zero) { $0 += $1 * $1 }
        let l2norm = String(format: "%.3f", sqrt(Double(varsum)))
        if let metadata = KokoroVoiceCatalog.voice(for: canonicalVoice) {
            logger.info(
                "Loaded voice embedding: \(metadata.id) [\(metadata.languageCode), \(metadata.gender.rawValue.capitalized)], dim=\(dim), l2norm=\(l2norm)"
            )
        } else {
            logger.info("Loaded voice embedding: \(canonicalVoice), dim=\(dim), l2norm=\(l2norm)")
        }

        return embedding
    }

    /// Helper to fetch ref_s expected dimension from model
    private static func refDimFromModel() -> Int {
        guard let model = kokoroModel else { return 256 }
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
            let shape = desc.multiArrayConstraint?.shape,
            shape.count >= 2
        {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }

    /// Synthesize a single chunk of text
    private static func synthesizeChunk(_ chunk: TextChunk, voice: String) async throws -> [Float] {
        // Convert phonemes to input IDs
        let phonemeSeq = chunk.phonemes
        // No language token prefix; avoid audible leading vowel
        let inputIds = phonemesToInputIds(phonemeSeq)

        guard !inputIds.isEmpty else {
            throw TTSError.processingFailed("No input IDs generated for chunk: \(chunk.words.joined(separator: " "))")
        }

        // Get voice embedding
        let refStyle = try loadVoiceEmbedding(voice: voice, phonemeCount: inputIds.count)

        // Determine target token length from model and pad/truncate accordingly
        let targetTokens = targetTokenLength()
        var trimmedIds = inputIds
        if trimmedIds.count > targetTokens {
            trimmedIds = Array(trimmedIds.prefix(targetTokens))
        } else if trimmedIds.count < targetTokens {
            var cycled: [Int32] = []
            var index = 0
            let baseCount = max(trimmedIds.count, 1)
            while trimmedIds.count + cycled.count < targetTokens {
                cycled.append(trimmedIds[index])
                index = (index + 1) % baseCount
            }
            let needed = targetTokens - trimmedIds.count
            trimmedIds.append(contentsOf: cycled.prefix(needed))
        }

        // Create model inputs
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)] as [NSNumber], dataType: .int32)
        for (i, id) in trimmedIds.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        // Create attention mask (1 for real tokens up to original count, 0 for padding)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)] as [NSNumber], dataType: .int32)
        let trueLen = min(inputIds.count, targetTokens)
        for i in 0..<targetTokens {
            attentionMask[i] = NSNumber(value: i < trueLen ? 1 : 0)
        }

        // Use zeros for phases for determinism (works well for 3s model)
        let phasesArray = try MLMultiArray(shape: [1, 9] as [NSNumber], dataType: .float32)
        let phasesPointer = phasesArray.dataPointer.bindMemory(to: Float.self, capacity: 9)
        phasesPointer.initialize(repeating: 0, count: 9)

        // Debug: print model IO

        // Run inference
        let modelInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputArray,
            "attention_mask": attentionMask,
            "ref_s": refStyle,
            "random_phases": phasesArray,
        ])

        // Time ONLY the model prediction
        guard let model = kokoroModel else {
            throw TTSError.modelNotFound("Kokoro model not initialized")
        }
        let predictionStart = Date()
        let primaryOptions = MLPredictionOptions()
        primaryOptions.usesCPUOnly = false
        let cpuFallbackOptions = MLPredictionOptions()
        cpuFallbackOptions.usesCPUOnly = true

        let predictSynchronously: (MLPredictionOptions) throws -> MLFeatureProvider = { options in
            try model.prediction(from: modelInput, options: options)
        }

        let output: MLFeatureProvider
        if #available(macOS 14.0, iOS 17.0, *) {
            do {
                output = try await model.prediction(from: modelInput, options: primaryOptions)
            } catch {
                logger.warning("Async CoreML prediction failed; retrying synchronous (hardware): \(error.localizedDescription)")
                do {
                    output = try predictSynchronously(primaryOptions)
                } catch {
                    logger.warning("Synchronous hardware prediction failed; retrying CPU-only: \(error.localizedDescription)")
                    output = try predictSynchronously(cpuFallbackOptions)
                }
            }
        } else {
            do {
                output = try predictSynchronously(primaryOptions)
            } catch {
                logger.warning("Synchronous hardware prediction failed; retrying CPU-only: \(error.localizedDescription)")
                output = try predictSynchronously(cpuFallbackOptions)
            }
        }
        let predictionTime = Date().timeIntervalSince(predictionStart)
        print("[PERF] Pure model.prediction() time: \(String(format: "%.3f", predictionTime))s")

        // Extract audio output explicitly by key used by model
        guard let audioArrayUnwrapped = output.featureValue(for: "audio")?.multiArrayValue,
            audioArrayUnwrapped.count > 0
        else {
            let names = Array(output.featureNames)
            throw TTSError.processingFailed("Failed to extract 'audio' output. Features: \(names)")
        }

        let rawCount = audioArrayUnwrapped.count
        var effectiveCount = rawCount
        if let lenFV = output.featureValue(for: "audio_length_samples") {
            var n: Int = 0
            if let lenArray = lenFV.multiArrayValue, lenArray.count > 0 {
                n = lenArray[0].intValue
            } else if lenFV.type == .int64 {
                n = Int(lenFV.int64Value)
            } else if lenFV.type == .double {
                n = Int(lenFV.doubleValue)
            }
            if n < 0 {
                logger.warning("audio_length_samples reported negative count (\(n)); ignoring")
            } else {
                if n > rawCount {
                    logger.warning("audio_length_samples (\(n)) exceeds tensor count (\(rawCount)); clamping")
                }
                effectiveCount = min(n, rawCount)
            }
        }

        // Convert to float samples
        var samples: [Float] = []
        for i in 0..<effectiveCount {
            samples.append(audioArrayUnwrapped[i].floatValue)
        }

        // Basic sanity logging
        let minVal = samples.min() ?? 0
        let maxVal = samples.max() ?? 0
        if maxVal - minVal == 0 {
            logger.warning("Prediction produced constant signal (min=max=\(minVal)).")
        } else {
            logger.info("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
        }

        return samples
    }

    /// Main synthesis function with chunking support
    public static func synthesize(
        text: String,
        voice: String = KokoroVoiceCatalog.defaultVoiceId
    ) async throws -> Data {
        let (samples, totalTime, _) = try await synthesizeSamplesInternal(text: text, voice: voice)
        let audioData = try AudioWAV.data(from: samples, sampleRate: 24000)
        logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
        logger.info("Audio size: \(audioData.count) bytes")
        return audioData
    }

    /// Generate normalized PCM samples without packaging them into a WAV container.
    public static func synthesizeSamples(
        text: String,
        voice: String = KokoroVoiceCatalog.defaultVoiceId
    ) async throws -> [Float] {
        let (samples, totalTime, _) = try await synthesizeSamplesInternal(text: text, voice: voice)
        logger.info(
            "Synthesis complete (samples only) in \(String(format: "%.3f", totalTime))s; sample count=\(samples.count)"
        )
        return samples
    }

    /// Synthesize speech and return chunk-level debug information.
    public static func synthesizeDetailed(
        text: String,
        voice: String = KokoroVoiceCatalog.defaultVoiceId,
        chunkOutputDirectory: URL? = nil
    ) async throws -> (audio: Data, chunks: [ChunkDebugInfo]) {
        let (samples, totalTime, chunks) = try await synthesizeSamplesInternal(
            text: text,
            voice: voice,
            chunkOutputDirectory: chunkOutputDirectory
        )
        let audioData = try AudioWAV.data(from: samples, sampleRate: 24000)
        logger.info("Detailed synthesis complete in \(String(format: "%.3f", totalTime))s (chunks=\(chunks.count))")
        let debugChunks: [ChunkDebugInfo] = chunks.enumerated().map { index, chunk in
            ChunkDebugInfo(
                index: index,
                text: chunk.words.joined(separator: " "),
                words: chunk.words,
                pauseAfterMs: chunk.pauseAfterMs,
                isForcedSplit: chunk.isForcedSplit,
                phonemeCount: chunk.phonemes.count,
                trailingPunctuation: chunk.trailingPunctuation,
                stillInSentence: !chunk.isSentenceBoundary
            )
        }
        return (audioData, debugChunks)
    }

    private static func synthesizeSamplesInternal(
        text: String,
        voice: String,
        chunkOutputDirectory: URL? = nil
    ) async throws -> ([Float], Double, [TextChunk]) {
        let synthesisStart = Date()

        logger.info("Starting synthesis: '\(text)'")
        logger.info("Input length: \(text.count) characters")

        let canonicalVoice = voiceSearchParameters(for: voice).canonical
        let voiceMetadata = KokoroVoiceCatalog.voice(for: canonicalVoice)
        if let metadata = voiceMetadata {
            logger.info(
                "Voice: \(metadata.label) [\(metadata.languageName), \(metadata.gender.rawValue.capitalized)]"
            )
        } else {
            logger.info("Voice: \(canonicalVoice)")
        }
        if canonicalVoice != voice {
            logger.debug("Voice alias '\(voice)' resolved to '\(canonicalVoice)'")
        }

        // Ensure required files are downloaded
        try await ensureRequiredFiles()
        // Ensure voice embedding if available
        do {
            try await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: canonicalVoice)
        } catch {
            logger.warning("Failed to prefetch voice embedding for \(canonicalVoice): \(error.localizedDescription)")
        }

        // Load model if needed
        if !isModelLoaded {
            try await loadModel()
        }

        // Preload dictionary for OOV detection
        try loadSimplePhonemeDictionary()
        let languageCode = voiceMetadata?.languageCode
        let phonemeDictionary = try phonemeDictionary(for: languageCode)

        do {
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
            var oov: [String] = []
            oov.reserveCapacity(8)

            for raw in tokens {
                let key = normalize(raw)
                if key.isEmpty { continue }
                if phonemeDictionary[key] == nil {
                    oov.append(key)
                    if oov.count >= 8 { break }
                }
            }
            #if canImport(ESpeakNG)
            if !oov.isEmpty {
                if EspeakG2P.isDataAvailable() == false {
                    do {
                        let modelsDirectory = try TtsModels.cacheDirectoryURL().appendingPathComponent("Models")
                        _ = try await DownloadUtils.ensureEspeakDataBundle(in: modelsDirectory)
                    } catch {
                        logger.warning("Failed to download eSpeak NG data: \(error.localizedDescription)")
                    }
                }

                if EspeakG2P.isDataAvailable() == false {
                    logger.warning(
                        "G2P (eSpeak NG) data unavailable; skipping OOV words: \(Set(oov).sorted().prefix(5).joined(separator: ", "))"
                    )
                }
            }
            #else
            if !oov.isEmpty {
                logger.warning(
                    "G2P (eSpeak NG) not included in this build; skipping OOV words: \(Set(oov).sorted().prefix(5).joined(separator: ", "))"
                )
            }
            #endif
        }

        let chunks = try chunkText(text, using: phonemeDictionary)
        if chunks.isEmpty {
            throw TTSError.processingFailed("No valid words found in text")
        }

        if let chunkOutputDirectory {
            try await saveChunksDebug(
                chunks: chunks,
                baseDirectory: chunkOutputDirectory,
                voice: canonicalVoice
            )
        }

        if chunks.count == 1 {
            logger.info("Text fits in single chunk")
            let chunk = chunks[0]
            logger.info("Processing chunk: \(chunk.words.count) words, \(chunk.totalFrames) frames")

            let samples = try await synthesizeChunk(chunk, voice: canonicalVoice)
            let maxVal = samples.map { abs($0) }.max() ?? 1.0
            let normalizedSamples = maxVal > 0 ? samples.map { $0 / maxVal } : samples
            let minVal = normalizedSamples.min() ?? 0
            let maxOut = normalizedSamples.max() ?? 0
            logger.info("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxOut))]")
            let totalTime = Date().timeIntervalSince(synthesisStart)
            return (normalizedSamples, totalTime, chunks)
        } else {
            logger.info("Text split into \(chunks.count) chunks")
            var allSamples: [Float] = []

            let crossfadeMs = 8
            let crossfadeN = max(0, Int(Double(crossfadeMs) * 24.0))

            for i in 0..<chunks.count {
                let chunk = chunks[i]
                logger.info(
                    "Processing chunk \(i+1)/\(chunks.count): \(chunk.words.count) words, \(chunk.totalFrames) frames"
                )
                logger.info("Chunk \(i+1) text: '\(chunk.words.joined(separator: " "))'")
                var chunkSamples = try await synthesizeChunk(chunk, voice: canonicalVoice)
                if i == 0 {
                    allSamples.append(contentsOf: chunkSamples)
                } else {
                    let prevPause = chunks[i - 1].pauseAfterMs
                    if prevPause > 0 {
                        let silenceCount = Int(Double(prevPause) * 24.0)
                        if silenceCount > 0 {
                            allSamples.append(contentsOf: Array(repeating: 0.0, count: silenceCount))
                        }
                        allSamples.append(contentsOf: chunkSamples)
                    } else {
                        let n = min(crossfadeN, allSamples.count, chunkSamples.count)
                        if n > 0 {
                            for k in 0..<n {
                                let aIdx = allSamples.count - n + k
                                let bIdx = k
                                let t = Float(k) / Float(n)
                                let fadeOut = 1.0 - t
                                let fadeIn = t
                                allSamples[aIdx] = allSamples[aIdx] * fadeOut + chunkSamples[bIdx] * fadeIn
                            }
                            if chunkSamples.count > n {
                                allSamples.append(contentsOf: chunkSamples[n...])
                            }
                        } else {
                            allSamples.append(contentsOf: chunkSamples)
                        }
                    }
                }
            }

            let maxVal = allSamples.map { abs($0) }.max() ?? 1.0
            let normalizedSamples = maxVal > 0 ? allSamples.map { $0 / maxVal } : allSamples
            let minVal = normalizedSamples.min() ?? 0
            let maxOut = normalizedSamples.max() ?? 0
            logger.info("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxOut))]")
            logger.info("Total samples: \(normalizedSamples.count)")
            let totalTime = Date().timeIntervalSince(synthesisStart)
            return (normalizedSamples, totalTime, chunks)
        }
    }

    private static func saveChunksDebug(
        chunks: [TextChunk],
        baseDirectory: URL,
        voice: String
    ) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: baseDirectory.path) {
            try fm.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        }

        for (index, chunk) in chunks.enumerated() {
            let words = chunk.words.joined(separator: " ")
            let filenameBase = String(format: "chunk_%03d", index)
            let textURL = baseDirectory.appendingPathComponent("\(filenameBase).txt")
            try words.write(to: textURL, atomically: true, encoding: .utf8)

            let chunkSamples = try await synthesizeChunk(chunk, voice: voice)
            let maxVal = chunkSamples.map { abs($0) }.max() ?? 1.0
            let normalized = maxVal > 0 ? chunkSamples.map { $0 / maxVal } : chunkSamples
            let audioData = try AudioWAV.data(from: normalized, sampleRate: 24000)
            let wavURL = baseDirectory.appendingPathComponent("\(filenameBase).wav")
            try audioData.write(to: wavURL)
        }
    }

    // convertSamplesToWAV moved to AudioWAV

    // convertToWAV removed (unused); use convertSamplesToWAV instead
}
