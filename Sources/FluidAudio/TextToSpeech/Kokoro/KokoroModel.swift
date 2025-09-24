import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using unified CoreML model
/// Supports both 5s and 15s variants with US English phoneme lexicons
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModel {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModel")

    /// Detailed synthesis output including audio data and per-chunk metadata.
    public struct SynthesisResult: Sendable {
        public let audio: Data
        public let chunks: [ChunkInfo]

        public init(audio: Data, chunks: [ChunkInfo]) {
            self.audio = audio
            self.chunks = chunks
        }
    }

    /// Metadata describing each chunk synthesized by the Kokoro pipeline.
    public struct ChunkInfo: Sendable {
        public let index: Int
        public let text: String
        public let wordCount: Int
        public let words: [String]
        public let atoms: [String]
        public let pauseAfterMs: Int
        public let tokenCount: Int
        public let samples: [Float]
        public let variant: ModelNames.TTS.Variant

        public init(
            index: Int,
            text: String,
            wordCount: Int,
            words: [String],
            atoms: [String],
            pauseAfterMs: Int,
            tokenCount: Int,
            samples: [Float],
            variant: ModelNames.TTS.Variant
        ) {
            self.index = index
            self.text = text
            self.wordCount = wordCount
            self.words = words
            self.atoms = atoms
            self.pauseAfterMs = pauseAfterMs
            self.tokenCount = tokenCount
            self.samples = samples
            self.variant = variant
        }
    }

    // Cached CoreML models per Kokoro variant
    private static var kokoroModels: [ModelNames.TTS.Variant: MLModel] = [:]
    private static var tokenLengthCache: [ModelNames.TTS.Variant: Int] = [:]
    private static var downloadedModelBundle: TtsModels?

    // Legacy: Phoneme dictionary with frame counts (kept for backward compatibility)
    private static var phonemeDictionary: [String: (frameCount: Float, phonemes: [String])] = [:]
    private static var isDictionaryLoaded = false

    // Preferred: Simple word -> phonemes mapping from US lexicon JSON files
    private static var wordToPhonemes: [String: [String]] = [:]
    private static var caseSensitiveWordToPhonemes: [String: [String]] = [:]
    private static var isSimpleDictLoaded = false

    // Model and data URLs
    private static let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"

    private static func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }

    private static func model(for variant: ModelNames.TTS.Variant) throws -> MLModel {
        guard let model = kokoroModels[variant] else {
            throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
        }
        return model
    }

    private static func inferTokenLength(from model: MLModel) -> Int {
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
        try await downloadFileIfNeeded(filename: "us_gold.json", urlPath: "us_gold.json")
        try await downloadFileIfNeeded(filename: "us_silver.json", urlPath: "us_silver.json")

        // Ensure eSpeak NG data bundle exists (download from HuggingFace Resources if missing)
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelsDirectory = cacheDir.appendingPathComponent("Models")
        _ = try? await DownloadUtils.ensureEspeakDataBundle(in: modelsDirectory)
    }

    /// Load Kokoro CoreML models for all supported variants.
    public static func loadModel() async throws {
        if kokoroModels.count == ModelNames.TTS.Variant.allCases.count {
            return
        }

        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        var variantsNeedingDownload: [ModelNames.TTS.Variant] = []
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        for variant in ModelNames.TTS.Variant.allCases where kokoroModels[variant] == nil {
            let fileName = ModelNames.TTS.bundle(for: variant)
            let localModelURL = cwd.appendingPathComponent(fileName)

            if fm.fileExists(atPath: localModelURL.path) {
                logger.info(
                    "Loading Kokoro \(variantDescription(variant)) model from local bundle: \(localModelURL.path)")
                let model: MLModel
                if localModelURL.pathExtension == "mlpackage" {
                    let compiledURL = try await MLModel.compileModel(at: localModelURL)
                    model = try MLModel(contentsOf: compiledURL, configuration: configuration)
                } else {
                    model = try MLModel(contentsOf: localModelURL, configuration: configuration)
                }
                kokoroModels[variant] = model
                tokenLengthCache[variant] = inferTokenLength(from: model)
            } else {
                variantsNeedingDownload.append(variant)
            }
        }

        if !variantsNeedingDownload.isEmpty {
            if downloadedModelBundle == nil {
                downloadedModelBundle = try await TtsModels.download()
            }
            guard let bundle = downloadedModelBundle else {
                throw TTSError.modelNotFound("Kokoro models unavailable after download attempt")
            }
            for variant in variantsNeedingDownload {
                let fileName = ModelNames.TTS.bundle(for: variant)
                guard let model = bundle.model(for: variant) else {
                    throw TTSError.modelNotFound(fileName)
                }
                kokoroModels[variant] = model
                tokenLengthCache[variant] = inferTokenLength(from: model)
                logger.info("Loaded Kokoro \(variantDescription(variant)) model from cache")
            }
        }

        let loadedVariants = kokoroModels.keys.map { variantDescription($0) }.sorted().joined(separator: ", ")
        logger.info("Kokoro models ready: [\(loadedVariants)]")
    }

    /// Load simple word->phonemes dictionary (preferred)
    /// Uses the richer US English lexicons (gold/silver) as the primary source.
    public static func loadSimplePhonemeDictionary() throws {
        if isSimpleDictLoaded && !caseSensitiveWordToPhonemes.isEmpty {
            return
        }

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let vocabulary = KokoroVocabulary.getVocabulary()
        let allowed = Set(vocabulary.keys)

        func filteredTokens(_ tokens: [String]) -> [String] {
            tokens.filter { allowed.contains($0) }
        }

        let lexiconFiles = ["us_gold.json", "us_silver.json"]
        var mapping: [String: [String]] = [:]
        var caseSensitive: [String: [String]] = [:]
        var totalAdded = 0

        for filename in lexiconFiles {
            let lexiconURL = kokoroDir.appendingPathComponent(filename)
            guard FileManager.default.fileExists(atPath: lexiconURL.path) else { continue }

            do {
                let rawLexicon = try Data(contentsOf: lexiconURL)
                guard let entries = try JSONSerialization.jsonObject(with: rawLexicon) as? [String: Any] else {
                    logger.warning("Skipping \(filename) (unexpected format)")
                    continue
                }

                var addedHere = 0
                for (key, value) in entries {
                    let normalizedKey = key.lowercased()
                    let tokens: [String]
                    if let stringValue = value as? String {
                        tokens = filteredTokens(tokenizeIPAString(stringValue))
                    } else if let arrayValue = value as? [String] {
                        tokens = filteredTokens(arrayValue)
                    } else {
                        continue
                    }

                    guard !tokens.isEmpty else { continue }

                    if caseSensitive[key] == nil {
                        caseSensitive[key] = tokens
                    }

                    if mapping[normalizedKey] == nil {
                        mapping[normalizedKey] = tokens
                        addedHere += 1
                    }
                }

                if addedHere > 0 {
                    totalAdded += addedHere
                    logger.info("Merged \(addedHere) entries from \(filename) into phoneme dictionary")
                }
            } catch {
                logger.warning("Failed to merge lexicon \(filename): \(error.localizedDescription)")
            }
        }

        guard !mapping.isEmpty else {
            throw TTSError.processingFailed("No US English lexicon entries found (missing us_gold.json/us_silver.json)")
        }

        wordToPhonemes = mapping
        caseSensitiveWordToPhonemes = caseSensitive
        isSimpleDictLoaded = true
        logger.info("Phoneme dictionary loaded from US lexicons: total=\(mapping.count) entries (merged=\(totalAdded))")
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
    private static func chunkText(_ text: String) throws -> [TextChunk] {
        try loadSimplePhonemeDictionary()
        let target = try tokenLength(for: .fifteenSecond)
        let hasLang = false
        return KokoroChunker.chunk(
            text: text,
            wordToPhonemes: wordToPhonemes,
            caseSensitiveLexicon: caseSensitiveWordToPhonemes,
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
    private static func tokenLength(for variant: ModelNames.TTS.Variant) throws -> Int {
        if let cached = tokenLengthCache[variant] {
            return cached
        }
        let model = try model(for: variant)
        let length = inferTokenLength(from: model)
        tokenLengthCache[variant] = length
        return length
    }

    private static func selectVariant(forTokenCount tokenCount: Int) throws -> ModelNames.TTS.Variant {
        let shortCapacity = try tokenLength(for: .fiveSecond)
        let longCapacity = try tokenLength(for: .fifteenSecond)
        guard tokenCount <= longCapacity else {
            throw TTSError.processingFailed(
                "Chunk token count \(tokenCount) exceeds supported capacities (short=\(shortCapacity), long=\(longCapacity))"
            )
        }

        let shortThreshold = min(71, shortCapacity)
        if tokenCount <= shortThreshold {
            return .fiveSecond
        }

        logger.notice(
            "Promoting chunk to Kokoro 15s variant: token count \(tokenCount) exceeds short threshold=\(shortThreshold) (short capacity=\(shortCapacity), long capacity=\(longCapacity))"
        )
        return .fifteenSecond
    }

    /// Load voice embedding (simplified for 3-second model)
    public static func loadVoiceEmbedding(voice: String = "af_heart", phonemeCount: Int) throws -> MLMultiArray {
        let voice = "af_heart"
        // Try to load from cache: ~/.cache/fluidaudio/Models/kokoro/voices/<voice>.json
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        // Try multiple candidates (prefer local repo files first)
        let candidates: [URL] = [
            cwd.appendingPathComponent("voices/\(voice).json"),
            cwd.appendingPathComponent("\(voice).json"),
            voicesDir.appendingPathComponent("\(voice).json"),
        ]
        let voiceJSON = candidates.first { FileManager.default.fileExists(atPath: $0.path) } ?? candidates[0]

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
                    if let embed = dict["embedding"], let arr = parseArray(embed) {
                        vector = arr
                    } else if let byVoice = dict[voice], let arr = parseArray(byVoice) {
                        vector = arr
                    } else {
                        let keys = dict.keys.compactMap { Int($0) }.sorted()
                        var chosen: [Float]? = nil
                        if let exact = dict["\(phonemeCount)"] {
                            chosen = parseArray(exact)
                        } else if let k = keys.last(where: { $0 <= phonemeCount }), let cand = dict["\(k)"] {
                            chosen = parseArray(cand)
                        }
                        if let c = chosen {
                            vector = c
                        } else if let any = dict.values.first {
                            vector = parseArray(any)
                        }
                    }
                }
            } catch {
                // Ignore parse errors; will fall back
            }
        }

        // Require a valid voice embedding; fail if missing or invalid
        let dim = refDimFromModel()
        guard let vec = vector, vec.count == dim else {
            throw TTSError.modelNotFound("Voice embedding for \(voice) not found or invalid at \(voiceJSON.path)")
        }
        let embedding = try MLMultiArray(shape: [1, NSNumber(value: dim)] as [NSNumber], dataType: .float32)
        var varsum: Float = 0
        for i in 0..<dim {
            let v = vec[i]
            embedding[i] = NSNumber(value: v)
            varsum += v * v
        }
        logger.info(
            "Loaded voice embedding: \(voice), dim=\(dim), l2norm=\(String(format: "%.3f", sqrt(Double(varsum))))")
        return embedding
    }

    /// Helper to fetch ref_s expected dimension from model
    private static func refDim(from model: MLModel) -> Int {
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
            let shape = desc.multiArrayConstraint?.shape,
            shape.count >= 2
        {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }

    private static func refDimFromModel() -> Int {
        if let defaultModel = kokoroModels[ModelNames.TTS.defaultVariant] {
            return refDim(from: defaultModel)
        }
        if let anyModel = kokoroModels.values.first {
            return refDim(from: anyModel)
        }
        return 256
    }

    /// Synthesize a single chunk of text using precomputed token IDs.
    private static func synthesizeChunk(
        _ chunk: TextChunk,
        voice: String,
        inputIds: [Int32],
        variant: ModelNames.TTS.Variant,
        targetTokens: Int
    ) async throws -> [Float] {
        guard !inputIds.isEmpty else {
            throw TTSError.processingFailed("No input IDs generated for chunk: \(chunk.words.joined(separator: " "))")
        }

        let kokoro = try model(for: variant)

        // Get voice embedding
        let refStyle = try loadVoiceEmbedding(voice: voice, phonemeCount: inputIds.count)

        // Pad or truncate to match model expectation
        var trimmedIds = inputIds
        if trimmedIds.count > targetTokens {
            logger.warning(
                "input_ids length (\(trimmedIds.count)) exceeds targetTokens=\(targetTokens) for chunk '\(chunk.text)' — truncating"
            )
            trimmedIds = Array(trimmedIds.prefix(targetTokens))
        } else if trimmedIds.count < targetTokens {
            logger.debug(
                "input_ids length (\(trimmedIds.count)) below targetTokens=\(targetTokens) for chunk '\(chunk.text)' — padding with zeros"
            )
            trimmedIds.append(contentsOf: Array(repeating: Int32(0), count: targetTokens - trimmedIds.count))
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
        for i in 0..<9 { phasesArray[i] = 0 }

        // Debug: print model IO

        // Run inference
        let modelInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputArray,
            "attention_mask": attentionMask,
            "ref_s": refStyle,
            "random_phases": phasesArray,
        ])

        // Time ONLY the model prediction
        let predictionStart = Date()
        let output = try kokoro.prediction(from: modelInput)
        let predictionTime = Date().timeIntervalSince(predictionStart)
        print(
            "[PERF] Pure model.prediction() time: \(String(format: "%.3f", predictionTime))s (variant=\(variantDescription(variant)))"
        )

        // Extract audio output explicitly by key used by model
        guard let audioArrayUnwrapped = output.featureValue(for: "audio")?.multiArrayValue,
            audioArrayUnwrapped.count > 0
        else {
            let names = Array(output.featureNames)
            throw TTSError.processingFailed("Failed to extract 'audio' output. Features: \(names)")
        }

        // Optional: trim to audio_length_samples if provided
        var effectiveCount = audioArrayUnwrapped.count
        if let lenFV = output.featureValue(for: "audio_length_samples") {
            var n: Int = 0
            if let lenArray = lenFV.multiArrayValue, lenArray.count > 0 {
                n = lenArray[0].intValue
            } else if lenFV.type == .int64 {
                n = Int(lenFV.int64Value)
            } else if lenFV.type == .double {
                n = Int(lenFV.doubleValue)
            }
            n = max(0, n)
            if n > 0 && n <= audioArrayUnwrapped.count {
                effectiveCount = n
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

    /// Main synthesis function returning audio bytes only.
    public static func synthesize(text: String, voice: String = "af_heart") async throws -> Data {
        let result = try await synthesizeDetailed(text: text, voice: voice)
        return result.audio
    }

    /// Synthesize audio while returning per-chunk metadata used during inference.
    public static func synthesizeDetailed(
        text: String,
        voice: String = "af_heart"
    ) async throws -> SynthesisResult {
        let synthesisStart = Date()

        logger.info("Starting synthesis: '\(text)'")
        logger.info("Input length: \(text.count) characters")

        try await ensureRequiredFiles()
        try? await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: voice)

        try await loadModel()

        try loadSimplePhonemeDictionary()

        // If there are OOV words and G2P data is missing, fail fast per policy.
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
                if wordToPhonemes[key] == nil {
                    oov.append(key)
                    if oov.count >= 8 { break }
                }
            }
            #if canImport(ESpeakNG) || canImport(CEspeakNG)
            if !oov.isEmpty && EspeakG2P.isDataAvailable() == false {
                throw TTSError.processingFailed(
                    "G2P (eSpeak NG) data missing but required for OOV words: \(Set(oov).sorted().prefix(5).joined(separator: ", ")). Ensure the eSpeak NG data bundle is available in the models cache (use DownloadUtils.ensureEspeakDataBundle)."
                )
            }
            #else
            if !oov.isEmpty {
                throw TTSError.processingFailed(
                    "G2P (eSpeak NG) not included in this build but required for OOV words: \(Set(oov).sorted().prefix(5).joined(separator: ", "))."
                )
            }
            #endif
        }

        let chunks = try chunkText(text)
        guard !chunks.isEmpty else {
            throw TTSError.processingFailed("No valid words found in text")
        }

        struct ChunkInfoTemplate {
            let index: Int
            let text: String
            let wordCount: Int
            let words: [String]
            let atoms: [String]
            let pauseAfterMs: Int
            let tokenCount: Int
            let variant: ModelNames.TTS.Variant
            let targetTokens: Int
        }

        struct ChunkEntry {
            let chunk: TextChunk
            let inputIds: [Int32]
            let template: ChunkInfoTemplate
        }

        var entries: [ChunkEntry] = []
        entries.reserveCapacity(chunks.count)

        for (index, chunk) in chunks.enumerated() {
            let inputIds = phonemesToInputIds(chunk.phonemes)
            guard !inputIds.isEmpty else {
                throw TTSError.processingFailed(
                    "No input IDs generated for chunk: \(chunk.words.joined(separator: " "))")
            }
            let variant = try selectVariant(forTokenCount: inputIds.count)
            let targetTokens = try tokenLength(for: variant)
            let template = ChunkInfoTemplate(
                index: index,
                text: chunk.text,
                wordCount: chunk.words.count,
                words: chunk.words,
                atoms: chunk.atoms,
                pauseAfterMs: chunk.pauseAfterMs,
                tokenCount: min(inputIds.count, targetTokens),
                variant: variant,
                targetTokens: targetTokens
            )
            entries.append(ChunkEntry(chunk: chunk, inputIds: inputIds, template: template))
        }

        if entries.count == 1 {
            logger.info("Text fits in single chunk")
        } else {
            logger.info("Text split into \(entries.count) chunks")
        }

        var allSamples: [Float] = []
        var chunkTemplates: [ChunkInfoTemplate] = []
        var chunkSampleBuffers: [[Float]] = []
        let crossfadeMs = 8
        let crossfadeN = max(0, Int(Double(crossfadeMs) * 24.0))

        for (index, entry) in entries.enumerated() {
            let chunk = entry.chunk
            logger.info(
                "Processing chunk \(index + 1)/\(entries.count): \(chunk.words.count) words")
            logger.info("Chunk \(index + 1) text: '\(entry.template.text)'")
            logger.info("Chunk \(index + 1) using Kokoro \(variantDescription(entry.template.variant)) model")
            let chunkSamples = try await synthesizeChunk(
                entry.chunk,
                voice: voice,
                inputIds: entry.inputIds,
                variant: entry.template.variant,
                targetTokens: entry.template.targetTokens)

            chunkSampleBuffers.append(chunkSamples)
            chunkTemplates.append(entry.template)

            if index == 0 {
                allSamples.append(contentsOf: chunkSamples)
                continue
            }

            let prevPause = entries[index - 1].chunk.pauseAfterMs
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
                        let t = Float(k) / Float(n)
                        allSamples[aIdx] = allSamples[aIdx] * (1.0 - t) + chunkSamples[k] * t
                    }
                    if chunkSamples.count > n {
                        allSamples.append(contentsOf: chunkSamples[n...])
                    }
                } else {
                    allSamples.append(contentsOf: chunkSamples)
                }
            }
        }

        guard !allSamples.isEmpty else {
            throw TTSError.processingFailed("Synthesis produced no samples")
        }

        let maxVal = allSamples.map { abs($0) }.max() ?? 1.0
        let normalize: (Float) -> Float = { sample in
            guard maxVal > 0 else { return sample }
            return sample / maxVal
        }

        let normalizedSamples = allSamples.map(normalize)
        let audioData = try AudioWAV.data(from: normalizedSamples, sampleRate: 24000)

        let normalizedChunkSamples = chunkSampleBuffers.map { buffer -> [Float] in
            buffer.map(normalize)
        }

        let chunkInfos = zip(chunkTemplates, normalizedChunkSamples).map { template, samples in
            ChunkInfo(
                index: template.index,
                text: template.text,
                wordCount: template.wordCount,
                words: template.words,
                atoms: template.atoms,
                pauseAfterMs: template.pauseAfterMs,
                tokenCount: template.tokenCount,
                samples: samples,
                variant: template.variant
            )
        }

        let totalTime = Date().timeIntervalSince(synthesisStart)
        logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
        logger.info("Audio size: \(audioData.count) bytes")
        logger.info("Total samples: \(allSamples.count)")

        return SynthesisResult(audio: audioData, chunks: chunkInfos)
    }

    // convertSamplesToWAV moved to AudioWAV

    // convertToWAV removed (unused); use convertSamplesToWAV instead
}
