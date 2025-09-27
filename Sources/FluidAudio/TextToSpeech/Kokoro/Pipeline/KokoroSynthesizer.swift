import Accelerate
import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Supports both 5s and 15s variants with US English phoneme lexicons
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroSynthesizer {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroSynthesizer")
    private static let sampleRateHz = 24_000
    private static let shortVariantGuardThresholdSeconds = 3.0
    private static let shortVariantGuardFrameCount = 4
    private static let kokoroFrameSamples = 600  // Kokoro vocoder output samples per frame

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

    private struct ChunkInfoTemplate: Sendable {
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

    private struct ChunkEntry: Sendable {
        let chunk: TextChunk
        let inputIds: [Int32]
        let template: ChunkInfoTemplate
    }

    private struct TokenCapacities: Sendable {
        let short: Int
        let long: Int

        func capacity(for variant: ModelNames.TTS.Variant) -> Int {
            switch variant {
            case .fiveSecond:
                return short
            case .fifteenSecond:
                return long
            }
        }
    }

    private struct MultiArrayKey: Hashable {
        let dataTypeRawValue: Int
        let shape: [Int]

        init(dataType: MLMultiArrayDataType, shape: [NSNumber]) {
            dataTypeRawValue = dataType.rawValue
            self.shape = shape.map { $0.intValue }
        }

        func hash(into hasher: inout Hasher) {
            hasher.combine(dataTypeRawValue)
            for dimension in shape {
                hasher.combine(dimension)
            }
        }
    }

    private actor MultiArrayPool {
        private var storage: [MultiArrayKey: [MLMultiArray]] = [:]

        func rent(
            shape: [NSNumber],
            dataType: MLMultiArrayDataType,
            zeroFill: Bool
        ) async throws -> MLMultiArray {
            let key = MultiArrayKey(dataType: dataType, shape: shape)
            let array: MLMultiArray
            if var cached = storage[key], let candidate = cached.popLast() {
                storage[key] = cached
                array = candidate
            } else {
                array = try MLMultiArray(shape: shape, dataType: dataType)
            }

            if zeroFill {
                zero(array)
            }
            return array
        }

        func recycle(_ array: MLMultiArray, zeroFill: Bool) {
            if zeroFill {
                zero(array)
            }
            let key = MultiArrayKey(dataType: array.dataType, shape: array.shape)
            storage[key, default: []].append(array)
        }

        func preallocate(
            shape: [NSNumber],
            dataType: MLMultiArrayDataType,
            count: Int,
            zeroFill: Bool
        ) async throws {
            guard count > 0 else { return }
            let key = MultiArrayKey(dataType: dataType, shape: shape)
            var pool = storage[key] ?? []
            if pool.count >= count {
                storage[key] = pool
                return
            }

            let additional = count - pool.count
            pool.reserveCapacity(count)
            for _ in 0..<additional {
                let array = try MLMultiArray(shape: shape, dataType: dataType)
                if zeroFill {
                    zero(array)
                }
                pool.append(array)
            }
            storage[key] = pool
        }

        private func zero(_ array: MLMultiArray) {
            let elementCount = array.count
            guard elementCount > 0 else { return }

            switch array.dataType {
            case .int32:
                let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: elementCount)
                pointer.initialize(repeating: 0, count: elementCount)
            case .float32:
                let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: elementCount)
                vDSP_vclr(pointer, 1, vDSP_Length(elementCount))
            case .double:
                let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: elementCount)
                vDSP_vclrD(pointer, 1, vDSP_Length(elementCount))
            case .float16:
                let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: elementCount)
                pointer.initialize(repeating: 0, count: elementCount)
            #if swift(>=6.2)
            case .int8:
                array.dataPointer.initializeMemory(as: Int8.self, repeating: 0, count: elementCount)
            #endif
            default:
                memset(array.dataPointer, 0, elementCount * MemoryLayout<Float>.stride)
            }
        }
    }

    // Cached CoreML models per Kokoro variant
    // Legacy: Phoneme dictionary with frame counts (kept for backward compatibility)
    private actor LexiconCache {
        private var wordToPhonemes: [String: [String]] = [:]
        private var caseSensitiveWordToPhonemes: [String: [String]] = [:]
        private var isLoaded = false

        private struct CachePayload: Codable {
            let lower: [String: [String]]
            let caseSensitive: [String: [String]]
        }

        func ensureLoaded(kokoroDirectory: URL, allowedTokens: Set<String>) async throws {
            if isLoaded && !caseSensitiveWordToPhonemes.isEmpty { return }

            let cacheURL = kokoroDirectory.appendingPathComponent("us_lexicon_cache.json")
            if await loadFromCache(cacheURL, allowedTokens: allowedTokens) {
                return
            }
            throw TTSError.processingFailed("No lexicon cache found and raw lexicon merge disabled")
        }

        func lexicons() -> (word: [String: [String]], caseSensitive: [String: [String]]) {
            (wordToPhonemes, caseSensitiveWordToPhonemes)
        }

        private func loadFromCache(_ url: URL, allowedTokens: Set<String>) async -> Bool {
            guard FileManager.default.fileExists(atPath: url.path) else { return false }
            do {
                let data = try Data(contentsOf: url)
                let payload = try JSONDecoder().decode(CachePayload.self, from: data)
                let filteredLower = payload.lower.mapValues { $0.filter { allowedTokens.contains($0) } }
                let filteredCase = payload.caseSensitive.mapValues { $0.filter { allowedTokens.contains($0) } }

                guard !filteredLower.isEmpty else { return false }

                wordToPhonemes = filteredLower
                caseSensitiveWordToPhonemes = filteredCase
                isLoaded = true
                KokoroSynthesizer.logger.info("Loaded lexicon cache: \(filteredLower.count) entries")
                return true
            } catch {
                KokoroSynthesizer.logger.warning("Failed to load lexicon cache: \(error.localizedDescription)")
                wordToPhonemes = [:]
                caseSensitiveWordToPhonemes = [:]
                isLoaded = false
                return false
            }
        }

        private func writeCache(payload: CachePayload, to url: URL) async {
            do {
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                let data = try encoder.encode(payload)
                try data.write(to: url, options: [.atomic])
                KokoroSynthesizer.logger.info("Wrote lexicon cache to \(url.path)")
            } catch {
                KokoroSynthesizer.logger.warning("Failed to persist lexicon cache: \(error.localizedDescription)")
            }
        }
    }

    private static let lexiconCache = LexiconCache()
    private static let multiArrayPool = MultiArrayPool()

    private struct VoiceEmbeddingPayload {
        let sourceURL: URL
        let json: Any
    }

    private struct VoiceEmbeddingCacheKey: Hashable {
        let voiceID: String
        let phonemeCount: Int
    }

    private struct VoiceEmbeddingData: Sendable {
        let voiceID: String
        let vector: [Float]
        let l2Norm: Float
    }

    private static var voiceEmbeddingPayloads: [String: VoiceEmbeddingPayload] = [:]
    private static var voiceEmbeddingVectors: [VoiceEmbeddingCacheKey: [Float]] = [:]
    private static let voiceEmbeddingLock = NSLock()

    private static func candidateVoiceEmbeddingURLs(
        for voiceID: String,
        cwd: URL,
        voicesDir: URL
    ) -> [URL] {
        [
            cwd.appendingPathComponent("voices/\(voiceID).json"),
            cwd.appendingPathComponent("\(voiceID).json"),
            voicesDir.appendingPathComponent("\(voiceID).json"),
        ]
    }

    private static func cachedVoiceEmbeddingPayload(
        for voiceID: String,
        candidates: [URL]
    ) throws -> VoiceEmbeddingPayload {
        voiceEmbeddingLock.lock()
        if let payload = voiceEmbeddingPayloads[voiceID] {
            voiceEmbeddingLock.unlock()
            return payload
        }
        voiceEmbeddingLock.unlock()

        guard let source = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            let checkedPaths = candidates.map { $0.path }.joined(separator: ", ")
            throw TTSError.modelNotFound(
                "Voice embedding unavailable for \(voiceID); checked paths: \(checkedPaths)")
        }

        let data = try Data(contentsOf: source)
        let json = try JSONSerialization.jsonObject(with: data)
        let payload = VoiceEmbeddingPayload(sourceURL: source, json: json)

        voiceEmbeddingLock.lock()
        voiceEmbeddingPayloads[voiceID] = payload
        voiceEmbeddingLock.unlock()

        return payload
    }

    private static func cachedVoiceEmbeddingVector(for key: VoiceEmbeddingCacheKey) -> [Float]? {
        voiceEmbeddingLock.lock()
        let vector = voiceEmbeddingVectors[key]
        voiceEmbeddingLock.unlock()
        return vector
    }

    private static func storeVoiceEmbeddingVector(_ vector: [Float], for key: VoiceEmbeddingCacheKey) {
        voiceEmbeddingLock.lock()
        voiceEmbeddingVectors[key] = vector
        voiceEmbeddingLock.unlock()
    }

    private static func fetchVoiceEmbeddingData(
        voice: String,
        phonemeCount: Int,
        expectedDimension: Int
    ) throws -> VoiceEmbeddingData {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        func resolveVector(for voiceID: String) throws -> ([Float]?, URL, Bool) {
            let candidates = candidateVoiceEmbeddingURLs(for: voiceID, cwd: cwd, voicesDir: voicesDir)
            do {
                let payload = try cachedVoiceEmbeddingPayload(for: voiceID, candidates: candidates)
                let key = VoiceEmbeddingCacheKey(voiceID: voiceID, phonemeCount: phonemeCount)
                if let cached = Self.cachedVoiceEmbeddingVector(for: key) {
                    return (cached, payload.sourceURL, true)
                }
                let vector = parseVoiceEmbeddingVector(payload.json, voiceID: voiceID, phonemeCount: phonemeCount)
                if let vector {
                    Self.storeVoiceEmbeddingVector(vector, for: key)
                } else {
                    Self.logger.warning(
                        "Voice embedding payload lacked usable vector for \(voiceID) at \(payload.sourceURL.path)"
                    )
                }
                return (vector, payload.sourceURL, false)
            } catch {
                Self.logger.warning("Failed to load voice embedding for \(voiceID): \(error.localizedDescription)")
                return (nil, candidates.first ?? voicesDir, false)
            }
        }

        var voiceUsed = voice
        var attemptedVoices = [voice]
        var (vector, sourceURL, wasCached) = try resolveVector(for: voiceUsed)

        if vector == nil && voice != "af_heart" {
            Self.logger.warning("Voice embedding for \(voice) not found; falling back to af_heart")
            voiceUsed = "af_heart"
            attemptedVoices.append("af_heart")
            (vector, sourceURL, wasCached) = try resolveVector(for: voiceUsed)
        }

        guard let resolvedVector = vector else {
            throw TTSError.modelNotFound(
                "Voice embedding unavailable for \(attemptedVoices.joined(separator: ", ")); checked \(sourceURL.path)"
            )
        }

        guard resolvedVector.count == expectedDimension else {
            throw TTSError.modelNotFound(
                "Voice embedding for \(voiceUsed) has unexpected length (expected \(expectedDimension), got \(resolvedVector.count))"
            )
        }

        var sumSquares: Float = 0
        resolvedVector.withUnsafeBufferPointer { sourcePointer in
            guard let baseAddress = sourcePointer.baseAddress, !sourcePointer.isEmpty else { return }
            vDSP_svesq(baseAddress, 1, &sumSquares, vDSP_Length(sourcePointer.count))
        }
        let norm = sqrt(Double(sumSquares))

        let formattedNorm = String(format: "%.3f", norm)
        if wasCached {
            Self.logger.debug("Reusing cached voice embedding: \(voiceUsed), dim=\(expectedDimension), l2norm=\(formattedNorm)")
        } else {
            Self.logger.info("Loaded voice embedding: \(voiceUsed), dim=\(expectedDimension), l2norm=\(formattedNorm)")
        }

        return VoiceEmbeddingData(voiceID: voiceUsed, vector: resolvedVector, l2Norm: Float(norm))
    }

    private static func isVoiceEmbeddingPayloadCached(for voiceID: String) -> Bool {
        voiceEmbeddingLock.lock()
        let cached = voiceEmbeddingPayloads[voiceID] != nil
        voiceEmbeddingLock.unlock()
        return cached
    }

    private static func parseVoiceEmbeddingVector(
        _ json: Any,
        voiceID: String,
        phonemeCount: Int
    ) -> [Float]? {
        func parseArray(_ any: Any) -> [Float]? {
            if let doubles = any as? [Double] { return doubles.map(Float.init) }
            if let floats = any as? [Float] { return floats }
            if let numbers = any as? [NSNumber] { return numbers.map { $0.floatValue } }
            if let anyArray = any as? [Any] {
                var out: [Float] = []
                out.reserveCapacity(anyArray.count)
                for value in anyArray {
                    if let num = value as? NSNumber {
                        out.append(num.floatValue)
                    } else if let dbl = value as? Double {
                        out.append(Float(dbl))
                    } else if let flt = value as? Float {
                        out.append(flt)
                    } else {
                        return nil
                    }
                }
                return out
            }
            return nil
        }

        if let direct = parseArray(json) {
            return direct
        }

        guard let dict = json as? [String: Any] else { return nil }

        if let embed = dict["embedding"], let parsed = parseArray(embed) {
            return parsed
        }

        if let voiceSpecific = dict[voiceID], let parsed = parseArray(voiceSpecific) {
            return parsed
        }

        var numericCandidates: [(Int, [Float])] = []
        numericCandidates.reserveCapacity(dict.count)
        for (key, value) in dict {
            guard let intKey = Int(key), let parsed = parseArray(value) else { continue }
            numericCandidates.append((intKey, parsed))
        }

        numericCandidates.sort { $0.0 < $1.0 }

        if let exact = numericCandidates.first(where: { $0.0 == phonemeCount }) {
            return exact.1
        }

        if let fallback = numericCandidates.last(where: { $0.0 <= phonemeCount }) {
            return fallback.1
        }

        for value in dict.values {
            if let parsed = parseArray(value) {
                return parsed
            }
        }

        return nil
    }

    // Model and data URLs
    private static func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }

    private static func model(for variant: ModelNames.TTS.Variant) async throws -> MLModel {
        try await KokoroModelCache.shared.model(for: variant)
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

    /// Ensure required dictionary files exist
    public static func ensureRequiredFiles() async throws {
        try await LexiconAssetManager.ensureCoreAssets()
    }

    /// Load Kokoro CoreML models for all supported variants.
    public static func loadModel() async throws {
        try await KokoroModelCache.shared.loadModelsIfNeeded()
    }

    /// Register a bundle of preloaded Core ML models so future calls to `loadModel()` reuse them.
    /// This allows higher-level managers to inject models that were downloaded elsewhere (e.g. pod lint).
    internal static func registerPreloadedModels(_ models: TtsModels) async {
        await KokoroModelCache.shared.registerPreloadedModels(models)
    }

    /// Load simple word->phonemes dictionary (preferred)
    /// Uses the richer US English lexicons (gold/silver) as the primary source.
    public static func loadSimplePhonemeDictionary() async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        let allowed = Set(vocabulary.keys)
        try await lexiconCache.ensureLoaded(kokoroDirectory: kokoroDir, allowedTokens: allowed)
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

    private static func validateTextHasDictionaryCoverage(_ text: String) async throws {
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

        #if canImport(ESpeakNG)
        if !oov.isEmpty && EspeakG2P.isDataAvailable() == false {
            let sample = Set(oov).sorted().prefix(5).joined(separator: ", ")
            throw TTSError.processingFailed(
                "G2P (eSpeak NG) data missing but required for OOV words: \(sample). Ensure the eSpeak NG data bundle is available in the models cache (use DownloadUtils.ensureEspeakDataBundle)."
            )
        }
        #else
        if !oov.isEmpty {
            let sample = Set(oov).sorted().prefix(5).joined(separator: ", ")
            throw TTSError.processingFailed(
                "G2P (eSpeak NG) not included in this build but required for OOV words: \(sample)."
            )
        }
        #endif
    }

    /// Structure to hold a chunk of text that fits within 3.17 seconds
    // TextChunk is defined in KokoroChunker.swift

    /// Chunk text into segments under the model token budget, using punctuation-driven pauses.
    private static func chunkText(
        _ text: String,
        vocabulary: [String: Int32],
        longVariantTokenBudget: Int
    ) async throws -> [TextChunk] {
        try await loadSimplePhonemeDictionary()
        let hasLang = false
        let lexicons = await lexiconCache.lexicons()
        return KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lexicons.word,
            caseSensitiveLexicon: lexicons.caseSensitive,
            targetTokens: longVariantTokenBudget,
            hasLanguageToken: hasLang,
            allowedPhonemes: Set(vocabulary.keys)
        )
    }

    private static func buildChunkEntries(
        from chunks: [TextChunk],
        vocabulary: [String: Int32],
        preference: ModelNames.TTS.Variant?,
        capacities: TokenCapacities
    ) throws -> [ChunkEntry] {
        var entries: [ChunkEntry] = []
        entries.reserveCapacity(chunks.count)

        for (index, chunk) in chunks.enumerated() {
            let inputIds = phonemesToInputIds(chunk.phonemes, vocabulary: vocabulary)
            guard !inputIds.isEmpty else {
                let joinedWords = chunk.words.joined(separator: " ")
                throw TTSError.processingFailed(
                    "No input IDs generated for chunk: \(joinedWords)")
            }
            let variant = try selectVariant(
                forTokenCount: inputIds.count,
                preference: preference,
                capacities: capacities
            )
            let targetTokens = capacities.capacity(for: variant)
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
            Self.logger.info("Text fits in single chunk")
        } else {
            Self.logger.info("Text split into \(entries.count) chunks")
        }

        return entries
    }

    /// Convert phonemes to input IDs
    public static func phonemesToInputIds(
        _ phonemes: [String],
        vocabulary: [String: Int32]
    ) -> [Int32] {
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
                Self.logger.warning(
                    "Found \(outOfRange.count) token IDs out of range [\(minId), \(maxId)] (excluding BOS/EOS=0)"
                )
            }
            Self.logger.debug("Tokenized \(ids.count) ids; first 32: \(ids.prefix(32))")
        }
        #endif

        return ids
    }

    public static func phonemesToInputIds(_ phonemes: [String]) async throws -> [Int32] {
        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        return phonemesToInputIds(phonemes, vocabulary: vocabulary)
    }

    /// Inspect model to determine the expected token length for input_ids
    private static func tokenLength(for variant: ModelNames.TTS.Variant) async throws -> Int {
        try await KokoroModelCache.shared.tokenLength(for: variant)
    }

    private static func tokenCapacities() async throws -> TokenCapacities {
        async let short = tokenLength(for: .fiveSecond)
        async let long = tokenLength(for: .fifteenSecond)
        return try await TokenCapacities(short: short, long: long)
    }

    private static func selectVariant(
        forTokenCount tokenCount: Int,
        preference: ModelNames.TTS.Variant?,
        capacities: TokenCapacities
    ) throws -> ModelNames.TTS.Variant {
        if let preference {
            let capacity = capacities.capacity(for: preference)
            if tokenCount <= capacity {
                return preference
            }

            logger.notice(
                "Requested \(variantDescription(preference)) variant but token count \(tokenCount) exceeds capacity=\(capacity); falling back to automatic selection"
            )
            if preference == .fifteenSecond {
                throw TTSError.processingFailed(
                    "Chunk token count \(tokenCount) exceeds \(variantDescription(preference)) capacity \(capacity)"
                )
            }
            // fall through to automatic selection
        }
        let shortCapacity = capacities.short
        let longCapacity = capacities.long
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

    public static func loadVoiceEmbedding(voice: String = "af_heart", phonemeCount: Int) async throws -> MLMultiArray {
        let expectedDimension = try await KokoroModelCache.shared.referenceEmbeddingDimension()
        let data = try fetchVoiceEmbeddingData(
            voice: voice,
            phonemeCount: phonemeCount,
            expectedDimension: expectedDimension
        )

        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: data.vector.count)] as [NSNumber],
            dataType: .float32
        )

        data.vector.withUnsafeBufferPointer { sourcePointer in
            guard let baseAddress = sourcePointer.baseAddress, !sourcePointer.isEmpty else { return }
            let destinationPointer = embedding.dataPointer.assumingMemoryBound(to: Float.self)
            destinationPointer.update(from: baseAddress, count: sourcePointer.count)
        }

        return embedding
    }

    /// Helper to fetch ref_s expected dimension from model
    internal static func refDim(from model: MLModel) -> Int {
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
            let shape = desc.multiArrayConstraint?.shape,
            shape.count >= 2
        {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }

    private static func prepareVoiceEmbeddingCache(
        voice: String,
        entries: [ChunkEntry],
        embeddingDimension: Int
    ) throws -> [Int: VoiceEmbeddingData] {
        let uniqueCounts = Set(entries.map { $0.inputIds.count })
        var cache: [Int: VoiceEmbeddingData] = [:]
        cache.reserveCapacity(uniqueCounts.count)

        for count in uniqueCounts {
            cache[count] = try fetchVoiceEmbeddingData(
                voice: voice,
                phonemeCount: count,
                expectedDimension: embeddingDimension
            )
        }

        return cache
    }

    /// Synthesize a single chunk of text using precomputed token IDs.
    private static func synthesizeChunk(
        _ chunk: TextChunk,
        inputIds: [Int32],
        variant: ModelNames.TTS.Variant,
        targetTokens: Int,
        referenceVector: [Float]
    ) async throws -> ([Float], TimeInterval) {
        guard !inputIds.isEmpty else {
            throw TTSError.processingFailed("No input IDs generated for chunk: \(chunk.words.joined(separator: " "))")
        }

        let kokoro = try await model(for: variant)

        let refShape: [NSNumber] = [1, NSNumber(value: referenceVector.count)]
        let refStyle = try await multiArrayPool.rent(
            shape: refShape,
            dataType: .float32,
            zeroFill: false
        )
        let refPointer = refStyle.dataPointer.bindMemory(to: Float.self, capacity: referenceVector.count)
        referenceVector.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            refPointer.update(from: baseAddress, count: buffer.count)
        }

        // Pad or truncate to match model expectation
        var trimmedIds = inputIds
        if trimmedIds.count > targetTokens {
            logger.warning(
                "input_ids length (\(trimmedIds.count)) exceeds targetTokens=\(targetTokens) for chunk '\(chunk.text)' — truncating"
            )
            trimmedIds = Array(trimmedIds.prefix(targetTokens))
        } else if trimmedIds.count < targetTokens {
            Self.logger.debug(
                "input_ids length (\(trimmedIds.count)) below targetTokens=\(targetTokens) for chunk '\(chunk.text)' — padding with zeros"
            )
            trimmedIds.append(contentsOf: Array(repeating: Int32(0), count: targetTokens - trimmedIds.count))
        }

        let inputShape: [NSNumber] = [1, NSNumber(value: targetTokens)]
        let phasesShape: [NSNumber] = [1, 9]
        let inputArray = try await multiArrayPool.rent(
            shape: inputShape,
            dataType: .int32,
            zeroFill: false
        )
        let attentionMask = try await multiArrayPool.rent(
            shape: inputShape,
            dataType: .int32,
            zeroFill: false
        )
        let phasesArray = try await multiArrayPool.rent(
            shape: phasesShape,
            dataType: .float32,
            zeroFill: true
        )

        func recycleModelArrays() async {
            await multiArrayPool.recycle(phasesArray, zeroFill: true)
            await multiArrayPool.recycle(attentionMask, zeroFill: false)
            await multiArrayPool.recycle(inputArray, zeroFill: false)
            await multiArrayPool.recycle(refStyle, zeroFill: false)
        }

        let inputPointer = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: targetTokens)
        inputPointer.initialize(repeating: 0, count: targetTokens)
        trimmedIds.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            inputPointer.update(from: baseAddress, count: buffer.count)
        }

        let maskPointer = attentionMask.dataPointer.bindMemory(to: Int32.self, capacity: targetTokens)
        maskPointer.initialize(repeating: 0, count: targetTokens)
        let trueLen = min(inputIds.count, targetTokens)
        if trueLen > 0 {
            for idx in 0..<trueLen {
                maskPointer[idx] = 1
            }
        }

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

        let predictionStart = Date()
        let output: MLFeatureProvider
        do {
            output = try await kokoro.compatPrediction(from: modelInput, options: MLPredictionOptions())
        } catch {
            await recycleModelArrays()
            throw error
        }
        let predictionTime = Date().timeIntervalSince(predictionStart)
        // Extract audio output explicitly by key used by model
        guard let audioArrayUnwrapped = output.featureValue(for: "audio")?.multiArrayValue,
            audioArrayUnwrapped.count > 0
        else {
            let names = Array(output.featureNames)
            await recycleModelArrays()
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

        if variant == .fiveSecond {
            let thresholdSamples = Int(shortVariantGuardThresholdSeconds * Double(sampleRateHz))
            if effectiveCount < thresholdSamples {
                let guardSamples = shortVariantGuardFrameCount * kokoroFrameSamples
                if effectiveCount > guardSamples {
                    effectiveCount -= guardSamples
                }
            }
        }

        // Convert to float samples
        let samples: [Float]
        if audioArrayUnwrapped.dataType == .float32 {
            let sourcePointer = audioArrayUnwrapped.dataPointer.bindMemory(
                to: Float.self, capacity: audioArrayUnwrapped.count)
            samples = Array(UnsafeBufferPointer(start: sourcePointer, count: effectiveCount))
        } else {
            var fallback: [Float] = []
            fallback.reserveCapacity(effectiveCount)
            for i in 0..<effectiveCount {
                fallback.append(audioArrayUnwrapped[i].floatValue)
            }
            samples = fallback
        }

        // Basic sanity logging
        let minVal = samples.min() ?? 0
        let maxVal = samples.max() ?? 0
        if maxVal - minVal == 0 {
            logger.warning("Prediction produced constant signal (min=max=\(minVal)).")
        } else {
            logger.info("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
        }

        await recycleModelArrays()
        return (samples, predictionTime)
    }

    /// Main synthesis function returning audio bytes only.
    public static func synthesize(
        text: String,
        voice: String = "af_heart",
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> Data {
        let startTime = Date()
        let result = try await synthesizeDetailed(
            text: text,
            voice: voice,
            variantPreference: variantPreference
        )
        let totalTime = Date().timeIntervalSince(startTime)
        Self.logger.info("Total synthesis time: \(String(format: "%.3f", totalTime))s for \(text.count) characters")
        return result.audio
    }

    /// Synthesize audio while returning per-chunk metadata used during inference.
    public static func synthesizeDetailed(
        text: String,
        voice: String = "af_heart",
        variantPreference: ModelNames.TTS.Variant? = nil
    ) async throws -> SynthesisResult {

        logger.info("Starting synthesis: '\(text)'")
        logger.info("Input length: \(text.count) characters")

        try await ensureRequiredFiles()
        if !isVoiceEmbeddingPayloadCached(for: voice) {
            try? await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: voice)
        }

        try await loadModel()

        try await loadSimplePhonemeDictionary()

        try await validateTextHasDictionaryCoverage(text)

        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        let capacities = try await tokenCapacities()

        let chunks = try await chunkText(
            text,
            vocabulary: vocabulary,
            longVariantTokenBudget: capacities.long
        )
        guard !chunks.isEmpty else {
            throw TTSError.processingFailed("No valid words found in text")
        }

        let entries = try buildChunkEntries(
            from: chunks,
            vocabulary: vocabulary,
            preference: variantPreference,
            capacities: capacities
        )

        struct ChunkSynthesisResult: Sendable {
            let index: Int
            let samples: [Float]
            let predictionTime: TimeInterval
        }

        let embeddingDimension = try await KokoroModelCache.shared.referenceEmbeddingDimension()
        let embeddingCache = try prepareVoiceEmbeddingCache(
            voice: voice,
            entries: entries,
            embeddingDimension: embeddingDimension
        )

        let totalChunks = entries.count
        let groupedByTargetTokens = Dictionary(grouping: entries, by: { $0.template.targetTokens })
        let phasesShape: [NSNumber] = [1, 9]
        try await multiArrayPool.preallocate(
            shape: phasesShape,
            dataType: .float32,
            count: max(1, totalChunks),
            zeroFill: true
        )
        for (targetTokens, group) in groupedByTargetTokens {
            let shape: [NSNumber] = [1, NSNumber(value: targetTokens)]
            try await multiArrayPool.preallocate(
                shape: shape,
                dataType: .int32,
                count: max(1, group.count * 2),
                zeroFill: false
            )
        }
        let refShape: [NSNumber] = [1, NSNumber(value: embeddingDimension)]
        try await multiArrayPool.preallocate(
            shape: refShape,
            dataType: .float32,
            count: max(1, totalChunks),
            zeroFill: false
        )
        let chunkTemplates = entries.map { $0.template }
        var chunkSampleBuffers = Array(repeating: [Float](), count: totalChunks)
        var allSamples: [Float] = []
        let crossfadeMs = 8
        let crossfadeN = max(0, Int(Double(crossfadeMs) * 24.0))
        var totalPredictionTime: TimeInterval = 0
        Self.logger.info("Starting audio inference across \(totalChunks) chunk(s)")

        let chunkOutputs = try await withThrowingTaskGroup(of: ChunkSynthesisResult.self) { group in
            for (index, entry) in entries.enumerated() {
                let chunk = entry.chunk
                let inputIds = entry.inputIds
                let template = entry.template
                let chunkIndex = index
                guard let embeddingData = embeddingCache[inputIds.count] else {
                    throw TTSError.processingFailed(
                        "Missing voice embedding for chunk \(index + 1) with \(inputIds.count) tokens"
                    )
                }
                let referenceVector = embeddingData.vector
                group.addTask(priority: .userInitiated) {
                    Self.logger.info(
                        "Processing chunk \(chunkIndex + 1)/\(totalChunks): \(chunk.words.count) words")
                    Self.logger.info("Chunk \(chunkIndex + 1) text: '\(template.text)'")
                    Self.logger.info(
                        "Chunk \(chunkIndex + 1) using Kokoro \(variantDescription(template.variant)) model")
                    let (chunkSamples, predictionTime) = try await synthesizeChunk(
                        chunk,
                        inputIds: inputIds,
                        variant: template.variant,
                        targetTokens: template.targetTokens,
                        referenceVector: referenceVector)
                    return ChunkSynthesisResult(
                        index: chunkIndex,
                        samples: chunkSamples,
                        predictionTime: predictionTime)
                }
            }

            var results: [ChunkSynthesisResult] = []
            results.reserveCapacity(totalChunks)
            for try await result in group {
                results.append(result)
            }
            return results
        }

        let sortedOutputs = chunkOutputs.sorted { $0.index < $1.index }

        for output in sortedOutputs {
            let index = output.index
            let chunkSamples = output.samples
            chunkSampleBuffers[index] = chunkSamples
            totalPredictionTime += output.predictionTime

            Self.logger.info(
                "Chunk \(index + 1) model prediction latency: \(String(format: "%.3f", output.predictionTime))s")
            let chunkDurationSeconds = Double(chunkSamples.count) / Double(sampleRateHz)
            let chunkFrameCount = kokoroFrameSamples > 0 ? chunkSamples.count / kokoroFrameSamples : 0
            Self.logger.info(
                "Chunk \(index + 1) duration: \(String(format: "%.3f", chunkDurationSeconds))s (\(chunkFrameCount) frames)"
            )

            if index == 0 {
                allSamples.append(contentsOf: chunkSamples)
                continue
            }

            let prevPause = entries[index - 1].chunk.pauseAfterMs
            if prevPause > 0 {
                let silenceCount = Int(Double(prevPause) * 24.0)
                let expectedAppend = chunkSamples.count + max(0, silenceCount)
                if expectedAppend > 0 {
                    allSamples.reserveCapacity(allSamples.count + expectedAppend)
                }
                if silenceCount > 0 {
                    allSamples.append(contentsOf: repeatElement(0.0, count: silenceCount))
                }
                allSamples.append(contentsOf: chunkSamples)
            } else {
                let n = min(crossfadeN, allSamples.count, chunkSamples.count)
                if n > 0 {
                    let appendCount = max(0, chunkSamples.count - n)
                    if appendCount > 0 {
                        allSamples.reserveCapacity(allSamples.count + appendCount)
                    }
                    let tailStartIndex = allSamples.count - n
                    var fadeIn = [Float](repeating: 0, count: n)
                    if n == 1 {
                        fadeIn[0] = 1
                    } else {
                        var start: Float = 0
                        var step: Float = 1.0 / Float(n - 1)
                        fadeIn.withUnsafeMutableBufferPointer { buffer in
                            guard let baseAddress = buffer.baseAddress else { return }
                            vDSP_vramp(&start, &step, baseAddress, 1, vDSP_Length(n))
                        }
                    }

                    var fadeOut = [Float](repeating: 1, count: n)
                    fadeIn.withUnsafeBufferPointer { fadeInBuffer in
                        fadeOut.withUnsafeMutableBufferPointer { fadeOutBuffer in
                            guard let fadeInBase = fadeInBuffer.baseAddress,
                                let fadeOutBase = fadeOutBuffer.baseAddress
                            else { return }
                            vDSP_vsub(fadeInBase, 1, fadeOutBase, 1, fadeOutBase, 1, vDSP_Length(n))
                        }
                    }

                    allSamples.withUnsafeMutableBufferPointer { allBuffer in
                        guard let allBase = allBuffer.baseAddress else { return }
                        let tailPointer = allBase.advanced(by: tailStartIndex)
                        fadeOut.withUnsafeBufferPointer { fadeOutBuffer in
                            guard let fadeOutBase = fadeOutBuffer.baseAddress else { return }
                            vDSP_vmul(tailPointer, 1, fadeOutBase, 1, tailPointer, 1, vDSP_Length(n))
                        }
                        chunkSamples.withUnsafeBufferPointer { chunkBuffer in
                            guard let chunkBase = chunkBuffer.baseAddress else { return }
                            fadeIn.withUnsafeBufferPointer { fadeInBuffer in
                                guard let fadeInBase = fadeInBuffer.baseAddress else { return }
                                vDSP_vma(chunkBase, 1, fadeInBase, 1, tailPointer, 1, tailPointer, 1, vDSP_Length(n))
                            }
                        }
                    }

                    if chunkSamples.count > n {
                        allSamples.append(contentsOf: chunkSamples[n...])
                    }
                } else {
                    allSamples.reserveCapacity(allSamples.count + chunkSamples.count)
                    allSamples.append(contentsOf: chunkSamples)
                }
            }
        }

        guard !allSamples.isEmpty else {
            throw TTSError.processingFailed("Synthesis produced no samples")
        }

        var maxMagnitude: Float = 0
        allSamples.withUnsafeBufferPointer { pointer in
            guard let baseAddress = pointer.baseAddress, !pointer.isEmpty else { return }
            vDSP_maxmgv(baseAddress, 1, &maxMagnitude, vDSP_Length(pointer.count))
        }

        if maxMagnitude > 0 {
            var divisor = maxMagnitude
            allSamples.withUnsafeMutableBufferPointer { destination in
                guard let destBase = destination.baseAddress else { return }
                vDSP_vsdiv(destBase, 1, &divisor, destBase, 1, vDSP_Length(destination.count))
            }
            for index in chunkSampleBuffers.indices {
                chunkSampleBuffers[index].withUnsafeMutableBufferPointer { destination in
                    guard let destBase = destination.baseAddress, !destination.isEmpty else { return }
                    var chunkDivisor = maxMagnitude
                    vDSP_vsdiv(destBase, 1, &chunkDivisor, destBase, 1, vDSP_Length(destination.count))
                }
            }
        }

        let audioData = try AudioWAV.data(from: allSamples, sampleRate: 24000)

        let chunkInfos = zip(chunkTemplates, chunkSampleBuffers).map { template, samples in
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

        Self.logger.notice(
            "Total model prediction time: \(String(format: "%.3f", totalPredictionTime))s for \(entries.count) chunk(s)"
        )
        return SynthesisResult(audio: audioData, chunks: chunkInfos)
    }

    // convertSamplesToWAV moved to AudioWAV

    // convertToWAV removed (unused); use convertSamplesToWAV instead
}
