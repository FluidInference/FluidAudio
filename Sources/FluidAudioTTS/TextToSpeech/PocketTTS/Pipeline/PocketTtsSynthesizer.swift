@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

/// PocketTTS flow-matching language model synthesizer.
///
/// Generates audio autoregressively: each generation step produces
/// an 80ms audio frame (1920 samples at 24kHz).
///
/// Long text is split into sentence-based chunks (≤50 tokens each)
/// to stay within the KV cache limit (200 positions).
///
/// Pipeline: text → chunk → [tokenize → embed → prefill KV → generate → flow decode → mimi decode] → WAV
public struct PocketTtsSynthesizer {

    static let logger = AppLogger(category: "PocketTtsSynthesizer")

    private enum Context {
        @TaskLocal static var modelCache: PocketTtsModelCache?
    }

    static func withModelCache<T>(
        _ cache: PocketTtsModelCache,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelCache.withValue(cache) {
            try await operation()
        }
    }

    static func currentModelCache() throws -> PocketTtsModelCache {
        guard let cache = Context.modelCache else {
            throw TTSError.processingFailed(
                "PocketTtsSynthesizer requires a model cache context.")
        }
        return cache
    }

    // MARK: - Public API

    /// Synthesize audio from text.
    ///
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voice: Voice identifier (default: "alba").
    ///   - temperature: Generation temperature (default: 0.7).
    ///   - seed: Random seed for reproducibility (nil for random).
    ///   - deEss: Whether to apply de-essing post-processing.
    /// - Returns: A synthesis result containing WAV audio data.
    public static func synthesize(
        text: String,
        voice: String = PocketTtsConstants.defaultVoice,
        temperature: Float = PocketTtsConstants.temperature,
        seed: UInt64? = nil,
        deEss: Bool = true
    ) async throws -> SynthesisResult {
        let cache = try currentModelCache()

        logger.info("PocketTTS synthesizing: '\(text)'")

        // 1. Load constants and voice
        let constants = try await cache.constants()
        let voiceData = try await cache.voiceData(for: voice)

        // 2. Split text into chunks that fit within KV cache capacity
        let chunks = chunkText(text, tokenizer: constants.tokenizer)
        logger.info("Split into \(chunks.count) chunk(s)")

        // 3. Set random seed if specified
        if let seed {
            srand48(Int(seed))
        }

        // 4. Load models
        let condModel = try await cache.condStep()
        let stepModel = try await cache.flowlmStep()
        let flowModel = try await cache.flowDecoder()
        let mimiModel = try await cache.mimiDecoder()

        // 5. Load Mimi initial state (continuous across chunks)
        let repoDir = try await resolveRepoDirectory(cache: cache)
        var mimiState = try loadMimiInitialState(from: repoDir)

        // 6. Create BOS embedding
        let bosEmb = try createBosEmbedding(constants.bosEmbedding)

        // 7. Generate audio for each chunk
        var audioChunks: [[Float]] = []
        var lastEosStep: Int?
        var totalSteps = 0

        let genStart = Date()

        for (chunkIdx, chunkText) in chunks.enumerated() {
            let (normalizedChunk, framesAfterEos) = normalizeText(chunkText)
            logger.info("Chunk \(chunkIdx + 1)/\(chunks.count): '\(normalizedChunk)'")

            // Tokenize and embed this chunk
            let tokenIds = constants.tokenizer.encode(normalizedChunk)
            let textEmbeddings = embedTokens(tokenIds, constants: constants)

            // Fresh KV cache per chunk
            let prefillStart = Date()
            var kvState = try await prefillKVCache(
                voiceData: voiceData,
                textEmbeddings: textEmbeddings,
                model: condModel
            )
            let prefillElapsed = Date().timeIntervalSince(prefillStart)
            logger.info(
                "Chunk \(chunkIdx + 1) prefill: \(String(format: "%.2f", prefillElapsed))s (\(tokenIds.count) tokens)"
            )

            // Generation loop for this chunk
            let maxGenLen = estimateMaxFrames(text: chunkText)
            var eosStep: Int?
            var sequence = try createNaNSequence()
            let totalFramesAfterEos =
                framesAfterEos + PocketTtsConstants.extraFramesAfterDetection

            for step in 0..<maxGenLen {
                let (transformerOut, eosLogit) = try await runFlowLMStep(
                    sequence: sequence,
                    bosEmb: bosEmb,
                    state: &kvState,
                    model: stepModel
                )

                if eosLogit > PocketTtsConstants.eosThreshold && eosStep == nil {
                    eosStep = step
                    logger.info("Chunk \(chunkIdx + 1) EOS at step \(step)")
                }
                if let eos = eosStep, step >= eos + totalFramesAfterEos {
                    break
                }

                let latent = try await flowDecode(
                    transformerOut: transformerOut,
                    numSteps: PocketTtsConstants.numLsdSteps,
                    temperature: temperature,
                    model: flowModel
                )

                let denormalized = denormalize(
                    latent, mean: constants.embMean, std: constants.embStd)
                let quantized = quantize(
                    denormalized, weight: constants.quantizerWeight)

                // Mimi state is continuous across chunks
                let frameSamples = try await runMimiDecoder(
                    quantized: quantized,
                    state: &mimiState,
                    model: mimiModel
                )
                audioChunks.append(frameSamples)

                sequence = try createSequenceFromLatent(latent)

                if step % 20 == 0 {
                    logger.info("Chunk \(chunkIdx + 1) step \(step)...")
                }
            }

            totalSteps += eosStep ?? maxGenLen
            lastEosStep = eosStep
        }

        let genElapsed = Date().timeIntervalSince(genStart)
        logger.info(
            "Generated \(audioChunks.count) frames in \(String(format: "%.2f", genElapsed))s")

        // 8. Concatenate audio (no peak normalization — preserve natural levels)
        var allSamples = audioChunks.flatMap { $0 }

        // De-essing
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &allSamples,
                sampleRate: Float(PocketTtsConstants.audioSampleRate),
                deEssAmount: -3.0,
                smoothing: false
            )
        }

        // 9. Encode WAV
        let audioData = try AudioWAV.data(
            from: allSamples,
            sampleRate: Double(PocketTtsConstants.audioSampleRate)
        )

        let duration = Double(allSamples.count) / Double(PocketTtsConstants.audioSampleRate)
        logger.info("Audio duration: \(String(format: "%.2f", duration))s")

        return SynthesisResult(
            audio: audioData,
            samples: allSamples,
            frameCount: audioChunks.count,
            eosStep: lastEosStep
        )
    }

    // MARK: - Text Processing

    /// Normalize a text chunk for PocketTTS (matching Python `prepare_text_prompt`).
    static func normalizeText(_ text: String) -> (text: String, framesAfterEos: Int) {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Collapse whitespace
        result = result.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression)

        // Strip trailing clause punctuation (commas, semicolons, colons)
        // before adding sentence-ending punctuation
        while let last = result.last, ",;:".contains(last) {
            result = String(result.dropLast())
        }
        result = result.trimmingCharacters(in: .whitespaces)

        // Capitalize first letter
        if let first = result.first, first.isLetter {
            result = first.uppercased() + result.dropFirst()
        }

        // Add period if no terminal punctuation
        if let last = result.last, !".!?".contains(last) {
            result += "."
        }

        // Pad short texts for better prosody
        let wordCount = result.split(separator: " ").count
        let framesAfterEos: Int
        if wordCount < PocketTtsConstants.shortTextWordThreshold {
            result = String(repeating: " ", count: 8) + result
            framesAfterEos = PocketTtsConstants.shortTextPadFrames
        } else {
            framesAfterEos = PocketTtsConstants.longTextExtraFrames
        }

        return (result, framesAfterEos)
    }

    /// Split text into chunks that fit within the KV cache token limit.
    ///
    /// Splits at sentence boundaries (`.!?`) and groups sentences into chunks
    /// where each chunk tokenizes to ≤ `maxTokensPerChunk` tokens.
    /// Oversized single sentences are further split at word boundaries.
    static func chunkText(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int = PocketTtsConstants.maxTokensPerChunk
    ) -> [String] {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // If it fits in one chunk, return as-is
        let tokenCount = tokenizer.encode(normalized).count
        if tokenCount <= maxTokens {
            return [normalized]
        }

        // Split into sentences at .!? boundaries
        let sentences = splitSentences(normalized)

        // Further split any oversized sentences at word boundaries
        var pieces: [String] = []
        for sentence in sentences {
            let sentenceTokens = tokenizer.encode(sentence).count
            if sentenceTokens <= maxTokens {
                pieces.append(sentence)
            } else {
                pieces.append(contentsOf: splitOversizedSentence(sentence, tokenizer: tokenizer, maxTokens: maxTokens))
            }
        }

        // Group pieces into chunks that fit the token limit
        var chunks: [String] = []
        var currentChunk = ""

        for piece in pieces {
            let candidate: String
            if currentChunk.isEmpty {
                candidate = piece
            } else {
                candidate = currentChunk + " " + piece
            }

            let candidateTokens = tokenizer.encode(candidate).count
            if candidateTokens <= maxTokens {
                currentChunk = candidate
            } else {
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                }
                currentChunk = piece
            }
        }

        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }

        return chunks.isEmpty ? [normalized] : chunks
    }

    /// Split an oversized sentence to fit within the token limit.
    ///
    /// First tries splitting at clause boundaries (commas, semicolons, colons).
    /// Falls back to word-boundary splitting for clauses that still exceed the limit.
    private static func splitOversizedSentence(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [String] {
        // First try: split at clause boundaries
        let clauseParts = splitAtClauseBoundaries(text)

        // Group clause parts into chunks that fit
        var result: [String] = []
        var currentPart = ""

        for part in clauseParts {
            let candidate = currentPart.isEmpty ? part : currentPart + " " + part
            let candidateTokens = tokenizer.encode(candidate).count

            if candidateTokens <= maxTokens {
                currentPart = candidate
            } else {
                if !currentPart.isEmpty {
                    result.append(currentPart)
                }
                // If single clause part still exceeds limit, split at word boundaries
                if tokenizer.encode(part).count > maxTokens {
                    result.append(contentsOf: splitAtWordBoundaries(part, tokenizer: tokenizer, maxTokens: maxTokens))
                    currentPart = ""
                } else {
                    currentPart = part
                }
            }
        }

        if !currentPart.isEmpty {
            result.append(currentPart)
        }

        return result.isEmpty ? [text] : result
    }

    /// Split text at clause punctuation (commas, semicolons, colons).
    ///
    /// Does not split at commas within numbers (e.g., "3,500").
    private static func splitAtClauseBoundaries(_ text: String) -> [String] {
        let clauseBreaks: Set<Character> = [",", ";", ":"]
        var parts: [String] = []
        var current = ""
        let chars = Array(text)

        for (i, char) in chars.enumerated() {
            current.append(char)

            guard clauseBreaks.contains(char) else { continue }

            // Don't split at commas between digits (e.g., "3,500")
            if char == "," {
                let prevIsDigit = i > 0 && chars[i - 1].isNumber
                let nextIsDigit = i + 1 < chars.count && chars[i + 1].isNumber
                if prevIsDigit && nextIsDigit {
                    continue
                }
            }

            let trimmed = current.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                parts.append(trimmed)
            }
            current = ""
        }

        let trimmed = current.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            parts.append(trimmed)
        }

        return parts
    }

    /// Split text at word boundaries to fit within the token limit.
    private static func splitAtWordBoundaries(
        _ text: String,
        tokenizer: SentencePieceTokenizer,
        maxTokens: Int
    ) -> [String] {
        let words = text.split(separator: " ").map(String.init)
        guard words.count > 1 else { return [text] }

        var chunks: [String] = []
        var currentWords: [String] = []

        for word in words {
            let candidate = (currentWords + [word]).joined(separator: " ")
            let tokens = tokenizer.encode(candidate).count

            if tokens > maxTokens && !currentWords.isEmpty {
                chunks.append(currentWords.joined(separator: " "))
                currentWords = [word]
            } else {
                currentWords.append(word)
            }
        }

        if !currentWords.isEmpty {
            chunks.append(currentWords.joined(separator: " "))
        }

        return chunks
    }

    /// Common abbreviations that end with a period but don't end a sentence.
    private static let abbreviations: Set<String> = [
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "dept", "univ", "govt", "approx",
        "avg", "est", "gen", "gov", "hon", "sgt", "cpl", "pvt", "capt",
        "lt", "col", "maj", "cmdr", "adm", "rev", "sen", "rep",
    ]

    /// Split text into sentences at `.!?` boundaries.
    ///
    /// Handles abbreviations (e.g., "Dr.", "Prof.") by not splitting after them.
    private static func splitSentences(_ text: String) -> [String] {
        var sentences: [String] = []
        var current = ""
        let chars = Array(text)

        for (i, char) in chars.enumerated() {
            current.append(char)

            guard ".!?".contains(char) else { continue }

            // For periods, check if this is an abbreviation
            if char == "." {
                let trimmed = current.trimmingCharacters(in: .whitespaces)
                // Get the last word before the period
                let withoutPeriod = String(trimmed.dropLast())
                let lastWord = withoutPeriod.split(separator: " ").last.map(String.init) ?? withoutPeriod

                // Skip if it's a known abbreviation
                if abbreviations.contains(lastWord.lowercased()) {
                    continue
                }

                // Skip if it's a single uppercase letter (e.g., "J." in initials)
                if lastWord.count == 1, lastWord.first?.isUppercase == true {
                    continue
                }

                // Skip if followed by a digit (e.g., "3.5")
                if i + 1 < chars.count, chars[i + 1].isNumber {
                    continue
                }
            }

            let trimmed = current.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                sentences.append(trimmed)
            }
            current = ""
        }

        // Remaining text without terminal punctuation
        let trimmed = current.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            sentences.append(trimmed)
        }

        return sentences
    }

    // MARK: - Embedding

    /// Look up text token embeddings from the embedding table.
    static func embedTokens(
        _ tokenIds: [Int], constants: PocketTtsConstantsBundle
    ) -> [[Float]] {
        let dim = constants.embeddingDim
        return tokenIds.map { id in
            let offset = id * dim
            return Array(constants.textEmbedTable[offset..<(offset + dim)])
        }
    }

    // MARK: - Helpers

    /// Estimate maximum generation frames based on text length.
    private static func estimateMaxFrames(text: String) -> Int {
        let wordCount = text.split(separator: " ").count
        let genLenSec = Double(wordCount) + 2.0
        return Int(genLenSec * 12.5)
    }

    /// Create the BOS embedding as an MLMultiArray [32].
    private static func createBosEmbedding(_ bos: [Float]) throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(shape: [NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        bos.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base, count: dim)
        }
        return array
    }

    /// Create a NaN-filled sequence [1, 1, 32] (signals BOS to the model).
    private static func createNaNSequence() throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        for i in 0..<dim {
            ptr[i] = .nan
        }
        return array
    }

    /// Create a sequence [1, 1, 32] from a latent vector.
    private static func createSequenceFromLatent(_ latent: [Float]) throws -> MLMultiArray {
        let dim = PocketTtsConstants.latentDim
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        latent.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base, count: dim)
        }
        return array
    }

    /// Resolve the repo directory from the cache.
    private static func resolveRepoDirectory(
        cache: PocketTtsModelCache
    ) async throws -> URL {
        // Trigger load if needed (this will set the repo directory internally)
        try await cache.loadIfNeeded()
        // We need to access the repo directory — use ensureModels() which returns it
        return try await PocketTtsResourceDownloader.ensureModels()
    }
}
