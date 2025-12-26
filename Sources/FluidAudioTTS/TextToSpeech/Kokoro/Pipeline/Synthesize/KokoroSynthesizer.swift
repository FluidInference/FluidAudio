import Accelerate
@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Supports both 5s and 15s variants with US English phoneme lexicons
public struct KokoroSynthesizer {
    static let logger = AppLogger(category: "KokoroSynthesizer")

    static let lexiconCache = LexiconCache()
    static let multiArrayPool = MultiArrayPool()

    private enum Context {
        @TaskLocal static var modelCache: KokoroModelCache?
        @TaskLocal static var lexiconAssets: LexiconAssetManager?
        @TaskLocal static var customLexicon: TtsCustomLexicon?
    }

    static func withModelCache<T>(
        _ cache: KokoroModelCache,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$modelCache.withValue(cache) {
            try await operation()
        }
    }

    static func withLexiconAssets<T>(
        _ assets: LexiconAssetManager,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$lexiconAssets.withValue(assets) {
            try await operation()
        }
    }

    static func withCustomLexicon<T>(
        _ lexicon: TtsCustomLexicon?,
        operation: () async throws -> T
    ) async rethrows -> T {
        try await Context.$customLexicon.withValue(lexicon) {
            try await operation()
        }
    }

    static func currentCustomLexicon() -> TtsCustomLexicon? {
        Context.customLexicon
    }

    static func currentModelCache() throws -> KokoroModelCache {
        guard let cache = Context.modelCache else {
            throw TTSError.processingFailed(
                "KokoroSynthesizer requires a model cache context. Use TtSManager or withModelCache(_:operation:)."
            )
        }
        return cache
    }

    static func currentLexiconAssets() throws -> LexiconAssetManager {
        guard let assets = Context.lexiconAssets else {
            throw TTSError.processingFailed(
                "KokoroSynthesizer requires lexicon assets context. Use TtSManager or withLexiconAssets(_:operation:)."
            )
        }
        return assets
    }

    // Voice embedding caches protected by voiceEmbeddingLock
    nonisolated(unsafe) static var voiceEmbeddingPayloads: [String: VoiceEmbeddingPayload] = [:]
    nonisolated(unsafe) static var voiceEmbeddingVectors: [VoiceEmbeddingCacheKey: [Float]] = [:]
    static let voiceEmbeddingLock = NSLock()

    private static func chunkText(
        _ text: String,
        vocabulary: [String: Int32],
        longVariantTokenBudget: Int,
        phoneticOverrides: [TtsPhoneticOverride]
    ) async throws -> [TextChunk] {
        try await loadSimplePhonemeDictionary()
        let hasLang = false
        let lexicons = await lexiconCache.lexicons()
        let customLexicon = currentCustomLexicon()
        return try KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lexicons.word,
            caseSensitiveLexicon: lexicons.caseSensitive,
            customLexicon: customLexicon,
            targetTokens: longVariantTokenBudget,
            hasLanguageToken: hasLang,
            allowedPhonemes: Set(vocabulary.keys),
            phoneticOverrides: phoneticOverrides
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
        let cache = try currentModelCache()
        return try await cache.tokenLength(for: variant)
    }

    public static func capacities(for preference: ModelNames.TTS.Variant?) async throws -> TokenCapacities {
        switch preference {
        case .fiveSecond?:
            let short = try await tokenLength(for: .fiveSecond)
            return TokenCapacities(short: short, long: short)
        case .fifteenSecond?:
            let long = try await tokenLength(for: .fifteenSecond)
            return TokenCapacities(short: long, long: long)
        case nil:
            async let short = tokenLength(for: .fiveSecond)
            async let long = tokenLength(for: .fifteenSecond)
            return try await TokenCapacities(short: short, long: long)
        }
    }

    public static func tokenBudget(for preference: ModelNames.TTS.Variant?) async throws -> Int {
        let capacities = try await capacities(for: preference)
        if let variant = preference {
            return capacities.capacity(for: variant)
        }
        return capacities.long
    }

    private static func selectVariant(
        forTokenCount tokenCount: Int,
        preference: ModelNames.TTS.Variant?,
        capacities: TokenCapacities
    ) throws -> ModelNames.TTS.Variant {
        if let preference {
            let capacity = capacities.capacity(for: preference)
            guard tokenCount <= capacity else {
                throw TTSError.processingFailed(
                    "Chunk token count \(tokenCount) exceeds \(variantDescription(preference)) capacity \(capacity)"
                )
            }
            return preference
        }
        let shortCapacity = capacities.short
        let longCapacity = capacities.long
        guard tokenCount <= longCapacity else {
            throw TTSError.processingFailed(
                "Chunk token count \(tokenCount) exceeds supported capacities (short=\(shortCapacity), long=\(longCapacity))"
            )
        }

        // Kokoro 5s CoreML export fixes the input_ids window at 71 tokens (shape [1, 71]) so clamp the
        // threshold to that empirically verified limit even if model metadata reports a larger capacity.
        let shortThreshold = min(71, shortCapacity)
        if tokenCount <= shortThreshold {
            return .fiveSecond
        }

        logger.notice(
            "Promoting chunk to Kokoro 15s variant: token count \(tokenCount) exceeds short threshold=\(shortThreshold) (short capacity=\(shortCapacity), long capacity=\(longCapacity))"
        )
        return .fifteenSecond
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
            let thresholdSamples = Int(
                TtsConstants.shortVariantGuardThresholdSeconds
                    * Double(TtsConstants.audioSampleRate)
            )
            if effectiveCount < thresholdSamples {
                let guardSamples =
                    TtsConstants.shortVariantGuardFrameCount * TtsConstants.kokoroFrameSamples
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
        voice: String = TtsConstants.recommendedVoice,
        voiceSpeed: Float = 1.0,
        variantPreference: ModelNames.TTS.Variant? = nil,
        phoneticOverrides: [TtsPhoneticOverride] = []
    ) async throws -> Data {
        let startTime = Date()
        let result = try await synthesizeDetailed(
            text: text,
            voice: voice,
            voiceSpeed: voiceSpeed,
            variantPreference: variantPreference,
            phoneticOverrides: phoneticOverrides
        )
        let totalTime = Date().timeIntervalSince(startTime)
        Self.logger.info("Total synthesis time: \(String(format: "%.3f", totalTime))s for \(text.count) characters")
        return result.audio
    }

    /// Synthesize audio while returning per-chunk metadata used during inference.
    public static func synthesizeDetailed(
        text: String,
        voice: String = TtsConstants.recommendedVoice,
        voiceSpeed: Float = 1.0,
        variantPreference: ModelNames.TTS.Variant? = nil,
        phoneticOverrides: [TtsPhoneticOverride] = []
    ) async throws -> SynthesisResult {

        logger.info("Starting synthesis: '\(text)'")
        logger.info("Input length: \(text.count) characters")
        if let variantPreference {
            logger.info("Variant preference requested: \(variantDescription(variantPreference))")
        } else {
            logger.info("Variant preference requested: automatic")
        }

        try await ensureRequiredFiles()
        if !isVoiceEmbeddingPayloadCached(for: voice) {
            try? await TtsResourceDownloader.ensureVoiceEmbedding(voice: voice)
        }

        try await loadModel(variant: variantPreference)

        try await loadSimplePhonemeDictionary()

        try await validateTextHasDictionaryCoverage(text)

        let modelCache = try currentModelCache()
        let vocabulary = try await KokoroVocabulary.shared.getVocabulary()
        let capacities = try await capacities(for: variantPreference)
        let lexiconMetrics = await lexiconCache.metrics()

        let chunks = try await chunkText(
            text,
            vocabulary: vocabulary,
            longVariantTokenBudget: capacities.long,
            phoneticOverrides: phoneticOverrides
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

        let embeddingDimension = try await modelCache.referenceEmbeddingDimension()
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
        let samplesPerMillisecond = Double(TtsConstants.audioSampleRate) / 1_000.0
        let crossfadeN = max(0, Int(Double(crossfadeMs) * samplesPerMillisecond))
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

        var totalFrameCount = 0
        for output in sortedOutputs {
            let index = output.index
            let chunkSamples = output.samples
            chunkSampleBuffers[index] = chunkSamples
            totalPredictionTime += output.predictionTime

            Self.logger.info(
                "Chunk \(index + 1) model prediction latency: \(String(format: "%.3f", output.predictionTime))s")
            let chunkDurationSeconds = Double(chunkSamples.count) / Double(TtsConstants.audioSampleRate)
            let chunkFrameCount =
                TtsConstants.kokoroFrameSamples > 0
                ? chunkSamples.count / TtsConstants.kokoroFrameSamples
                : 0
            if TtsConstants.kokoroFrameSamples > 0 {
                totalFrameCount += chunkFrameCount
            }
            Self.logger.info(
                "Chunk \(index + 1) duration: \(String(format: "%.3f", chunkDurationSeconds))s (\(chunkFrameCount) frames)"
            )

            if index == 0 {
                allSamples.append(contentsOf: chunkSamples)
                continue
            }

            let prevPause = entries[index - 1].chunk.pauseAfterMs
            if prevPause > 0 {
                let silenceCount = Int(Double(prevPause) * samplesPerMillisecond)
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

        let audioData = try AudioWAV.data(
            from: allSamples,
            sampleRate: Double(TtsConstants.audioSampleRate)
        )

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

        if TtsConstants.kokoroFrameSamples > 0 {
            let frameLabel = totalFrameCount == 1 ? "frame" : "frames"
            Self.logger.notice(
                "Total model prediction time: \(String(format: "%.3f", totalPredictionTime))s for \(entries.count) chunk(s), \(totalFrameCount) total \(frameLabel)"
            )
        } else {
            Self.logger.notice(
                "Total model prediction time: \(String(format: "%.3f", totalPredictionTime))s for \(entries.count) chunk(s)"
            )
        }
        let variantsUsed = Set(entries.map { $0.template.variant })
        var footprints: [ModelNames.TTS.Variant: Int] = [:]
        for variant in variantsUsed {
            if let bundleURL = try? modelBundleURL(for: variant) {
                footprints[variant] = directorySize(at: bundleURL)
            }
        }

        let diagnostics = Diagnostics(
            variantFootprints: footprints,
            lexiconEntryCount: lexiconMetrics.entryCount,
            lexiconEstimatedBytes: lexiconMetrics.estimatedBytes,
            audioSampleBytes: allSamples.count * MemoryLayout<Float>.size,
            outputWavBytes: audioData.count
        )
        let baseResult = SynthesisResult(audio: audioData, chunks: chunkInfos, diagnostics: diagnostics)

        let factor = max(0.1, voiceSpeed)
        if abs(factor - 1.0) < 0.01 {
            return baseResult
        }

        let adjustedChunks = baseResult.chunks.map { chunk -> ChunkInfo in
            let stretched = adjustSamples(chunk.samples, factor: factor)
            return ChunkInfo(
                index: chunk.index,
                text: chunk.text,
                wordCount: chunk.wordCount,
                words: chunk.words,
                atoms: chunk.atoms,
                pauseAfterMs: chunk.pauseAfterMs,
                tokenCount: chunk.tokenCount,
                samples: stretched,
                variant: chunk.variant
            )
        }

        let combinedSamples = adjustedChunks.flatMap { $0.samples }
        let adjustedAudio = try AudioWAV.data(
            from: combinedSamples,
            sampleRate: Double(TtsConstants.audioSampleRate)
        )
        let updatedDiagnostics = baseResult.diagnostics?.updating(
            audioSampleBytes: combinedSamples.count * MemoryLayout<Float>.size,
            outputWavBytes: adjustedAudio.count
        )

        return SynthesisResult(
            audio: adjustedAudio,
            chunks: adjustedChunks,
            diagnostics: updatedDiagnostics
        )
    }

    private static func adjustSamples(_ samples: [Float], factor: Float) -> [Float] {
        let clamped = max(0.1, factor)
        if abs(clamped - 1.0) < 0.01 { return samples }

        if clamped < 1.0 {
            let repeatCount = max(1, Int(round(1.0 / clamped)))
            var stretched: [Float] = []
            stretched.reserveCapacity(samples.count * repeatCount)
            for sample in samples {
                for _ in 0..<repeatCount {
                    stretched.append(sample)
                }
            }
            return stretched
        }

        let step = Int(clamped)
        guard step > 1 else { return samples }
        var compressed: [Float] = []
        compressed.reserveCapacity(samples.count / step + 1)
        var index = 0
        while index < samples.count {
            compressed.append(samples[index])
            index += step
        }
        return compressed
    }

    static func removeDelimiterCharacters(from text: String) -> String {
        String(text.filter { !TtsConstants.delimiterCharacters.contains($0) })
    }

    static func collapseWhitespace(in text: String) -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let collapsed = TtsConstants.whitespacePattern.stringByReplacingMatches(
            in: text,
            options: [],
            range: range,
            withTemplate: " "
        )
        return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // convertSamplesToWAV moved to AudioWAV

    // convertToWAV removed (unused); use convertSamplesToWAV instead
}
