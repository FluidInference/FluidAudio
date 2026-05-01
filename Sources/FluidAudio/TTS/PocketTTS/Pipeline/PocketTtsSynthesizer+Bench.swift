@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Time `prefillKVCache` end-to-end for a single text + voice. Used by
    /// `PocketTtsManager.benchmarkCondStepPrefill` and the `pocket-tts-cond-bench`
    /// CLI subcommand to A/B compare the legacy chunk-1 and hybrid chunk-N
    /// dispatch paths.
    ///
    /// Must be called within a `withModelStore` context. The store's
    /// `condStepMode` decides whether the hybrid path runs.
    static func benchmarkCondStepPrefill(
        text: String,
        voice: String,
        warmup: Int,
        iters: Int
    ) async throws -> PocketTtsManager.CondStepPrefillBenchmarkResult {
        let store = try currentModelStore()
        let voiceData = try await store.voiceData(for: voice)
        return try await benchmarkCondStepPrefill(
            text: text, voiceData: voiceData, warmup: warmup, iters: iters
        )
    }

    /// Same as the voice-name overload but takes pre-resolved voice data.
    static func benchmarkCondStepPrefill(
        text: String,
        voiceData: PocketTtsVoiceData,
        warmup: Int,
        iters: Int
    ) async throws -> PocketTtsManager.CondStepPrefillBenchmarkResult {
        let store = try currentModelStore()

        let constants = try await store.constants()
        let condModel = try await store.condStep()
        let condLayerKeys = try await store.condStepLayerKeys()

        // Tokenize + embed once. The benchmark times only the prefill loop —
        // tokenization is cheap and identical across iterations and across
        // legacy vs chunked dispatch.
        let (normalizedChunk, _) = normalizeText(text)
        let tokenIds = constants.tokenizer.encode(normalizedChunk)
        let textEmbeddings = embedTokens(tokenIds, constants: constants)

        // Resolve chunk resources only when the store is in chunked mode.
        let chunkSize = await store.condStepChunkSize()
        let chunkModel: MLModel?
        let chunkLayerKeys: PocketTtsLayerKeys?
        if chunkSize != nil {
            chunkModel = try await store.condStepChunkModel()
            chunkLayerKeys = try await store.condStepChunkLayerKeys()
        } else {
            chunkModel = nil
            chunkLayerKeys = nil
        }

        // Voice token count for reporting. Snapshot voices skip cond_step
        // entirely on the voice side (`promptLength == 0` in that case).
        let voiceTokens: Int = {
            if voiceData.cacheSnapshot != nil { return 0 }
            return voiceData.promptLength
        }()

        // Warmup iterations (not recorded). Each runs a full prefill so the
        // CoreML graph, MIL caches, and ANE/GPU pipelines are warm before
        // we start sampling.
        for _ in 0..<warmup {
            _ = try await runPrefill(
                voiceData: voiceData,
                textEmbeddings: textEmbeddings,
                model: condModel,
                layerKeys: condLayerKeys,
                chunkModel: chunkModel,
                chunkLayerKeys: chunkLayerKeys,
                chunkSize: chunkSize
            )
        }

        // Timed iterations. Use `Date` (wall clock) instead of
        // `ContinuousClock` to keep iOS 16 / macOS 13 compatibility
        // — the rest of the pipeline targets the same baseline.
        var durations: [TimeInterval] = []
        durations.reserveCapacity(iters)
        for _ in 0..<iters {
            let start = Date()
            _ = try await runPrefill(
                voiceData: voiceData,
                textEmbeddings: textEmbeddings,
                model: condModel,
                layerKeys: condLayerKeys,
                chunkModel: chunkModel,
                chunkLayerKeys: chunkLayerKeys,
                chunkSize: chunkSize
            )
            durations.append(Date().timeIntervalSince(start))
        }

        return PocketTtsManager.CondStepPrefillBenchmarkResult(
            textTokens: textEmbeddings.count,
            voiceTokens: voiceTokens,
            durations: durations,
            usingChunked: chunkSize != nil,
            chunkSize: chunkSize
        )
    }

    /// Single prefill call routed through the optional chunk params. Used
    /// only by the benchmark — production paths call `prefillKVCache`
    /// directly with whichever args they need.
    private static func runPrefill(
        voiceData: PocketTtsVoiceData,
        textEmbeddings: [[Float]],
        model: MLModel,
        layerKeys: PocketTtsLayerKeys,
        chunkModel: MLModel?,
        chunkLayerKeys: PocketTtsLayerKeys?,
        chunkSize: Int?
    ) async throws -> KVCacheState {
        if let cm = chunkModel, let ck = chunkLayerKeys, let cs = chunkSize {
            return try await prefillKVCache(
                voiceData: voiceData,
                textEmbeddings: textEmbeddings,
                model: model,
                layerKeys: layerKeys,
                chunkModel: cm,
                chunkLayerKeys: ck,
                chunkSize: cs
            )
        }
        return try await prefillKVCache(
            voiceData: voiceData,
            textEmbeddings: textEmbeddings,
            model: model,
            layerKeys: layerKeys
        )
    }
}
