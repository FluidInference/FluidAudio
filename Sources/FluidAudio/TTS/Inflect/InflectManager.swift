@preconcurrency import CoreML
import Foundation

/// Public API for the Inflect v2 (Micro / Nano) CoreML TTS backend.
///
/// Ultra-tiny VITS-family English TTS (9.4M / 4.0M params, 24 kHz). The
/// pipeline is a fixed-shape encoder + duration predictor, host-side duration
/// expansion and prior sampling, then a bucketed HiFiGAN synthesizer. See
/// `FluidInference/inflect-v2-coreml`.
///
/// > Phonemizer parity gap — same as StyleTTS2.
/// > Inflect trained on `phonemizer`/espeak-ng en-us IPA. FluidAudio can't
/// > ship the GPL espeak C library, so the text path reuses the shared
/// > Misaki-lexicon + BART G2P frontend (`StyleTTS2Phonemizer`), which
/// > approximates espeak IPA. Output is intelligible but stress markers can
/// > differ. Callers with a real espeak phonemizer should feed IPA directly
/// > via ``synthesize(ipa:)``.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor InflectManager {

    private let logger = AppLogger(category: "InflectManager")

    private let variant: InflectVariant
    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let noiseScale: Float
    private let speed: Float

    private var store: InflectModelStore?
    private var synthesizer: InflectSynthesizer?
    private var phonemizer: StyleTTS2Phonemizer?

    public nonisolated var sampleRate: Int { InflectConstants.sampleRate }

    public init(
        variant: InflectVariant = .micro,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndGPU,
        noiseScale: Float = InflectConstants.defaultNoiseScale,
        speed: Float = InflectConstants.defaultSpeed
    ) {
        self.variant = variant
        self.directory = directory
        self.computeUnits = computeUnits
        self.noiseScale = noiseScale
        self.speed = speed
    }

    public var isAvailable: Bool { synthesizer != nil }

    /// Convenience factory: download assets and return a ready manager.
    public static func downloadAndCreate(
        variant: InflectVariant = .micro,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> InflectManager {
        let manager = InflectManager(
            variant: variant, directory: directory, computeUnits: computeUnits)
        try await manager.initialize()
        return manager
    }

    /// Download + load the variant's CoreML bundles and the shared English
    /// frontend (Misaki lexicon cache + BART G2P), then build the phonemizer.
    public func initialize() async throws {
        if synthesizer != nil { return }

        let store = InflectModelStore(
            variant: variant, directory: directory, computeUnits: computeUnits)
        try await store.loadIfNeeded()
        self.store = store
        self.synthesizer = InflectSynthesizer(store: store, variant: variant)

        // Reuse the shared English frontend (same espeak-approximation used by
        // StyleTTS2). The lexicon cache + BART G2P assets live in the kokoro
        // cache dir and are shared across backends.
        do {
            try await StyleTTS2ResourceDownloader.ensureG2PAssets(directory: directory)
            let kokoroDir = try await StyleTTS2ResourceDownloader.ensureLexiconCache()
            try await G2PModel.shared.ensureModelsAvailable()

            let allowedTokens = Set(InflectSymbols.symbols.map { String($0) })
            let lexiconCache = LexiconAssetCache()
            try await lexiconCache.ensureLoaded(
                kokoroDirectory: kokoroDir, allowedTokens: allowedTokens)
            let lexicons = await lexiconCache.lexicons()
            self.phonemizer = StyleTTS2Phonemizer(
                wordToPhonemes: lexicons.word,
                caseSensitiveWordToPhonemes: lexicons.caseSensitive)
            logger.info(
                "Inflect \(variant.rawValue) ready — lexicon \(lexicons.word.count) entries, "
                    + "compute \(computeUnits.inflectDescription)")
        } catch {
            logger.warning(
                "English frontend load failed (\(error)); text synthesis unavailable, "
                    + "use synthesize(ipa:). Detail: \(error.localizedDescription)")
            self.phonemizer = StyleTTS2Phonemizer()
        }
    }

    // MARK: - Text path

    /// Phonemize `text` (Misaki + BART G2P), split into encoder-sized chunks,
    /// and synthesize. Returns 24 kHz mono Float32 PCM.
    public func synthesize(
        text: String,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        guard let phonemizer else { throw InflectError.notInitialized }
        let ipa: String
        do {
            ipa = try await phonemizer.phonemize(text)
        } catch {
            throw InflectError.phonemizationFailed("\(error)")
        }
        return try await synthesize(ipa: ipa, noiseSeed: noiseSeed)
    }

    // MARK: - IPA path (espeak-parity escape hatch)

    /// Synthesize directly from an IPA phoneme string (keithito/espeak symbol
    /// set). Bypasses the lexicon + G2P — feed the espeak IPA the model was
    /// trained on for best quality. Chunks longer than the encoder axis are
    /// split at punctuation/whitespace and concatenated with short pauses.
    public func synthesize(
        ipa: String,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        guard let synthesizer else { throw InflectError.notInitialized }
        let chunks = PhonemeChunker.chunk(ipa, maxLength: InflectConstants.maxPhonemeChunkChars)
        guard !chunks.isEmpty else {
            throw InflectError.inputProcessingFailed("no speakable phonemes in input")
        }

        var samples: [Float] = []
        for (index, chunk) in chunks.enumerated() {
            let tokens = InflectSymbols.encode(chunk)
            guard tokens.count > 1 else { continue }
            if index > 0 {
                samples.append(
                    contentsOf: [Float](
                        repeating: 0, count: pauseSamples(afterChunk: chunks[index - 1])))
            }
            let chunkSamples = try await synthesizer.synthesize(
                tokens: tokens,
                noiseScale: noiseScale,
                speed: speed,
                noiseSeed: noiseSeed &+ UInt64(index))
            samples.append(contentsOf: chunkSamples)
        }
        guard !samples.isEmpty else {
            throw InflectError.inputProcessingFailed("synthesis produced no audio")
        }
        return samples
    }

    public func cleanup() async {
        await store?.unload()
        store = nil
        synthesizer = nil
        phonemizer = nil
    }

    // MARK: - Helpers

    /// Inter-chunk silence, keyed on the preceding chunk's final punctuation
    /// (mirrors upstream `boundary_pause_seconds`).
    private func pauseSamples(afterChunk chunk: String) -> Int {
        let ending = chunk.trimmingCharacters(in: .whitespaces).last
        let seconds: Double
        switch ending {
        case "?": seconds = 0.28
        case "!": seconds = 0.24
        case ".": seconds = 0.22
        case ";": seconds = 0.16
        case ":": seconds = 0.13
        case ",": seconds = 0.09
        default: seconds = 0.08
        }
        return Int(Double(InflectConstants.sampleRate) * seconds)
    }
}

extension MLComputeUnits {
    fileprivate var inflectDescription: String {
        switch self {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .all: return "all"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown"
        }
    }
}
