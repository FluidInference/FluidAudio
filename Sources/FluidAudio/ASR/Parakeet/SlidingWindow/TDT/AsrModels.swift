@preconcurrency import CoreML
import Foundation

/// ASR model version enum
public enum AsrModelVersion: Sendable {
    case v2
    case v3
    /// Parakeet Redux (moondream): ternary re-training of v3. Same vocabulary,
    /// blank id, window and JointDecisionv3 contract; only the weights differ,
    /// so every v3 decode path applies (see `isV3Family`).
    case redux
    /// Parakeet Ultra (moondream): full-precision post-training of v3. Same
    /// architecture and contract as v3; the encoder ships int8-linear.
    case ultra
    /// Phonon-2 (Fermion Research): quantization-aware re-training of v3,
    /// English only. Same architecture and contract as v3; the encoder ships
    /// as exact five-value palettes (iOS 18+).
    case phonon2
    /// Phonon-2.5 multilingual: v3-family contract with a dense LUT3 encoder.
    /// The encoder defaults to CPU/GPU; ANE activation parity is not validated.
    case phonon25
    /// Underdog Bark 3.0 (Conway Research): post-training of v3, English, verbatim.
    /// Same architecture and contract as v3; the encoder ships as exact five-value
    /// palettes (iOS 18+). The model emits `<unk>` for quotation marks; transcripts drop it.
    case bark3
    /// Underdog Bark 3.0 Clean: drops fillers, false starts and repeated words.
    /// Same contract and encoder encoding as `.bark3`.
    case bark3Clean
    /// 110M parameter hybrid TDT-CTC model with fused preprocessor+encoder
    case tdtCtc110m
    /// 600M parameter TDT model for Japanese (ja) - hybrid CTC preprocessor/encoder + TDT decoder/joint v2
    case tdtJa

    var repo: Repo {
        switch self {
        case .v2: return .parakeetV2
        case .v3: return .parakeetV3
        case .redux: return .parakeetRedux
        case .ultra: return .parakeetUltra
        case .phonon2: return .phonon2
        case .phonon25: return .phonon25
        case .bark3: return .bark3
        case .bark3Clean: return .bark3Clean
        case .tdtCtc110m: return .parakeetTdtCtc110m
        case .tdtJa: return .parakeetJa
        }
    }

    /// Whether this model version uses a fused preprocessor+encoder (no separate Encoder model)
    public var hasFusedEncoder: Bool {
        switch self {
        case .tdtCtc110m: return true
        default: return false
        }
    }

    /// Encoder hidden dimension for this model version
    public var encoderHiddenSize: Int {
        switch self {
        case .tdtCtc110m: return 512
        case .tdtJa: return 1024
        default: return 1024
        }
    }

    /// Blank token ID for this model version
    public var blankId: Int {
        switch self {
        case .v2, .tdtCtc110m: return 1024
        case .v3, .redux, .ultra, .phonon2, .phonon25, .bark3, .bark3Clean: return 8192
        case .tdtJa: return 3072
        }
    }

    /// Number of LSTM layers in the decoder prediction network
    public var decoderLayers: Int {
        switch self {
        case .tdtCtc110m: return 1
        default: return 2
        }
    }

    /// v3 and its weight-compatible derivatives (Redux, Ultra, Phonon-2/2.5, Bark). Gates the v3-only
    /// decode behaviours: `TdtDecoderV3`, silence-aligned chunking, blank
    /// recovery and the JointDecisionv3 top-K contract.
    public var isV3Family: Bool {
        switch self {
        case .v3, .redux, .ultra, .phonon2, .phonon25, .bark3, .bark3Clean: return true
        default: return false
        }
    }
}

public struct AsrModels: Sendable {

    /// Required model names for ASR
    public static let requiredModelNames = ModelNames.ASR.requiredModels

    /// Separate encoder model (nil for fused models like tdtCtc110m where preprocessor includes encoder)
    public let encoder: MLModel?
    public let preprocessor: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    /// Optional CTC decoder head for custom vocabulary (encoder features → CTC logits)
    public let ctcHead: MLModel?
    /// Shorter-window functions of a multifunction encoder, ascending by
    /// length; empty for a single-shape encoder. A window runs on the
    /// smallest function that holds it, so it pays Neural Engine time for
    /// the audio it carries rather than for the full 15 s shape.
    public let encoderLadder: [EncoderLadderFunction]
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let version: AsrModelVersion

    private static let logger = AppLogger(category: "AsrModels")

    public init(
        encoder: MLModel?,
        preprocessor: MLModel,
        decoder: MLModel,
        joint: MLModel,
        ctcHead: MLModel? = nil,
        encoderLadder: [EncoderLadderFunction] = [],
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        version: AsrModelVersion
    ) {
        self.encoder = encoder
        self.encoderLadder = encoderLadder
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.joint = joint
        self.ctcHead = ctcHead
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.version = version
    }

    /// Whether this model uses a separate preprocessor and encoder (true for 0.6B, false for 110m fused)
    public var usesSplitFrontend: Bool {
        !version.hasFusedEncoder
    }
}

/// One window length of a multifunction encoder (`enc_<milliseconds>ms`).
public struct EncoderLadderFunction: Sendable {
    public let seconds: Double
    /// Mel frames the function consumes (1 + 100 × seconds).
    public let melFrames: Int
    public let model: MLModel

    /// Multifunction encoder function names are `enc_<milliseconds>ms`
    /// (integer: Core ML's Neural Engine compiler crashes on a '.' in a
    /// function name).
    static func seconds(fromFunctionName name: String) -> Double? {
        guard name.hasPrefix("enc_"), name.hasSuffix("ms"), let ms = Int(name.dropFirst(4).dropLast(2)) else {
            return nil
        }
        return Double(ms) / 1000
    }
}

extension AsrModels {
    /// Load every `enc_<N>ms` function of a multifunction encoder at `path`.
    /// Returns an empty array for a single-function encoder, or when the
    /// platform predates multifunction models (macOS 15 / iOS 18).
    static func loadEncoderLadder(
        at path: URL, computeUnits: MLComputeUnits, defaultEncoder: MLModel
    ) async -> [EncoderLadderFunction] {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else { return [] }
        let names: [String]
        do {
            names = try await MLModelAsset(url: path).functionNames
        } catch {
            return []
        }
        // The default function is already loaded as `defaultEncoder`; reuse it
        // rather than hold a second instance of the same program.
        let defaultMelFrames =
            defaultEncoder.modelDescription.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape.last?.intValue
        var ladder: [EncoderLadderFunction] = []
        for name in names {
            guard let seconds = EncoderLadderFunction.seconds(fromFunctionName: name) else { continue }
            if let defaultMelFrames, defaultMelFrames == 1 + Int(seconds * 100) {
                ladder.append(
                    EncoderLadderFunction(seconds: seconds, melFrames: defaultMelFrames, model: defaultEncoder))
                continue
            }
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            config.allowLowPrecisionAccumulationOnGPU = true
            config.functionName = name
            let start = Date()
            do {
                let model = try MLModel(contentsOf: path, configuration: config)
                let melFrames =
                    model.modelDescription.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape.last?.intValue
                    ?? (1 + Int(seconds * 100))
                ladder.append(EncoderLadderFunction(seconds: seconds, melFrames: melFrames, model: model))
                logger.info(
                    "Loaded encoder function \(name) in \(String(format: "%.0f", Date().timeIntervalSince(start) * 1000)) ms"
                )
            } catch {
                logger.warning("Encoder function \(name) failed to load: \(error.localizedDescription)")
            }
        }
        return ladder.sorted { $0.seconds < $1.seconds }
    }
}

extension AsrModels {

    private struct ModelSpec {
        let fileName: String
        let computeUnits: MLComputeUnits
    }

    /// Select encoder placement separately from decoder/joint configuration.
    /// Phonon-2.5's published dense encoder has passed the GPU gate, while the
    /// sparse ANE conversion has not. Keep ANE opt-in unavailable for this model:
    /// without an explicit request its encoder runs on CPU/GPU, or on the CPU
    /// when the configuration is CPU-only (background-safe / CI configurations).
    static func resolvedEncoderComputeUnits(
        for version: AsrModelVersion,
        configuration: MLComputeUnits,
        requested: MLComputeUnits? = nil
    ) throws -> MLComputeUnits {
        guard version == .phonon25 else { return requested ?? configuration }
        let units = requested ?? (configuration == .cpuOnly ? .cpuOnly : .cpuAndGPU)
        guard units == .cpuAndGPU || units == .cpuOnly else {
            throw AsrModelsError.loadingFailed(
                "Phonon-2.5 supports the GPU encoder or CPU-only diagnostics; "
                    + "ANE encoder parity is not validated. Use encoderComputeUnits: .cpuAndGPU.")
        }
        return units
    }

    private static func createModelSpecs(
        using config: MLModelConfiguration,
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision,
        encoderComputeUnits: MLComputeUnits? = nil
    ) throws -> [ModelSpec] {
        if version.hasFusedEncoder {
            // Fused preprocessor+encoder runs on ANE (it contains the conformer encoder)
            return [
                ModelSpec(fileName: Names.preprocessorFile, computeUnits: config.computeUnits)
            ]
        }
        let fileNames = getModelFileNames(version: version, encoderPrecision: encoderPrecision)
        return [
            // Preprocessor ops map to CPU-only across all platforms. XCode profiling shows
            // that 100% of the the operations map to the CPU anyways.
            ModelSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),

            // The conformer encoder defaults to the configuration's compute units
            // (`.cpuAndNeuralEngine`). It can be overridden to `.cpuAndGPU`, which
            // is ~24% faster on Apple Silicon GPUs (17.8ms vs 23.5ms per 15s window,
            // M-series) for ~+8% end-to-end RTFx, WER-neutral. ANE remains the default
            // because it is far more power-efficient on iOS; opt into GPU for
            // throughput-oriented / plugged-in workloads. See docs/perf/parakeet-v3-encoder.md.
            // Phonon-2.5 is the exception: CPU/GPU by default, ANE placements rejected
            // (see resolvedEncoderComputeUnits).
            ModelSpec(
                fileName: fileNames.encoder,
                computeUnits: try resolvedEncoderComputeUnits(
                    for: version, configuration: config.computeUnits, requested: encoderComputeUnits)
            ),
        ]
    }

    /// Redux's 2-bit, Phonon-2's / Bark's five-value and Phonon-2.5's LUT3 encoders use iOS 18
    /// Core ML ops, so they have no iOS 17 build. Fails fast there and points at `.ultra`.
    static func checkPlatformSupport(for version: AsrModelVersion) throws {
        let name: String
        switch version {
        case .redux: name = "Parakeet Redux"
        case .phonon2: name = "Phonon-2"
        case .phonon25: name = "Phonon-2.5"
        case .bark3: name = "Bark 3.0"
        case .bark3Clean: name = "Bark 3.0 Clean"
        default: return
        }
        if #available(macOS 15, iOS 18, tvOS 18, *) { return }
        throw AsrModelsError.loadingFailed(
            "\(name) requires iOS 18 / macOS 15 (its compressed encoder uses iOS 18 Core ML ops). "
                + "Use AsrModelVersion.ultra on iOS 17 / macOS 14.")
    }

    /// Helper to get the repo path from a models directory
    private static func repoPath(from modelsDirectory: URL, version: AsrModelVersion = .v3) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(version.repo.folderName)
    }

    static func inferredVersion(from directory: URL) -> AsrModelVersion? {
        let directoryPath = directory.path.lowercased()
        let knownVersions: [AsrModelVersion] = [
            .tdtCtc110m, .v2, .v3, .redux, .ultra, .phonon25, .phonon2, .bark3Clean, .bark3, .tdtJa,
        ]

        // The model folder itself wins over a parent folder that merely contains another name
        // (".../phonon-2.5-eval/phonon-2" is Phonon-2).
        let folder = directory.lastPathComponent.lowercased()
        if let exact = knownVersions.first(where: { $0.repo.folderName.lowercased() == folder }) {
            return exact
        }
        for version in knownVersions {
            if directoryPath.contains(version.repo.folderName.lowercased()) {
                return version
            }
        }

        return nil
    }

    // Use centralized model names
    private typealias Names = ModelNames.ASR

    private static func getModelFileNames(
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision
    ) -> (encoder: String, decoder: String, joint: String, vocabulary: String) {
        switch version {
        case .tdtJa:
            return (
                encoder: ModelNames.TDTJa.encoderFile,
                decoder: ModelNames.TDTJa.decoderFile,
                joint: ModelNames.TDTJa.jointFile,
                vocabulary: ModelNames.TDTJa.vocabularyFile
            )
        case .v3:
            // v3 uses JointDecisionv3 exclusively. Top-K outputs (`top_k_ids`,
            // `top_k_logits`) are always computed by the joint graph, but
            // Swift-side extraction is gated by `needsTopK` in
            // `TdtModelInference.runJointPrepared` so callers that don't pass
            // `language:` pay no extra allocations per step.
            return (
                encoder: encoderPrecision.encoderFileName,
                decoder: Names.decoderFile,
                joint: Names.jointV3File,
                vocabulary: Names.vocabularyFile
            )
        case .redux, .ultra, .phonon2, .phonon25, .bark3, .bark3Clean:
            // Same contract as v3 with a single encoder build.
            return (
                encoder: Names.encoderFile,
                decoder: Names.decoderFile,
                joint: Names.jointV3File,
                vocabulary: Names.vocabularyFile
            )
        default:
            return (
                encoder: Names.encoderFile,
                decoder: Names.decoderFile,
                joint: Names.jointFile,
                vocabulary: Names.vocabularyFile
            )
        }
    }

    private static func getRequiredModels(
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision
    ) -> Set<String> {
        switch version {
        case .tdtJa:
            return ModelNames.TDTJa.requiredModels
        case .v3:
            return Names.requiredModelsV3(precision: encoderPrecision)
        case .redux, .ultra, .phonon2, .phonon25, .bark3, .bark3Clean:
            return Names.requiredModelsV3()
        default:
            return version.hasFusedEncoder ? Names.requiredModelsFused : Names.requiredModels
        }
    }

    /// Load compiled models from this exact directory without downloading or resolving a repository.
    ///
    /// Use this for compatible community weights such as Orukeet. The directory must contain
    /// the version's `.mlmodelc` components and vocabulary; missing files fail locally.
    /// Compile portable `.mlpackage` files on the destination device with `MLModel.compileModel(at:)`.
    public static func loadLocal(
        from directory: URL,
        version: AsrModelVersion = .v3,
        configuration: MLModelConfiguration? = nil,
        encoderPrecision: ParakeetEncoderPrecision = .int8,
        encoderComputeUnits: MLComputeUnits? = nil
    ) throws -> AsrModels {
        try checkPlatformSupport(for: version)
        let config = configuration ?? defaultConfiguration()
        _ = try resolvedEncoderComputeUnits(
            for: version, configuration: config.computeUnits, requested: encoderComputeUnits)
        let names = getModelFileNames(version: version, encoderPrecision: encoderPrecision)
        func component(_ name: String, units: MLComputeUnits) throws -> MLModel {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw AsrModelsError.modelNotFound(name, url)
            }
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = units
            modelConfig.allowLowPrecisionAccumulationOnGPU = config.allowLowPrecisionAccumulationOnGPU
            return try MLModel(contentsOf: url, configuration: modelConfig)
        }
        let vocabularyURL = directory.appendingPathComponent(names.vocabulary)
        let vocabulary = try parseVocabulary(at: vocabularyURL)
        guard (0..<version.blankId).allSatisfy({ vocabulary[$0] != nil }) else {
            throw AsrModelsError.loadingFailed("Local vocabulary must contain every token before the blank ID")
        }
        let encoder =
            try version.hasFusedEncoder
            ? nil
            : component(
                names.encoder,
                units: resolvedEncoderComputeUnits(
                    for: version, configuration: config.computeUnits, requested: encoderComputeUnits))
        let ctcURL = directory.appendingPathComponent(Names.ctcHeadFile)
        let ctcHead =
            try FileManager.default.fileExists(atPath: ctcURL.path)
            ? component(Names.ctcHeadFile, units: config.computeUnits) : nil
        return try AsrModels(
            encoder: encoder,
            preprocessor: component(
                Names.preprocessorFile, units: version.hasFusedEncoder ? config.computeUnits : .cpuOnly),
            decoder: component(names.decoder, units: config.computeUnits),
            joint: component(names.joint, units: config.computeUnits),
            ctcHead: ctcHead,
            configuration: config,
            vocabulary: vocabulary,
            version: version
        )
    }

    /// Load ASR models from a directory
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model files
    ///   - configuration: Optional MLModel configuration. When provided, the configuration's
    ///                   computeUnits will be respected (Phonon-2.5's encoder: see below). When
    ///                   nil, platform-optimized defaults are used (per-model optimization).
    ///   - version: ASR model version to load (defaults to v3)
    ///   - encoderComputeUnits: Optional override for the conformer encoder's compute units.
    ///                   When `nil` (default) the encoder uses the configuration's compute
    ///                   units (`.cpuAndNeuralEngine`). Pass `.cpuAndGPU` to run the encoder
    ///                   on the GPU (~+8% RTFx, WER-neutral on Apple Silicon) for
    ///                   throughput-oriented workloads; ANE stays the default for power
    ///                   efficiency on iOS. See docs/perf/parakeet-v3-encoder.md.
    ///                   Phonon-2.5 only accepts `.cpuAndGPU` (its default) or `.cpuOnly`
    ///                   (also chosen when the configuration is CPU-only); ANE placements throw.
    ///
    /// - Returns: Loaded ASR models
    ///
    /// - Note: The default configuration pins the preprocessor to CPU and every other
    ///         Parakeet component to `.cpuAndNeuralEngine` to avoid GPU dispatch, which keeps
    ///         background execution permitted on iOS. Phonon-2.5 is an exception:
    ///         its encoder defaults to CPU/GPU and is intended for foreground use.
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3,
        encoderPrecision: ParakeetEncoderPrecision = .int8,
        encoderComputeUnits: MLComputeUnits? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> AsrModels {
        try checkPlatformSupport(for: version)
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()

        let parentDirectory = directory.deletingLastPathComponent()
        let downloadVariant: String? = (version == .v3) ? encoderPrecision.rawValue : nil
        let specs = try createModelSpecs(
            using: config, version: version, encoderPrecision: encoderPrecision,
            encoderComputeUnits: encoderComputeUnits)

        var loadedModels: [String: MLModel] = [:]

        for spec in specs {
            let models = try await ModelHub.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDirectory,
                computeUnits: spec.computeUnits,
                variant: downloadVariant,
                progressHandler: progressHandler
            )

            if let model = models[spec.fileName] {
                loadedModels[spec.fileName] = model
                let unitsText = Self.describeComputeUnits(spec.computeUnits)
                logger.info("Loaded \(spec.fileName) with compute units: \(unitsText)")
            }
        }

        guard let preprocessorModel = loadedModels[Names.preprocessorFile] else {
            throw AsrModelsError.loadingFailed("Failed to load preprocessor model")
        }

        let fileNames = getModelFileNames(version: version, encoderPrecision: encoderPrecision)
        let encoderModel = loadedModels[fileNames.encoder]  // nil for fused models

        if !version.hasFusedEncoder && encoderModel == nil {
            throw AsrModelsError.loadingFailed("Failed to load encoder model (required for split frontend)")
        }
        var encoderLadder: [EncoderLadderFunction] = []
        if let encoderModel, let encoderSpec = specs.first(where: { $0.fileName == fileNames.encoder }) {
            encoderLadder = await loadEncoderLadder(
                at: directory.appendingPathComponent(fileNames.encoder),
                computeUnits: encoderSpec.computeUnits,
                defaultEncoder: encoderModel
            )
            if !encoderLadder.isEmpty {
                logger.info("Encoder ladder: \(encoderLadder.map { "\($0.seconds) s" }.joined(separator: ", "))")
            }
        }

        // Load decoder first
        let decoderModels = try await ModelHub.loadModels(
            version.repo,
            modelNames: [fileNames.decoder],
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            variant: downloadVariant,
            progressHandler: progressHandler
        )

        guard let decoderModel = decoderModels[fileNames.decoder] else {
            throw AsrModelsError.loadingFailed("Failed to load decoder model")
        }

        // Load the joint model. For v3 this is JointDecisionv3.mlmodelc (with
        // top-K outputs); for v2 / 110m / tdtJa it is the version-specific
        // JointDecision.mlmodelc.
        let jointModels = try await ModelHub.loadModels(
            version.repo,
            modelNames: [fileNames.joint],
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            variant: downloadVariant,
            progressHandler: progressHandler
        )

        guard let jointModel = jointModels[fileNames.joint] else {
            let hint =
                version.isV3Family
                ? " (required for v3; delete the models directory to force a fresh download)"
                : ""
            throw AsrModelsError.loadingFailed("Failed to load joint model \(fileNames.joint)\(hint)")
        }
        logger.info("Loaded \(fileNames.joint)")

        // [Beta] Optionally load CTC head model for custom vocabulary.
        // Supports two paths:
        //   v1: CtcHead.mlmodelc placed manually in the TDT model directory
        //   v2: Auto-download from FluidInference/parakeet-ctc-110m-coreml HF repo
        var ctcHeadModel: MLModel?
        if version == .tdtCtc110m {
            // v1: Check local TDT model directory first
            let repoDir = repoPath(from: directory, version: version)
            let ctcHeadPath = repoDir.appendingPathComponent(Names.ctcHeadFile)
            if FileManager.default.fileExists(atPath: ctcHeadPath.path) {
                let ctcConfig = MLModelConfiguration()
                ctcConfig.computeUnits = config.computeUnits
                ctcHeadModel = try? MLModel(contentsOf: ctcHeadPath, configuration: ctcConfig)
                if ctcHeadModel != nil {
                    logger.info("[Beta] Loaded CTC head model from local directory")
                } else {
                    logger.warning("CTC head model found but failed to load: \(ctcHeadPath.path)")
                }
            }

            // v2: Fall back to downloading from parakeet-ctc-110m HF repo
            if ctcHeadModel == nil {
                do {
                    let ctcModels = try await ModelHub.loadModels(
                        .parakeetCtc110m,
                        modelNames: [Names.ctcHeadFile],
                        directory: parentDirectory,
                        computeUnits: config.computeUnits,
                        progressHandler: progressHandler
                    )
                    ctcHeadModel = ctcModels[Names.ctcHeadFile]
                    if ctcHeadModel != nil {
                        logger.info("[Beta] Loaded CTC head model from HF repo")
                    }
                } catch {
                    logger.warning("CTC head model not available: \(error.localizedDescription)")
                }
            }
        }

        let asrModels = AsrModels(
            encoder: encoderModel,
            preprocessor: preprocessorModel,
            decoder: decoderModel,
            joint: jointModel,
            ctcHead: ctcHeadModel,
            encoderLadder: encoderLadder,
            configuration: config,
            vocabulary: try loadVocabulary(from: directory, version: version),
            version: version
        )
        logger.info("Successfully loaded all ASR models with optimized compute units")
        return asrModels
    }

    private static func loadVocabulary(from directory: URL, version: AsrModelVersion) throws -> [Int: String] {
        let vocabularyFileName = getModelFileNames(version: version, encoderPrecision: .int8).vocabulary
        let vocabPath = repoPath(from: directory, version: version).appendingPathComponent(vocabularyFileName)

        return try parseVocabulary(at: vocabPath)
    }

    private static func parseVocabulary(at vocabPath: URL) throws -> [Int: String] {
        let vocabularyFileName = vocabPath.lastPathComponent
        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "Vocabulary file not found at \(vocabPath.path). Please ensure the vocab file is downloaded with the models."
            )
            throw AsrModelsError.modelNotFound(vocabularyFileName, vocabPath)
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let json = try JSONSerialization.jsonObject(with: data)

            var vocabulary: [Int: String] = [:]

            if let jsonArray = json as? [String] {
                // Array format (110m hybrid): index = token ID
                for (index, token) in jsonArray.enumerated() {
                    vocabulary[index] = token
                }
            } else if let jsonDict = json as? [String: String] {
                // Dictionary format (0.6B v2/v3): key = token ID string
                for (key, value) in jsonDict {
                    if let tokenId = Int(key) {
                        vocabulary[tokenId] = value
                    }
                }
            } else {
                throw AsrModelsError.loadingFailed("Vocabulary file has unexpected format")
            }

            logger.info("Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch let error as AsrModelsError {
            throw error
        } catch {
            logger.error(
                "Failed to load or parse vocabulary file at \(vocabPath.path): \(error.localizedDescription)"
            )
            throw AsrModelsError.loadingFailed("Vocabulary parsing failed")
        }
    }

    public static func loadFromCache(
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3,
        encoderComputeUnits: MLComputeUnits? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory(for: version)
        return try await load(
            from: cacheDir, configuration: configuration, version: version,
            encoderComputeUnits: encoderComputeUnits,
            progressHandler: progressHandler)
    }

    /// Load models with automatic recovery on compilation failures
    public static func loadWithAutoRecovery(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        encoderComputeUnits: MLComputeUnits? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> AsrModels {
        let targetDir = directory ?? defaultCacheDirectory()
        return try await load(
            from: targetDir, configuration: configuration,
            encoderComputeUnits: encoderComputeUnits, progressHandler: progressHandler)
    }

    private static func describeComputeUnits(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly:
            return "cpuOnly"
        case .cpuAndGPU:
            return "cpuAndGPU"
        case .cpuAndNeuralEngine:
            return "cpuAndNeuralEngine"
        case .all:
            return "all"
        @unknown default:
            return "unknown(\(units.rawValue))"
        }
    }

    public static func defaultConfiguration() -> MLModelConfiguration {
        // Prefer Neural Engine across platforms for ASR inference to avoid GPU dispatch.
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine)
    }

    /// Create optimized configuration for model inference
    public static func optimizedConfiguration(
        enableFP16: Bool = true
    ) -> MLModelConfiguration {
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        let config = MLModelConfigurationUtils.defaultConfiguration(
            computeUnits: isCI ? .cpuOnly : .cpuAndNeuralEngine
        )
        config.allowLowPrecisionAccumulationOnGPU = enableFP16
        return config
    }

    /// Create optimized prediction options for inference
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()

        // Enable batching for better GPU utilization
        options.outputBackings = [:]  // Reuse output buffers

        return options
    }
}

extension AsrModels {

    @discardableResult
    /// Download (and compile-warm) the models for `version`.
    ///
    /// - Parameter encoderComputeUnits: compute units the encoder is warmed with; `nil` uses the
    ///   version's default placement (see `load(from:configuration:version:encoderPrecision:encoderComputeUnits:)`).
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        version: AsrModelVersion = .v3,
        encoderPrecision: ParakeetEncoderPrecision = .int8,
        encoderComputeUnits: MLComputeUnits? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        try checkPlatformSupport(for: version)
        let targetDir = directory ?? defaultCacheDirectory(for: version)
        logger.info("Downloading ASR models to: \(targetDir.path)")
        let parentDir = targetDir.deletingLastPathComponent()
        let downloadVariant: String? = (version == .v3) ? encoderPrecision.rawValue : nil
        // A pinned repo (Phonon-2.5) is only "present" with its revision marker; ModelHub re-checks it on load.
        let revision = ModelRegistry.mapRevision(version.repo.remotePath, default: version.repo.revision)
        let revisionMatches = ModelCache.matchesRevision(
            at: repoPath(from: targetDir, version: version), revision: revision)

        if !force && revisionMatches
            && modelsExist(at: targetDir, version: version, encoderPrecision: encoderPrecision)
        {
            logger.info("ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        struct DownloadSpec {
            let fileName: String
            let computeUnits: MLComputeUnits
        }

        let defaultUnits = defaultConfiguration().computeUnits
        let fileNames = getModelFileNames(version: version, encoderPrecision: encoderPrecision)

        let specs: [DownloadSpec]
        if version.hasFusedEncoder {
            specs = [
                // Fused preprocessor+encoder runs on ANE
                DownloadSpec(fileName: Names.preprocessorFile, computeUnits: defaultUnits),
                DownloadSpec(fileName: fileNames.decoder, computeUnits: defaultUnits),
                DownloadSpec(fileName: fileNames.joint, computeUnits: defaultUnits),
            ]
        } else {
            specs = [
                // Preprocessor ops map to CPU-only across all platforms.
                DownloadSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),
                DownloadSpec(
                    fileName: fileNames.encoder,
                    computeUnits: try resolvedEncoderComputeUnits(
                        for: version, configuration: defaultUnits, requested: encoderComputeUnits)),
                DownloadSpec(fileName: fileNames.decoder, computeUnits: defaultUnits),
                DownloadSpec(fileName: fileNames.joint, computeUnits: defaultUnits),
            ]
        }

        for spec in specs {
            _ = try await ModelHub.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDir,
                computeUnits: spec.computeUnits,
                variant: downloadVariant,
                progressHandler: progressHandler
            )
        }

        // The compiled-model download above doesn't cover the vocab JSON that
        // `load()` needs; guarantee it here (#748).
        try await ensureVocabularyDownloaded(
            version: version, encoderPrecision: encoderPrecision, targetDir: targetDir)

        logger.info("Successfully downloaded ASR models")
        return targetDir
    }

    /// On-disk location of the version's vocabulary JSON in a cache rooted at `targetDir`.
    static func vocabularyFileURL(
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision,
        targetDir: URL
    ) -> URL {
        let vocabularyFileName = getModelFileNames(
            version: version, encoderPrecision: encoderPrecision
        ).vocabulary
        return repoPath(from: targetDir, version: version)
            .appendingPathComponent(vocabularyFileName)
    }

    /// Fetch the vocabulary JSON from HuggingFace if missing; no-op when already on disk (#748).
    static func ensureVocabularyDownloaded(
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision,
        targetDir: URL
    ) async throws {
        let vocabURL = vocabularyFileURL(
            version: version, encoderPrecision: encoderPrecision, targetDir: targetDir)
        let vocabularyFileName = vocabURL.lastPathComponent

        if FileManager.default.fileExists(atPath: vocabURL.path) {
            return
        }

        logger.info("Vocabulary \(vocabularyFileName) missing after model download; fetching directly")
        let revision = ModelRegistry.mapRevision(version.repo.remotePath, default: version.repo.revision)
        let remoteURL = try ModelRegistry.resolveModel(version.repo.remotePath, vocabularyFileName, revision: revision)
        let data = try await ModelHub.fetchFile(
            from: remoteURL, description: vocabularyFileName)
        try FileManager.default.createDirectory(
            at: vocabURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: vocabURL, options: [.atomic])
        logger.info("Downloaded vocabulary \(vocabularyFileName) (\(data.count / 1024) KB)")
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3,
        encoderPrecision: ParakeetEncoderPrecision = .int8,
        encoderComputeUnits: MLComputeUnits? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> AsrModels {
        // Same order as loadLocal/load: platform before encoder placement, both before any download.
        try checkPlatformSupport(for: version)
        let resolvedEncoder = try resolvedEncoderComputeUnits(
            for: version, configuration: (configuration ?? defaultConfiguration()).computeUnits,
            requested: encoderComputeUnits)
        let targetDir = try await download(
            to: directory,
            version: version,
            encoderPrecision: encoderPrecision,
            encoderComputeUnits: resolvedEncoder,
            progressHandler: progressHandler
        )
        return try await load(
            from: targetDir,
            configuration: configuration,
            version: version,
            encoderPrecision: encoderPrecision,
            encoderComputeUnits: resolvedEncoder,
            progressHandler: progressHandler)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let detectedVersion = inferredVersion(from: directory) ?? .v3
        return modelsExist(at: directory, version: detectedVersion)
    }

    public static func modelsExist(
        at directory: URL,
        version: AsrModelVersion,
        encoderPrecision: ParakeetEncoderPrecision = .int8
    ) -> Bool {
        let fileManager = FileManager.default
        let requiredFiles = getRequiredModels(version: version, encoderPrecision: encoderPrecision)

        // Check in the ModelHub repo structure
        let repoPath = repoPath(from: directory, version: version)

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        // Also check for vocabulary file associated with the version
        let vocabularyFileName = getModelFileNames(version: version, encoderPrecision: encoderPrecision).vocabulary
        let vocabPath = repoPath.appendingPathComponent(vocabularyFileName)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    public static func isModelValid(
        version: AsrModelVersion = .v3,
        encoderPrecision: ParakeetEncoderPrecision = .int8
    ) async throws -> Bool {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform("Parakeet models require Apple Silicon")
        }

        let cacheDir = defaultCacheDirectory(for: version)
        guard modelsExist(at: cacheDir, version: version, encoderPrecision: encoderPrecision) else {
            logger.info("Model validation failed: model files not found")
            return false
        }

        let repoPath = repoPath(from: cacheDir, version: version)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        let fileNames = getModelFileNames(version: version, encoderPrecision: encoderPrecision)
        var modelsToValidate = [
            ("Preprocessor", ModelNames.ASR.preprocessorFile),
            ("Decoder", fileNames.decoder),
            ("Joint", fileNames.joint),
        ]
        if !version.hasFusedEncoder {
            modelsToValidate.insert(("Encoder", fileNames.encoder), at: 1)
        }

        for (name, fileName) in modelsToValidate {
            let modelPath = repoPath.appendingPathComponent(fileName)
            do {
                _ = try MLModel(contentsOf: modelPath, configuration: config)
            } catch {
                logger.error("Model validation failed: \(name) - \(error.localizedDescription)")
                return false
            }
        }

        return true
    }

    public static func defaultCacheDirectory(for version: AsrModelVersion = .v3) -> URL {
        MLModelConfigurationUtils.defaultModelsDirectory(for: version.repo)
    }
}

public enum AsrModelsError: LocalizedError, Sendable {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)
    case modelCompilationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "ASR model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download ASR models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load ASR models: \(reason)"
        case .modelCompilationFailed(let reason):
            return
                "Failed to compile ASR models: \(reason). Try deleting the models and re-downloading."
        }
    }
}
