@preconcurrency import CoreML
import Foundation

/// Immutable bundle of CoreML models + tokenizer + config that can be
/// shared across N independent `StreamingNemotronMultilingualAsrManager`
/// instances. Use this for multi-stream parallel inference where each
/// stream has its own cache/state but reuses the same compiled model
/// graphs to avoid O(N) memory blowup.
///
/// MLModel is thread-safe for `prediction(from:)` calls — multiple
/// streams may dispatch predictions concurrently against the same
/// model object. Per-stream mutable state (caches, hState/cState,
/// melCache, prediction output backings) stays inside the manager
/// actor.
public struct SharedNemotronMultilingualModels: @unchecked Sendable {
    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    /// B1 fusion (decoder + joint merged). May be nil.
    public let decoderJoint: MLModel?
    /// B2 triple-fusion (decoder + joint + argmax). May be nil.
    public let decoderJointArgmax: MLModel?
    /// B3+B1 fusion (decoder + joint-without-encproj). May be nil.
    public let decoderJointNoEncProj: MLModel?
    /// Speculative batched joint. May be nil.
    public let jointBatched: MLModel?
    /// True iff the encoder uses MLState for cache (iOS 18+ stateful path).
    public let encoderIsStateful: Bool
    public let config: NemotronMultilingualStreamingConfig
    public let tokenizer: NemotronMultilingualTokenizer
    /// Optional native-Accelerate RNN-T weights blob directory.
    public let nativeWeightsDir: URL?
    /// MLModelConfiguration used to load these. Each manager uses the
    /// same configuration to stay on the same compute units.
    public let mlConfiguration: MLModelConfiguration

    fileprivate init(
        preprocessor: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        decoderJoint: MLModel?,
        decoderJointArgmax: MLModel?,
        decoderJointNoEncProj: MLModel?,
        jointBatched: MLModel?,
        encoderIsStateful: Bool,
        config: NemotronMultilingualStreamingConfig,
        tokenizer: NemotronMultilingualTokenizer,
        nativeWeightsDir: URL?,
        mlConfiguration: MLModelConfiguration
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.decoderJoint = decoderJoint
        self.decoderJointArgmax = decoderJointArgmax
        self.decoderJointNoEncProj = decoderJointNoEncProj
        self.jointBatched = jointBatched
        self.encoderIsStateful = encoderIsStateful
        self.config = config
        self.tokenizer = tokenizer
        self.nativeWeightsDir = nativeWeightsDir
        self.mlConfiguration = mlConfiguration
    }
}

extension StreamingNemotronMultilingualAsrManager {

    /// Load all CoreML models + tokenizer + config ONCE, producing a
    /// shareable bundle that N managers can consume via
    /// `loadFromShared(_:)`. The single load cost is paid once; each
    /// consumer pays only its own per-stream state allocation.
    ///
    /// Memory footprint at N managers:
    /// - With per-manager loadModels(): N × (~1.5 GB models + ~50 MB state)
    /// - With shared+loadFromShared(): 1 × ~1.5 GB models + N × ~50 MB state
    ///
    /// `configuration` defaults to `.cpuAndNeuralEngine` (ANE path).
    public static func preloadShared(
        from directory: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SharedNemotronMultilingualModels {
        let logger = AppLogger(category: "NemotronMultilingualStreaming")

        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron multilingual int8 streaming models require Apple Silicon (ANE)."
            )
        }

        let mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        logger.info("Preloading shared Nemotron multilingual models from \(directory.path)...")

        let metadataPath = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ASRError.processingFailed(
                "metadata.json not found at \(metadataPath.path)."
            )
        }
        let config = try NemotronMultilingualStreamingConfig(from: metadataPath)
        logger.info(
            "Loaded multilingual config: \(config.chunkMs)ms chunks, vocab=\(config.vocabSize), \(config.numPrompts) prompts"
        )

        let preprocessor = try await Self.loadShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.preprocessorFile,
            packageName: ModelNames.NemotronMultilingualStreaming.preprocessorPackage,
            configuration: mlConfiguration
        )

        let encoder = try await Self.loadShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.encoderFile,
            packageName: ModelNames.NemotronMultilingualStreaming.encoderPackage,
            configuration: mlConfiguration
        )
        let encoderIsStateful: Bool
        if #available(macOS 15, iOS 18, *) {
            encoderIsStateful = !encoder.modelDescription.stateDescriptionsByName.isEmpty
            if encoderIsStateful {
                logger.info("Encoder has MLState — per-stream state will be allocated on consumer init")
            }
        } else {
            encoderIsStateful = false
        }

        let decoder = try await Self.loadShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.decoderFile,
            packageName: ModelNames.NemotronMultilingualStreaming.decoderPackage,
            configuration: mlConfiguration
        )

        let joint = try await Self.loadShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.jointFile,
            packageName: ModelNames.NemotronMultilingualStreaming.jointPackage,
            configuration: mlConfiguration
        )

        // Optional fusion mlpackages (B2 > B3+B1 > B1 priority — same
        // precedence as the per-manager loadModels path)
        let decoderJointArgmax = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "decoder_joint_argmax.mlmodelc",
            packageName: "decoder_joint_argmax.mlpackage",
            configuration: mlConfiguration,
            logName: "decoder_joint_argmax",
            logger: logger
        )
        var decoderJointNoEncProj: MLModel? = nil
        if decoderJointArgmax == nil {
            decoderJointNoEncProj = try await Self.loadOptionalShared(
                directory: directory,
                compiledName: "decoder_joint_noencproj.mlmodelc",
                packageName: "decoder_joint_noencproj.mlpackage",
                configuration: mlConfiguration,
                logName: "decoder_joint_noencproj",
                logger: logger
            )
        }
        var decoderJoint: MLModel? = nil
        if decoderJointArgmax == nil && decoderJointNoEncProj == nil {
            decoderJoint = try await Self.loadOptionalShared(
                directory: directory,
                compiledName: "decoder_joint.mlmodelc",
                packageName: "decoder_joint.mlpackage",
                configuration: mlConfiguration,
                logName: "decoder_joint",
                logger: logger
            )
        }
        let jointBatched = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "joint_batched.mlmodelc",
            packageName: "joint_batched.mlpackage",
            configuration: mlConfiguration,
            logName: "joint_batched",
            logger: logger
        )

        // Tokenizer
        let tokenizerURL = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.tokenizer)
        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tokenizerURL,
            langTagTokenIds: config.langTagTokenIds
        )

        let nativeWeightsDir = directory.appendingPathComponent("native_weights")
        let nativeAvailable = FileManager.default.fileExists(
            atPath: nativeWeightsDir.appendingPathComponent("weights.bin").path
        )

        logger.info("Shared models preload complete — ready for N consumers")

        return SharedNemotronMultilingualModels(
            preprocessor: preprocessor,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            decoderJoint: decoderJoint,
            decoderJointArgmax: decoderJointArgmax,
            decoderJointNoEncProj: decoderJointNoEncProj,
            jointBatched: jointBatched,
            encoderIsStateful: encoderIsStateful,
            config: config,
            tokenizer: tokenizer,
            nativeWeightsDir: nativeAvailable ? nativeWeightsDir : nil,
            mlConfiguration: mlConfiguration
        )
    }

    /// Initialize this manager from a pre-loaded shared model bundle.
    /// Each manager builds its OWN per-stream state (caches, MLState
    /// instance, prediction options with output backings, step buffers,
    /// NativeRnntInner) — only the MLModel handles are shared.
    public func loadFromShared(_ shared: SharedNemotronMultilingualModels) async throws {
        // Adopt shared configuration so prediction calls route through
        // the same compute units. Without this, the manager's default
        // MLModelConfiguration may differ from the shared bundle's.
        self.mlConfiguration = shared.mlConfiguration

        self.config = shared.config
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)

        // Adopt shared MLModel references
        self.preprocessor = shared.preprocessor
        self.encoder = shared.encoder
        self.decoder = shared.decoder
        self.joint = shared.joint
        self.decoderJoint = shared.decoderJoint
        self.decoderJointArgmax = shared.decoderJointArgmax
        self.decoderJointNoEncProj = shared.decoderJointNoEncProj
        self.jointBatched = shared.jointBatched
        self.tokenizer = shared.tokenizer

        // Per-stream MLState instance (each stream gets its own).
        // makeState() returns a fresh zero-initialized state.
        if #available(macOS 15, iOS 18, *) {
            if shared.encoderIsStateful {
                self.encoderState = shared.encoder.makeState()
            }
        }

        // Per-stream NativeRnntInner (has its own LSTM state buffers).
        if let nativeDir = shared.nativeWeightsDir {
            self.nativeRnnt = NativeRnntInner(directory: nativeDir)
        }

        // Per-stream cache/state init
        try resetStates()

        // Per-stream MLPredictionOptions (each contains pre-allocated
        // output buffers — CANNOT be shared across streams).
        self.encoderPredictionOptions = Self.makePredictionOptions(for: self.encoder)
        self.decoderPredictionOptions = Self.makePredictionOptions(for: self.decoder)
        self.jointPredictionOptions = Self.makePredictionOptions(for: self.joint)
        self.decoderJointPredictionOptions = Self.makePredictionOptions(for: self.decoderJoint)
        self.decoderJointArgmaxPredictionOptions = Self.makePredictionOptions(for: self.decoderJointArgmax)
        self.decoderJointNoEncProjPredictionOptions = Self.makePredictionOptions(for: self.decoderJointNoEncProj)

        // Per-stream inner-loop step buffers
        self.encoderStepBuf = try? MLMultiArray(shape: [1, NSNumber(value: config.encoderDim), 1], dataType: .float32)
        self.encoderProjStepBuf = try? MLMultiArray(shape: [1, 1, NSNumber(value: 640)], dataType: .float32)

        // Per-stream token input buffers
        if let tokInput = try? MLMultiArray(shape: [1, 1], dataType: .int32) {
            self.tokenInputBuf = tokInput
        }
        if let tokLen = try? MLMultiArray(shape: [1], dataType: .int32) {
            tokLen[0] = 1
            self.tokenLenBuf = tokLen
        }

        // Skip warmup — the shared models are already compiled & resident
        // from preloadShared(). The first real chunk pays no cold-start
        // penalty in any consumer.

        logger.info(
            "Nemotron multilingual manager initialized from shared models (\(config.chunkMs)ms chunks)."
        )
    }

    /// Compile-if-needed + load helper for required model files.
    private static func loadShared(
        directory: URL,
        compiledName: String,
        packageName: String,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiledURL = directory.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        }
        let packageURL = directory.appendingPathComponent(packageName)
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw ASRError.processingFailed(
                "Neither \(compiledName) nor \(packageName) found in \(directory.path)"
            )
        }
        let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
        return try await MLModel.load(contentsOf: tempCompiledURL, configuration: configuration)
    }

    /// Compile-if-needed + load helper for optional fusion bundles.
    /// Returns nil if neither the compiled nor the package form is present.
    private static func loadOptionalShared(
        directory: URL,
        compiledName: String,
        packageName: String,
        configuration: MLModelConfiguration,
        logName: String,
        logger: AppLogger
    ) async throws -> MLModel? {
        let compiledURL = directory.appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            let m = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
            logger.info("Loaded shared \(compiledName)")
            return m
        }
        let packageURL = directory.appendingPathComponent(packageName)
        if FileManager.default.fileExists(atPath: packageURL.path) {
            let tempCompiledURL = try await MLModel.compileModel(at: packageURL)
            let m = try await MLModel.load(contentsOf: tempCompiledURL, configuration: configuration)
            logger.info("Compiled + loaded shared \(packageName)")
            return m
        }
        return nil
    }
}
