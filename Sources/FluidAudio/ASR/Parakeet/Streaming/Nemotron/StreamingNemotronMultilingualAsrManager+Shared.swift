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
public struct SharedNemotronMultilingualModels: Sendable {
    // The mel front-end is computed natively in Swift per manager
    // (`NemotronMelExtractor`); no CoreML preprocessor is shared.
    public let encoder: MLModel
    /// Bare prediction LSTM. Optional: a lean ship may omit it when B1
    /// (`decoderJoint`) covers the standard path and no smart-spec (K=4)
    /// asset is present (the smart-spec path is the only consumer of the
    /// unfused decoder, for `dec_out`).
    public let decoder: MLModel?
    /// Bare joint. Optional: only the smart-spec drain and the hybrid path
    /// use it standalone; the standard path uses B1.
    public let joint: MLModel?
    /// B1 fusion (decoder + joint merged). May be nil.
    public let decoderJoint: MLModel?
    /// B2 triple-fusion (decoder + joint + argmax). May be nil.
    public let decoderJointArgmax: MLModel?
    /// B3+B1 fusion (decoder + joint-without-encproj). May be nil.
    public let decoderJointNoEncProj: MLModel?
    /// Smart-speculative batched joint. May be nil.
    public let jointNoEncProjBatched: MLModel?
    /// True iff the encoder uses MLState for cache (iOS 18+ stateful path).
    public let encoderIsStateful: Bool
    public let config: NemotronMultilingualStreamingConfig
    public let tokenizer: NemotronMultilingualTokenizer
    /// MLModelConfiguration used to load these. Each manager uses the
    /// same configuration to stay on the same compute units.
    public let mlConfiguration: MLModelConfiguration

    fileprivate init(
        encoder: MLModel,
        decoder: MLModel?,
        joint: MLModel?,
        decoderJoint: MLModel?,
        decoderJointArgmax: MLModel?,
        decoderJointNoEncProj: MLModel?,
        jointNoEncProjBatched: MLModel?,
        encoderIsStateful: Bool,
        config: NemotronMultilingualStreamingConfig,
        tokenizer: NemotronMultilingualTokenizer,
        mlConfiguration: MLModelConfiguration
    ) {
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.decoderJoint = decoderJoint
        self.decoderJointArgmax = decoderJointArgmax
        self.decoderJointNoEncProj = decoderJointNoEncProj
        self.jointNoEncProjBatched = jointNoEncProjBatched
        self.encoderIsStateful = encoderIsStateful
        self.config = config
        self.tokenizer = tokenizer
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

        // Bare decoder + joint are now OPTIONAL. A lean B1 ship can omit them;
        // they're only consumed by the smart-spec (K=4) path and the hybrid
        // path. The standard decode path uses B1 (`decoderJoint`). A valid
        // decode path is enforced after the fused assets load (below).
        let decoder = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.decoderFile,
            packageName: ModelNames.NemotronMultilingualStreaming.decoderPackage,
            configuration: mlConfiguration,
            logName: "decoder",
            logger: logger
        )

        let joint = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: ModelNames.NemotronMultilingualStreaming.jointFile,
            packageName: ModelNames.NemotronMultilingualStreaming.jointPackage,
            configuration: mlConfiguration,
            logName: "joint",
            logger: logger
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
        let jointNoEncProjBatched = try await Self.loadOptionalShared(
            directory: directory,
            compiledName: "joint_noencproj_batched.mlmodelc",
            packageName: "joint_noencproj_batched.mlpackage",
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_JOINT_BATCHED_CU", base: mlConfiguration, logger: logger),
            logName: "joint_noencproj_batched",
            logger: logger
        )

        // Validate a usable decode path exists now that decoder/joint are
        // optional. Standard path needs a fused decoder_joint (B1/B3/B2) or
        // the bare decoder+joint pair. Smart-spec (K=4) consumes the bare
        // decoder (for dec_out) and bare joint (drain), so if it's present
        // both must be too — otherwise its force-unwraps would crash.
        let hasStandardPath =
            decoderJoint != nil || decoderJointNoEncProj != nil || decoderJointArgmax != nil
            || (decoder != nil && joint != nil)
        guard hasStandardPath else {
            throw ASRError.processingFailed(
                "No decode path in \(directory.path): provide a fused decoder_joint (B1/B3) "
                    + "or both bare decoder.mlmodelc + joint.mlmodelc.")
        }
        if jointNoEncProjBatched != nil && (decoder == nil || joint == nil) {
            throw ASRError.processingFailed(
                "Smart-spec asset joint_noencproj_batched present but bare decoder/joint missing "
                    + "— K=4 needs both. Either add them or remove the smart-spec asset.")
        }
        if decoder == nil && joint == nil {
            logger.info("Lean B1 ship: bare decoder/joint omitted; using fused decode path only.")
        }

        // Tokenizer
        let tokenizerURL = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.tokenizer)
        let tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tokenizerURL,
            langTagTokenIds: config.langTagTokenIds
        )

        logger.info("Shared models preload complete — ready for N consumers")

        return SharedNemotronMultilingualModels(
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            decoderJoint: decoderJoint,
            decoderJointArgmax: decoderJointArgmax,
            decoderJointNoEncProj: decoderJointNoEncProj,
            jointNoEncProjBatched: jointNoEncProjBatched,
            encoderIsStateful: encoderIsStateful,
            config: config,
            tokenizer: tokenizer,
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

        // Each manager builds its own (non-thread-safe) mel extractors; only
        // the heavyweight MLModel handles are shared across streams.
        self.melExtractor = NemotronMelExtractor(nMels: shared.config.melFeatures)
        self.prefetchMelExtractor = NemotronMelExtractor(nMels: shared.config.melFeatures)

        // Adopt shared MLModel references
        self.encoder = shared.encoder
        self.decoder = shared.decoder
        self.joint = shared.joint
        self.decoderJoint = shared.decoderJoint
        self.decoderJointArgmax = shared.decoderJointArgmax
        self.decoderJointNoEncProj = shared.decoderJointNoEncProj
        self.jointNoEncProjBatched = shared.jointNoEncProjBatched
        self.tokenizer = shared.tokenizer

        if let m = self.jointNoEncProjBatched,
            let constraint = m.modelDescription.inputDescriptionsByName["encoder_proj"]?.multiArrayConstraint,
            constraint.shape.count >= 2
        {
            let kFromModel = constraint.shape[1].intValue
            if kFromModel > 0 {
                self.jointNoEncProjBatchedK = kFromModel
            }
        }

        // Per-stream MLState instance (each stream gets its own).
        // makeState() returns a fresh zero-initialized state.
        if #available(macOS 15, iOS 18, *) {
            if shared.encoderIsStateful {
                self.encoderState = shared.encoder.makeState()
            }
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
        self.jointNoEncProjBatchedPredictionOptions = Self.makePredictionOptions(for: self.jointNoEncProjBatched)

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

    /// Map a language hint (e.g. "en-US", "zh-CN", "de-DE", "auto") to the
    /// model folder in the HuggingFace repo.
    ///
    /// The repo ships two models: `latin` (a Latin-script-pruned vocab shared by
    /// en/es/fr/it/pt/de — smaller, faster joint) and `multilingual` (the full
    /// 13087-token vocab covering every language, incl. zh/ja). Latin-script
    /// language hints route to `latin`; everything else, and "auto", falls back
    /// to the full-vocab `multilingual` model.
    public static func languageDirectory(for languageCode: String) -> String {
        let c = languageCode.lowercased()
        let latinPrefixes = ["en", "es", "fr", "it", "pt", "de"]
        if latinPrefixes.contains(where: { c.hasPrefix($0) }) { return "latin" }
        return "multilingual"
    }

    /// Download the requested `<language>/<chunkMs>ms` variant from the
    /// HuggingFace repo (compiled `.mlmodelc` only) and preload it.
    ///
    /// - Parameters:
    ///   - languageCode: Language hint, e.g. "en-US", "zh-CN", "de-DE", or
    ///     "auto"/"multilingual" for the full-vocab model. Per-language ships
    ///     are vocab-pruned (faster); the multilingual ship covers 100+ langs.
    ///   - chunkMs: Chunk size tier — 560, 1120, 2240 (recommended), or 4480.
    ///     Note: on long continuous sessions the 560 ms tier emits increasingly
    ///     sparse punctuation (words are unaffected); prefer 1120 ms+ when
    ///     punctuation matters, or reset the session on silence. See issue #687.
    ///   - directory: Model cache root (default: Application Support/FluidAudio/Models).
    public static func downloadAndPreloadShared(
        languageCode: String = "auto",
        chunkMs: Int = 2240,
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> SharedNemotronMultilingualModels {
        let variantDir = try await downloadVariant(
            languageCode: languageCode, chunkMs: chunkMs,
            to: directory, progressHandler: progressHandler)
        return try await preloadShared(from: variantDir, configuration: configuration)
    }

    /// Download the requested `<language>/<chunkMs>ms` variant from the
    /// HuggingFace repo (compiled `.mlmodelc` only) and return the local
    /// variant directory. Cached downloads are reused.
    public static func downloadVariant(
        languageCode: String = "auto",
        chunkMs: Int = 2240,
        to directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let logger = AppLogger(category: "NemotronMultilingualStreaming")
        let langDir = languageDirectory(for: languageCode)
        let subdirectory = "\(langDir)/\(chunkMs)ms"

        let modelsBaseDir =
            directory
            ?? FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first!
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)

        let repoCacheDir = modelsBaseDir.appendingPathComponent(Repo.nemotronMultilingual.folderName)
        let variantDir = repoCacheDir.appendingPathComponent(subdirectory)
        let metadataPath = variantDir.appendingPathComponent(
            ModelNames.NemotronMultilingualStreaming.metadata)

        if FileManager.default.fileExists(atPath: metadataPath.path) {
            // Repair pass for variants cached before 2026-05-31: the latin
            // tokenizer.json shipped with a corrupt entry — id 2224 read
            // "<blank>" (duplicating the real blank at 2828) instead of
            // "▁there", so every "there" in a transcript decoded as the
            // literal text "<blank>". The repo file is fixed upstream;
            // re-fetch the tokenizer when the cached copy still has the
            // corruption (https://github.com/FluidInference/FluidAudio/issues/687).
            let tokenizerPath = variantDir.appendingPathComponent(
                ModelNames.NemotronMultilingualStreaming.tokenizer)
            if Self.tokenizerHasCorruptBlankEntry(tokenizerPath: tokenizerPath, metadataPath: metadataPath) {
                logger.warning(
                    "Cached tokenizer.json at \(tokenizerPath.path) has a corrupt duplicate '<blank>' entry — re-downloading the fixed file"
                )
                // Move the corrupt file aside so download(subdirectory:) re-fetches
                // it; restore on failure so offline users keep a working variant.
                let backupPath = tokenizerPath.appendingPathExtension("corrupt")
                try? FileManager.default.removeItem(at: backupPath)
                try FileManager.default.moveItem(at: tokenizerPath, to: backupPath)
                do {
                    try await ModelHub.download(
                        .nemotronMultilingual,
                        subdirectory: subdirectory,
                        to: repoCacheDir,
                        progressHandler: progressHandler,
                        shouldSkip: { $0.contains(".mlpackage") }
                    )
                    try? FileManager.default.removeItem(at: backupPath)
                    logger.info("Repaired tokenizer.json for \(subdirectory)")
                } catch {
                    try? FileManager.default.moveItem(at: backupPath, to: tokenizerPath)
                    logger.warning(
                        "Could not re-download tokenizer.json (\(error.localizedDescription)); keeping the cached copy"
                    )
                }
            } else {
                logger.info("Using cached multilingual variant at \(variantDir.path)")
            }
        } else {
            logger.info("Downloading multilingual variant \(subdirectory) (.mlmodelc only)...")
            try await ModelHub.download(
                .nemotronMultilingual,
                subdirectory: subdirectory,
                to: repoCacheDir,
                progressHandler: progressHandler,
                shouldSkip: { $0.contains(".mlpackage") }
            )
        }
        return variantDir
    }

    /// Detect the known tokenizer.json corruption shipped with latin variants
    /// before 2026-05-31: a "<blank>" piece at an id other than the blank
    /// index (id 2224, which should be "▁there", duplicated the real blank
    /// entry at the blank index). A healthy tokenizer maps every id below
    /// `blank_idx` to a real piece — "<blank>" may only appear at `blank_idx`.
    ///
    /// Returns `false` when either file is missing or unreadable; missing
    /// files are handled by the normal download path.
    internal static func tokenizerHasCorruptBlankEntry(tokenizerPath: URL, metadataPath: URL) -> Bool {
        guard let config = try? NemotronMultilingualStreamingConfig(from: metadataPath),
            let data = try? Data(contentsOf: tokenizerPath),
            let vocab = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        else { return false }
        return vocab.contains { entry in
            guard entry.value == "<blank>", let id = Int(entry.key) else { return false }
            return id != config.blankIdx
        }
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
