@preconcurrency import CoreML
import Foundation

/// ASR model version enum
public enum AsrModelVersion: Sendable {
    case v2
    case v3
    /// 110M parameter hybrid TDT-CTC model with fused preprocessor+encoder
    case tdtCtc110m
    /// 600M parameter CTC-only model for Mandarin Chinese (zh-CN)
    case ctcZhCn
    /// 600M parameter CTC-only model for Japanese (ja)
    case ctcJa
    /// 600M parameter TDT model for Japanese (ja) - hybrid CTC preprocessor/encoder + TDT decoder/joint v2
    case tdtJa

    var repo: Repo {
        switch self {
        case .v2: return .parakeetV2
        case .v3: return .parakeet
        case .tdtCtc110m: return .parakeetTdtCtc110m
        case .ctcZhCn: return .parakeetCtcZhCn
        case .ctcJa: return .parakeetCtcJa
        case .tdtJa: return .parakeetCtcJa  // TDT v2 models uploaded to CTC repo
        }
    }

    /// Whether this model version uses a fused preprocessor+encoder (no separate Encoder model)
    public var hasFusedEncoder: Bool {
        switch self {
        case .tdtCtc110m: return true
        default: return false
        }
    }

    /// Whether this model is CTC-only (no TDT decoder+joint)
    public var isCtcOnly: Bool {
        switch self {
        case .ctcZhCn, .ctcJa: return true
        default: return false
        }
    }

    /// Encoder hidden dimension for this model version
    public var encoderHiddenSize: Int {
        switch self {
        case .tdtCtc110m: return 512
        case .ctcZhCn, .ctcJa, .tdtJa: return 1024
        default: return 1024
        }
    }

    /// Blank token ID for this model version
    public var blankId: Int {
        switch self {
        case .v2, .tdtCtc110m: return 1024
        case .v3: return 8192
        case .ctcZhCn: return 7000
        case .ctcJa, .tdtJa: return 3072
        }
    }

    /// Number of LSTM layers in the decoder prediction network
    public var decoderLayers: Int {
        switch self {
        case .tdtCtc110m: return 1
        default: return 2
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
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        version: AsrModelVersion
    ) {
        self.encoder = encoder
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

extension AsrModels {

    private struct ModelSpec {
        let fileName: String
        let computeUnits: MLComputeUnits
    }

    private static func createModelSpecs(
        using config: MLModelConfiguration, version: AsrModelVersion
    ) -> [ModelSpec] {
        if version.hasFusedEncoder {
            // Fused preprocessor+encoder runs on ANE (it contains the conformer encoder)
            return [
                ModelSpec(fileName: Names.preprocessorFile, computeUnits: config.computeUnits)
            ]
        }
        return [
            // Preprocessor ops map to CPU-only across all platforms. XCode profiling shows
            // that 100% of the the operations map to the CPU anyways.
            ModelSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),

            ModelSpec(fileName: Names.encoderFile, computeUnits: config.computeUnits),
        ]
    }

    /// Helper to get the repo path from a models directory
    private static func repoPath(from modelsDirectory: URL, version: AsrModelVersion = .v3) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(version.repo.folderName)
    }

    private static func inferredVersion(from directory: URL) -> AsrModelVersion? {
        let directoryPath = directory.path.lowercased()
        let knownVersions: [AsrModelVersion] = [.tdtCtc110m, .v2, .v3]

        for version in knownVersions {
            if directoryPath.contains(version.repo.folderName.lowercased()) {
                return version
            }
        }

        return nil
    }

    // Use centralized model names
    private typealias Names = ModelNames.ASR

    /// Load ASR models from a directory
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model files
    ///   - configuration: Optional MLModel configuration. When provided, the configuration's
    ///                   computeUnits will be respected. When nil, platform-optimized defaults
    ///                   are used (per-model optimization based on model type).
    ///   - version: ASR model version to load (defaults to v3)
    ///
    /// - Returns: Loaded ASR models
    ///
    /// - Note: The default configuration pins the preprocessor to CPU and every other
    ///         Parakeet component to `.cpuAndNeuralEngine` to avoid GPU dispatch, which keeps
    ///         background execution permitted on iOS.
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> AsrModels {
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()

        let parentDirectory = directory.deletingLastPathComponent()
        // Load preprocessor and encoder first; decoder and joint are loaded below as well.
        let specs = createModelSpecs(using: config, version: version)

        var loadedModels: [String: MLModel] = [:]

        for spec in specs {
            let models = try await DownloadUtils.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDirectory,
                computeUnits: spec.computeUnits,
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
        let encoderModel = loadedModels[Names.encoderFile]  // nil for fused models

        if !version.hasFusedEncoder && encoderModel == nil {
            throw AsrModelsError.loadingFailed("Failed to load encoder model (required for split frontend)")
        }

        // Load decoder and joint as well
        let decoderAndJoint = try await DownloadUtils.loadModels(
            version.repo,
            modelNames: [Names.decoderFile, Names.jointFile],
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            progressHandler: progressHandler
        )

        guard let decoderModel = decoderAndJoint[Names.decoderFile],
            let jointModel = decoderAndJoint[Names.jointFile]
        else {
            throw AsrModelsError.loadingFailed("Failed to load decoder or joint model")
        }

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
                    let ctcModels = try await DownloadUtils.loadModels(
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
            configuration: config,
            vocabulary: try loadVocabulary(from: directory, version: version),
            version: version
        )
        logger.info("Successfully loaded all ASR models with optimized compute units")
        return asrModels
    }

    private static func loadVocabulary(from directory: URL, version: AsrModelVersion) throws -> [Int: String] {
        let vocabPath = repoPath(from: directory, version: version).appendingPathComponent(
            Names.vocabulary(for: version.repo))

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "Vocabulary file not found at \(vocabPath.path). Please ensure the vocab file is downloaded with the models."
            )
            throw AsrModelsError.modelNotFound(Names.vocabulary(for: version.repo), vocabPath)
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
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory(for: version)
        return try await load(
            from: cacheDir, configuration: configuration, version: version,
            progressHandler: progressHandler)
    }

    /// Load models with automatic recovery on compilation failures
    public static func loadWithAutoRecovery(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> AsrModels {
        let targetDir = directory ?? defaultCacheDirectory()
        return try await load(from: targetDir, configuration: configuration, progressHandler: progressHandler)
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
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        version: AsrModelVersion = .v3,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(for: version)
        logger.info("Downloading ASR models to: \(targetDir.path)")
        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir, version: version) {
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

        let specs: [DownloadSpec]
        if version.hasFusedEncoder {
            specs = [
                // Fused preprocessor+encoder runs on ANE
                DownloadSpec(fileName: Names.preprocessorFile, computeUnits: defaultUnits),
                DownloadSpec(fileName: Names.decoderFile, computeUnits: defaultUnits),
                DownloadSpec(fileName: Names.jointFile, computeUnits: defaultUnits),
            ]
        } else {
            specs = [
                // Preprocessor ops map to CPU-only across all platforms.
                DownloadSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),
                DownloadSpec(fileName: Names.encoderFile, computeUnits: defaultUnits),
                DownloadSpec(fileName: Names.decoderFile, computeUnits: defaultUnits),
                DownloadSpec(fileName: Names.jointFile, computeUnits: defaultUnits),
            ]
        }

        for spec in specs {
            _ = try await DownloadUtils.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDir,
                computeUnits: spec.computeUnits,
                progressHandler: progressHandler
            )
        }

        logger.info("Successfully downloaded ASR models")
        return targetDir
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> AsrModels {
        let targetDir = try await download(to: directory, version: version, progressHandler: progressHandler)
        return try await load(
            from: targetDir, configuration: configuration, version: version,
            progressHandler: progressHandler)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let detectedVersion = inferredVersion(from: directory) ?? .v3
        return modelsExist(at: directory, version: detectedVersion)
    }

    public static func modelsExist(at directory: URL, version: AsrModelVersion) -> Bool {
        let fileManager = FileManager.default
        let requiredFiles =
            version.hasFusedEncoder ? ModelNames.ASR.requiredModelsFused : ModelNames.ASR.requiredModels

        // Check in the DownloadUtils repo structure
        let repoPath = repoPath(from: directory, version: version)

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        // Also check for vocabulary file associated with the version
        let vocabPath = repoPath.appendingPathComponent(Names.vocabulary(for: version.repo))
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    public static func isModelValid(version: AsrModelVersion = .v3) async throws -> Bool {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform("Parakeet models require Apple Silicon")
        }

        let cacheDir = defaultCacheDirectory(for: version)
        guard modelsExist(at: cacheDir, version: version) else {
            logger.info("Model validation failed: model files not found")
            return false
        }

        let repoPath = repoPath(from: cacheDir, version: version)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        var modelsToValidate = [
            ("Preprocessor", ModelNames.ASR.preprocessorFile),
            ("Decoder", ModelNames.ASR.decoderFile),
            ("Joint", ModelNames.ASR.jointFile),
        ]
        if !version.hasFusedEncoder {
            modelsToValidate.insert(("Encoder", ModelNames.ASR.encoderFile), at: 1)
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

    // Legacy method for backward compatibility
    public static func defaultCacheDirectory() -> URL {
        return defaultCacheDirectory(for: .v3)
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
