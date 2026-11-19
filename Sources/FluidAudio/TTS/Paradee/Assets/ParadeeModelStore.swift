#if TTS
@preconcurrency import CoreML
import Foundation

/// Actor store for one Paradee release and variant: `ParadeeText` +
/// `ParadeeAcoustic` CoreML bundles and the phoneme vocab, downloaded from
/// `FluidInference/paradee-8m-coreml/[v1.<2|3>/]<variant>/` on first use, and
/// for v1.3 the eSpeak-free G2P from `v1.3/g2p/`.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor ParadeeModelStore {

    private let logger = AppLogger(category: "ParadeeModelStore")

    private let version: ParadeeVersion
    private let variant: ParadeeVariant
    /// Models root every asset of this store resolves under (`nil` = the
    /// default TTS cache root).
    nonisolated let directory: URL?
    private let computeUnits: MLComputeUnits

    private var textModel: MLModel?
    private var acousticModel: MLModel?
    private var vocab: KokoroAneVocab?
    private var g2p: ParadeeG2P?

    public init(
        version: ParadeeVersion = .v13,
        variant: ParadeeVariant = .int8,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly
    ) {
        self.version = version
        self.variant = variant
        self.directory = directory
        self.computeUnits = computeUnits
    }

    /// Download (if missing) and load both models + vocab.
    public func loadIfNeeded() async throws {
        if textModel != nil, acousticModel != nil, vocab != nil { return }
        guard Self.isSupported(computeUnits) else {
            throw ParadeeError.unsupportedComputeUnits(Self.describe(computeUnits))
        }

        let repoDir = try await ensureModels()
        textModel = try loadModel(repoDir: repoDir, fileName: ModelNames.Paradee.textFile)
        acousticModel = try loadModel(repoDir: repoDir, fileName: ModelNames.Paradee.acousticFile)
        do {
            vocab = try KokoroAneVocab.load(from: repoDir.appendingPathComponent(ModelNames.Paradee.vocabFile))
        } catch {
            throw ParadeeError.modelFileNotFound("\(ModelNames.Paradee.vocabFile): \(error.localizedDescription)")
        }
        logger.info(
            "Paradee \(version.rawValue) \(variant.rawValue) loaded from \(repoDir.path) (\(Self.describe(computeUnits)))"
        )
    }

    public func models() throws -> (text: MLModel, acoustic: MLModel) {
        guard let textModel, let acousticModel else { throw ParadeeError.notInitialized }
        return (textModel, acousticModel)
    }

    public func vocabulary() throws -> KokoroAneVocab {
        guard let vocab else { throw ParadeeError.notInitialized }
        return vocab
    }

    /// Download (if missing) and load the eSpeak-free G2P (`ParadeeG2P` +
    /// `paradee_g2p.json`). It always runs on the CPU: its input lengths vary
    /// per word.
    func loadG2PIfNeeded() async throws {
        if g2p != nil { return }
        let repoDir = try await ensureDownloaded(.paradeeV13G2P, required: ModelNames.Paradee.g2pRequiredModels)
        let model = try loadModel(repoDir: repoDir, fileName: ModelNames.Paradee.g2pFile, computeUnits: .cpuOnly)
        let dataURL = repoDir.appendingPathComponent(ModelNames.Paradee.g2pDataFile)
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw ParadeeError.modelFileNotFound(ModelNames.Paradee.g2pDataFile)
        }
        g2p = ParadeeG2P(data: try ParadeeG2PData.load(from: dataURL), model: model)
        logger.info("Paradee G2P loaded from \(repoDir.path)")
    }

    func paradeeG2P() throws -> ParadeeG2P {
        guard let g2p else { throw ParadeeError.notInitialized }
        return g2p
    }

    public func unload() {
        textModel = nil
        acousticModel = nil
        vocab = nil
        g2p = nil
    }

    /// `.all` and `.cpuAndGPU` abort in MPSGraph (`GPURNNOps … JIT not supported`).
    static func isSupported(_ units: MLComputeUnits) -> Bool {
        units == .cpuOnly || units == .cpuAndNeuralEngine
    }

    static func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .all: return "all"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "unknown"
        }
    }

    // MARK: - Helpers

    private func ensureModels() async throws -> URL {
        try await ensureDownloaded(variant.repo(for: version), required: ModelNames.Paradee.requiredModels)
    }

    private func ensureDownloaded(_ repo: Repo, required: Set<String>) async throws -> URL {
        let modelsRoot = try directory ?? Self.defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(repo.folderName)
        let allPresent = required.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if allPresent { return repoDir }

        logger.info("Downloading \(repo.folderName) from HuggingFace…")
        do {
            try await ModelHub.download(repo, to: modelsRoot)
        } catch {
            throw ParadeeError.downloadFailed("\(error)")
        }
        return repoDir
    }

    private func loadModel(repoDir: URL, fileName: String, computeUnits: MLComputeUnits? = nil) throws -> MLModel {
        let url = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ParadeeError.modelFileNotFound(fileName)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits ?? self.computeUnits
        do {
            return try MLModel(contentsOf: url, configuration: config)
        } catch {
            throw ParadeeError.corruptedModel(fileName, underlying: "\(error)")
        }
    }

    private static func defaultCacheRoot() throws -> URL {
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}
#endif
