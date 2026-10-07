@preconcurrency import CoreML
import Foundation

/// Actor store for one Paradee variant: `ParadeeText` + `ParadeeAcoustic`
/// CoreML bundles and the phoneme vocab, downloaded from
/// `FluidInference/paradee-8m-coreml/<variant>/` on first use.
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
public actor ParadeeModelStore {

    private let logger = AppLogger(category: "ParadeeModelStore")

    private let variant: ParadeeVariant
    private let directory: URL?
    private let computeUnits: MLComputeUnits

    private var textModel: MLModel?
    private var acousticModel: MLModel?
    private var vocab: KokoroAneVocab?

    public init(
        variant: ParadeeVariant = .int8,
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly
    ) {
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
        logger.info("Paradee \(variant.rawValue) loaded from \(repoDir.path) (\(Self.describe(computeUnits)))")
    }

    public func models() throws -> (text: MLModel, acoustic: MLModel) {
        guard let textModel, let acousticModel else { throw ParadeeError.notInitialized }
        return (textModel, acousticModel)
    }

    public func vocabulary() throws -> KokoroAneVocab {
        guard let vocab else { throw ParadeeError.notInitialized }
        return vocab
    }

    public func unload() {
        textModel = nil
        acousticModel = nil
        vocab = nil
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
        let repo = variant.repo
        let modelsRoot = try directory ?? Self.defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(repo.folderName)
        let allPresent = ModelNames.Paradee.requiredModels.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if allPresent { return repoDir }

        logger.info("Downloading Paradee \(variant.rawValue) CoreML models from HuggingFace…")
        do {
            try await ModelHub.download(repo, to: modelsRoot)
        } catch {
            throw ParadeeError.downloadFailed("\(error)")
        }
        return repoDir
    }

    private func loadModel(repoDir: URL, fileName: String) throws -> MLModel {
        let url = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ParadeeError.modelFileNotFound(fileName)
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
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
