@preconcurrency import CoreML
import Foundation

/// Downloads and loads the Chatterbox Multilingual CoreML assets from
/// `FluidInference/chatterbox-multilingual-coreml` (compiled `.mlmodelc`
/// bundles at the repo root, plus `tables/*.safetensors` and
/// `tokenizer/*.json`).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
@available(macOS 15.0, iOS 18.0, *)
struct ChatterboxModels: Sendable {

    private static let logger = AppLogger(category: "ChatterboxModels")

    let prefill: MLModel
    let decode: MLModel
    let flow: MLModel
    let vocoder: MLModel
    let tokenizer: ChatterboxTokenizer
    let tables: ChatterboxTables
    let voice: ChatterboxTables.Voice
    let repoDir: URL

    static func load(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> ChatterboxModels {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.chatterbox.folderName)

        let requiredPaths =
            ModelNames.Chatterbox.requiredModels.map { $0 } + ModelNames.Chatterbox.auxFiles
        let allPresent = requiredPaths.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if !allPresent {
            logger.info("Downloading Chatterbox Multilingual CoreML assets from HuggingFace…")
            try await ModelHub.download(
                .chatterbox, to: modelsRoot,
                progressHandler: progressHandler)
            // The repo walk only descends into the required .mlmodelc bundles;
            // the tables + tokenizer assets live in subdirectories and are
            // fetched individually.
            try await ensureAuxAssets(repoDir: repoDir)
        } else {
            logger.info("Chatterbox assets found in cache at \(repoDir.path)")
        }

        // ⚠️ Never .cpuOnly for the T3 packages: prediction hard-crashes the
        // process (also reported by other Chatterbox CoreML ports). The ANE
        // compiler rejects the T3 graphs (stacked KV I/O), so GPU is the
        // productive unit for the whole chain.
        func makeConfig(_ units: MLComputeUnits) -> MLModelConfiguration {
            let config = MLModelConfiguration()
            config.computeUnits = units
            return config
        }

        let prefill = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.Chatterbox.prefillFile),
            configuration: makeConfig(.cpuAndGPU))
        let decode = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.Chatterbox.decodeFile),
            configuration: makeConfig(.cpuAndGPU))
        let flow = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.Chatterbox.flowFile),
            configuration: makeConfig(.cpuAndGPU))
        let vocoder = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.Chatterbox.vocoderFile),
            configuration: makeConfig(.cpuAndGPU))

        let tokenizer = try ChatterboxTokenizer(
            tokenizerJsonURL: repoDir.appendingPathComponent(
                ModelNames.Chatterbox.tokenizerFile))
        let tables = try ChatterboxTables.load(
            tablesURL: repoDir.appendingPathComponent(ModelNames.Chatterbox.tablesFile))
        let voice = try ChatterboxTables.loadVoice(
            voiceURL: repoDir.appendingPathComponent(ModelNames.Chatterbox.defaultVoiceFile))

        return ChatterboxModels(
            prefill: prefill,
            decode: decode,
            flow: flow,
            vocoder: vocoder,
            tokenizer: tokenizer,
            tables: tables,
            voice: voice,
            repoDir: repoDir)
    }

    private static func ensureAuxAssets(repoDir: URL) async throws {
        for relative in ModelNames.Chatterbox.auxFiles {
            let localURL = repoDir.appendingPathComponent(relative)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }
            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let remoteURL = try ModelRegistry.resolveModel(
                Repo.chatterbox.remotePath, relative)
            let data = try await AssetDownloader.fetchData(
                from: remoteURL, description: "chatterbox \(relative)", logger: logger)
            try data.write(to: localURL, options: [.atomic])
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
