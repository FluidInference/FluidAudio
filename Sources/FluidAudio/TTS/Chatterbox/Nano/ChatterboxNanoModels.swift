@preconcurrency import CoreML
import Foundation

/// Downloads and loads the Chatterbox Nano CoreML assets from
/// `FluidInference/chatterbox-nano-coreml` (compiled `.mlmodelc` bundles at
/// the repo root, plus `tables/*.safetensors` and `tokenizer/*`).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
@available(macOS 15.0, iOS 18.0, *)
struct ChatterboxNanoModels: Sendable {

    private static let logger = AppLogger(category: "ChatterboxNanoModels")

    let prefill: MLModel
    let decode: MLModel
    let flow: MLModel
    let vocoder: MLModel
    let tokenizer: ChatterboxNanoTokenizer
    let tables: ChatterboxTables.Nano
    let voice: ChatterboxTables.Voice
    let repoDir: URL

    static func load(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> ChatterboxNanoModels {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.chatterboxNano.folderName)

        let requiredPaths =
            ModelNames.ChatterboxNano.requiredModels.map { $0 }
            + ModelNames.ChatterboxNano.auxFiles
        let allPresent = requiredPaths.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if !allPresent {
            logger.info("Downloading Chatterbox Nano CoreML assets from HuggingFace…")
            try await ModelHub.download(
                .chatterboxNano, to: modelsRoot,
                progressHandler: progressHandler)
            // The repo walk only descends into the required .mlmodelc bundles;
            // the tables + tokenizer assets live in subdirectories and are
            // fetched individually.
            try await ensureAuxAssets(repoDir: repoDir)
        } else {
            logger.info("Chatterbox Nano assets found in cache at \(repoDir.path)")
        }

        // The sibling Multilingual T3 packages hard-crash under .cpuOnly and
        // the ANE compiler rejects the stacked KV I/O; keep the whole chain
        // on GPU (untested on .cpuOnly — not worth diverging from MTL).
        func makeConfig(_ units: MLComputeUnits) -> MLModelConfiguration {
            let config = MLModelConfiguration()
            config.computeUnits = units
            return config
        }

        let prefill = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.prefillFile),
            configuration: makeConfig(.cpuAndGPU))
        let decode = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.decodeFile),
            configuration: makeConfig(.cpuAndGPU))
        let flow = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.flowFile),
            configuration: makeConfig(.cpuAndGPU))
        let vocoder = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.vocoderFile),
            configuration: makeConfig(.cpuAndGPU))

        func loadAux() throws -> (ChatterboxNanoTokenizer, ChatterboxTables.Nano, ChatterboxTables.Voice) {
            let tokenizer = try ChatterboxNanoTokenizer(
                vocabURL: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.vocabFile),
                mergesURL: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.mergesFile),
                addedTokensURL: repoDir.appendingPathComponent(
                    ModelNames.ChatterboxNano.addedTokensFile))
            let tables = try ChatterboxTables.loadNano(
                tablesURL: repoDir.appendingPathComponent(ModelNames.ChatterboxNano.tablesFile))
            let voice = try ChatterboxTables.loadVoice(
                voiceURL: repoDir.appendingPathComponent(
                    ModelNames.ChatterboxNano.defaultVoiceFile))
            try ChatterboxTables.validate(tables, voice: voice)
            try tokenizer.validate(embeddingRows: tables.textEmb.rows)
            return (tokenizer, tables, voice)
        }

        // Cache checks are existence-only, so a truncated/corrupt aux file
        // would otherwise fail every launch — drop and re-fetch once
        // (never in offline mode; see loadAuxWithRecovery).
        let (tokenizer, tables, voice) = try await ChatterboxMLSupport.loadAuxWithRecovery(
            repoDir: repoDir,
            auxFiles: ModelNames.ChatterboxNano.auxFiles,
            logger: logger,
            refetch: { try await ensureAuxAssets(repoDir: repoDir) },
            load: loadAux)

        return ChatterboxNanoModels(
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
        for relative in ModelNames.ChatterboxNano.auxFiles {
            let localURL = repoDir.appendingPathComponent(relative)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }
            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let remoteURL = try ModelRegistry.resolveModel(
                Repo.chatterboxNano.remotePath, relative)
            let data = try await AssetDownloader.fetchData(
                from: remoteURL, description: "chatterbox-nano \(relative)", logger: logger)
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
