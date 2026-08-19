@preconcurrency import CoreML
import Foundation

/// Downloads and loads the NeuTTS-2E CoreML assets from
/// `FluidInference/neutts-2e-coreml` (compiled `.mlmodelc` bundles at the
/// repo root, plus `tokenizer.json` and `samples/<speaker>.json`).
///
/// - Note: Beta — this is a beta model conversion; API, model artifacts, and accuracy may change.
@available(macOS 15.0, iOS 18.0, *)
struct NeuTtsModels: Sendable {

    private static let logger = AppLogger(category: "NeuTtsModels")

    let prefill: MLModel
    let decode: MLModel
    let codec: MLModel
    let tokenizer: NeuTtsBpeTokenizer
    let repoDir: URL

    static func load(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> NeuTtsModels {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.neuTts.folderName)

        let requiredPaths =
            ModelNames.NeuTts.requiredModels.map { $0 }
            + [ModelNames.NeuTts.tokenizerFile]
            + NeuTtsConstants.speakers.map { "samples/\($0).json" }
        let allPresent = requiredPaths.allSatisfy {
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if !allPresent {
            logger.info("Downloading NeuTTS-2E CoreML assets from HuggingFace…")
            try await ModelHub.download(
                .neuTts, to: modelsRoot,
                progressHandler: progressHandler)
            // The repo walk only descends into the required .mlmodelc bundles;
            // the tiny speaker-reference JSONs live under samples/ and are
            // fetched individually.
            try await ensureSpeakerReferences(repoDir: repoDir)
        } else {
            logger.info("NeuTTS-2E assets found in cache at \(repoDir.path)")
        }

        // LM runs GPU-dominant (the ANE compiler rejects the decode graph and
        // E5RT falls back); the codec is ~2× faster on the Neural Engine.
        func makeConfig(_ units: MLComputeUnits) -> MLModelConfiguration {
            let config = MLModelConfiguration()
            config.computeUnits = units
            return config
        }

        let prefill = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.NeuTts.prefillFile),
            configuration: makeConfig(.all))
        // .cpuAndGPU: the ANE compiler rejects this graph (ANECCompile -14);
        // requesting .all just adds a noisy E5RT failure log before the same
        // GPU fallback (identical measured speed).
        let decode = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.NeuTts.decodeFile),
            configuration: makeConfig(.cpuAndGPU))
        let codec = try await MLModel.load(
            contentsOf: repoDir.appendingPathComponent(ModelNames.NeuTts.codecFile),
            configuration: makeConfig(.cpuAndNeuralEngine))

        let tokenizer = try NeuTtsBpeTokenizer(
            tokenizerJsonURL: repoDir.appendingPathComponent(ModelNames.NeuTts.tokenizerFile))

        return NeuTtsModels(
            prefill: prefill,
            decode: decode,
            codec: codec,
            tokenizer: tokenizer,
            repoDir: repoDir)
    }

    func speakerReference(_ name: String) throws -> NeuTtsPrompt.SpeakerReference {
        guard NeuTtsConstants.speakers.contains(name) else {
            throw NeuTtsPrompt.PromptError.unknownSpeaker(name)
        }
        let url = repoDir.appendingPathComponent("samples/\(name).json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(NeuTtsPrompt.SpeakerReference.self, from: data)
    }

    private static func ensureSpeakerReferences(repoDir: URL) async throws {
        let samplesDir = repoDir.appendingPathComponent("samples")
        try FileManager.default.createDirectory(at: samplesDir, withIntermediateDirectories: true)
        for speaker in NeuTtsConstants.speakers {
            let localURL = samplesDir.appendingPathComponent("\(speaker).json")
            if FileManager.default.fileExists(atPath: localURL.path) { continue }
            let remoteURL = try ModelRegistry.resolveModel(
                Repo.neuTts.remotePath, "samples/\(speaker).json")
            let data = try await AssetDownloader.fetchData(
                from: remoteURL, description: "neutts samples/\(speaker).json", logger: logger)
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
