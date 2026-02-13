@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

/// Actor-based store for Qwen3-TTS CoreML models.
///
/// Manages loading and storing of the CoreML models:
/// - LM Prefill (qwen3_tts_lm_prefill_v9.mlmodelc)
/// - LM Decode V10 (qwen3_tts_lm_decode_v10.mlmodelc)
/// - CP Prefill (qwen3_tts_cp_prefill.mlmodelc)
/// - CP Decode (qwen3_tts_cp_decode.mlmodelc)
/// - Audio Decoder (qwen3_tts_decoder_10s.mlmodelc)
/// - CP Embeddings (cp_embeddings.npy) [15, 2048, 1024]
public actor Qwen3TtsModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "Qwen3TtsModelStore")

    private var prefillModel: MLModel?
    private var decodeModel: MLModel?
    private var cpPrefillModel: MLModel?
    private var cpDecodeModel: MLModel?
    private var audioDecoderModel: MLModel?
    private var speakerEmbedding: [Float]?
    private var ttsEmbeddings: (bos: [Float], pad: [Float], eos: [Float])?
    /// Code predictor embedding tables [15][2048][1024].
    private var cpEmbeddings: [[[Float]]]?
    private var repoDirectory: URL?

    public init() {}

    /// Download models from HuggingFace and load them.
    public func loadIfNeeded() async throws {
        guard prefillModel == nil else { return }

        let repoDir = try await Qwen3TtsResourceDownloader.ensureModels()
        try await loadFromDirectory(repoDir)
    }

    /// Load all CoreML models from a local directory.
    ///
    /// - Parameter directory: Directory containing the .mlmodelc bundles and .npy files.
    public func loadFromDirectory(_ directory: URL) async throws {
        guard prefillModel == nil else { return }

        self.repoDirectory = directory

        logger.info("Loading Qwen3-TTS CoreML models from \(directory.path)...")

        // Use CPU+GPU for float32 precision (matches Python reference)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        let modelFiles = [
            (ModelNames.Qwen3TTS.lmPrefillFile, "LM prefill"),
            (ModelNames.Qwen3TTS.lmDecodeFile, "LM decode"),
            (ModelNames.Qwen3TTS.cpPrefillFile, "CP prefill"),
            (ModelNames.Qwen3TTS.cpDecodeFile, "CP decode"),
            (ModelNames.Qwen3TTS.audioDecoderFile, "audio decoder"),
        ]

        var loadedModels: [MLModel] = []
        for (file, name) in modelFiles {
            let modelURL = directory.appendingPathComponent(file)
            let model = try loadModel(at: modelURL, config: config, name: name)
            loadedModels.append(model)
        }

        prefillModel = loadedModels[0]
        decodeModel = loadedModels[1]
        cpPrefillModel = loadedModels[2]
        cpDecodeModel = loadedModels[3]
        audioDecoderModel = loadedModels[4]

        // Load code predictor embedding tables
        let cpEmbedURL = directory.appendingPathComponent(ModelNames.Qwen3TTS.cpEmbeddingsFile)
        if FileManager.default.fileExists(atPath: cpEmbedURL.path) {
            cpEmbeddings = try loadNumpyFloat3DArray(from: cpEmbedURL, shape: (15, 2048, 1024))
            logger.info("Loaded CP embeddings [15, 2048, 1024]")
        }

        // Load speaker embedding if available
        let speakerURL = directory.appendingPathComponent(
            ModelNames.Qwen3TTS.speakerEmbeddingFile)
        if FileManager.default.fileExists(atPath: speakerURL.path) {
            speakerEmbedding = try loadNumpyFloatArray(from: speakerURL)
            logger.info("Loaded speaker embedding")
        }

        // Load TTS embeddings if available
        let bosURL = directory.appendingPathComponent(ModelNames.Qwen3TTS.ttsBosEmbedFile)
        let padURL = directory.appendingPathComponent(ModelNames.Qwen3TTS.ttsPadEmbedFile)
        let eosURL = directory.appendingPathComponent(ModelNames.Qwen3TTS.ttsEosEmbedFile)
        if FileManager.default.fileExists(atPath: bosURL.path),
            FileManager.default.fileExists(atPath: padURL.path),
            FileManager.default.fileExists(atPath: eosURL.path)
        {
            let bos = try loadNumpyFloatArray(from: bosURL)
            let pad = try loadNumpyFloatArray(from: padURL)
            let eos = try loadNumpyFloatArray(from: eosURL)
            ttsEmbeddings = (bos: bos, pad: pad, eos: eos)
            logger.info("Loaded TTS embeddings")
        }

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All Qwen3-TTS models loaded in \(String(format: "%.2f", elapsed))s")
    }

    /// The LM prefill model (context preparation).
    public func prefill() throws -> MLModel {
        guard let model = prefillModel else {
            throw TTSError.modelNotFound("Qwen3-TTS prefill model not loaded")
        }
        return model
    }

    /// The LM decode model (autoregressive CB0 generation).
    public func decode() throws -> MLModel {
        guard let model = decodeModel else {
            throw TTSError.modelNotFound("Qwen3-TTS decode model not loaded")
        }
        return model
    }

    /// The code predictor prefill model (initial 2-token processing).
    public func cpPrefill() throws -> MLModel {
        guard let model = cpPrefillModel else {
            throw TTSError.modelNotFound("Qwen3-TTS CP prefill model not loaded")
        }
        return model
    }

    /// The code predictor decode model (per-step with KV cache).
    public func cpDecode() throws -> MLModel {
        guard let model = cpDecodeModel else {
            throw TTSError.modelNotFound("Qwen3-TTS CP decode model not loaded")
        }
        return model
    }

    /// The code predictor embedding tables [15][2048][1024].
    public func getCpEmbeddings() throws -> [[[Float]]] {
        guard let embeds = cpEmbeddings else {
            throw TTSError.modelNotFound("Qwen3-TTS CP embeddings not loaded")
        }
        return embeds
    }

    /// The audio decoder model (codes → waveform).
    public func audioDecoder() throws -> MLModel {
        guard let model = audioDecoderModel else {
            throw TTSError.modelNotFound("Qwen3-TTS audio decoder model not loaded")
        }
        return model
    }

    /// The pre-loaded speaker embedding.
    public func speaker() -> [Float]? {
        speakerEmbedding
    }

    /// The pre-loaded TTS embeddings (BOS, PAD, EOS).
    public func getTtsEmbeddings() -> (bos: [Float], pad: [Float], eos: [Float])? {
        ttsEmbeddings
    }

    /// The repository directory containing models.
    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw TTSError.modelNotFound("Qwen3-TTS repository not loaded")
        }
        return dir
    }

    /// Check if models are loaded.
    public var isLoaded: Bool {
        prefillModel != nil && decodeModel != nil && cpPrefillModel != nil
            && cpDecodeModel != nil && audioDecoderModel != nil
    }

    // MARK: - Private Helpers

    /// Load a CoreML model from mlmodelc bundle or mlpackage (compiling if needed).
    private func loadModel(
        at url: URL,
        config: MLModelConfiguration,
        name: String
    ) throws -> MLModel {
        let ext = url.pathExtension

        if ext == "mlpackage" {
            logger.info("Compiling \(name) model...")
            let compiledURL = try MLModel.compileModel(at: url)
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            logger.info("Loaded \(name) model (compiled)")
            return model
        }

        let model = try MLModel(contentsOf: url, configuration: config)
        logger.info("Loaded \(name) model")
        return model
    }

    /// Load a numpy .npy file containing float32 array.
    private func loadNumpyFloatArray(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)

        // NPY header: magic (6 bytes) + version (2 bytes) + header_len (2 or 4 bytes) + header
        guard data.count >= 10 else {
            throw TTSError.processingFailed("Invalid NPY file: too small")
        }

        // Check magic number
        let magic = data.prefix(6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw TTSError.processingFailed("Invalid NPY magic number")
        }

        // Get version
        let majorVersion = data[6]

        // Get header length based on version
        let headerLen: Int
        let headerOffset: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerOffset = 10
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerOffset = 12
        }

        let dataOffset = headerOffset + headerLen

        // Read float32 data
        let floatData = data.dropFirst(dataOffset)
        let count = floatData.count / 4
        var result = [Float](repeating: 0, count: count)

        floatData.withUnsafeBytes { buffer in
            let floatBuffer = buffer.bindMemory(to: Float.self)
            for i in 0..<count {
                result[i] = floatBuffer[i]
            }
        }

        return result
    }

    /// Load a numpy .npy file containing float32 3D array [d0, d1, d2].
    private func loadNumpyFloat3DArray(from url: URL, shape: (Int, Int, Int)) throws -> [[[Float]]] {
        let flat = try loadNumpyFloatArray(from: url)
        let (d0, d1, d2) = shape
        let expectedCount = d0 * d1 * d2

        guard flat.count == expectedCount else {
            throw TTSError.processingFailed(
                "NPY shape mismatch: expected \(expectedCount) floats, got \(flat.count)")
        }

        var result = [[[Float]]](
            repeating: [[Float]](
                repeating: [Float](repeating: 0, count: d2),
                count: d1
            ),
            count: d0
        )

        for i in 0..<d0 {
            for j in 0..<d1 {
                let offset = (i * d1 + j) * d2
                for k in 0..<d2 {
                    result[i][j][k] = flat[offset + k]
                }
            }
        }

        return result
    }
}
