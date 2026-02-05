@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

/// Actor-based store for Qwen3-TTS CoreML models.
///
/// Manages loading and storing of the CoreML models:
/// - LM Prefill (qwen3_tts_lm_prefill_v9.mlpackage)
/// - LM Decode V10 (qwen3_tts_lm_decode_v10.mlpackage)
/// - CP Prefill (qwen3_tts_cp_prefill.mlpackage)
/// - CP Decode (qwen3_tts_cp_decode.mlpackage)
/// - Audio Decoder (qwen3_tts_decoder_10s.mlpackage)
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

    /// Load all CoreML models from a local directory.
    ///
    /// - Parameter directory: Directory containing the .mlpackage files.
    public func loadFromDirectory(_ directory: URL) async throws {
        guard prefillModel == nil else { return }

        self.repoDirectory = directory

        logger.info("Loading Qwen3-TTS CoreML models from \(directory.path)...")

        // Use CPU+GPU for float32 precision (matches Python reference)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        // Load LM prefill model
        let prefillURL = directory.appendingPathComponent("qwen3_tts_lm_prefill_v9.mlpackage")
        prefillModel = try loadModel(at: prefillURL, config: config, name: "LM prefill")

        // Load LM decode V10 model (takes CB0 + CB1-15 as inputs)
        let decodeURL = directory.appendingPathComponent("qwen3_tts_lm_decode_v10.mlpackage")
        decodeModel = try loadModel(at: decodeURL, config: config, name: "LM decode V10")

        // Load code predictor prefill model
        let cpPrefillURL = directory.appendingPathComponent("qwen3_tts_cp_prefill.mlpackage")
        cpPrefillModel = try loadModel(at: cpPrefillURL, config: config, name: "CP prefill")

        // Load code predictor decode model
        let cpDecodeURL = directory.appendingPathComponent("qwen3_tts_cp_decode.mlpackage")
        cpDecodeModel = try loadModel(at: cpDecodeURL, config: config, name: "CP decode")

        // Load audio decoder model
        let decoderURL = directory.appendingPathComponent("qwen3_tts_decoder_10s.mlpackage")
        audioDecoderModel = try loadModel(at: decoderURL, config: config, name: "audio decoder")

        // Load code predictor embedding tables
        let cpEmbedURL = directory.appendingPathComponent("cp_embeddings.npy")
        if FileManager.default.fileExists(atPath: cpEmbedURL.path) {
            cpEmbeddings = try loadNumpyFloat3DArray(from: cpEmbedURL, shape: (15, 2048, 1024))
            logger.info("Loaded CP embeddings [15, 2048, 1024]")
        }

        // Load speaker embedding if available
        let speakerURL = directory.appendingPathComponent("speaker_embedding_official.npy")
        if FileManager.default.fileExists(atPath: speakerURL.path) {
            speakerEmbedding = try loadNumpyFloatArray(from: speakerURL)
            logger.info("Loaded speaker embedding")
        }

        // Load TTS embeddings if available
        let bosURL = directory.appendingPathComponent("tts_bos_embed.npy")
        let padURL = directory.appendingPathComponent("tts_pad_embed.npy")
        let eosURL = directory.appendingPathComponent("tts_eos_embed.npy")
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

    /// Load a CoreML model, compiling mlpackage if needed.
    private func loadModel(
        at url: URL,
        config: MLModelConfiguration,
        name: String
    ) throws -> MLModel {
        let ext = url.pathExtension

        if ext == "mlpackage" {
            // Need to compile mlpackage first
            logger.info("Compiling \(name) model...")
            let compiledURL = try MLModel.compileModel(at: url)
            let model = try MLModel(contentsOf: compiledURL, configuration: config)
            logger.info("Loaded \(name) model (compiled)")
            return model
        } else {
            // Already compiled (.mlmodelc)
            let model = try MLModel(contentsOf: url, configuration: config)
            logger.info("Loaded \(name) model")
            return model
        }
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
