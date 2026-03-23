@preconcurrency import CoreML
import Foundation
import OSLog

/// Actor-based store for the 6 Qwen3-TTS CoreML models.
///
/// Models:
/// - TextProjector — text token → embedding
/// - CodeEmbedder — codec token → embedding
/// - MultiCodeEmbedder — linearized codebook token → embedding
/// - CodeDecoder — 28-layer transformer with KV cache (generates CB0)
/// - MultiCodeDecoder — 5-layer transformer with KV cache (generates CB1-CB15)
/// - SpeechDecoder — codec frames → audio waveform
public actor Qwen3TtsModelStore {

    private let logger = AppLogger(category: "Qwen3TtsModelStore")

    private var textProjectorModel: MLModel?
    private var codeEmbedderModel: MLModel?
    private var multiCodeEmbedderModel: MLModel?
    private var codeDecoderModel: MLModel?
    private var multiCodeDecoderModel: MLModel?
    private var speechDecoderModel: MLModel?
    private var speakerEmbedding: [Float]?
    private var repoDirectory: URL?

    public init() {}

    /// Download models from HuggingFace and load them.
    public func loadIfNeeded() async throws {
        guard textProjectorModel == nil else { return }

        let repoDir = try await Qwen3TtsResourceDownloader.ensureModels()
        try await loadFromDirectory(repoDir)
    }

    /// Load all CoreML models from a local directory.
    public func loadFromDirectory(_ directory: URL) async throws {
        guard textProjectorModel == nil else { return }

        self.repoDirectory = directory

        logger.info("Loading Qwen3-TTS CoreML models from \(directory.path)...")

        // Embedding models and SpeechDecoder use CPU+GPU (float32)
        let f32Config = MLModelConfiguration()
        f32Config.computeUnits = .cpuAndGPU

        // CodeDecoder also uses CPU+GPU to prevent inf/NaN from ANE float16 overflow
        let allConfig = MLModelConfiguration()
        allConfig.computeUnits = .cpuAndGPU

        let loadStart = Date()

        textProjectorModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.textProjectorFile),
            config: f32Config, name: "TextProjector")
        codeEmbedderModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.codeEmbedderFile),
            config: f32Config, name: "CodeEmbedder")
        multiCodeEmbedderModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.multiCodeEmbedderFile),
            config: f32Config, name: "MultiCodeEmbedder")
        codeDecoderModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.codeDecoderFile),
            config: allConfig, name: "CodeDecoder")
        // MultiCodeDecoder MUST use CPU_ONLY (all other configs produce NaN)
        let mcdConfig = MLModelConfiguration()
        mcdConfig.computeUnits = .cpuOnly
        multiCodeDecoderModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.multiCodeDecoderFile),
            config: mcdConfig, name: "MultiCodeDecoder")
        speechDecoderModel = try loadModel(
            at: directory.appendingPathComponent(ModelNames.Qwen3TTS.speechDecoderFile),
            config: f32Config, name: "SpeechDecoder")

        // Load optional speaker embedding
        let speakerURL = directory.appendingPathComponent(
            ModelNames.Qwen3TTS.speakerEmbeddingFile)
        if FileManager.default.fileExists(atPath: speakerURL.path) {
            speakerEmbedding = try loadNumpyFloatArray(from: speakerURL)
            logger.info("Loaded speaker embedding (\(speakerEmbedding!.count) floats)")
        }

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All Qwen3-TTS models loaded in \(String(format: "%.2f", elapsed))s")
    }

    // MARK: - Accessors

    public func textProjector() throws -> MLModel {
        guard let model = textProjectorModel else {
            throw TTSError.modelNotFound("TextProjector model not loaded")
        }
        return model
    }

    public func codeEmbedder() throws -> MLModel {
        guard let model = codeEmbedderModel else {
            throw TTSError.modelNotFound("CodeEmbedder model not loaded")
        }
        return model
    }

    public func multiCodeEmbedder() throws -> MLModel {
        guard let model = multiCodeEmbedderModel else {
            throw TTSError.modelNotFound("MultiCodeEmbedder model not loaded")
        }
        return model
    }

    public func codeDecoder() throws -> MLModel {
        guard let model = codeDecoderModel else {
            throw TTSError.modelNotFound("CodeDecoder model not loaded")
        }
        return model
    }

    public func multiCodeDecoder() throws -> MLModel {
        guard let model = multiCodeDecoderModel else {
            throw TTSError.modelNotFound("MultiCodeDecoder model not loaded")
        }
        return model
    }

    public func speechDecoder() throws -> MLModel {
        guard let model = speechDecoderModel else {
            throw TTSError.modelNotFound("SpeechDecoder model not loaded")
        }
        return model
    }

    public func speaker() -> [Float]? {
        speakerEmbedding
    }

    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw TTSError.modelNotFound("Qwen3-TTS repository not loaded")
        }
        return dir
    }

    public var isLoaded: Bool {
        textProjectorModel != nil && codeEmbedderModel != nil
            && multiCodeEmbedderModel != nil && codeDecoderModel != nil
            && multiCodeDecoderModel != nil && speechDecoderModel != nil
    }

    public func reset() {
        textProjectorModel = nil
        codeEmbedderModel = nil
        multiCodeEmbedderModel = nil
        codeDecoderModel = nil
        multiCodeDecoderModel = nil
        speechDecoderModel = nil
        speakerEmbedding = nil
        repoDirectory = nil
    }

    // MARK: - Private Helpers

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

        guard data.count >= 12 else {
            throw TTSError.processingFailed("Invalid NPY file: too small")
        }

        let magic = data.prefix(6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw TTSError.processingFailed("Invalid NPY magic number")
        }

        let majorVersion = data[6]

        let headerLen: Int
        let headerOffset: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerOffset = 10
        } else {
            headerLen =
                Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerOffset = 12
        }

        let dataOffset = headerOffset + headerLen

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
}
