import CoreML
import Foundation

/// Errors thrown by the LS-EEND inference pipeline.
public enum LSEENDError: Error, LocalizedError {
    /// The model metadata JSON is malformed or contains invalid values.
    case invalidMetadata(String)
    /// A matrix operation received dimensions that don't match.
    case invalidMatrixShape(String)
    /// The audio input format is unsupported (e.g. wrong sample rate or empty buffer).
    case unsupportedAudio(String)
    /// CoreML prediction failed during a model forward pass.
    case modelPredictionFailed(String)
    /// A required output feature is missing from the CoreML prediction result.
    case missingFeature(String)
    /// A file path could not be resolved.
    case invalidPath(String)
    /// The CoreML model could not be loaded or compiled.
    case modelLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let message):
            return "Invalid LS-EEND metadata: \(message)"
        case .invalidMatrixShape(let message):
            return "Invalid LS-EEND matrix shape: \(message)"
        case .unsupportedAudio(let message):
            return "Unsupported LS-EEND audio input: \(message)"
        case .modelPredictionFailed(let message):
            return "LS-EEND CoreML prediction failed: \(message)"
        case .missingFeature(let message):
            return "Missing CoreML feature: \(message)"
        case .invalidPath(let message):
            return "Invalid LS-EEND path: \(message)"
        case .modelLoadFailed(let message):
            return "Failed to load LS-EEND model: \(message)"
        }
    }
}

/// A row-major 2D matrix of `Float` values used throughout the LS-EEND pipeline.
///
/// Rows typically represent time frames and columns represent speaker channels or feature dimensions.
/// All operations return new matrices (value semantics); the underlying `values` array is stored flat
/// in row-major order.
public struct LSEENDMatrix: Sendable, Equatable {
    /// The number of rows (typically time frames).
    public let rows: Int
    /// The number of columns (typically speakers or feature dimensions).
    public let columns: Int
    /// Flat row-major storage. Element at `(row, col)` is at index `row * columns + col`.
    public var values: [Float]

    /// Creates a matrix with validated dimensions.
    ///
    /// - Parameters:
    ///   - rows: Number of rows (must be non-negative).
    ///   - columns: Number of columns (must be non-negative).
    ///   - values: Flat row-major values. Count must equal `rows * columns`.
    /// - Throws: ``LSEENDError/invalidMatrixShape(_:)`` if dimensions are negative or values count doesn't match.
    public init(rows: Int, columns: Int, values: [Float]) throws {
        guard rows >= 0, columns >= 0 else {
            throw LSEENDError.invalidMatrixShape("Negative dimensions are not supported.")
        }
        guard values.count == rows * columns else {
            throw LSEENDError.invalidMatrixShape(
                "Expected \(rows * columns) values, received \(values.count)."
            )
        }
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    /// Creates a matrix without validating that `values.count == rows * columns`.
    ///
    /// Use this initializer only when the caller has already guaranteed dimensional consistency.
    public init(validatingRows rows: Int, columns: Int, values: [Float]) {
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    /// Creates a zero-filled matrix with the given dimensions.
    public static func zeros(rows: Int, columns: Int) -> LSEENDMatrix {
        LSEENDMatrix(
            validatingRows: rows, columns: columns, values: [Float](repeating: 0, count: max(0, rows * columns)))
    }

    /// Creates an empty matrix (zero rows) with the given column count.
    public static func empty(columns: Int) -> LSEENDMatrix {
        zeros(rows: 0, columns: columns)
    }

    /// Whether the matrix contains no data (zero rows, zero columns, or empty values).
    public var isEmpty: Bool {
        rows == 0 || columns == 0 || values.isEmpty
    }

    /// Accesses the element at the given row and column.
    public subscript(row: Int, column: Int) -> Float {
        get {
            values[(row * columns) + column]
        }
        set {
            values[(row * columns) + column] = newValue
        }
    }

    /// Returns the values of a single row as an `ArraySlice`.
    public func row(_ index: Int) -> ArraySlice<Float> {
        let start = index * columns
        return values[start..<(start + columns)]
    }

    /// Returns a new matrix containing only the first `count` columns of each row.
    public func prefixingColumns(_ count: Int) -> LSEENDMatrix {
        let clipped = max(0, min(count, columns))
        guard clipped < columns else { return self }
        guard rows > 0 else { return .empty(columns: clipped) }
        var out = [Float](repeating: 0, count: rows * clipped)
        for rowIndex in 0..<rows {
            let srcStart = rowIndex * columns
            let dstStart = rowIndex * clipped
            for columnIndex in 0..<clipped {
                out[dstStart + columnIndex] = values[srcStart + columnIndex]
            }
        }
        return LSEENDMatrix(validatingRows: rows, columns: clipped, values: out)
    }

    /// Converts the matrix to an array of per-row arrays.
    public func rowMajorRows() -> [[Float]] {
        guard rows > 0, columns > 0 else { return [] }
        return (0..<rows).map { Array(row($0)) }
    }

    /// Returns a new matrix formed by appending the rows of `other` below this matrix.
    ///
    /// Both matrices must have the same column count.
    public func appendingRows(_ other: LSEENDMatrix) -> LSEENDMatrix {
        if isEmpty { return other }
        if other.isEmpty { return self }
        precondition(columns == other.columns, "Column count mismatch")
        return LSEENDMatrix(validatingRows: rows + other.rows, columns: columns, values: values + other.values)
    }

    /// Returns a new matrix with the first `count` rows removed.
    public func droppingFirstRows(_ count: Int) -> LSEENDMatrix {
        let clipped = max(0, min(count, rows))
        guard clipped > 0 else { return self }
        let start = clipped * columns
        return LSEENDMatrix(
            validatingRows: rows - clipped, columns: columns, values: Array(values[start..<values.count]))
    }

    /// Returns a submatrix containing rows in the half-open range `[start, end)`.
    ///
    /// Indices are clamped to valid bounds.
    public func slicingRows(start: Int, end: Int) -> LSEENDMatrix {
        let lower = max(0, min(start, rows))
        let upper = max(lower, min(end, rows))
        guard lower < upper else { return .empty(columns: columns) }
        let slice = Array(values[(lower * columns)..<(upper * columns)])
        return LSEENDMatrix(validatingRows: upper - lower, columns: columns, values: slice)
    }

    /// Returns a new matrix with the element-wise sigmoid function applied to all values.
    ///
    /// Converts logits to probabilities: `σ(x) = 1 / (1 + exp(-x))`.
    public func applyingSigmoid() -> LSEENDMatrix {
        guard !values.isEmpty else { return self }
        var output = values
        for index in output.indices {
            output[index] = 1.0 / (1.0 + expf(-values[index]))
        }
        return LSEENDMatrix(validatingRows: rows, columns: columns, values: output)
    }
}

/// The result of a complete (offline) LS-EEND inference pass.
///
/// Contains both "real" outputs (speaker tracks only) and "full" outputs
/// (including the two boundary tracks the model uses internally).
public struct LSEENDInferenceResult: Sendable {
    /// Speaker logits with boundary tracks removed. Shape: `[frames, realOutputDim]`.
    public let logits: LSEENDMatrix
    /// Speaker probabilities (sigmoid of ``logits``). Shape: `[frames, realOutputDim]`.
    public let probabilities: LSEENDMatrix
    /// Raw model logits including boundary tracks. Shape: `[frames, fullOutputDim]`.
    public let fullLogits: LSEENDMatrix
    /// Probabilities including boundary tracks (sigmoid of ``fullLogits``).
    public let fullProbabilities: LSEENDMatrix
    /// Output frame rate in Hz (e.g. 10.0 means one frame per 100 ms).
    public let frameHz: Double
    /// Duration of the input audio in seconds.
    public let durationSeconds: Double
}

/// An incremental update from a streaming LS-EEND session.
///
/// Each update contains two regions:
/// - **Committed** (`logits` / `probabilities`): frames that have passed through the
///   full encoder and are final.
/// - **Preview** (`previewLogits` / `previewProbabilities`): speculative frames decoded
///   by flushing pending state with zero-padded input. These will be refined by future audio.
public struct LSEENDStreamingUpdate: Sendable {
    /// Frame index where the committed region begins.
    public var startFrame: Int
    /// Committed speaker logits (boundary tracks removed).
    public var logits: LSEENDMatrix
    /// Committed speaker probabilities (sigmoid of ``logits``).
    public var probabilities: LSEENDMatrix
    /// Frame index where the preview region begins (equal to ``totalEmittedFrames``).
    public var previewStartFrame: Int
    /// Speculative speaker logits for frames not yet fully committed.
    public var previewLogits: LSEENDMatrix
    /// Speculative speaker probabilities (sigmoid of ``previewLogits``).
    public var previewProbabilities: LSEENDMatrix
    /// Output frame rate in Hz.
    public var frameHz: Double
    /// Total audio duration processed so far, in seconds.
    public var durationSeconds: Double
    /// Running total of committed frames emitted across all updates.
    public var totalEmittedFrames: Int
}

/// Progress information for a single chunk in a streaming simulation.
///
/// Used by ``LSEENDInferenceEngine/simulateStreaming(audioFileURL:chunkSeconds:)``
/// to report per-chunk statistics.
public struct LSEENDStreamingProgress: Sendable, Codable {
    /// One-based index of the chunk being processed.
    public let chunkIndex: Int
    /// Cumulative audio duration fed to the session, in seconds.
    public let bufferSeconds: Double
    /// Number of new committed frames emitted by this chunk.
    public let numFramesEmitted: Int
    /// Running total of committed frames across all chunks so far.
    public let totalFramesEmitted: Int
    /// Whether this entry represents the final flush (finalization).
    public let flush: Bool
}

/// Combined result of a streaming simulation, pairing the final inference output
/// with per-chunk progress entries.
public struct LSEENDStreamingSimulationResult: Sendable {
    /// The complete inference result after all chunks have been processed and finalized.
    public let result: LSEENDInferenceResult
    /// Per-chunk progress entries logged during the simulation.
    public let updates: [LSEENDStreamingProgress]
}

/// The LS-EEND model variant (dataset the model was trained on).
///
/// Maps directly to ``ModelNames/LSEEND/Variant``. Each variant corresponds
/// to a different training dataset and produces slightly different diarization behavior.
public typealias LSEENDVariant = ModelNames.LSEEND.Variant

extension LSEENDVariant: Identifiable {
    /// The dataset name used as a stable identifier (e.g. `"DIHARD III"`).
    public var id: String { rawValue }
}

/// Locates the CoreML model and metadata files for a specific LS-EEND variant.
///
/// Pass the descriptor to ``LSEENDInferenceEngine/init(descriptor:computeUnits:)``
/// or ``LSEENDDiarizer/initialize(descriptor:)`` to load the model.
public struct LSEENDModelDescriptor: Sendable {
    /// The model variant (training dataset).
    public let variant: LSEENDVariant
    /// URL of the compiled CoreML model (`.mlmodelc`) or model package (`.mlpackage`).
    public let modelURL: URL
    /// URL of the JSON metadata file describing model dimensions and audio parameters.
    public let metadataURL: URL

    private static let logger = AppLogger(category: "LSEENDModelDescriptor")

    /// Creates a descriptor from explicit file paths.
    ///
    /// - Parameters:
    ///   - variant: The model variant.
    ///   - modelURL: Path to the `.mlmodelc` or `.mlpackage` file.
    ///   - metadataURL: Path to the JSON metadata file.
    public init(
        variant: LSEENDVariant,
        modelURL: URL,
        metadataURL: URL
    ) {
        self.variant = variant
        self.modelURL = modelURL
        self.metadataURL = metadataURL
    }

    /// Download LS-EEND models from HuggingFace and construct a descriptor.
    ///
    /// Downloads all variant files on first call; subsequent calls use the cache.
    /// The returned descriptor points at the cached `.mlmodelc` and `.json` files.
    ///
    /// - Parameters:
    ///   - variant: The model variant to load (default: `.dihard3`).
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: Model compute units (.cpuOnly seems to be fastest for this model)
    /// - Returns: A descriptor ready for ``LSEENDInferenceEngine/init(descriptor:computeUnits:)``.
    public static func loadFromHuggingFace(
        variant: LSEENDVariant = .dihard3,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> LSEENDModelDescriptor {
        await SystemInfo.logOnce(using: logger)

        let directory =
            cacheDirectory
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models")

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let repo = Repo.lseend
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant.stem)

        let allModelsExist = requiredModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await DownloadUtils.downloadRepo(
                repo,
                to: directory,
                variant: variant.stem,
                progressHandler: progressHandler
            )
        }

        let modelURL = repoPath.appendingPathComponent(variant.modelFile)
        let metadataURL = repoPath.appendingPathComponent(variant.configFile)

        return LSEENDModelDescriptor(
            variant: variant,
            modelURL: modelURL,
            metadataURL: metadataURL
        )
    }
}

/// Tensor shapes for the six recurrent state buffers carried between LS-EEND inference steps.
///
/// Each property is an array of dimension sizes (e.g. `[layers, heads, keyDim, bufferLen]`).
/// These shapes are read from the model metadata JSON and used to allocate
/// zero-initialized `MLMultiArray` tensors at session start.
public struct LSEENDStateShapes: Decodable, Sendable {
    /// Encoder retention key-value cache shape.
    public let encRetKv: [Int]
    /// Encoder retention scale buffer shape.
    public let encRetScale: [Int]
    /// Encoder convolutional cache shape.
    public let encConvCache: [Int]
    /// Decoder retention key-value cache shape.
    public let decRetKv: [Int]
    /// Decoder retention scale buffer shape.
    public let decRetScale: [Int]
    /// Top-level buffer shape (used for cross-attention between encoder and decoder).
    public let topBuffer: [Int]

    enum CodingKeys: String, CodingKey {
        case encRetKv = "enc_ret_kv"
        case encRetScale = "enc_ret_scale"
        case encConvCache = "enc_conv_cache"
        case decRetKv = "dec_ret_kv"
        case decRetScale = "dec_ret_scale"
        case topBuffer = "top_buffer"
    }
}

/// Model configuration decoded from the LS-EEND metadata JSON file.
///
/// Contains all architectural parameters (layer counts, dimensions, state shapes)
/// and audio processing parameters (sample rate, FFT settings, mel bands).
/// Optional audio fields (`sampleRate`, `winLength`, etc.) fall back to defaults
/// via the `resolved*` computed properties.
public struct LSEENDModelMetadata: Decodable, Sendable {
    /// Input feature dimension per frame (nMels × splice window width).
    public let inputDim: Int
    /// Total output dimension including boundary tracks.
    public let fullOutputDim: Int
    /// Number of real speaker output tracks (excludes boundary tracks).
    public let realOutputDim: Int
    /// Number of encoder transformer layers.
    public let encoderLayers: Int
    /// Number of decoder transformer layers.
    public let decoderLayers: Int
    /// Hidden dimension of the encoder.
    public let encoderDim: Int
    /// Number of attention heads.
    public let numHeads: Int
    /// Key dimension per attention head.
    public let keyDim: Int
    /// Value dimension per attention head.
    public let headDim: Int
    /// Length of the encoder convolutional cache (number of frames buffered).
    public let encoderConvCacheLen: Int
    /// Length of the top-level cross-attention buffer.
    public let topBufferLen: Int
    /// Number of initial frames consumed before the decoder begins producing output.
    public let convDelay: Int
    /// Maximum number of speaker slots in the model output.
    public let maxNspks: Int
    /// Output frame rate in Hz (frames per second).
    public let frameHz: Double
    /// Target audio sample rate the model expects.
    public let targetSampleRate: Int
    /// Compute precision used during export (informational, e.g. `"float32"`).
    public let computePrecision: String?
    /// Tensor shapes for the six recurrent state buffers.
    public let stateShapes: LSEENDStateShapes
    /// Explicit sample rate override (falls back to ``targetSampleRate`` if nil).
    public let sampleRate: Int?
    /// STFT window length in samples (defaults to 200 if nil).
    public let winLength: Int?
    /// STFT hop length in samples (defaults to 80 if nil).
    public let hopLength: Int?
    /// FFT size (defaults to next power of 2 ≥ ``resolvedWinLength`` if nil).
    public let nFFT: Int?
    /// Number of mel filterbank channels (inferred from ``inputDim`` if nil).
    public let nMels: Int?
    /// Context receptive field half-width for splice-and-subsample (inferred if nil).
    public let contextRecp: Int?
    /// Subsampling factor for feature frames (inferred from frame rate if nil).
    public let subsampling: Int?
    /// Feature type identifier (informational, e.g. `"logmel_cmvn"`).
    public let featType: String?

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case fullOutputDim = "full_output_dim"
        case realOutputDim = "real_output_dim"
        case encoderLayers = "encoder_layers"
        case decoderLayers = "decoder_layers"
        case encoderDim = "encoder_dim"
        case numHeads = "num_heads"
        case keyDim = "key_dim"
        case headDim = "head_dim"
        case encoderConvCacheLen = "encoder_conv_cache_len"
        case topBufferLen = "top_buffer_len"
        case convDelay = "conv_delay"
        case maxNspks = "max_nspks"
        case frameHz = "frame_hz"
        case targetSampleRate = "target_sample_rate"
        case computePrecision = "compute_precision"
        case stateShapes = "state_shapes"
        case sampleRate = "sample_rate"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case nFFT = "n_fft"
        case nMels = "n_mels"
        case contextRecp = "context_recp"
        case subsampling
        case featType = "feat_type"
    }

    /// Effective sample rate: uses ``sampleRate`` if present, otherwise ``targetSampleRate``.
    public var resolvedSampleRate: Int {
        sampleRate ?? targetSampleRate
    }

    /// Effective STFT window length in samples (defaults to 200).
    public var resolvedWinLength: Int {
        winLength ?? 200
    }

    /// Effective STFT hop length in samples (defaults to 80).
    public var resolvedHopLength: Int {
        hopLength ?? 80
    }

    /// Effective FFT size. Uses ``nFFT`` if present, otherwise the smallest power of 2
    /// that is ≥ ``resolvedWinLength``.
    public var resolvedFFTSize: Int {
        if let nFFT {
            return nFFT
        }
        var fft = 1
        while fft < resolvedWinLength {
            fft <<= 1
        }
        return fft
    }

    /// Effective number of mel filterbank channels. Uses ``nMels`` if present,
    /// otherwise inferred from ``inputDim`` and ``resolvedContextRecp``.
    public var resolvedMelCount: Int {
        if let nMels {
            return nMels
        }
        let inferred = inputDim / max(1, (2 * resolvedContextRecp) + 1)
        return max(1, inferred)
    }

    /// Effective context receptive field half-width for the splice-and-subsample step.
    /// Uses ``contextRecp`` if present, otherwise inferred from ``inputDim`` and mel count.
    public var resolvedContextRecp: Int {
        if let contextRecp {
            return contextRecp
        }
        let melCount = max(1, nMels ?? 23)
        return max(0, ((inputDim / melCount) - 1) / 2)
    }

    /// Effective subsampling factor (how many STFT frames map to one model frame).
    /// Uses ``subsampling`` if present, otherwise derived from ``frameHz`` and hop length.
    public var resolvedSubsampling: Int {
        if let subsampling {
            return subsampling
        }
        let denominator = Int(round(frameHz * Double(resolvedHopLength)))
        return max(1, resolvedSampleRate / max(1, denominator))
    }

    /// Minimum streaming latency in seconds before the model can produce its first output frame.
    ///
    /// Accounts for the FFT center padding, context receptive field, and convolutional delay.
    public var streamingLatencySeconds: Double {
        let fftSize = resolvedFFTSize
        return Double(
            (fftSize / 2) + (resolvedContextRecp * resolvedHopLength)
                + (convDelay * resolvedSubsampling * resolvedHopLength))
            / Double(max(resolvedSampleRate, 1))
    }
}
