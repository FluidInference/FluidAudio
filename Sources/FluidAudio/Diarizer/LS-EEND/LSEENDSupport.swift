import CoreML
import Foundation

public enum LSEENDError: Error, LocalizedError {
    case invalidMetadata(String)
    case invalidMatrixShape(String)
    case unsupportedAudio(String)
    case modelPredictionFailed(String)
    case missingFeature(String)
    case invalidPath(String)

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
        }
    }
}

public struct LSEENDMatrix: Sendable, Equatable {
    public let rows: Int
    public let columns: Int
    public var values: [Float]

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

    public init(validatingRows rows: Int, columns: Int, values: [Float]) {
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    public static func zeros(rows: Int, columns: Int) -> LSEENDMatrix {
        LSEENDMatrix(validatingRows: rows, columns: columns, values: [Float](repeating: 0, count: max(0, rows * columns)))
    }

    public static func empty(columns: Int) -> LSEENDMatrix {
        zeros(rows: 0, columns: columns)
    }

    public var isEmpty: Bool {
        rows == 0 || columns == 0 || values.isEmpty
    }

    public subscript(row: Int, column: Int) -> Float {
        get {
            values[(row * columns) + column]
        }
        set {
            values[(row * columns) + column] = newValue
        }
    }

    public func row(_ index: Int) -> ArraySlice<Float> {
        let start = index * columns
        return values[start..<(start + columns)]
    }

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

    public func rowMajorRows() -> [[Float]] {
        guard rows > 0, columns > 0 else { return [] }
        return (0..<rows).map { Array(row($0)) }
    }

    public func appendingRows(_ other: LSEENDMatrix) -> LSEENDMatrix {
        if isEmpty { return other }
        if other.isEmpty { return self }
        precondition(columns == other.columns, "Column count mismatch")
        return LSEENDMatrix(validatingRows: rows + other.rows, columns: columns, values: values + other.values)
    }

    public func droppingFirstRows(_ count: Int) -> LSEENDMatrix {
        let clipped = max(0, min(count, rows))
        guard clipped > 0 else { return self }
        let start = clipped * columns
        return LSEENDMatrix(validatingRows: rows - clipped, columns: columns, values: Array(values[start..<values.count]))
    }

    public func slicingRows(start: Int, end: Int) -> LSEENDMatrix {
        let lower = max(0, min(start, rows))
        let upper = max(lower, min(end, rows))
        guard lower < upper else { return .empty(columns: columns) }
        let slice = Array(values[(lower * columns)..<(upper * columns)])
        return LSEENDMatrix(validatingRows: upper - lower, columns: columns, values: slice)
    }

    public func applyingSigmoid() -> LSEENDMatrix {
        guard !values.isEmpty else { return self }
        var output = values
        for index in output.indices {
            output[index] = 1.0 / (1.0 + expf(-values[index]))
        }
        return LSEENDMatrix(validatingRows: rows, columns: columns, values: output)
    }
}

public struct LSEENDInferenceResult: Sendable {
    public let logits: LSEENDMatrix
    public let probabilities: LSEENDMatrix
    public let fullLogits: LSEENDMatrix
    public let fullProbabilities: LSEENDMatrix
    public let frameHz: Double
    public let durationSeconds: Double
}

public struct LSEENDStreamingUpdate: Sendable {
    public var startFrame: Int
    public var logits: LSEENDMatrix
    public var probabilities: LSEENDMatrix
    public var previewStartFrame: Int
    public var previewLogits: LSEENDMatrix
    public var previewProbabilities: LSEENDMatrix
    public var frameHz: Double
    public var durationSeconds: Double
    public var totalEmittedFrames: Int
}

public struct LSEENDStreamingProgress: Sendable, Codable {
    public let chunkIndex: Int
    public let bufferSeconds: Double
    public let numFramesEmitted: Int
    public let totalFramesEmitted: Int
    public let flush: Bool
}

public struct LSEENDStreamingSimulationResult: Sendable {
    public let result: LSEENDInferenceResult
    public let updates: [LSEENDStreamingProgress]
}

public enum LSEENDModelVariant: String, CaseIterable, Identifiable, Sendable {
    case ami = "AMI"
    case callhome = "CALLHOME"
    case dihard2 = "DIHARD II"
    case dihard3 = "DIHARD III"

    public var id: String { rawValue }

    public var artifactStem: String {
        switch self {
        case .ami:
            return "ls_eend_ami_step"
        case .callhome:
            return "ls_eend_callhome_step"
        case .dihard2:
            return "ls_eend_dih2_step"
        case .dihard3:
            return "ls_eend_dih3_step"
        }
    }

    public var checkpointName: String {
        switch self {
        case .ami:
            return "ls_eend_ami_allspk_model.ckpt"
        case .callhome:
            return "ls_eend_ch_allspk_model.ckpt"
        case .dihard2:
            return "ls_eend_dih2_allspk_model.ckpt"
        case .dihard3:
            return "ls_eend_dih3_allspk_model.ckpt"
        }
    }

    public var configName: String {
        switch self {
        case .ami:
            return "spk_onl_conformer_retention_enc_dec_nonautoreg_ami_infer.yaml"
        case .callhome:
            return "spk_onl_conformer_retention_enc_dec_nonautoreg_callhome_infer.yaml"
        case .dihard2:
            return "spk_onl_conformer_retention_enc_dec_nonautoreg_dihard2_infer.yaml"
        case .dihard3:
            return "spk_onl_conformer_retention_enc_dec_nonautoreg_dihard3_infer.yaml"
        }
    }
}

public struct LSEENDModelDescriptor: Sendable {
    public let variant: LSEENDModelVariant
    public let modelURL: URL
    public let metadataURL: URL
    public let checkpointURL: URL
    public let configURL: URL

    public init(
        variant: LSEENDModelVariant,
        modelURL: URL,
        metadataURL: URL,
        checkpointURL: URL,
        configURL: URL
    ) {
        self.variant = variant
        self.modelURL = modelURL
        self.metadataURL = metadataURL
        self.checkpointURL = checkpointURL
        self.configURL = configURL
    }

    public static func defaultDescriptor(for variant: LSEENDModelVariant) -> LSEENDModelDescriptor {
        let root = LSEENDWorkspace.rootURL
        let artifacts = root.appendingPathComponent("artifacts/coreml", isDirectory: true)
        return LSEENDModelDescriptor(
            variant: variant,
            modelURL: artifacts.appendingPathComponent("\(variant.artifactStem).mlpackage"),
            metadataURL: artifacts.appendingPathComponent("\(variant.artifactStem).json"),
            checkpointURL: root.appendingPathComponent(variant.checkpointName),
            configURL: root.appendingPathComponent("conf/\(variant.configName)")
        )
    }
}

public struct LSEENDStateShapes: Decodable, Sendable {
    public let encRetKv: [Int]
    public let encRetScale: [Int]
    public let encConvCache: [Int]
    public let decRetKv: [Int]
    public let decRetScale: [Int]
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

public struct LSEENDModelMetadata: Decodable, Sendable {
    public let checkpoint: String?
    public let config: String?
    public let inputDim: Int
    public let fullOutputDim: Int
    public let realOutputDim: Int
    public let encoderLayers: Int
    public let decoderLayers: Int
    public let encoderDim: Int
    public let numHeads: Int
    public let keyDim: Int
    public let headDim: Int
    public let encoderConvCacheLen: Int
    public let topBufferLen: Int
    public let convDelay: Int
    public let maxNspks: Int
    public let frameHz: Double
    public let targetSampleRate: Int
    public let computePrecision: String?
    public let stateShapes: LSEENDStateShapes
    public let sampleRate: Int?
    public let winLength: Int?
    public let hopLength: Int?
    public let nFFT: Int?
    public let nMels: Int?
    public let contextRecp: Int?
    public let subsampling: Int?
    public let featType: String?

    enum CodingKeys: String, CodingKey {
        case checkpoint
        case config
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

    public var resolvedSampleRate: Int {
        sampleRate ?? targetSampleRate
    }

    public var resolvedWinLength: Int {
        winLength ?? 200
    }

    public var resolvedHopLength: Int {
        hopLength ?? 80
    }

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

    public var resolvedMelCount: Int {
        if let nMels {
            return nMels
        }
        let inferred = inputDim / max(1, (2 * resolvedContextRecp) + 1)
        return max(1, inferred)
    }

    public var resolvedContextRecp: Int {
        if let contextRecp {
            return contextRecp
        }
        let melCount = max(1, nMels ?? 23)
        return max(0, ((inputDim / melCount) - 1) / 2)
    }

    public var resolvedSubsampling: Int {
        if let subsampling {
            return subsampling
        }
        let denominator = Int(round(frameHz * Double(resolvedHopLength)))
        return max(1, resolvedSampleRate / max(1, denominator))
    }

    public var streamingLatencySeconds: Double {
        let fftSize = resolvedFFTSize
        return Double((fftSize / 2) + (resolvedContextRecp * resolvedHopLength) + (convDelay * resolvedSubsampling * resolvedHopLength))
            / Double(max(resolvedSampleRate, 1))
    }
}

public enum LSEENDWorkspace {
    public static let rootURL: URL = {
        if let override = ProcessInfo.processInfo.environment["LSEEND_WORKSPACE_ROOT"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        var url = URL(fileURLWithPath: #filePath)
        while url.lastPathComponent != "LS-EEND" && url.path != "/" {
            url.deleteLastPathComponent()
        }
        return url
    }()
}
