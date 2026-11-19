#if TTS && arch(arm64)
@preconcurrency import CoreML
import Foundation

/// LFM3-Audio speech constants: 12 codebooks of 2016 codes (+ EOS code 2016) per 80 ms frame (12.5 Hz).
enum LFM3AudioSpeechConfig {
    static let codebooks = 12
    static let audioVocab = 2017
    static let eosCode: Int32 = 2016
    static let depthHidden = 1024
    /// Codebook-sum embedding table rows: 12 x 2017.
    static let codebookRows = codebooks * audioVocab
    static let codecFrames = [1, 4, 25]
}

/// Speech sampling of LFM3-Audio-2B (generation_config: temperature 1.0, top-k 128, presence penalty 1.1 over
/// the last 4096 frames).
struct LFM3AudioSpeechSampling: Sendable {
    var temperature: Float = 1.0
    var topK = 128
    var penalty: Float = 1.1
    var window = 4096
}

/// LiquidCodec synthesis head (host fp32): lin 512 -> 1282, exp / polar, windowed irfft(1280) as one sgemm with a
/// `[1282, 1280]` real basis (window and 1/N folded in).
struct LFM3AudioCodecHead: Sendable {
    static let latent = 512
    static let projection = 1282
    static let nFFT = 1280
    let linearWeight: [Float]  // [1282, 512]
    let linearBias: [Float]  // [1282]
    let basis: [Float]  // [1282, 1280]
    let window: [Float]  // [1280]

    init(url: URL) throws {
        let nW = Self.projection * Self.latent
        let all = try LFM3AudioTables.float32(url, count: nW + Self.projection + Self.nFFT)
        linearWeight = Array(all[0..<nW])
        linearBias = Array(all[nW..<(nW + Self.projection)])
        let window = Array(all[(nW + Self.projection)...])
        self.window = window
        // x[n] = w[n]/N [Re X0 + (-1)^n Re X_{N/2} + 2 sum_{k=1}^{N/2-1} (Re Xk cos - Im Xk sin)]
        let n = Self.nFFT
        let half = n / 2 + 1
        var b = [Float](repeating: 0, count: 2 * half * n)
        for k in 0..<half {
            let weight = (k == 0 || k == half - 1) ? 1.0 : 2.0
            for t in 0..<n {
                let angle = 2.0 * Double.pi * Double((k * t) % n) / Double(n)
                let w = Double(window[t]) / Double(n)
                b[k * n + t] = Float(w * weight * cos(angle))
                if k != 0 && k != half - 1 { b[(half + k) * n + t] = Float(-w * 2.0 * sin(angle)) }
            }
        }
        basis = b
    }
}

/// Loaded TTS components.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
struct LFM3AudioSpeechModels: Sendable {
    enum Depth: Sendable {
        /// Core ML per-pass functions + the `[12, 2017, 1024]` code embeddings and `[1024]` mask embedding (fp16).
        case coreML(depthIn: MLModel, depthPass: MLModel, tables: Data)
        /// Core AI bundle with the fused `depth_frame` function, loaded by the manager (macOS 27 / iOS 27).
        case coreAI(bundle: URL)
    }

    let depth: Depth
    let codec: [Int: MLModel]
    /// `[24204, 2048]` fp16: LM input embedding of each (codebook, code).
    let codebook: Data
    let codecHead: LFM3AudioCodecHead

    /// - Parameter coreAIDepth: Core AI depth bundle; when set the Core ML depth decoder is not loaded.
    static func load(
        from directory: URL, coreAIDepth: URL?, computeUnits: LFM3AudioComputeUnits
    ) throws -> LFM3AudioSpeechModels {
        let names = ModelNames.LFM3Audio.self
        let c = LFM3AudioSpeechConfig.self
        var codec: [Int: MLModel] = [:]
        for frames in c.codecFrames {
            codec[frames] = try LFM3AudioModels.loadModel(
                names.codec, from: directory, function: "f\(frames)", units: computeUnits.codec)
        }
        let depth: Depth
        if let coreAIDepth {
            depth = .coreAI(bundle: coreAIDepth)
        } else {
            depth = .coreML(
                depthIn: try LFM3AudioModels.loadModel(
                    names.depthDecoder, from: directory, function: "depth_in", units: computeUnits.depthDecoder),
                depthPass: try LFM3AudioModels.loadModel(
                    names.depthDecoder, from: directory, function: "depth_pass", units: computeUnits.depthDecoder),
                tables: try LFM3AudioTables.mapped(
                    directory.appendingPathComponent(names.depthTablesFile),
                    expectedBytes: (c.codebookRows * c.depthHidden + c.depthHidden) * 2))
        }
        return LFM3AudioSpeechModels(
            depth: depth, codec: codec,
            codebook: try LFM3AudioTables.mapped(
                directory.appendingPathComponent(names.codebookFile),
                expectedBytes: c.codebookRows * LFM3AudioConfig.hiddenSize * 2),
            codecHead: try LFM3AudioCodecHead(url: directory.appendingPathComponent(names.codecHeadFile)))
    }
}
#endif
