import Foundation

/// On-disk schema of the upstream `tts.json` config.
///
/// The reference Swift CLI only consumes two scalars from each section
/// (`sample_rate`, `base_chunk_size`, `chunk_compress_factor`, `latent_dim`).
/// The Swift port duplicates those fields here so a downloaded `tts.json`
/// can override the compile-time defaults in `Supertonic3Constants` when
/// FluidInference republishes a tuned variant.
public struct Supertonic3Config: Codable, Sendable {

    public struct AEConfig: Codable, Sendable {
        public let sampleRate: Int
        public let baseChunkSize: Int

        public init(sampleRate: Int, baseChunkSize: Int) {
            self.sampleRate = sampleRate
            self.baseChunkSize = baseChunkSize
        }

        private enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case baseChunkSize = "base_chunk_size"
        }
    }

    public struct TTLConfig: Codable, Sendable {
        public let chunkCompressFactor: Int
        public let latentDim: Int

        public init(chunkCompressFactor: Int, latentDim: Int) {
            self.chunkCompressFactor = chunkCompressFactor
            self.latentDim = latentDim
        }

        private enum CodingKeys: String, CodingKey {
            case chunkCompressFactor = "chunk_compress_factor"
            case latentDim = "latent_dim"
        }
    }

    public let ae: AEConfig
    public let ttl: TTLConfig

    public init(ae: AEConfig, ttl: TTLConfig) {
        self.ae = ae
        self.ttl = ttl
    }

    /// Fallback config that matches `Supertonic3Constants` — used when the
    /// caller cannot supply a `tts.json` (e.g. embedded resource scenarios).
    public static let defaults = Supertonic3Config(
        ae: .init(
            sampleRate: Supertonic3Constants.sampleRate,
            baseChunkSize: Supertonic3Constants.baseChunkSize),
        ttl: .init(
            chunkCompressFactor: Supertonic3Constants.chunkCompressFactor,
            latentDim: Supertonic3Constants.latentDim))
}

/// Weight-quantization level for the VectorEstimator stage. All three are
/// post-training, weight-only compressions that leave placement and speed
/// unchanged — they only shrink the on-disk / in-memory model:
///   - `.int8` — linear per-channel symmetric int8 (≈64 MB, transparent).
///   - `.int6` — 6-bit k-means palettization (≈48 MB, very good).
///   - `.int4` — 4-bit k-means palettization (≈32 MB, perceptually clean).
public enum Supertonic3Quantization: String, Sendable, Equatable, CaseIterable {
    case int8
    case int6
    case int4
}

/// Selects which VectorEstimator build the pipeline downloads and runs.
///
/// VectorEstimator is the heaviest stage (run `totalSteps`× per utterance).
/// Two independent axes — weight precision (size) and shape mode (compute
/// device):
///
/// - `.fp16Dynamic` (default): the original FP16 RangeDim model. Dynamic
///   shapes ⇒ CPU/GPU; preserves pre-existing behavior.
/// - `.dynamic(q)`: a weight-quantized RangeDim model. Smaller download, same
///   placement (dynamic shapes cannot use the ANE).
/// - `.aneBucketed(q)`: fixed-length L∈{128,256,512} models that land ~94% on
///   the Neural Engine (~2.7× faster end-to-end). The synthesizer pads each
///   chunk's latent up to the smallest bucket ≥ its length. Per-chunk length is
///   bounded by the text chunker, so the 128 bucket covers the common case.
public enum Supertonic3VectorEstimator: Sendable, Equatable {
    case fp16Dynamic
    case dynamic(Supertonic3Quantization)
    case aneBucketed(Supertonic3Quantization)

    /// Default: ANE-bucketed int4 — ~94% on the ANE, ~2.7× faster end-to-end,
    /// 4-bit k-means palettization that is perceptually clean. The historical
    /// fp16 dynamic build stays available via `--ve-variant fp16`.
    public static let `default`: Supertonic3VectorEstimator = .aneBucketed(.int4)

    /// `nil` for FP16; the rawValue (`"int8"`/`"int6"`/`"int4"`) otherwise.
    var precisionSuffix: String? {
        switch self {
        case .fp16Dynamic: return nil
        case .dynamic(let q), .aneBucketed(let q): return q.rawValue
        }
    }

    var isBucketed: Bool {
        if case .aneBucketed = self { return true }
        return false
    }

    /// Variant token passed to `ModelHub.download` / `getRequiredModelNames`
    /// so only the selected VectorEstimator file(s) are fetched.
    var downloadVariant: String? {
        switch self {
        case .fp16Dynamic: return nil
        case .dynamic(let q): return "dyn-\(q.rawValue)"
        case .aneBucketed(let q): return "ane-\(q.rawValue)"
        }
    }
}

/// The 10 built-in Supertonic-3 voice styles published at
/// `FluidInference/supertonic-3-coreml/voice_styles/`: female `f1`-`f5`,
/// male `m1`-`m5`. Fetch one with
/// `Supertonic3ResourceDownloader.downloadVoiceStyle(_:)` (or download + decode
/// in one call via `loadVoiceStyle(_:)`). Custom styles can still be supplied
/// as any file via `Supertonic3VoiceStyle.load(from:)`.
public enum Supertonic3Voice: String, CaseIterable, Sendable {
    case f1 = "F1"
    case f2 = "F2"
    case f3 = "F3"
    case f4 = "F4"
    case f5 = "F5"
    case m1 = "M1"
    case m2 = "M2"
    case m3 = "M3"
    case m4 = "M4"
    case m5 = "M5"

    /// Default voice (`M1`), the style shipped before the others were added.
    public static let `default`: Supertonic3Voice = .m1

    /// Repo-relative path of this voice's style JSON
    /// (e.g. `voice_styles/F3.json`).
    public var fileName: String { "voice_styles/\(rawValue).json" }

    /// Parse a voice name case-insensitively, e.g. `"f3"` or `"M1"`.
    /// Returns `nil` for unknown names.
    public init?(name: String) {
        self.init(rawValue: name.uppercased())
    }
}

/// On-disk schema of a Supertonic-3 voice style JSON file (the `M1` /
/// `F1` / etc. presets shipped under `assets/voice_styles/` in the
/// reference repo).
///
/// `style_ttl` feeds the text encoder + vector estimator; `style_dp` feeds
/// the duration predictor. Both components encode the same 3-D tensor
/// `[1, D1, D2]` as a nested array; `dims` records the original shape so
/// the loader can validate against the model's expected input shape.
public struct Supertonic3VoiceStyleData: Codable, Sendable {

    public struct Component: Codable, Sendable {
        public let data: [[[Float]]]
        public let dims: [Int]
        public let type: String

        public init(data: [[[Float]]], dims: [Int], type: String) {
            self.data = data
            self.dims = dims
            self.type = type
        }
    }

    public let styleTtl: Component
    public let styleDp: Component

    public init(styleTtl: Component, styleDp: Component) {
        self.styleTtl = styleTtl
        self.styleDp = styleDp
    }

    private enum CodingKeys: String, CodingKey {
        case styleTtl = "style_ttl"
        case styleDp = "style_dp"
    }
}

/// Decoded voice style ready to bind into CoreML feature dictionaries.
///
/// Both tensors are flattened row-major matching the dims `[bsz, D1, D2]`
/// stored on disk. The synthesizer wraps these into `MLMultiArray` instances
/// at call time so the same `Supertonic3VoiceStyle` can be shared across
/// many synthesis calls without re-parsing the JSON.
public struct Supertonic3VoiceStyle: Sendable {
    public let name: String
    public let ttlValues: [Float]
    public let ttlDims: [Int]
    public let dpValues: [Float]
    public let dpDims: [Int]

    public init(
        name: String,
        ttlValues: [Float],
        ttlDims: [Int],
        dpValues: [Float],
        dpDims: [Int]
    ) {
        self.name = name
        self.ttlValues = ttlValues
        self.ttlDims = ttlDims
        self.dpValues = dpValues
        self.dpDims = dpDims
    }

    /// Decode a JSON-encoded voice style file into the flattened
    /// representation expected by the synthesizer.
    public static func load(from url: URL) throws -> Supertonic3VoiceStyle {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw Supertonic3Error.voiceStyleLoadFailed(
                path: url.path, underlying: "\(error)")
        }
        let decoded: Supertonic3VoiceStyleData
        do {
            decoded = try JSONDecoder().decode(Supertonic3VoiceStyleData.self, from: data)
        } catch {
            throw Supertonic3Error.voiceStyleLoadFailed(
                path: url.path, underlying: "decode: \(error)")
        }

        let expectedTtl = [1, Supertonic3Constants.ttlStyleTokens, Supertonic3Constants.ttlStyleDim]
        if decoded.styleTtl.dims != expectedTtl {
            throw Supertonic3Error.voiceStyleShapeMismatch(
                component: "style_ttl", expected: expectedTtl, got: decoded.styleTtl.dims)
        }
        let expectedDp = [1, Supertonic3Constants.dpStyleTokens, Supertonic3Constants.dpStyleDim]
        if decoded.styleDp.dims != expectedDp {
            throw Supertonic3Error.voiceStyleShapeMismatch(
                component: "style_dp", expected: expectedDp, got: decoded.styleDp.dims)
        }

        let ttlFlat = flatten(decoded.styleTtl.data, dims: decoded.styleTtl.dims)
        let dpFlat = flatten(decoded.styleDp.data, dims: decoded.styleDp.dims)

        return Supertonic3VoiceStyle(
            name: url.deletingPathExtension().lastPathComponent,
            ttlValues: ttlFlat,
            ttlDims: decoded.styleTtl.dims,
            dpValues: dpFlat,
            dpDims: decoded.styleDp.dims)
    }

    private static func flatten(_ data: [[[Float]]], dims: [Int]) -> [Float] {
        var out: [Float] = []
        let totalCount = dims.reduce(1, *)
        out.reserveCapacity(totalCount)
        for plane in data {
            for row in plane {
                out.append(contentsOf: row)
            }
        }
        return out
    }
}
