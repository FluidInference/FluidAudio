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
