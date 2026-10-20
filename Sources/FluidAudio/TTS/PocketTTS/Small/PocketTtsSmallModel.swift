#if TTS
import Foundation

/// The small English Pocket TTS models Kyutai released with pocket-tts 4.0.0
/// (2026-10-20): 18.2M, 6.9M and 1.9M parameters, each with a reward
/// fine-tuned (`refl`) twin. The raw value is the upstream config name, which
/// is also the pack directory under `v4/` on `FluidInference/pocket-tts-coreml`.
///
/// They are not the 100M architecture: grouped-query attention, a `z_proj`
/// before the heads, a low-rank text table, one-step flow heads and a separate
/// 128-d Mimi decoder. Everything a host needs is read from the pack's
/// `arch.json` / `pocket_state.json` (`PocketTtsSmallArch`,
/// `PocketTtsSmallStateInfo`), so nothing here hard-codes a dimension.
public enum PocketTtsSmallModel: String, Sendable, CaseIterable {
    case english18m = "english_18m_2026-10"
    case english18mRefl = "english_18m_refl_2026-10"
    case english7m = "english_7m_2026-10"
    case english7mRefl = "english_7m_refl_2026-10"
    case english2m = "english_2m_2026-10"
    case english2mRefl = "english_2m_refl_2026-10"

    /// Pack directory under the `FluidInference/pocket-tts-coreml` repo root.
    public var repoSubdirectory: String { "v4/\(rawValue)" }

    /// Parse a CLI / config string: the upstream id, or a short alias
    /// (`18m`, `7m`, `2m`, `18m-refl`, ...).
    public init?(name: String) {
        let lowered = name.lowercased()
        if let exact = PocketTtsSmallModel(rawValue: lowered) {
            self = exact
            return
        }
        switch lowered.replacingOccurrences(of: "_", with: "-") {
        case "18m", "english-18m": self = .english18m
        case "18m-refl", "english-18m-refl": self = .english18mRefl
        case "7m", "english-7m": self = .english7m
        case "7m-refl", "english-7m-refl": self = .english7mRefl
        case "2m", "english-2m": self = .english2m
        case "2m-refl", "english-2m-refl": self = .english2mRefl
        default: return nil
        }
    }
}

/// `arch.json` of a small-model pack: every dimension and threshold the host
/// needs, written by the converter from the loaded PyTorch model.
public struct PocketTtsSmallArch: Codable, Sendable, Equatable {
    public let modelId: String
    public let dModel: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKvHeads: Int
    public let headDim: Int
    public let attnDim: Int
    public let kvDim: Int
    public let ffDim: Int
    /// Width of `transformer_out` (the `z_proj` output).
    public let zDim: Int
    /// Audio latent width (32).
    public let ldim: Int
    public let flowType: String
    public let ropeMaxPeriod: Double
    /// Stop once the EOS logit exceeds this (−1.0 for the small models).
    public let eosThreshold: Float
    /// Default sampling temperature (noise std = sqrt(temperature)).
    public let temperature: Float
    /// EOS is ignored during the first frames of a chunk.
    public let minFramesBeforeEos: Int
    public let textVocab: Int
    public let textRank: Int?
    public let insertBosBeforeVoice: Bool
    public let sampleRate: Int
    public let frameSize: Int
    public let upsampleStride: Int
    public let mimiDecoderDim: Int
    public let mimiDecoderLayers: Int
    public let mimiDecoderHeads: Int
    public let mimiDecoderHeadDim: Int
    public let mimiDecoderContext: Int

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case dModel = "d_model"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case numKvHeads = "num_kv_heads"
        case headDim = "head_dim"
        case attnDim = "attn_dim"
        case kvDim = "kv_dim"
        case ffDim = "ff_dim"
        case zDim = "z_dim"
        case ldim
        case flowType = "flow_type"
        case ropeMaxPeriod = "rope_max_period"
        case eosThreshold = "eos_threshold"
        case temperature
        case minFramesBeforeEos = "min_frames_before_eos"
        case textVocab = "text_vocab"
        case textRank = "text_rank"
        case insertBosBeforeVoice = "insert_bos_before_voice"
        case sampleRate = "sample_rate"
        case frameSize = "frame_size"
        case upsampleStride = "upsample_stride"
        case mimiDecoderDim = "mimi_decoder_dim"
        case mimiDecoderLayers = "mimi_decoder_layers"
        case mimiDecoderHeads = "mimi_decoder_heads"
        case mimiDecoderHeadDim = "mimi_decoder_head_dim"
        case mimiDecoderContext = "mimi_decoder_context"
    }

    public static func load(from packDirectory: URL) throws -> PocketTtsSmallArch {
        let url = packDirectory.appendingPathComponent("arch.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PocketTTSError.modelNotFound("arch.json missing in \(packDirectory.path)")
        }
        return try JSONDecoder().decode(PocketTtsSmallArch.self, from: Data(contentsOf: url))
    }
}

/// `pocket_state.json` of a small-model pack: what the MLState multifunction
/// package contains and how to drive its Mimi half.
public struct PocketTtsSmallStateInfo: Decodable, Sendable {
    /// Functions in `pocket_state.mlmodelc`. Always `prefill`, `generate`,
    /// `generate_decode`; `write_state` / `decode` only where the ANE planner
    /// accepted them.
    public let functions: [String]
    /// FlowLM KV cache length L (slots per layer).
    public let kvCacheLen: Int
    /// Conditioning block length compiled into `prefill`.
    public let condPrefillTMax: Int
    /// Number of Mimi phases (`2 * capacity / tokens_per_frame`).
    public let mimiPhases: Int
    public let mimiCapacity: Int
    public let mimiTokensPerFrame: Int
    /// Mimi conv histories / overlap-add partials that travel as ordinary
    /// inputs (`<name>`) and outputs (`<name>_out`), zero at utterance start.
    public let mimiConvIO: [String: [Int]]
    /// Mimi KV cache state buffers (`st_...`), zero at utterance start.
    public let mimiState: [String: [Int]]

    enum CodingKeys: String, CodingKey {
        case functions
        case kvCacheLen = "kv_cache_len"
        case condPrefillTMax = "cond_prefill_t_max"
        case mimiPhases = "mimi_phases"
        case mimiCapacity = "mimi_capacity"
        case mimiTokensPerFrame = "mimi_tokens_per_frame"
        case mimiConvIO = "mimi_conv_io"
        case mimiState = "mimi_state"
    }

    public var hasFusedGenerateDecode: Bool { functions.contains("generate_decode") }

    /// Phase of the `frame`-th Mimi call of an utterance (0-based): warm-up
    /// phases while the cache fills, then steady-state phases cycling through
    /// the write offset. Mirrors `phase_index` in the converter.
    public func mimiPhase(frame: Int) -> Int {
        let n = mimiCapacity / mimiTokensPerFrame
        return frame < n ? frame : n + frame % n
    }

    public static func load(from packDirectory: URL) throws -> PocketTtsSmallStateInfo {
        let url = packDirectory.appendingPathComponent("pocket_state.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PocketTTSError.modelNotFound("pocket_state.json missing in \(packDirectory.path)")
        }
        return try JSONDecoder().decode(PocketTtsSmallStateInfo.self, from: Data(contentsOf: url))
    }
}

/// A shipped voice for a small model: the upstream per-layer FlowLM KV
/// snapshot, split into K and V, as flat row-major `[n, Hkv, D]` fp32.
public struct PocketTtsSmallVoice: Sendable {
    public let name: String
    /// Per layer: K then V, each `promptLength * numKvHeads * headDim` floats.
    public let kLayers: [[Float]]
    public let vLayers: [[Float]]
    /// Positions already filled (= KV position after injection).
    public let promptLength: Int
}

/// Binary constants of a small-model pack (`constants_bin/`).
public struct PocketTtsSmallConstants: Sendable {
    public let arch: PocketTtsSmallArch
    public let state: PocketTtsSmallStateInfo
    /// `[ldim]`: the first `sequence` of every chunk.
    public let bosEmbedding: [Float]
    /// `[textVocab, dModel]`, low-rank projection already folded in.
    public let textEmbedTable: [Float]
    public let tokenizer: SentencePieceTokenizer
    /// `[dModel]`: prepended to a cloned voice's conditioning block.
    public let bosBeforeVoice: [Float]
    /// `[dModel, ldim]`: Mimi latent -> voice conditioning (cloning).
    public let speakerProjWeight: [Float]
    public let constantsBinDirectory: URL

    public static func load(from packDirectory: URL) throws -> PocketTtsSmallConstants {
        let arch = try PocketTtsSmallArch.load(from: packDirectory)
        let state = try PocketTtsSmallStateInfo.load(from: packDirectory)
        let bin = packDirectory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let bos = try loadFloats(bin.appendingPathComponent("bos_emb.bin"), expected: arch.ldim)
        let table = try loadFloats(
            bin.appendingPathComponent("text_embed_table.bin"), expected: arch.textVocab * arch.dModel)
        let bosBeforeVoice = try loadFloats(
            bin.appendingPathComponent("bos_before_voice.bin"), expected: arch.dModel)
        let speakerProj = try loadFloats(
            bin.appendingPathComponent("speaker_proj_weight.bin"), expected: arch.dModel * arch.ldim)
        let tokenizerURL = bin.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw PocketTtsConstantsLoader.LoadError.fileNotFound("tokenizer.model")
        }
        let tokenizer = try SentencePieceTokenizer(modelData: Data(contentsOf: tokenizerURL))
        return PocketTtsSmallConstants(
            arch: arch, state: state, bosEmbedding: bos, textEmbedTable: table, tokenizer: tokenizer,
            bosBeforeVoice: bosBeforeVoice, speakerProjWeight: speakerProj, constantsBinDirectory: bin)
    }

    /// Load `<voice>.safetensors` (upstream KV snapshot computed with this
    /// model's weights): `transformer.layers.{i}.self_attn/cache`
    /// `[2, 1, n, Hkv, D]` fp32 + `/offset` `[1]` int64 (+ a zero `/pad`).
    public func loadVoice(_ voice: String) throws -> PocketTtsSmallVoice {
        let sanitized = voice.replacingOccurrences(of: "/", with: "")
        let url = constantsBinDirectory.appendingPathComponent("\(sanitized).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PocketTtsConstantsLoader.LoadError.fileNotFound("\(sanitized).safetensors")
        }
        let data = try Data(contentsOf: url)
        let tensors = try PocketTtsConstantsLoader.parseSafetensors(data, fileLabel: sanitized)
        var kLayers: [[Float]] = []
        var vLayers: [[Float]] = []
        var promptLength: Int?
        for layer in 0..<arch.numLayers {
            let base = "transformer.layers.\(layer).self_attn"
            guard let cache = tensors["\(base)/cache"], let offset = tensors["\(base)/offset"] else {
                throw PocketTtsConstantsLoader.LoadError.malformed(
                    "\(sanitized).safetensors: missing layer \(layer)")
            }
            guard cache.dtype == "F32", cache.shape.count == 5, cache.shape[0] == 2, cache.shape[1] == 1,
                cache.shape[3] == arch.numKvHeads, cache.shape[4] == arch.headDim
            else {
                throw PocketTtsConstantsLoader.LoadError.malformed(
                    "\(sanitized).safetensors: layer \(layer) cache shape \(cache.shape) "
                        + "does not match [2,1,n,\(arch.numKvHeads),\(arch.headDim)] F32")
            }
            let n = cache.shape[2]
            let offsetValue = data.withUnsafeBytes { raw -> Int in
                Int(raw.loadUnaligned(fromByteOffset: offset.byteOffset, as: Int64.self))
            }
            guard offsetValue == n else {
                throw PocketTtsConstantsLoader.LoadError.malformed(
                    "\(sanitized).safetensors: layer \(layer) offset \(offsetValue) != cache length \(n)")
            }
            if let p = promptLength, p != n {
                throw PocketTtsConstantsLoader.LoadError.malformed(
                    "\(sanitized).safetensors: layers differ in length")
            }
            promptLength = n
            let perKV = n * arch.numKvHeads * arch.headDim
            let floats: [Float] = data.withUnsafeBytes { raw in
                let start = raw.baseAddress!.advanced(by: cache.byteOffset)
                var out = [Float](repeating: 0, count: 2 * perKV)
                out.withUnsafeMutableBytes { dst in
                    dst.copyMemory(from: UnsafeRawBufferPointer(start: start, count: 2 * perKV * 4))
                }
                return out
            }
            kLayers.append(Array(floats[0..<perKV]))
            vLayers.append(Array(floats[perKV..<(2 * perKV)]))
        }
        return PocketTtsSmallVoice(
            name: sanitized, kLayers: kLayers, vLayers: vLayers, promptLength: promptLength ?? 0)
    }

    /// Row-major `[tokens, dModel]` text embeddings for `tokenIds`.
    public func embed(_ tokenIds: [Int]) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(tokenIds.count * arch.dModel)
        for id in tokenIds {
            let row = min(max(id, 0), arch.textVocab - 1) * arch.dModel
            out.append(contentsOf: textEmbedTable[row..<(row + arch.dModel)])
        }
        return out
    }

    private static func loadFloats(_ url: URL, expected: Int) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PocketTtsConstantsLoader.LoadError.fileNotFound(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        guard count == expected else {
            throw PocketTtsConstantsLoader.LoadError.invalidSize(
                url.lastPathComponent, expected: expected, actual: count)
        }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
#endif
