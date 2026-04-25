import Foundation
import OSLog

/// Pre-loaded binary constants for PocketTTS inference.
public struct PocketTtsConstantsBundle: Sendable {
    public let bosEmbedding: [Float]
    public let textEmbedTable: [Float]
    public let tokenizer: SentencePieceTokenizer
}

/// Pre-loaded voice conditioning data.
///
/// Two sources populate this:
///  - **Pre-baked KV cache snapshot** (on-disk `<voice>.safetensors` from a
///    v2 language pack): per-layer K/V tensors already shaped to drop
///    directly into the LM transformer's `cache{i}` slots. Skips
///    `cond_step` voice prefill.
///  - **Flat audio prompt** (runtime voice cloning via `mimi_encoder`):
///    a `[1, promptLength, 1024]` Float32 tensor that the synthesizer
///    feeds through `cond_step` token-by-token to populate the LM
///    transformer KV cache.
///
/// Exactly one of the two is populated. The synthesizer's `prefillKVCache`
/// branches on `cacheSnapshot != nil` to choose the path.
public struct PocketTtsVoiceData: Sendable {
    /// Flattened audio prompt: [1, promptLength, 1024]. Empty when
    /// `cacheSnapshot` is non-nil. Populated by runtime voice cloning.
    public let audioPrompt: [Float]
    /// Number of voice conditioning tokens (typically 125 for prebakes,
    /// up to 250 for cloned voices). Zero when `cacheSnapshot` is non-nil.
    public let promptLength: Int
    /// Pre-baked LM transformer KV cache loaded from a v2 voice safetensors
    /// file. `nil` for cloned voices.
    public let cacheSnapshot: PocketTtsVoiceCacheSnapshot?

    public init(
        audioPrompt: [Float],
        promptLength: Int,
        cacheSnapshot: PocketTtsVoiceCacheSnapshot? = nil
    ) {
        self.audioPrompt = audioPrompt
        self.promptLength = promptLength
        self.cacheSnapshot = cacheSnapshot
    }
}

/// Pre-baked KV cache snapshot for v2 voice packs.
///
/// One `LayerCache` per LM transformer layer, in layer order. Each
/// `cache` is a flat Float32 array shaped `[2, 1, seqLen, 16, 64]`
/// (K and V interleaved at outer dim) matching the model's `cache{i}`
/// input layout, and `offset` is the number of tokens already filled
/// (= seqLen for a fully-prebaked snapshot).
public struct PocketTtsVoiceCacheSnapshot: Sendable {
    public struct LayerCache: Sendable {
        public let cache: [Float]
        public let offset: Int
    }

    public let layers: [LayerCache]
    /// Sequence length per layer in the source snapshot (e.g. 126).
    public let cacheSeqLen: Int
}

/// Loads PocketTTS constants from raw `.bin` Float32 files on disk.
public enum PocketTtsConstantsLoader {

    private static let logger = AppLogger(category: "PocketTtsConstantsLoader")

    public enum LoadError: Error {
        case fileNotFound(String)
        case invalidSize(String, expected: Int, actual: Int)
        case tokenizerLoadFailed(String)
    }

    /// Load all constants from the given directory.
    public static func load(from directory: URL) throws -> PocketTtsConstantsBundle {
        let constantsDir = directory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)

        let bosEmb = try loadFloatArray(
            from: constantsDir.appendingPathComponent("bos_emb.bin"),
            expectedCount: PocketTtsConstants.latentDim,
            name: "bos_emb"
        )
        // text_embed_table vocab size is language-dependent — English ships
        // 4001 rows, other languages may differ. Validate only that the file
        // is a clean multiple of embeddingDim; the synthesizer indexes into
        // this table by token ID at runtime.
        let embedTable = try loadFloatArrayMultipleOf(
            from: constantsDir.appendingPathComponent("text_embed_table.bin"),
            rowSize: PocketTtsConstants.embeddingDim,
            name: "text_embed_table"
        )

        let tokenizerURL = constantsDir.appendingPathComponent("tokenizer.model")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw LoadError.fileNotFound("tokenizer.model")
        }
        let tokenizerData = try Data(contentsOf: tokenizerURL)
        let tokenizer: SentencePieceTokenizer
        do {
            tokenizer = try SentencePieceTokenizer(modelData: tokenizerData)
        } catch {
            throw LoadError.tokenizerLoadFailed(error.localizedDescription)
        }

        logger.info("Loaded PocketTTS constants from \(directory.lastPathComponent)")

        return PocketTtsConstantsBundle(
            bosEmbedding: bosEmb,
            textEmbedTable: embedTable,
            tokenizer: tokenizer
        )
    }

    /// Load voice conditioning data from the given directory.
    ///
    /// Reads `<voice>.safetensors` from the language pack's `constants_bin/`
    /// directory — every v2 language pack ships pre-baked LM KV cache
    /// snapshots in this format (per-layer `[2,1,seqLen,16,64]` F32 plus
    /// `[1]` I64 offset).
    public static func loadVoice(
        _ voice: String, from directory: URL
    ) throws -> PocketTtsVoiceData {
        // Sanitize voice name to prevent path traversal
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw LoadError.fileNotFound("invalid voice name: \(voice)")
        }

        let constantsDir = directory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let safetensorsURL = constantsDir.appendingPathComponent("\(sanitized).safetensors")

        guard FileManager.default.fileExists(atPath: safetensorsURL.path) else {
            throw LoadError.fileNotFound("\(sanitized).safetensors")
        }

        let snapshot = try loadVoiceSnapshot(from: safetensorsURL, voiceName: sanitized)
        logger.info(
            "Loaded PocketTTS voice '\(sanitized)' from safetensors (\(snapshot.layers.count) layers, seq=\(snapshot.cacheSeqLen))"
        )
        return PocketTtsVoiceData(
            audioPrompt: [],
            promptLength: 0,
            cacheSnapshot: snapshot
        )
    }

    /// Parse a v2 voice safetensors file into a `PocketTtsVoiceCacheSnapshot`.
    ///
    /// Expected schema (kyutai pocket-tts v2):
    ///  - `transformer.layers.{N}.self_attn/cache` F32 `[2, 1, seqLen, 16, 64]`
    ///  - `transformer.layers.{N}.self_attn/offset` I64 `[1]`
    ///
    /// All layers share the same `seqLen`. Layer count is auto-detected (6
    /// for non-24L packs, 24 for `*_24l` packs).
    private static func loadVoiceSnapshot(
        from url: URL, voiceName: String
    ) throws -> PocketTtsVoiceCacheSnapshot {
        let raw = try Data(contentsOf: url)
        let parsed = try parseSafetensors(raw, fileLabel: voiceName)

        // Collect cache + offset entries by layer index.
        var caches: [Int: SafetensorsTensor] = [:]
        var offsets: [Int: SafetensorsTensor] = [:]
        let cachePrefix = "transformer.layers."
        let cacheSuffix = ".self_attn/cache"
        let offsetSuffix = ".self_attn/offset"

        for (key, tensor) in parsed.tensors {
            guard key.hasPrefix(cachePrefix) else { continue }
            let stripped = String(key.dropFirst(cachePrefix.count))
            if let dotRange = stripped.firstIndex(of: ".") {
                let idxStr = String(stripped[..<dotRange])
                let rest = stripped[dotRange...]
                guard let layerIdx = Int(idxStr) else { continue }
                if rest == Substring(cacheSuffix) {
                    caches[layerIdx] = tensor
                } else if rest == Substring(offsetSuffix) {
                    offsets[layerIdx] = tensor
                }
            }
        }

        guard !caches.isEmpty else {
            throw LoadError.invalidSize(
                "\(voiceName).safetensors: no transformer.layers.*.self_attn/cache entries",
                expected: 1,
                actual: 0
            )
        }
        guard caches.count == offsets.count else {
            throw LoadError.invalidSize(
                "\(voiceName).safetensors: cache/offset count mismatch",
                expected: caches.count,
                actual: offsets.count
            )
        }

        let sortedLayers = caches.keys.sorted()
        // Layer indices must be 0..N-1 contiguous.
        for (i, k) in sortedLayers.enumerated() where i != k {
            throw LoadError.invalidSize(
                "\(voiceName).safetensors: layer indices not contiguous (gap at \(i))",
                expected: i,
                actual: k
            )
        }

        // All layer caches must share shape [2, 1, seqLen, 16, 64].
        let firstShape = caches[sortedLayers[0]]!.shape
        guard firstShape.count == 5,
            firstShape[0] == 2,
            firstShape[1] == 1,
            firstShape[3] == 16,
            firstShape[4] == 64
        else {
            throw LoadError.invalidSize(
                "\(voiceName).safetensors: unexpected cache shape \(firstShape)",
                expected: 0,
                actual: 0
            )
        }
        let seqLen = firstShape[2]

        var layers: [PocketTtsVoiceCacheSnapshot.LayerCache] = []
        layers.reserveCapacity(sortedLayers.count)

        for layerIdx in sortedLayers {
            guard let cacheTensor = caches[layerIdx], let offsetTensor = offsets[layerIdx]
            else { continue }
            guard cacheTensor.shape == firstShape else {
                throw LoadError.invalidSize(
                    "\(voiceName).safetensors: cache shape mismatch at layer \(layerIdx)",
                    expected: 0,
                    actual: 0
                )
            }
            guard cacheTensor.dtype == "F32" else {
                throw LoadError.invalidSize(
                    "\(voiceName).safetensors: layer \(layerIdx) cache dtype \(cacheTensor.dtype) (want F32)",
                    expected: 0,
                    actual: 0
                )
            }
            let floatCount = cacheTensor.byteCount / MemoryLayout<Float>.size
            let cacheFloats: [Float] = raw.withUnsafeBytes { rawBuf -> [Float] in
                let base = rawBuf.baseAddress!.advanced(by: cacheTensor.byteOffset)
                let typed = base.assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: typed, count: floatCount))
            }

            // offset is I64 [1]
            guard offsetTensor.dtype == "I64", offsetTensor.shape == [1] else {
                throw LoadError.invalidSize(
                    "\(voiceName).safetensors: layer \(layerIdx) offset shape/dtype unexpected",
                    expected: 0,
                    actual: 0
                )
            }
            let offsetVal: Int = raw.withUnsafeBytes { rawBuf -> Int in
                let base = rawBuf.baseAddress!.advanced(by: offsetTensor.byteOffset)
                let typed = base.assumingMemoryBound(to: Int64.self)
                return Int(typed.pointee)
            }

            layers.append(.init(cache: cacheFloats, offset: offsetVal))
        }

        return PocketTtsVoiceCacheSnapshot(layers: layers, cacheSeqLen: seqLen)
    }

    // MARK: - Private

    /// Load a raw Float32 binary file into a [Float] array.
    private static func loadFloatArray(
        from url: URL, expectedCount: Int, name: String
    ) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        let actualCount = data.count / MemoryLayout<Float>.size

        guard actualCount == expectedCount else {
            throw LoadError.invalidSize(name, expected: expectedCount, actual: actualCount)
        }

        return data.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
    }

    // MARK: - safetensors

    /// One tensor entry from a safetensors file.
    fileprivate struct SafetensorsTensor {
        let dtype: String
        let shape: [Int]
        /// Absolute byte offset into the original file (already includes
        /// the 8-byte size prefix and JSON header length).
        let byteOffset: Int
        let byteCount: Int
    }

    fileprivate struct SafetensorsParsed {
        let tensors: [String: SafetensorsTensor]
    }

    /// Minimal safetensors reader: 8-byte LE u64 header length, then JSON
    /// header, then raw tensor bytes. Only used for v2 voice prebakes — no
    /// need to support arbitrary safetensors features.
    fileprivate static func parseSafetensors(
        _ data: Data, fileLabel: String
    ) throws -> SafetensorsParsed {
        guard data.count >= 8 else {
            throw LoadError.invalidSize(
                "\(fileLabel).safetensors: too small",
                expected: 8,
                actual: data.count
            )
        }
        // Header length: little-endian u64 at byte 0.
        let headerLen: UInt64 = data.withUnsafeBytes { rawBuf -> UInt64 in
            let typed = rawBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            return UInt64(littleEndian: typed.pointee)
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLen)
        guard headerEnd <= data.count else {
            throw LoadError.invalidSize(
                "\(fileLabel).safetensors: header overflow",
                expected: headerEnd,
                actual: data.count
            )
        }
        let headerBytes = data.subdata(in: headerStart..<headerEnd)
        let json: Any
        do {
            json = try JSONSerialization.jsonObject(with: headerBytes, options: [])
        } catch {
            throw LoadError.invalidSize(
                "\(fileLabel).safetensors: header JSON parse failed: \(error)",
                expected: 0,
                actual: 0
            )
        }
        guard let dict = json as? [String: Any] else {
            throw LoadError.invalidSize(
                "\(fileLabel).safetensors: header is not a JSON object",
                expected: 0,
                actual: 0
            )
        }

        let dataStart = headerEnd
        var tensors: [String: SafetensorsTensor] = [:]
        for (key, value) in dict {
            if key == "__metadata__" { continue }
            guard let entry = value as? [String: Any] else { continue }
            guard let dtype = entry["dtype"] as? String,
                let shapeArr = entry["shape"] as? [Any],
                let offsets = entry["data_offsets"] as? [Any],
                offsets.count == 2
            else {
                throw LoadError.invalidSize(
                    "\(fileLabel).safetensors: malformed entry '\(key)'",
                    expected: 0,
                    actual: 0
                )
            }
            let shape: [Int] = shapeArr.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeArr.count else {
                throw LoadError.invalidSize(
                    "\(fileLabel).safetensors: bad shape for '\(key)'",
                    expected: 0,
                    actual: 0
                )
            }
            guard let startNum = offsets[0] as? NSNumber,
                let endNum = offsets[1] as? NSNumber
            else {
                throw LoadError.invalidSize(
                    "\(fileLabel).safetensors: bad offsets for '\(key)'",
                    expected: 0,
                    actual: 0
                )
            }
            let start = startNum.intValue
            let end = endNum.intValue
            guard end >= start else {
                throw LoadError.invalidSize(
                    "\(fileLabel).safetensors: negative span for '\(key)'",
                    expected: start,
                    actual: end
                )
            }
            let absStart = dataStart + start
            let span = end - start
            guard absStart + span <= data.count else {
                throw LoadError.invalidSize(
                    "\(fileLabel).safetensors: tensor '\(key)' overflows file",
                    expected: data.count,
                    actual: absStart + span
                )
            }
            tensors[key] = SafetensorsTensor(
                dtype: dtype,
                shape: shape,
                byteOffset: absStart,
                byteCount: span
            )
        }
        return SafetensorsParsed(tensors: tensors)
    }

    /// Load a raw Float32 binary file whose length must be a non-zero multiple
    /// of `rowSize`. Used for tensors whose leading dimension varies per
    /// language (e.g. `text_embed_table` vocab size).
    private static func loadFloatArrayMultipleOf(
        from url: URL, rowSize: Int, name: String
    ) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        let actualCount = data.count / MemoryLayout<Float>.size

        guard actualCount > 0, actualCount % rowSize == 0 else {
            throw LoadError.invalidSize(name, expected: rowSize, actual: actualCount)
        }

        return data.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
    }
}
