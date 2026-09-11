import Foundation

/// Host-side runtime tables for Chatterbox: T3 embedding / positional tables
/// (`tables/tables.safetensors`) and a precomputed voice
/// (`tables/voice-<name>.safetensors`), both exported by mobius
/// `models/tts/chatterbox/coreml/export-tables.py`.
struct ChatterboxTables: Sendable {

    /// Row-major `[rows, cols]` fp32 matrix backing an embedding lookup.
    struct Table: Sendable {
        let rows: Int
        let cols: Int
        let values: [Float]

        func row(_ index: Int) -> ArraySlice<Float> {
            let base = index * cols
            return values[base..<(base + cols)]
        }
    }

    struct Voice: Sendable {
        /// T3 conditioning embeds (condLength × hidden), exaggeration baked in.
        let condEmb: Table
        /// S3Gen reference: prompt speech tokens (25 Hz).
        let promptTokens: [Int32]
        /// S3Gen reference mel (frames × 80).
        let promptFeat: Table
        /// CAMPPlus x-vector (192).
        let embedding: [Float]
    }

    let textEmb: Table
    let speechEmb: Table
    let textPos: Table
    let speechPos: Table

    static func load(tablesURL: URL) throws -> ChatterboxTables {
        let tensors = try SafetensorsFile(url: tablesURL)
        return ChatterboxTables(
            textEmb: try tensors.table("text_emb"),
            speechEmb: try tensors.table("speech_emb"),
            textPos: try tensors.table("text_pos_emb"),
            speechPos: try tensors.table("speech_pos_emb"))
    }

    static func loadVoice(voiceURL: URL) throws -> Voice {
        let tensors = try SafetensorsFile(url: voiceURL)
        let condEmb = try tensors.table("t3_cond_emb")
        let promptFeat = try tensors.table("prompt_feat")
        let promptTokens = try tensors.int32Values("prompt_token")
        let embedding = try tensors.table("embedding")
        return Voice(
            condEmb: condEmb,
            promptTokens: promptTokens,
            promptFeat: promptFeat,
            embedding: embedding.values)
    }
}

/// Minimal safetensors reader (F32 / F16 / I32), sufficient for the
/// Chatterbox table exports.
private struct SafetensorsFile {
    struct Entry {
        let dtype: String
        let shape: [Int]
        let range: Range<Int>
    }

    let data: Data
    let entries: [String: Entry]
    let dataStart: Int

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 8 else {
            throw ChatterboxError.malformedAsset("\(url.lastPathComponent): truncated header")
        }
        let headerLen = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self).littleEndian
        }
        let headerEnd = 8 + Int(headerLen)
        guard headerEnd <= data.count,
            let header = try JSONSerialization.jsonObject(
                with: data.subdata(in: 8..<headerEnd)) as? [String: Any]
        else {
            throw ChatterboxError.malformedAsset("\(url.lastPathComponent): bad JSON header")
        }

        var entries = [String: Entry]()
        for (name, value) in header where name != "__metadata__" {
            guard let obj = value as? [String: Any],
                let dtype = obj["dtype"] as? String,
                let shape = obj["shape"] as? [Int],
                let offsets = obj["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw ChatterboxError.malformedAsset("\(url.lastPathComponent): entry \(name)")
            }
            entries[name] = Entry(dtype: dtype, shape: shape, range: offsets[0]..<offsets[1])
        }
        self.data = data
        self.entries = entries
        self.dataStart = headerEnd
    }

    private func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else {
            throw ChatterboxError.malformedAsset("missing tensor '\(name)'")
        }
        return entry
    }

    /// Read a tensor as a 2-D fp32 table (leading singleton dims collapsed).
    func table(_ name: String) throws -> ChatterboxTables.Table {
        let entry = try entry(name)
        let dims = entry.shape.drop { $0 == 1 }
        let cols = dims.last ?? 1
        let rows = dims.dropLast().reduce(1, *)
        let values = try floatValues(name)
        guard values.count == rows * cols else {
            throw ChatterboxError.malformedAsset("tensor '\(name)' shape/data mismatch")
        }
        return ChatterboxTables.Table(rows: rows, cols: cols, values: values)
    }

    func floatValues(_ name: String) throws -> [Float] {
        let entry = try entry(name)
        let bytes = data.subdata(
            in: (dataStart + entry.range.lowerBound)..<(dataStart + entry.range.upperBound))
        switch entry.dtype {
        case "F32":
            let count = bytes.count / 4
            var out = [Float](repeating: 0, count: count)
            bytes.withUnsafeBytes { raw in
                out.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(
                        from: raw.bindMemory(to: Float.self).baseAddress!, count: count)
                }
            }
            return out
        case "F16":
            let count = bytes.count / 2
            var out = [Float](repeating: 0, count: count)
            bytes.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: UInt16.self).baseAddress!
                out.withUnsafeMutableBufferPointer { dst in
                    Float16Conversion.toFloat32(src: src, dst: dst.baseAddress!, count: count)
                }
            }
            return out
        default:
            throw ChatterboxError.malformedAsset("tensor '\(name)': unsupported dtype \(entry.dtype)")
        }
    }

    func int32Values(_ name: String) throws -> [Int32] {
        let entry = try entry(name)
        guard entry.dtype == "I32" else {
            throw ChatterboxError.malformedAsset("tensor '\(name)': expected I32, got \(entry.dtype)")
        }
        let bytes = data.subdata(
            in: (dataStart + entry.range.lowerBound)..<(dataStart + entry.range.upperBound))
        let count = bytes.count / 4
        var out = [Int32](repeating: 0, count: count)
        bytes.withUnsafeBytes { raw in
            out.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(
                    from: raw.bindMemory(to: Int32.self).baseAddress!, count: count)
            }
        }
        return out
    }
}
