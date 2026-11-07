#if arch(arm64)
import Foundation

/// Rows appended to the LM context.
enum LFM3AudioRows: Sendable {
    /// Token ids (embedded from the fp16 embedding table).
    case ids([Int32])
    /// Explicit fp16 input rows `[n, hidden]` (speech-frame embeddings).
    case embeddings([Float16])
    /// Token ids where each `placeholder` id takes the next row of `rows` (audio-encoder rows in a prompt).
    case mixed(ids: [Int32], rows: [Float16], placeholder: Int32)

    func count(hidden: Int) -> Int {
        switch self {
        case .ids(let ids): return ids.count
        case .embeddings(let rows): return rows.count / hidden
        case .mixed(let ids, _, _): return ids.count
        }
    }

    /// Writes rows `lo ..< lo + n` into `dst` (`[n, hidden]` fp16). `rowCursor` tracks consumed placeholder rows.
    func write(
        _ lo: Int, _ n: Int, into dst: UnsafeMutablePointer<Float16>, embedding: LFM3AudioEmbeddingTable,
        rowCursor: inout Int
    ) throws {
        let h = embedding.hidden
        switch self {
        case .ids(let ids):
            for j in 0..<n { try embedding.gather(ids[lo + j], into: dst + j * h) }
        case .embeddings(let rows):
            guard (lo + n) * h <= rows.count else { throw LFM3AudioError.processingFailed("row overflow") }
            rows.withUnsafeBufferPointer { src in
                guard let base = src.baseAddress else { return }
                dst.update(from: base + lo * h, count: n * h)
            }
        case .mixed(let ids, let rows, let placeholder):
            for j in 0..<n {
                let id = ids[lo + j]
                if id == placeholder {
                    guard (rowCursor + 1) * h <= rows.count else {
                        throw LFM3AudioError.processingFailed("more placeholders than audio rows")
                    }
                    let cursor = rowCursor
                    rows.withUnsafeBufferPointer { src in
                        guard let base = src.baseAddress else { return }
                        (dst + j * h).update(from: base + cursor * h, count: h)
                    }
                    rowCursor += 1
                } else {
                    try embedding.gather(id, into: dst + j * h)
                }
            }
        }
    }
}

/// What the caller needs from the last fed row.
enum LFM3AudioWant: Sendable {
    /// Final-norm hidden state (speech frames → depth decoder).
    case hidden
    /// Greedy next token; ids in `suppress` are never chosen.
    case token(suppress: Set<Int32>)
}

struct LFM3AudioFeedResult: Sendable {
    var token: Int32 = -1
    /// Final-norm hidden of the last row (`[hidden]` fp16); empty unless `.hidden` was requested.
    var hidden: [Float16] = []
}

/// The LFM3 backbone (2048 x 24: conv / full / sliding-attention layers) with its KV / conv state.
/// Implemented by the Core ML pack (`LFM3AudioCoreMLBackbone`) and the Core AI bundle (`LFM3AudioCoreAIBackbone`).
protocol LFM3AudioLanguageModel: Actor {
    /// Clears the context (new MLState / zeroed Core AI states).
    func reset() throws
    /// Loads whatever `rows`-row calls need now instead of on their first call.
    func prepare(rows: Int) throws
    /// Appends `rows` at the current position and returns what `want` asks for from the last row.
    func feed(_ rows: LFM3AudioRows, want: LFM3AudioWant) async throws -> LFM3AudioFeedResult
}

/// Memory-mapped fp16 token-embedding table `[vocab, hidden]` (host gather of LM input rows).
struct LFM3AudioEmbeddingTable: Sendable {
    let vocab: Int
    let hidden: Int
    private let data: Data

    init(url: URL, vocab: Int, hidden: Int) throws {
        data = try LFM3AudioTables.mapped(url, expectedBytes: vocab * hidden * 2)
        self.vocab = vocab
        self.hidden = hidden
    }

    func gather(_ id: Int32, into dst: UnsafeMutablePointer<Float16>) throws {
        guard id >= 0, Int(id) < vocab else { throw LFM3AudioError.processingFailed("token id \(id) out of range") }
        let h = hidden
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            dst.update(from: base.assumingMemoryBound(to: Float16.self) + Int(id) * h, count: h)
        }
    }
}
#endif
