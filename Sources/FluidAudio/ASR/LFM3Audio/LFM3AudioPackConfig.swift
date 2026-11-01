#if arch(arm64)
import Foundation

/// Host contract of the LFM3 Core ML backbone pack (`assets/config.json`): stage packages, function buckets,
/// RoPE / sliding-window parameters and the chunked top-k head layout.
struct LFM3AudioPackConfig: Codable, Sendable {
    struct Rope: Codable, Sendable {
        let theta: Double
        let pairs: Int
    }
    struct Embedding: Codable, Sendable {
        let file: String
        let shape: [Int]
    }
    struct Function: Codable, Sendable {
        let name: String
        let q: Int
        let span: Int?
        let role: String  // "decode" | "prefill" | "decode_hidden"
        let calls: [String]?  // per-stage function names (default: `name` in every stage)
    }
    struct Stage: Codable, Sendable {
        let package: String
        let functions: [Function]
    }
    struct Prefill: Codable, Sendable {
        let buckets: [Int]
        let padFirst: Bool

        enum CodingKeys: String, CodingKey {
            case buckets
            case padFirst = "pad_first"
        }
    }
    struct Head: Codable, Sendable {
        let package: String
    }
    struct TopK: Codable, Sendable {
        let k: Int
        let chunks: Int
        let chunkSize: Int

        enum CodingKeys: String, CodingKey {
            case k, chunks
            case chunkSize = "chunk_size"
        }
    }

    let hiddenSize: Int
    let vocabSize: Int
    let headDim: Int
    let cacheLen: Int
    let fifoLen: Int
    let slidingWindow: Int
    let rope: [String: Rope]
    let embedding: Embedding
    let stages: [Stage]
    let prefill: Prefill
    let topk: TopK?
    let head: Head

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case headDim = "head_dim"
        case cacheLen = "cache_len"
        case fifoLen = "fifo_len"
        case slidingWindow = "sliding_window"
        case rope, embedding, stages, prefill, topk, head
    }

    static func load(_ url: URL) throws -> LFM3AudioPackConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LFM3AudioError.modelNotFound(url.lastPathComponent)
        }
        let config = try JSONDecoder().decode(LFM3AudioPackConfig.self, from: Data(contentsOf: url))
        guard config.hiddenSize == LFM3AudioConfig.hiddenSize, config.rope["full"] != nil,
            config.rope["sliding"] != nil, !config.stages.isEmpty
        else {
            throw LFM3AudioError.invalidModel("unexpected backbone pack config")
        }
        return config
    }

    /// Distinct per-stage call lists of the decode / prefill functions, keyed by rows per call.
    var callPlans: [(q: Int, span: Int, calls: [String])] {
        guard let last = stages.last else { return [] }
        var seen = Set<String>()
        var out: [(Int, Int, [String])] = []
        for function in last.functions where function.role == "decode" || function.role == "prefill" {
            let calls = function.calls ?? Array(repeating: function.name, count: stages.count)
            let key = calls.joined(separator: "|")
            guard seen.insert(key).inserted else { continue }
            out.append((function.q, function.span ?? cacheLen, calls))
        }
        return out
    }
}
#endif
