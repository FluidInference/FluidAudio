import Foundation

/// Binary loader for the jieba HMM tables shipped at
/// `FluidInference/kokoro-82m-coreml/ANE-zh/assets/`.
///
/// The Python jieba project distributes its character HMM as three text
/// files (`prob_start.py`, `prob_trans.py`, `prob_emit.py`). Those are
/// pre-converted to compact little-endian binary by a one-shot mobius
/// script so the runtime cost is a `Data(contentsOf:)` plus a linear
/// parse — no Python interpreter at startup.
///
/// Wire format for the three artefacts:
///
///   * `jieba_hmm_start.bin` — 4 × `Float32_LE`. The fixed B/M/E/S state
///     order matches `jieba.finalseg.prob_start.P`.
///
///   * `jieba_hmm_trans.bin` — 16 × `Float32_LE`. Row-major 4×4 with the
///     same B/M/E/S order on both axes (`trans[from*4 + to]`).
///
///   * `jieba_hmm_emit.bin` — repeat:
///     ```
///     u32_le  unicode codepoint of the emitting character
///     4×f32_le  log-prob for B / M / E / S
///     ```
///     Codepoints are written in arbitrary order; the loader builds a
///     dictionary keyed by `Character` for O(1) lookups.
///
/// Log-probs are natural log; the Viterbi decoder sums them additively.
/// Codepoints absent from `emit` are treated as having a uniform
/// log-prob of `MandarinJiebaHmmTables.unknownCharLogProb` (≈ -3.14e+38),
/// preventing the decoder from getting stuck on OOV characters while
/// still strongly preferring the trained vocabulary.
public struct MandarinJiebaHmmTables: Sendable {

    /// Start log-probabilities, indexed by `JiebaHmmState.rawValue`.
    public let start: [Double]
    /// Transition log-probabilities, `trans[from][to]`. Always 4×4.
    public let trans: [[Double]]
    /// Emission log-probabilities, keyed by character. Each value is a
    /// length-4 array indexed by `JiebaHmmState.rawValue`.
    public let emit: [Character: [Double]]

    /// Sentinel log-prob for characters not present in `emit`. Picked to
    /// be effectively negative infinity without overflowing addition.
    public static let unknownCharLogProb: Double = -3.14e38

    public init(
        start: [Double],
        trans: [[Double]],
        emit: [Character: [Double]]
    ) {
        precondition(
            start.count == JiebaHmmState.allCases.count,
            "start must have one log-prob per HMM state")
        precondition(
            trans.count == JiebaHmmState.allCases.count
                && trans.allSatisfy { $0.count == JiebaHmmState.allCases.count },
            "trans must be a 4×4 matrix indexed by JiebaHmmState")
        self.start = start
        self.trans = trans
        self.emit = emit
    }

    public enum LoadError: Swift.Error, LocalizedError {
        case truncated(String)
        case wrongSize(String, expected: Int, actual: Int)

        public var errorDescription: String? {
            switch self {
            case .truncated(let what):
                return "Jieba HMM table \(what) is truncated"
            case .wrongSize(let what, let expected, let actual):
                return "Jieba HMM table \(what) has size \(actual) bytes, expected \(expected)"
            }
        }
    }

    /// Load all three tables from the directory that holds them.
    /// Convenience wrapper for the typical KokoroAneModelStore call site.
    public static func load(directory: URL) throws -> MandarinJiebaHmmTables {
        let startURL = directory.appendingPathComponent(
            KokoroAneConstants.jiebaHmmStartFile)
        let transURL = directory.appendingPathComponent(
            KokoroAneConstants.jiebaHmmTransFile)
        let emitURL = directory.appendingPathComponent(
            KokoroAneConstants.jiebaHmmEmitFile)
        let startData = try Data(contentsOf: startURL)
        let transData = try Data(contentsOf: transURL)
        let emitData = try Data(contentsOf: emitURL)
        return try MandarinJiebaHmmTables(
            startData: startData,
            transData: transData,
            emitData: emitData
        )
    }

    /// Construct from raw `Data` payloads — split out from `load(directory:)`
    /// so tests can synthesise tables in-memory without touching the disk.
    public init(startData: Data, transData: Data, emitData: Data) throws {
        let stateCount = JiebaHmmState.allCases.count

        // start: 4 floats
        let startBytes = stateCount * MemoryLayout<Float32>.size
        guard startData.count == startBytes else {
            throw LoadError.wrongSize(
                "start", expected: startBytes, actual: startData.count)
        }
        var start: [Double] = []
        start.reserveCapacity(stateCount)
        for i in 0..<stateCount {
            start.append(Double(Self.readFloatLE(startData, at: i * 4)))
        }

        // trans: 16 floats, row-major
        let transBytes = stateCount * stateCount * MemoryLayout<Float32>.size
        guard transData.count == transBytes else {
            throw LoadError.wrongSize(
                "trans", expected: transBytes, actual: transData.count)
        }
        var trans: [[Double]] = Array(
            repeating: Array(repeating: 0.0, count: stateCount), count: stateCount)
        for from in 0..<stateCount {
            for to in 0..<stateCount {
                let offset = (from * stateCount + to) * 4
                trans[from][to] = Double(Self.readFloatLE(transData, at: offset))
            }
        }

        // emit: stream of (u32 codepoint, 4×f32 logprobs)
        let recordSize = 4 + stateCount * MemoryLayout<Float32>.size
        guard emitData.count % recordSize == 0 else {
            throw LoadError.truncated("emit (size not a multiple of \(recordSize))")
        }
        var emit: [Character: [Double]] = [:]
        emit.reserveCapacity(emitData.count / recordSize)
        var pos = 0
        while pos < emitData.count {
            let cp = Self.readUInt32LE(emitData, at: pos)
            pos += 4
            guard let scalar = Unicode.Scalar(cp) else {
                throw LoadError.truncated("emit codepoint U+\(String(cp, radix: 16))")
            }
            var logProbs: [Double] = []
            logProbs.reserveCapacity(stateCount)
            for _ in 0..<stateCount {
                logProbs.append(Double(Self.readFloatLE(emitData, at: pos)))
                pos += 4
            }
            emit[Character(scalar)] = logProbs
        }

        self.init(start: start, trans: trans, emit: emit)
    }

    @inline(__always)
    private static func readUInt32LE(_ data: Data, at offset: Int) -> UInt32 {
        UInt32(data[data.startIndex + offset])
            | (UInt32(data[data.startIndex + offset + 1]) << 8)
            | (UInt32(data[data.startIndex + offset + 2]) << 16)
            | (UInt32(data[data.startIndex + offset + 3]) << 24)
    }

    @inline(__always)
    private static func readFloatLE(_ data: Data, at offset: Int) -> Float32 {
        let bits = readUInt32LE(data, at: offset)
        return Float32(bitPattern: bits)
    }

    /// Encode the in-memory tables back to the binary wire format. Used
    /// by tests to round-trip a synthetic fixture through the loader.
    public func encoded() -> (start: Data, trans: Data, emit: Data) {
        let stateCount = JiebaHmmState.allCases.count

        var startData = Data(capacity: stateCount * 4)
        for v in start {
            var bits = Float32(v).bitPattern
            withUnsafeBytes(of: &bits) { startData.append(contentsOf: $0) }
        }

        var transData = Data(capacity: stateCount * stateCount * 4)
        for row in trans {
            for v in row {
                var bits = Float32(v).bitPattern
                withUnsafeBytes(of: &bits) { transData.append(contentsOf: $0) }
            }
        }

        var emitData = Data(capacity: emit.count * (4 + stateCount * 4))
        // Iterate in deterministic order (codepoint ascending) so two
        // encoders produce byte-identical output for the same logical
        // table. Important for snapshot tests.
        let sorted = emit.sorted { lhs, rhs in
            (lhs.key.unicodeScalars.first?.value ?? 0)
                < (rhs.key.unicodeScalars.first?.value ?? 0)
        }
        for (ch, logProbs) in sorted {
            guard let scalar = ch.unicodeScalars.first else { continue }
            var cp = scalar.value
            withUnsafeBytes(of: &cp) { emitData.append(contentsOf: $0) }
            for v in logProbs {
                var bits = Float32(v).bitPattern
                withUnsafeBytes(of: &bits) { emitData.append(contentsOf: $0) }
            }
        }
        return (startData, transData, emitData)
    }
}

/// HMM hidden states for jieba's character-position tagger.
///
/// State semantics (per `jieba.finalseg`):
///
///   * `begin`  — first character of a multi-char word.
///   * `middle` — interior character of a 3+ char word.
///   * `end`    — final character of a multi-char word.
///   * `single` — a one-character word.
///
/// `rawValue` doubles as the stable index used by every table in this
/// file, so reordering this enum would re-interpret the binary tables
/// (the index order matches the original jieba `B/M/E/S` enumeration).
public enum JiebaHmmState: Int, CaseIterable, Sendable {
    case begin = 0
    case middle = 1
    case end = 2
    case single = 3
}
