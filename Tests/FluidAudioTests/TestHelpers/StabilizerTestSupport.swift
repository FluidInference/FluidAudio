import Foundation

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
struct StabilizerTestTokenizer {
    static let shared = StabilizerTestTokenizer()

    private let mapping: [Int: String] = [
        1: "▁hello",
        3: "▁world",
        4: "!",
        5: "▁stabilized",
        6: "▁test",
        7: "ing",
        8: ",",
        9: "▁streaming",
        10: "streaming",
        11: "##ing",
        12: "▁sta",
        13: "bil",
        14: "ized",
        15: "▁co",
        16: "mmit",
        17: "ment",
        18: "▁again",
        19: "again",
        20: "▁latency",
        21: "▁ms",
        22: "ms",
    ]

    func decode(_ tokenId: Int) -> String? {
        mapping[tokenId]
    }

    func decodeToText(_ tokens: [Int]) -> String {
        tokens.compactMap { decode($0) }.joined().replacingOccurrences(of: "▁", with: " ")
    }
}

struct SeededGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state = 6_364_136_223_846_793_005 &* state &+ 1
        return state
    }
}
