import XCTest

@testable import FluidAudio

final class NMESCClusteringTests: XCTestCase {

    /// Deterministic pseudo-random unit vector near a base direction.
    private func jitteredVector(base: [Double], seed: Int, jitter: Double) -> [Double] {
        var v = base
        var state = UInt64(seed &* 2_654_435_761)
        for i in 0..<v.count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let r = Double(state >> 11) / Double(UInt64.max >> 11) - 0.5
            v[i] += r * jitter
        }
        let norm = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return v.map { $0 / norm }
    }

    // n is chosen to match the production regime (~300-600 window embeddings
    // per scene). At tiny n the pruning grid degenerates (keep=2 neighbors)
    // and even clean blobs fragment — a separate known small-n limitation.
    func testSingleVoiceCollapsesToOneCluster() {
        var base = [Double](repeating: 0, count: 32)
        base[0] = 1
        let points = (0..<150).map { jitteredVector(base: base, seed: $0, jitter: 0.15) }
        let labels = NMESCClustering().cluster(embeddingFeatures: points)
        XCTAssertEqual(Set(labels).count, 1, "solo-voice cloud must collapse to k=1")
    }

    func testTwoSeparatedVoicesStayTwoClusters() {
        var a = [Double](repeating: 0, count: 32)
        a[0] = 1
        var b = [Double](repeating: 0, count: 32)
        b[1] = 1
        let points =
            (0..<150).map { jitteredVector(base: a, seed: $0, jitter: 0.1) }
            + (0..<150).map { jitteredVector(base: b, seed: 1000 + $0, jitter: 0.1) }
        let labels = NMESCClustering().cluster(embeddingFeatures: points)
        XCTAssertEqual(Set(labels).count, 2, "two distinct voices must stay k=2")
    }
}
