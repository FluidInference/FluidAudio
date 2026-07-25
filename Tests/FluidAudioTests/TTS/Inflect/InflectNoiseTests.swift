import XCTest

@testable import FluidAudio

final class InflectNoiseTests: XCTestCase {

    func testSameSeedIsDeterministic() {
        var a = InflectNoise(seed: 42)
        var b = InflectNoise(seed: 42)
        for _ in 0..<1000 {
            XCTAssertEqual(a.nextGaussian(), b.nextGaussian())
        }
    }

    func testDifferentSeedsDiverge() {
        var a = InflectNoise(seed: 1)
        var b = InflectNoise(seed: 2)
        var anyDifferent = false
        for _ in 0..<100 where a.nextGaussian() != b.nextGaussian() {
            anyDifferent = true
        }
        XCTAssertTrue(anyDifferent)
    }

    func testApproximatelyStandardNormal() {
        var rng = InflectNoise(seed: 7)
        let n = 50_000
        var sum: Double = 0
        var sumSq: Double = 0
        for _ in 0..<n {
            let x = Double(rng.nextGaussian())
            sum += x
            sumSq += x * x
        }
        let mean = sum / Double(n)
        let variance = sumSq / Double(n) - mean * mean
        // Loose bounds — just confirm it's a unit Gaussian, not degenerate.
        XCTAssertEqual(mean, 0, accuracy: 0.05)
        XCTAssertEqual(variance, 1, accuracy: 0.1)
    }

    func testFillMatchesSequentialDraws() {
        var a = InflectNoise(seed: 99)
        var b = InflectNoise(seed: 99)
        var buffer = [Float](repeating: 0, count: 16)
        a.fill(&buffer)
        for i in 0..<16 {
            XCTAssertEqual(buffer[i], b.nextGaussian())
        }
    }
}
