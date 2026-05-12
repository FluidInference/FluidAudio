import XCTest

@testable import FluidAudio

final class StyleTTS2DiffusionScheduleTests: XCTestCase {

    // MARK: - Karras sigmas

    func testKarrasSigmasReturnLengthIsNumStepsPlusPad() {
        let sigmas = StyleTTS2DiffusionSchedule.karrasSigmas(numSteps: 5)
        // numSteps + 1 (final padded zero terminator)
        XCTAssertEqual(sigmas.count, 6)
    }

    func testKarrasSigmasFirstEqualsSigmaMaxAndLastBeforePadEqualsSigmaMin() {
        let sigmaMin = 0.05
        let sigmaMax = 4.0
        let rho = 7.0
        let sigmas = StyleTTS2DiffusionSchedule.karrasSigmas(
            numSteps: 8, sigmaMin: sigmaMin, sigmaMax: sigmaMax, rho: rho)
        // Boundary conditions: at i=0, frac=0 → inner=sigmaMax^(1/ρ) → sigma=sigmaMax.
        // At i=numSteps-1, frac=1 → inner=sigmaMin^(1/ρ) → sigma=sigmaMin.
        XCTAssertEqual(sigmas[0], sigmaMax, accuracy: 1e-9)
        XCTAssertEqual(sigmas[7], sigmaMin, accuracy: 1e-9)
    }

    func testKarrasSigmasTerminatorIsZero() {
        let sigmas = StyleTTS2DiffusionSchedule.karrasSigmas(numSteps: 4)
        XCTAssertEqual(sigmas.last, 0)
    }

    func testKarrasSigmasMonotonicallyDecreasing() {
        let sigmas = StyleTTS2DiffusionSchedule.karrasSigmas(numSteps: 16)
        // All non-pad entries strictly decreasing.
        for i in 1..<(sigmas.count - 1) {
            XCTAssertLessThan(sigmas[i], sigmas[i - 1])
        }
        // Pad terminator is below the smallest interior sigma.
        XCTAssertLessThan(sigmas[sigmas.count - 1], sigmas[sigmas.count - 2])
    }

    // MARK: - StyleTTS2NoiseSource RNG

    func testNoiseSourceIsDeterministicForSameSeed() {
        var a = StyleTTS2NoiseSource(seed: 42)
        var b = StyleTTS2NoiseSource(seed: 42)
        let arrA = a.nextGaussianArray(count: 64)
        let arrB = b.nextGaussianArray(count: 64)
        XCTAssertEqual(arrA, arrB)
    }

    func testNoiseSourceDifferentSeedsDiverge() {
        var a = StyleTTS2NoiseSource(seed: 1)
        var b = StyleTTS2NoiseSource(seed: 2)
        let arrA = a.nextGaussianArray(count: 32)
        let arrB = b.nextGaussianArray(count: 32)
        XCTAssertNotEqual(arrA, arrB)
    }

    func testNoiseSourceZeroSeedIsHandled() {
        // Seed 0 collides with the SplitMix64 trivial-state issue; the
        // implementation reroutes to a fixed non-zero state.
        var rng = StyleTTS2NoiseSource(seed: 0)
        let v = rng.nextGaussianArray(count: 8)
        XCTAssertEqual(v.count, 8)
        // At least one non-zero sample (probability of all-zero is 0).
        XCTAssertTrue(v.contains { $0 != 0 })
    }

    func testNoiseSourceGaussianStatsRoughlyZeroMeanUnitVar() {
        var rng = StyleTTS2NoiseSource(seed: 0xC0FFEE)
        let n = 8192
        let samples = rng.nextGaussianArray(count: n)
        let mean = samples.reduce(0.0) { $0 + Double($1) } / Double(n)
        let variance = samples.reduce(0.0) { $0 + Double($1 - Float(mean)) * Double($1 - Float(mean)) } / Double(n)
        // Loose bounds — Box-Muller off SplitMix64 should comfortably fit.
        XCTAssertEqual(mean, 0.0, accuracy: 0.1)
        XCTAssertEqual(variance, 1.0, accuracy: 0.15)
    }
}
