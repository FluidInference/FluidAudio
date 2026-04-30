import XCTest

@testable import FluidAudio

/// Behavior tests for `MagpieMT19937` and the sampler RNG wrapper.
///
/// Bit-exact parity against NumPy (`np.random.get_state()`, `random_sample`,
/// `np.random.choice`) lives in the mobius reference repo — it's a one-time
/// port-verification artifact, not a runtime invariant. Here we only assert
/// the production-relevant properties: same seed → same draws, different seeds
/// diverge, and the fp32 sampler path stays in lock-step with the fp64 RNG
/// reference for exactly-representable probabilities.
final class MagpieMT19937Tests: XCTestCase {

    // MARK: - Float-overload sanity (sampler path)

    /// The sampler hands fp32 probability vectors to the RNG. The fp32
    /// `MagpieSamplerRng.numpyChoice(probs:)` should match the fp64
    /// `MagpieMT19937.numpyChoice` when probabilities are exactly representable
    /// in fp32.
    func testSamplerRngMatchesNumpyChoiceForExactFp32Probs() {
        // Exactly representable fp32 values (powers of 2 inverses).
        let probs32: [Float] = [0.5, 0.25, 0.125, 0.0625, 0.0625]
        let probs64: [Double] = probs32.map { Double($0) }

        // Reference uses MT19937 directly with fp64 probs.
        let referenceMt = MagpieMT19937(seed: 12_345)
        var referenceDraws: [Int] = []
        for _ in 0..<32 {
            referenceDraws.append(referenceMt.numpyChoice(probs: probs64))
        }

        // Sampler RNG goes through the fp32 path with the same seed.
        let samplerRng = MagpieSamplerRng(seed: 12_345)
        var samplerDraws: [Int] = []
        for _ in 0..<32 {
            samplerDraws.append(samplerRng.numpyChoice(probs: probs32))
        }

        XCTAssertEqual(samplerDraws, referenceDraws)
    }

    // MARK: - Determinism

    func testTwoInstancesWithSameSeedProduceSameSequence() {
        let a = MagpieMT19937(seed: 0xDEAD_BEEF)
        let b = MagpieMT19937(seed: 0xDEAD_BEEF)
        for _ in 0..<1_000 {
            XCTAssertEqual(a.genrandInt32(), b.genrandInt32())
        }
    }

    func testDifferentSeedsDiverge() {
        let a = MagpieMT19937(seed: 1)
        let b = MagpieMT19937(seed: 2)
        var diff = 0
        for _ in 0..<256 {
            if a.genrandInt32() != b.genrandInt32() { diff += 1 }
        }
        XCTAssertGreaterThan(diff, 200, "different seeds should diverge in nearly every draw")
    }
}
