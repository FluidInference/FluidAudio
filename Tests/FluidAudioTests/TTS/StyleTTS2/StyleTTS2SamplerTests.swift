import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for the Karras + ADPM2 sampler.
///
/// All numeric expectations are ground-truthed against the upstream
/// `99b_e2e_coreml.py` reference implementation, run on the same inputs
/// in a numpy session.
final class StyleTTS2SamplerTests: XCTestCase {

    // MARK: - karrasSigmas

    func testKarrasSigmasDefaults() {
        // numpy reference, steps=5, sigma_min=1e-4, sigma_max=3.0, rho=9.0
        let expected: [Float] = [
            3.0,
            0.5579148530960083,
            0.07036224007606506,
            0.0047578150406479836,
            9.999999747378752e-05,
            0.0,
        ]
        let got = StyleTTS2Sampler.karrasSigmas(steps: 5)
        XCTAssertEqual(got.count, expected.count)
        for (g, e) in zip(got, expected) {
            XCTAssertEqual(g, e, accuracy: 1e-6)
        }
    }

    func testKarrasSigmasSingleStep() {
        // steps=1 → just [sigma_max, 0.0].
        let got = StyleTTS2Sampler.karrasSigmas(steps: 1)
        XCTAssertEqual(got.count, 2)
        XCTAssertEqual(got[0], 3.0, accuracy: 1e-6)
        XCTAssertEqual(got[1], 0.0)
    }

    func testKarrasSigmasNonDefaultRho() {
        // numpy reference, steps=3, rho=7.0
        let expected: [Float] = [
            3.0,
            0.09943192452192307,
            9.999999747378752e-05,
            0.0,
        ]
        let got = StyleTTS2Sampler.karrasSigmas(steps: 3, rho: 7.0)
        XCTAssertEqual(got.count, expected.count)
        for (g, e) in zip(got, expected) {
            XCTAssertEqual(g, e, accuracy: 1e-6)
        }
    }

    // MARK: - adpm2Sample

    private func paddedNoise(_ values: [Float]) -> [Float] {
        // Pad a small numeric fixture out to 256 floats (the rest stay zero,
        // and a 0.5-scaling denoiser keeps them at zero forever).
        var noise = [Float](repeating: 0, count: 256)
        for (i, v) in values.enumerated() { noise[i] = v }
        return noise
    }

    /// Closure-based denoiser: `denoised(x, sigma) = x * 0.5`. Pure, no
    /// hidden state, so the only thing being exercised is the sampler math.
    private static let halfDenoise: StyleTTS2Sampler.DenoiseStep = { x, _ in
        x.map { $0 * 0.5 }
    }

    func testAdpm2Sample3StepsMatchesNumpy() async throws {
        // numpy reference: noise=[1,-1,2,0.5,0,…], denoise=x*0.5, steps=3.
        let noise = paddedNoise([1.0, -1.0, 2.0, 0.5])
        let out = try await StyleTTS2Sampler.adpm2Sample(
            steps: 3, noise: noise, denoise: Self.halfDenoise)

        XCTAssertEqual(out.count, 256)
        XCTAssertEqual(out[0], 7.383052349090576, accuracy: 1e-3)
        XCTAssertEqual(out[1], -7.383052349090576, accuracy: 1e-3)
        XCTAssertEqual(out[2], 14.766104698181152, accuracy: 1e-3)
        XCTAssertEqual(out[3], 3.691526174545288, accuracy: 1e-3)
        // Padding remains zero.
        for i in 4..<256 {
            XCTAssertEqual(out[i], 0.0, accuracy: 1e-6)
        }
    }

    func testAdpm2Sample2StepsMatchesNumpy() async throws {
        // numpy reference: same noise, steps=2.
        let noise = paddedNoise([1.0, -1.0, 2.0, 0.5])
        let out = try await StyleTTS2Sampler.adpm2Sample(
            steps: 2, noise: noise, denoise: Self.halfDenoise)

        XCTAssertEqual(out.count, 256)
        XCTAssertEqual(out[0], -63.824729919433594, accuracy: 1e-2)
        XCTAssertEqual(out[1], 63.824729919433594, accuracy: 1e-2)
        XCTAssertEqual(out[2], -127.64945983886719, accuracy: 1e-2)
        XCTAssertEqual(out[3], -31.912364959716797, accuracy: 1e-2)
    }

    func testAdpm2SampleCallsDenoiseExpectedNumberOfTimes() async throws {
        // For N steps the loop runs N times. Each non-final iteration calls
        // denoise twice (regular + midpoint); the final iteration (s_next=0)
        // calls it once. So total = 2*(N-1) + 1 = 2N - 1.
        let counter = SamplerCounter()
        let denoise: StyleTTS2Sampler.DenoiseStep = { x, _ in
            await counter.increment()
            return x.map { $0 * 0.5 }
        }
        let noise = paddedNoise([1.0])
        _ = try await StyleTTS2Sampler.adpm2Sample(
            steps: 5, noise: noise, denoise: denoise)
        let count = await counter.value
        XCTAssertEqual(count, 2 * 5 - 1)
    }

    func testAdpm2SamplePropagatesDenoiseErrors() async {
        struct Boom: Error {}
        let denoise: StyleTTS2Sampler.DenoiseStep = { _, _ in throw Boom() }
        let noise = [Float](repeating: 0, count: 256)
        do {
            _ = try await StyleTTS2Sampler.adpm2Sample(
                steps: 3, noise: noise, denoise: denoise)
            XCTFail("expected throw")
        } catch is Boom {
            // ok
        } catch {
            XCTFail("expected Boom, got \(error)")
        }
    }
}

/// Tiny actor for counting denoise invocations from an async closure.
private actor SamplerCounter {
    private(set) var value: Int = 0
    func increment() { value += 1 }
}
