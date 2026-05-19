import XCTest

@testable import FluidAudio

final class Supertonic3LatentSamplerTests: XCTestCase {

    // MARK: - mask()

    func testMaskShapeAndOnesPerRow() {
        let m = Supertonic3LatentSampler.mask(lengths: [3, 1, 5], maxLen: 5)
        XCTAssertEqual(m.count, 3 * 5)
        // Row 0: three ones then two zeros.
        XCTAssertEqual(Array(m[0..<5]), [1, 1, 1, 0, 0])
        XCTAssertEqual(Array(m[5..<10]), [1, 0, 0, 0, 0])
        XCTAssertEqual(Array(m[10..<15]), [1, 1, 1, 1, 1])
    }

    func testMaskClampsLengthAtMaxLen() {
        let m = Supertonic3LatentSampler.mask(lengths: [9], maxLen: 4)
        XCTAssertEqual(m, [1, 1, 1, 1])
    }

    // MARK: - sampleNoisyLatent()

    func testNoisyLatentDimensions() {
        // 1 s of audio at 44.1 kHz with baseChunkSize 512 and chunkCompress 6
        // → chunkSize 3072, latentLen ceil(44100 / 3072) = 15.
        let (noisy, mask, dims) = Supertonic3LatentSampler.sampleNoisyLatent(
            durations: [1.0],
            sampleRate: 44_100,
            baseChunkSize: 512,
            chunkCompress: 6,
            latentDim: 24,
            rng: { 0.5 })
        XCTAssertEqual(dims.bsz, 1)
        XCTAssertEqual(dims.channels, 24 * 6)
        XCTAssertEqual(dims.length, 15)
        XCTAssertEqual(noisy.count, dims.bsz * dims.channels * dims.length)
        XCTAssertEqual(mask.count, dims.bsz * dims.length)
    }

    func testSeededRngProducesDeterministicOutput() {
        // Deterministic RNG → identical buffers across runs (Box-Muller is
        // pure given the same uniform sequence).
        let stream: [Float] = (0..<10_000).map { Float(($0 + 7) % 97) / 97.0 }
        var idxA = 0
        var idxB = 0
        let (a, _, _) = Supertonic3LatentSampler.sampleNoisyLatent(
            durations: [0.2], sampleRate: 44_100,
            baseChunkSize: 512, chunkCompress: 6, latentDim: 24,
            rng: {
                defer { idxA += 1 }
                return stream[idxA % stream.count]
            })
        let (b, _, _) = Supertonic3LatentSampler.sampleNoisyLatent(
            durations: [0.2], sampleRate: 44_100,
            baseChunkSize: 512, chunkCompress: 6, latentDim: 24,
            rng: {
                defer { idxB += 1 }
                return stream[idxB % stream.count]
            })
        XCTAssertEqual(a.count, b.count)
        XCTAssertEqual(a, b)
    }

    func testPaddingPositionsAreZeroed() {
        // Batch of two utterances where one is shorter; padding columns
        // beyond the shorter utterance's valid length must be zero.
        let (noisy, _, dims) = Supertonic3LatentSampler.sampleNoisyLatent(
            durations: [0.2, 0.6],
            sampleRate: 44_100,
            baseChunkSize: 512,
            chunkCompress: 6,
            latentDim: 24,
            rng: { 0.5 })
        let chunkSize = 512 * 6
        let shortValid = (Int(0.2 * 44_100) + chunkSize - 1) / chunkSize
        let row0Start = 0
        // Sample channel 0, padding column shortValid..<dims.length must be 0.
        for t in shortValid..<dims.length {
            XCTAssertEqual(
                noisy[row0Start + t], 0,
                "padding position \(t) for short utterance must be zero")
        }
    }

    func testEmptyDurationsProducesEmptyTensors() {
        let (noisy, mask, dims) = Supertonic3LatentSampler.sampleNoisyLatent(
            durations: [],
            sampleRate: 44_100,
            baseChunkSize: 512,
            chunkCompress: 6,
            latentDim: 24,
            rng: { 0.5 })
        XCTAssertEqual(dims.bsz, 0)
        XCTAssertEqual(noisy, [])
        XCTAssertEqual(mask, [])
    }
}
