import XCTest

@testable import FluidAudio

/// The audio-in streaming front end must reproduce the batch path exactly: the same
/// mel frames as center-padded extraction and the same chunk sequence as
/// `Nemotron3FeatureLoader`, regardless of how the audio is batched into `append`.
final class Nemotron3StreamingFrontendTests: XCTestCase {

    /// Deterministic audio with speech-like structure (tones + noise), sized to not be a
    /// multiple of the mel hop so edge handling is exercised.
    private func makeAudio(seconds: Double) -> [Float] {
        let count = Int(16000 * seconds) + 137
        srand48(11)
        return (0..<count).map { i in
            let t = Float(i) / 16000
            let tone = 0.3 * sin(2 * .pi * 180 * t) + 0.15 * sin(2 * .pi * 733 * t)
            return tone + Float(drand48() - 0.5) * 0.05
        }
    }

    private typealias Chunk = (features: [Float], length: Int, leftOffset: Int, rightOffset: Int)

    private func batchChunks(config: Nemotron3Config, audio: [Float]) -> [Chunk] {
        let (feat, featLength, featSeqLength) = AudioMelSpectrogram().computeFlatTransposed(audio: audio)
        var loader = Nemotron3FeatureLoader(
            config: config, featSeq: feat, featLength: featLength, featSeqLength: featSeqLength)
        var chunks: [Chunk] = []
        while let c = loader.next() { chunks.append(c) }
        return chunks
    }

    private func streamChunks(config: Nemotron3Config, audio: [Float], batchSize: Int) -> [Chunk] {
        var frontend = Nemotron3StreamingFrontend(config: config)
        var chunks: [Chunk] = []
        var fed = 0
        while fed < audio.count {
            let end = min(fed + batchSize, audio.count)
            frontend.append(Array(audio[fed..<end]))
            fed = end
            while let c = frontend.nextChunk(final: false) { chunks.append(c) }
        }
        while let c = frontend.nextChunk(final: true) { chunks.append(c) }
        return chunks
    }

    private func assertSameChunks(
        _ a: [Chunk], _ b: [Chunk], tolerance: Float, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.count, b.count, "chunk count", file: file, line: line)
        for (i, (x, y)) in zip(a, b).enumerated() {
            XCTAssertEqual(x.length, y.length, "chunk \(i) length", file: file, line: line)
            XCTAssertEqual(x.leftOffset, y.leftOffset, "chunk \(i) leftOffset", file: file, line: line)
            XCTAssertEqual(x.rightOffset, y.rightOffset, "chunk \(i) rightOffset", file: file, line: line)
            XCTAssertEqual(x.features.count, y.features.count, "chunk \(i) capacity", file: file, line: line)
            var maxDiff: Float = 0
            for (p, q) in zip(x.features, y.features) { maxDiff = max(maxDiff, abs(p - q)) }
            XCTAssertLessThanOrEqual(maxDiff, tolerance, "chunk \(i) mel values", file: file, line: line)
        }
    }

    /// Streaming chunks equal the batch loader's chunks for the card-standard `low`
    /// profile (short core, right context) at several feeding granularities.
    func testStreamMatchesBatchLoaderLowProfile() {
        let config = Nemotron3Config.low
        let audio = makeAudio(seconds: 9.3)
        let reference = batchChunks(config: config, audio: audio)
        XCTAssertGreaterThan(reference.count, 5)
        // 160 = one hop; 1600 = 100 ms mic callback; 4093 = prime; whole file at once.
        for batch in [160, 1600, 4093, audio.count] {
            assertSameChunks(streamChunks(config: config, audio: audio, batchSize: batch), reference, tolerance: 1e-4)
        }
    }

    /// Same for `fast32` (long core), whose tail chunk is short and right-context-trimmed.
    func testStreamMatchesBatchLoaderFast32Tail() {
        let config = Nemotron3Config.fast32
        let audio = makeAudio(seconds: 7.1)
        let reference = batchChunks(config: config, audio: audio)
        let last = reference.last!
        XCTAssertLessThan(last.length, config.chunkMelFrames, "tail chunk should be short")
        XCTAssertEqual(last.rightOffset, 0, "tail chunk has no right context left")
        assertSameChunks(streamChunks(config: config, audio: audio, batchSize: 1600), reference, tolerance: 1e-4)
    }

    /// Chunks are only released once their right context is fully buffered (no
    /// premature emission), and the total mel frame count after finish is batch-exact.
    func testChunkReleaseWaitsForRightContext() {
        let config = Nemotron3Config.low
        var frontend = Nemotron3StreamingFrontend(config: config)
        let hop = 160
        let latencyFrames = (config.chunkLen + config.chunkRightContext) * config.subsamplingFactor
        let audio = makeAudio(seconds: 3)
        // One sample short of what the first chunk needs: last needed frame is
        // latencyFrames - 1, whose window ends at (latencyFrames - 1) * hop + 256.
        let needed = (latencyFrames - 1) * hop + 256
        frontend.append(Array(audio[0..<(needed - 1)]))
        XCTAssertNil(frontend.nextChunk(final: false))
        frontend.append([audio[needed - 1]])
        XCTAssertNotNil(frontend.nextChunk(final: false))

        frontend.append(Array(audio[needed...]))
        while frontend.nextChunk(final: false) != nil {}
        while frontend.nextChunk(final: true) != nil {}
        let (_, batchFrames, _) = AudioMelSpectrogram().computeFlatTransposed(audio: audio)
        XCTAssertEqual(frontend.melFramesComputed, batchFrames)
    }

    func testResetRestartsStream() {
        let config = Nemotron3Config.low
        let audio = makeAudio(seconds: 4)
        var frontend = Nemotron3StreamingFrontend(config: config)
        frontend.append(audio)
        while frontend.nextChunk(final: false) != nil {}
        frontend.reset()
        XCTAssertEqual(frontend.melFramesComputed, 0)
        XCTAssertNil(frontend.nextChunk(final: false))
        assertSameChunks(
            streamChunks(config: config, audio: audio, batchSize: 1600),
            batchChunks(config: config, audio: audio), tolerance: 1e-4)
    }

    /// The final flush right-pads the stream, so the last frames' windows extend past the
    /// received audio. For lengths where `(N + 112) mod 160 < 15` the buffer trim used to
    /// ask for more samples than remained and trapped. Cover the boundary and a mid case.
    func testFinalFlushDoesNotOverTrimForAnyLengthResidue() {
        let config = Nemotron3Config.low
        let base = makeAudio(seconds: 3)
        for residue in [0, 7, 14, 15, 89] {
            // N with (N + 112) mod 160 == residue
            let n = 160 * 250 - 112 + residue
            let audio = Array(base.prefix(n))
            var frontend = Nemotron3StreamingFrontend(config: config)
            frontend.append(audio)
            while frontend.nextChunk(final: false) != nil {}
            while frontend.nextChunk(final: true) != nil {}
            let (_, batchFrames, _) = AudioMelSpectrogram().computeFlatTransposed(audio: audio)
            XCTAssertEqual(frontend.melFramesComputed, batchFrames, "residue \(residue)")
        }
    }
}
