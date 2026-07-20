import XCTest

@testable import FluidAudio

/// Incremental (streaming) mel preprocessing must reproduce the batch
/// center-padded mel stream frame-for-frame, no matter how the audio is
/// batched into `addAudio` calls. Re-padding on every preprocess call used
/// to emit ~17 ms of phantom timeline per call, so with real-time feeding
/// the finalized frame stream ran ~3.5% faster than the audio and every
/// frame-derived timestamp drifted progressively late.
final class SortformerStreamingMelTests: XCTestCase {

    private let config = SortformerConfig.fastV2_1

    /// Deterministic audio with speech-like structure (tones + noise), sized
    /// to not be a multiple of the mel hop so edge handling is exercised.
    private func makeAudio(seconds: Double) -> [Float] {
        let count = Int(16000 * seconds) + 137
        srand48(7)
        return (0..<count).map { i in
            let t = Float(i) / 16000
            let tone = 0.3 * sin(2 * .pi * 220 * t) + 0.15 * sin(2 * .pi * 517 * t)
            return tone + Float(drand48() - 0.5) * 0.05
        }
    }

    /// Feed `audio` in `batchSize`-sample calls, draining every available
    /// feature chunk after each call; optionally flush the trailing frames
    /// the way `finalizeSession` does. Returns the extracted chunks.
    private func streamChunks(
        audio: [Float], batchSize: Int, finalize: Bool
    ) -> (chunks: [[Float]], framesEmitted: Int) {
        let diarizer = SortformerDiarizer(config: config)
        var chunks: [[Float]] = []
        var fed = 0
        while fed < audio.count {
            let end = min(fed + batchSize, audio.count)
            diarizer.addAudio(Array(audio[fed..<end]))
            fed = end
            while let (mel, _) = diarizer.getNextChunkFeatures() {
                chunks.append(mel)
            }
        }
        if finalize {
            diarizer.padAndEmitRemainingMel()
            while let (mel, _) = diarizer.getNextChunkFeatures() {
                chunks.append(mel)
            }
        }
        return (chunks, diarizer.melFramesEmitted)
    }

    /// Any feeding granularity must produce the identical chunk sequence and
    /// the identical total mel frame count.
    func testFeedingGranularityDoesNotChangeFramesOrChunks() throws {
        let audio = makeAudio(seconds: 12)
        let reference = streamChunks(audio: audio, batchSize: audio.count, finalize: true)

        // 160 = one hop; 1600 = 100 ms (typical mic callback); 4093 = prime,
        // misaligned with every internal boundary; 16000 = 1 s.
        for batchSize in [160, 1600, 4093, 16000] {
            let result = streamChunks(audio: audio, batchSize: batchSize, finalize: true)

            XCTAssertEqual(
                result.framesEmitted, reference.framesEmitted,
                "Mel frame count depends on feeding granularity (batchSize \(batchSize))")
            XCTAssertEqual(
                result.chunks.count, reference.chunks.count,
                "Chunk count depends on feeding granularity (batchSize \(batchSize))")
            for (i, (chunk, referenceChunk)) in zip(result.chunks, reference.chunks).enumerated() {
                XCTAssertEqual(chunk.count, referenceChunk.count, "Chunk \(i) size mismatch")
                for j in 0..<chunk.count where chunk[j] != referenceChunk[j] {
                    XCTFail(
                        "Chunk \(i) differs at \(j) for batchSize \(batchSize): "
                            + "\(chunk[j]) != \(referenceChunk[j])")
                    return
                }
            }
        }
    }

    /// The finalized stream must contain exactly the frames batch
    /// center-padded computation produces: `1 + (n + nFFT - melWindow) / hop`.
    func testFinalizedFrameCountIsBatchExact() throws {
        let audio = makeAudio(seconds: 9)
        let melSpectrogram = AudioMelSpectrogram()
        let expected = 1 + (audio.count + melSpectrogram.nFFT - config.melWindow) / config.melStride

        for batchSize in [1600, audio.count] {
            let result = streamChunks(audio: audio, batchSize: batchSize, finalize: true)
            XCTAssertEqual(
                result.framesEmitted, expected,
                "Finalized mel frame count must match batch formula (batchSize \(batchSize))")
        }
    }

    /// Streamed values must equal the batch center-padded mel of the whole
    /// audio, including the session-final frames whose windows overlap the
    /// right pad.
    func testStreamedMelMatchesBatchValues() throws {
        let audio = makeAudio(seconds: 9)
        let melSpectrogram = AudioMelSpectrogram()
        let (batchMel, batchFrames, _) = melSpectrogram.computeFlatTransposed(
            audio: audio, lastAudioSample: 0, paddingMode: .center, expectedFrameCount: nil)

        let diarizer = SortformerDiarizer(config: config)
        for start in stride(from: 0, to: audio.count, by: 1600) {
            diarizer.addAudio(Array(audio[start..<min(start + 1600, audio.count)]))
        }
        diarizer.padAndEmitRemainingMel()
        XCTAssertEqual(diarizer.melFramesEmitted, batchFrames)

        // First chunk covers mel frames [0, core + rc) with no left context,
        // an absolute anchor to the start of the batch stream.
        guard let (firstChunk, firstLength) = diarizer.getNextChunkFeatures() else {
            return XCTFail("No chunk available")
        }
        XCTAssertEqual(firstLength, config.coreFrames + config.chunkRightContext * config.subsamplingFactor)
        for i in 0..<(firstLength * config.melFeatures) {
            XCTAssertEqual(firstChunk[i], batchMel[i], accuracy: 1e-5, "First chunk mismatch at \(i)")
        }

        // Drain the rest; the feature buffer's tail must then equal the batch
        // stream's tail, which pins the right-pad frames emitted at finalize.
        while diarizer.getNextChunkFeatures() != nil {}
        let tail = diarizer.featureBuffer.suffix(64 * config.melFeatures)
        let batchTail = batchMel.prefix(batchFrames * config.melFeatures).suffix(tail.count)
        XCTAssertFalse(tail.isEmpty)
        for (i, (streamed, batch)) in zip(tail, batchTail).enumerated() {
            XCTAssertEqual(streamed, batch, accuracy: 1e-5, "Tail mismatch at \(i)")
        }
    }

    /// After the mel stream is exhausted, further audio is dropped (not
    /// buffered) and reset() must fully restart the stream.
    func testResetAfterExhaustionRestartsMelStream() throws {
        let audio = makeAudio(seconds: 6)
        let reference = streamChunks(audio: audio, batchSize: audio.count, finalize: false)

        let diarizer = SortformerDiarizer(config: config)
        diarizer.addAudio(audio)
        diarizer.padAndEmitRemainingMel()
        let exhaustedFrames = diarizer.melFramesEmitted
        diarizer.addAudio(audio)
        XCTAssertEqual(diarizer.melFramesEmitted, exhaustedFrames, "Audio after exhaustion must be ignored")

        diarizer.reset()
        diarizer.addAudio(audio)
        var chunks: [[Float]] = []
        while let (mel, _) = diarizer.getNextChunkFeatures() { chunks.append(mel) }
        XCTAssertEqual(chunks.count, reference.chunks.count)
        for (i, (chunk, referenceChunk)) in zip(chunks, reference.chunks).enumerated() {
            XCTAssertEqual(chunk, referenceChunk, "Chunk \(i) differs after reset from exhaustion")
        }
    }

    /// A reset diarizer must reproduce the session from scratch (the left pad
    /// is re-seeded, counters cleared).
    func testResetRestartsMelStream() throws {
        let audio = makeAudio(seconds: 6)
        let diarizer = SortformerDiarizer(config: config)

        diarizer.addAudio(audio)
        var firstRun: [[Float]] = []
        while let (mel, _) = diarizer.getNextChunkFeatures() { firstRun.append(mel) }

        diarizer.reset()
        diarizer.addAudio(audio)
        var secondRun: [[Float]] = []
        while let (mel, _) = diarizer.getNextChunkFeatures() { secondRun.append(mel) }

        XCTAssertEqual(firstRun.count, secondRun.count)
        for (i, (first, second)) in zip(firstRun, secondRun).enumerated() {
            XCTAssertEqual(first, second, "Chunk \(i) differs after reset")
        }
    }
}
