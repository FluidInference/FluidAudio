import Foundation
import XCTest

@testable import FluidAudio

final class Nemotron3StateUpdaterTests: XCTestCase {

    private var config: Nemotron3Config { .low }

    private func makeUpdater() -> Nemotron3StateUpdater {
        Nemotron3StateUpdater(
            config: config,
            silenceEmbedding: [Float](repeating: 0.01, count: config.preEncoderDims))
    }

    /// Packed predictions sized for the current state: [spkcache | fifo | chunk] left-packed.
    private func makePredictions(
        state: Nemotron3StreamingState, chunkLen: Int, value: Float = 0.9
    ) -> (preds: [Float], hires: [Float]) {
        let s = config.numSpeakers
        let packed = config.packedFrames
        var preds = [Float](repeating: 0, count: packed * s)
        let valid = state.spkcacheLength + state.fifoLength + chunkLen
        for frame in 0..<valid {
            preds[frame * s] = value  // speaker 0 active
        }
        var hires = [Float](repeating: 0, count: packed * config.upsampleFactor * s)
        for frame in 0..<(valid * config.upsampleFactor) {
            hires[frame * s] = value
        }
        return (preds, hires)
    }

    private func makeChunkEmbeddings(frames: Int, fill: Float = 0.5) -> [Float] {
        [Float](repeating: fill, count: config.chunkEncFrames * config.preEncoderDims)
    }

    // MARK: - FIFO accumulation

    func testFirstChunkGoesToFifo() throws {
        let updater = makeUpdater()
        var state = Nemotron3StreamingState(config: config)
        let chunkLen = config.chunkLen  // full chunk, rc consumed
        let (preds, hires) = makePredictions(state: state, chunkLen: chunkLen)

        let result = try updater.update(
            state: &state,
            chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
            chunkEncLength: config.chunkEncFrames,
            predictions: preds,
            highResPredictions: hires,
            lc: 0,
            rc: config.chunkRightContext
        )

        XCTAssertEqual(state.fifoLength, chunkLen, "core frames should land in FIFO")
        XCTAssertEqual(state.spkcacheLength, 0, "no cache update before FIFO overflow")
        XCTAssertEqual(result.frameCount, chunkLen * config.upsampleFactor, "10ms output per core frame")
        XCTAssertEqual(result.probabilities.count, result.frameCount * config.numSpeakers)
        XCTAssertEqual(result.probabilities[0], 0.9, accuracy: 1e-6)
    }

    func testFifoPopMovesFramesToSpkcache() throws {
        let updater = makeUpdater()
        var state = Nemotron3StreamingState(config: config)

        // Fill FIFO just below capacity, then push one more chunk to trigger a pop.
        var steps = 0
        while state.fifoLength + config.chunkLen <= config.fifoLen {
            let (preds, hires) = makePredictions(state: state, chunkLen: config.chunkLen)
            _ = try updater.update(
                state: &state,
                chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
                chunkEncLength: config.chunkEncFrames,
                predictions: preds, highResPredictions: hires,
                lc: 0, rc: config.chunkRightContext)
            steps += 1
        }
        XCTAssertEqual(state.spkcacheLength, 0)
        let fifoBefore = state.fifoLength

        let (preds, hires) = makePredictions(state: state, chunkLen: config.chunkLen)
        _ = try updater.update(
            state: &state,
            chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
            chunkEncLength: config.chunkEncFrames,
            predictions: preds, highResPredictions: hires,
            lc: 0, rc: config.chunkRightContext)

        // NeMo pop rule: pop = min(combined, max(updatePeriod, overflow))
        let combined = fifoBefore + config.chunkLen
        let expectedPop = min(combined, max(config.spkcacheUpdatePeriod, combined - config.fifoLen))
        XCTAssertEqual(state.spkcacheLength, expectedPop)
        XCTAssertEqual(state.fifoLength, combined - expectedPop)
    }

    func testZeroChunkFlushesFifo() throws {
        let updater = makeUpdater()
        var state = Nemotron3StreamingState(config: config)

        let (preds1, hires1) = makePredictions(state: state, chunkLen: config.chunkLen)
        _ = try updater.update(
            state: &state,
            chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
            chunkEncLength: config.chunkEncFrames,
            predictions: preds1, highResPredictions: hires1,
            lc: 0, rc: config.chunkRightContext)
        let fifoBefore = state.fifoLength
        XCTAssertGreaterThan(fifoBefore, 0)

        let (preds2, hires2) = makePredictions(state: state, chunkLen: 0)
        let result = try updater.update(
            state: &state,
            chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
            chunkEncLength: 0,
            predictions: preds2, highResPredictions: hires2,
            lc: 0, rc: 0)

        XCTAssertEqual(state.fifoLength, 0, "zero-length chunk must flush the FIFO")
        XCTAssertEqual(state.spkcacheLength, fifoBefore, "flushed frames land in the cache")
        XCTAssertEqual(result.frameCount, 0)
    }

    // MARK: - Compression

    func testCompressionCapsSpkcacheAtCapacity() throws {
        let updater = makeUpdater()
        var state = Nemotron3StreamingState(config: config)

        // Stream enough active chunks to overflow the speaker cache.
        // Each pop moves updatePeriod (222) frames; capacity 264 -> second pop compresses.
        var iterations = 0
        while !state.spkcacheCompressed && iterations < 200 {
            let (preds, hires) = makePredictions(state: state, chunkLen: config.chunkLen)
            _ = try updater.update(
                state: &state,
                chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames),
                chunkEncLength: config.chunkEncFrames,
                predictions: preds, highResPredictions: hires,
                lc: 0, rc: config.chunkRightContext)
            iterations += 1
            XCTAssertLessThanOrEqual(state.spkcacheLength, config.spkcacheLen)
        }
        XCTAssertTrue(state.spkcacheCompressed, "cache should compress after sustained speech")
        XCTAssertEqual(state.spkcacheLength, config.spkcacheLen)
    }

    func testCompressionInsertsSilenceEmbeddingForDisabledSlots() throws {
        let updater = makeUpdater()
        var state = Nemotron3StreamingState(config: config)

        // All-silence predictions: every score disables, so compression fills slots with the
        // learned silence embedding.
        var iterations = 0
        while !state.spkcacheCompressed && iterations < 200 {
            let (_, hires) = makePredictions(state: state, chunkLen: config.chunkLen, value: 0.0)
            let silent = [Float](repeating: 0, count: config.packedFrames * config.numSpeakers)
            _ = try updater.update(
                state: &state,
                chunkEmbeddings: makeChunkEmbeddings(frames: config.chunkEncFrames, fill: 0.7),
                chunkEncLength: config.chunkEncFrames,
                predictions: silent, highResPredictions: hires,
                lc: 0, rc: config.chunkRightContext)
            iterations += 1
        }
        XCTAssertTrue(state.spkcacheCompressed)
        // All frames were silent -> every selected slot should carry the silence embedding.
        XCTAssertEqual(state.spkcache[0], 0.01, accuracy: 1e-6)
        // Predictions for silence slots are zeroed.
        XCTAssertEqual(state.spkcachePreds[0], 0, accuracy: 1e-6)
    }

    // MARK: - Config invariants

    func testPresetShapes() {
        XCTAssertEqual(Nemotron3Config.low.chunkMelFrames, 104)
        XCTAssertEqual(Nemotron3Config.low.packedFrames, 541)
        XCTAssertEqual(Nemotron3Config.veryLow.chunkMelFrames, 64)
        XCTAssertEqual(Nemotron3Config.veryLow.packedFrames, 536)
        XCTAssertEqual(Nemotron3Config.ultraLow.chunkMelFrames, 32)
        XCTAssertEqual(Nemotron3Config.ultraLow.packedFrames, 532)
        XCTAssertEqual(Nemotron3Config.offline.chunkMelFrames, 3040)
        XCTAssertEqual(Nemotron3Config.offline.packedFrames, 684)
    }

    func testPresetLookup() {
        XCTAssertNotNil(Nemotron3Config.preset(named: "low"))
        XCTAssertNotNil(Nemotron3Config.preset(named: "offline"))
        XCTAssertNil(Nemotron3Config.preset(named: "bogus"))
    }
}

final class Nemotron3FeatureLoaderTests: XCTestCase {

    func testLoaderEmitsTailWithShrunkRightContext() {
        let config = Nemotron3Config.low
        let mel = config.melFeatures
        // 2.5 core chunks of mel frames, no full right context at the tail.
        let core = config.chunkLen * config.subsamplingFactor
        let frames = core * 2 + core / 2
        let featSeq = [Float](repeating: 1, count: frames * mel)

        var loader = Nemotron3FeatureLoader(
            config: config, featSeq: featSeq, featLength: frames, featSeqLength: frames)
        var chunks: [(length: Int, right: Int)] = []
        while let c = loader.next() {
            chunks.append((c.length, c.rightOffset))
            XCTAssertEqual(c.features.count, config.chunkMelFrames * mel, "fixed capacity padding")
        }

        XCTAssertEqual(chunks.count, 3, "tail chunk must still be emitted")
        XCTAssertEqual(chunks[0].right, config.chunkRightContext * config.subsamplingFactor)
        XCTAssertLessThan(chunks[2].right, config.chunkRightContext * config.subsamplingFactor)
    }
}
