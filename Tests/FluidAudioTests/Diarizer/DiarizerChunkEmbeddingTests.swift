import Foundation
import XCTest

@testable import FluidAudio

/// Tests for `DiarizerManager.buildChunkEmbeddings`, the mapping behind
/// `DiarizerConfig.exposeChunkEmbeddings`.
final class DiarizerChunkEmbeddingTests: XCTestCase {

    private let window = SlidingWindow(start: 20.0, duration: 0.0619, step: 0.016875)

    /// Frames × local speakers, 1.0 where the speaker is active.
    private func activity(frames: Int, active: [Int: ClosedRange<Int>]) -> [[[Float]]] {
        let speakers = 3
        var data = [[Float]](repeating: [Float](repeating: 0, count: speakers), count: frames)
        for (speaker, range) in active {
            for frame in range { data[frame][speaker] = 1.0 }
        }
        return [data]
    }

    func testOneEntryPerLocalSpeakerWithAnId() {
        let embeddings: [[Float]] = [
            [Float](repeating: 0.1, count: 256),
            [Float](repeating: 0.2, count: 256),
            [Float](repeating: 0.3, count: 256),
        ]
        let result = DiarizerManager.buildChunkEmbeddings(
            chunkIndex: 2,
            speakerIds: ["1", "", "3"],
            embeddings: embeddings,
            binarizedSegments: activity(frames: 100, active: [0: 10...39, 1: 0...99, 2: 50...59]),
            slidingWindow: window
        )

        XCTAssertEqual(result.map { $0.speakerId }, ["1", "3"])
        XCTAssertEqual(result.map { $0.speakerIndex }, [0, 2])
        XCTAssertEqual(result.map { $0.chunkIndex }, [2, 2])
        XCTAssertEqual(result[0].embedding256, embeddings[0])
        XCTAssertEqual(result[1].embedding256, embeddings[2])
        XCTAssertTrue(result.allSatisfy { $0.rho128.isEmpty })
    }

    func testSpanRunsFromFirstToLastActiveFrame() {
        let result = DiarizerManager.buildChunkEmbeddings(
            chunkIndex: 0,
            speakerIds: ["1", "", ""],
            embeddings: [[Float](repeating: 0.1, count: 256), [], []],
            binarizedSegments: activity(frames: 100, active: [0: 10...39]),
            slidingWindow: window
        )

        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].startTimeSeconds, window.time(forFrame: 10), accuracy: 1e-9)
        XCTAssertEqual(result[0].endTimeSeconds, window.time(forFrame: 40), accuracy: 1e-9)
    }

    func testSpeakerWithIdButNoActiveFrameIsSkipped() {
        let result = DiarizerManager.buildChunkEmbeddings(
            chunkIndex: 0,
            speakerIds: ["1", "", ""],
            embeddings: [[Float](repeating: 0.1, count: 256), [], []],
            binarizedSegments: activity(frames: 100, active: [:]),
            slidingWindow: window
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testEmptyInputProducesNoEntries() {
        let result = DiarizerManager.buildChunkEmbeddings(
            chunkIndex: 0,
            speakerIds: [],
            embeddings: [],
            binarizedSegments: [],
            slidingWindow: window
        )
        XCTAssertTrue(result.isEmpty)
    }

    func testDefaultConfigDoesNotExposeChunkEmbeddings() {
        XCTAssertFalse(DiarizerConfig().exposeChunkEmbeddings)
        XCTAssertTrue(DiarizerConfig(exposeChunkEmbeddings: true).exposeChunkEmbeddings)
    }
}
