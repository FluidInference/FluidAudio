@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// The multifunction encoder ladder: function-name parsing, mel slicing to
/// a shorter function, and the late-pause window cuts it enables.
final class EncoderLadderTests: XCTestCase {

    private let sampleRate = ASRConstants.sampleRate
    private let frameSamples = ASRConstants.samplesPerEncoderFrame

    // MARK: - Function names

    func testFunctionNameParsing() {
        XCTAssertEqual(EncoderLadderFunction.seconds(fromFunctionName: "enc_5040ms"), 5.04)
        XCTAssertEqual(EncoderLadderFunction.seconds(fromFunctionName: "enc_10160ms"), 10.16)
        XCTAssertEqual(EncoderLadderFunction.seconds(fromFunctionName: "enc_15000ms"), 15)
        XCTAssertNil(EncoderLadderFunction.seconds(fromFunctionName: "main"))
        XCTAssertNil(EncoderLadderFunction.seconds(fromFunctionName: "enc_5.04s"))
        XCTAssertNil(EncoderLadderFunction.seconds(fromFunctionName: "enc_ms"))
    }

    // MARK: - Mel slicing

    private func melProvider(frames: Int, channels: Int = 128) throws -> MLFeatureProvider {
        let mel = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: frames)], dataType: .float32)
        let pointer = mel.dataPointer.assumingMemoryBound(to: Float.self)
        for channel in 0..<channels {
            for frame in 0..<frames {
                pointer[channel * frames + frame] = Float(channel * 10_000 + frame)
            }
        }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: 801)
        return try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: length),
        ])
    }

    func testSlicingMelKeepsLeadingFramesAndOtherFeatures() throws {
        let input = try melProvider(frames: 1501)
        let sliced = try AsrManager.slicingMel(input, toFrames: 1017)

        let mel = try XCTUnwrap(sliced.featureValue(for: "mel")?.multiArrayValue)
        XCTAssertEqual(mel.shape.map(\.intValue), [1, 128, 1017])
        let pointer = mel.dataPointer.assumingMemoryBound(to: Float.self)
        XCTAssertEqual(pointer[0], 0)
        XCTAssertEqual(pointer[1016], 1016)
        XCTAssertEqual(pointer[1017], 10_000)  // channel 1, frame 0
        XCTAssertEqual(pointer[127 * 1017 + 1016], Float(127 * 10_000 + 1016))
        XCTAssertEqual(sliced.featureValue(for: "mel_length")?.multiArrayValue?[0].intValue, 801)
    }

    func testSlicingMelToItsOwnLengthReturnsTheInput() throws {
        let input = try melProvider(frames: 1017)
        let sliced = try AsrManager.slicingMel(input, toFrames: 1017)
        XCTAssertTrue(sliced as AnyObject === input as AnyObject)
    }

    func testSlicingMelBeyondItsLengthThrows() throws {
        let input = try melProvider(frames: 505)
        XCTAssertThrowsError(try AsrManager.slicingMel(input, toFrames: 1017))
    }

    // MARK: - Late pause cuts

    /// Constant-level samples with quiet stretches (sample-exact zeros) at
    /// the given second ranges.
    private func samples(seconds: Double, quiet: [(Double, Double)]) -> [Float] {
        var audio = [Float](repeating: 0.05, count: Int(seconds * Double(sampleRate)))
        for (start, end) in quiet {
            for index in Int(start * Double(sampleRate))..<Int(end * Double(sampleRate)) {
                audio[index] = 0
            }
        }
        return audio
    }

    func testFrameEnergySumsCoverEveryFrame() throws {
        let processor = ChunkProcessor(audioSamples: samples(seconds: 2.0, quiet: [(1.0, 1.08)]))
        let sums = try processor.frameEnergySums()
        XCTAssertEqual(sums.count, 2 * sampleRate / frameSamples)
        XCTAssertEqual(sums[0], Float(frameSamples) * 0.05 * 0.05, accuracy: 1e-3)
        // The quiet stretch 16 000–17 280 straddles frames 12 (15 360–16 640) and 13 (16 640–17 920).
        XCTAssertLessThan(sums[12], sums[0])
        XCTAssertLessThan(sums[13], sums[0])
    }

    func testLatestPauseCutPicksTheLastQuietRunInTheBand() throws {
        // Window [0, 15 s); band = last 5 s. Quiet stretches at 11.0–11.5 s and 13.0–13.4 s:
        // the later run wins, cut at its midpoint.
        let processor = ChunkProcessor(
            audioSamples: samples(seconds: 40, quiet: [(11.0, 11.5), (13.0, 13.4)]))
        let sums = try processor.frameEnergySums()
        let cut = try XCTUnwrap(
            processor.latestPauseCut(previousStart: 0, chunkSamples: 15 * sampleRate, frameEnergySums: sums))
        XCTAssertEqual(cut % frameSamples, 0)
        XCTAssertEqual(Double(cut) / Double(sampleRate), 13.2, accuracy: 0.17)
    }

    func testLatestPauseCutIgnoresAQuietBlipAndPausesBeforeTheBand() throws {
        // One quiet frame (80 ms) at 14 s is a blip, not a pause; the real pause at 7 s precedes the band.
        let processor = ChunkProcessor(
            audioSamples: samples(seconds: 40, quiet: [(7.0, 7.6), (14.0, 14.08)]))
        let sums = try processor.frameEnergySums()
        XCTAssertNil(processor.latestPauseCut(previousStart: 0, chunkSamples: 15 * sampleRate, frameEnergySums: sums))
    }

    func testLatestPauseCutIsRelativeToTheWindowStart() throws {
        let processor = ChunkProcessor(audioSamples: samples(seconds: 60, quiet: [(33.0, 33.4)]))
        let sums = try processor.frameEnergySums()
        let start = 20 * sampleRate
        let cut = try XCTUnwrap(
            processor.latestPauseCut(previousStart: start, chunkSamples: 15 * sampleRate, frameEnergySums: sums))
        XCTAssertEqual(Double(cut) / Double(sampleRate), 33.2, accuracy: 0.17)
        XCTAssertNil(processor.latestPauseCut(previousStart: 0, chunkSamples: 15 * sampleRate, frameEnergySums: sums))
    }

    func testSilenceAlignedStartsMarkPauseCutsOnlyWithTheLatePauseRule() throws {
        let audio = samples(seconds: 45, quiet: [(13.0, 13.4), (27.0, 27.4)])
        let processor = ChunkProcessor(audioSamples: audio)
        let chunkSamples = 15 * sampleRate - 160 / frameSamples * frameSamples
        let stride = chunkSamples - 2 * sampleRate

        let late = try processor.silenceAlignedChunkStarts(
            chunkSamples: chunkSamples, strideSamples: stride, canUseWarmupPrefix: false, preferLatePause: true)
        XCTAssertGreaterThanOrEqual(late.count, 3)
        XCTAssertFalse(late[0].pauseCut)
        XCTAssertTrue(late[1].pauseCut)
        XCTAssertEqual(Double(late[1].start) / Double(sampleRate), 13.2, accuracy: 0.17)
        XCTAssertTrue(late[2].pauseCut)
        XCTAssertEqual(Double(late[2].start) / Double(sampleRate), 27.2, accuracy: 0.17)

        // The two-tier search flags its near-silence starts too (a pause is a
        // pause; `process` only trims to the flag with a ladder loaded). Every
        // flagged start must sit inside a quiet stretch.
        let overlap = try processor.silenceAlignedChunkStarts(
            chunkSamples: chunkSamples, strideSamples: stride, canUseWarmupPrefix: false)
        for decision in overlap where decision.pauseCut {
            let seconds = Double(decision.start) / Double(sampleRate)
            XCTAssertTrue(
                (12.9...13.5).contains(seconds) || (26.9...27.5).contains(seconds),
                "pause-cut start at \(seconds) s is not inside a quiet stretch")
        }
    }
}
