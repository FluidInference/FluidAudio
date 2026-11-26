import CoreML
import XCTest

@testable import FluidAudio

final class KokoroAneV3AlignmentTests: XCTestCase {

    func testRepeatsEachTokenOverItsFrames() throws {
        // Three tokens of two channels; the second token has no frames.
        let d = try KokoroAneV3Arrays.float16Array(shape: [1, 3, 2], from: [1, 2, 3, 4, 5, 6])
        let tEn = try KokoroAneV3Arrays.float16Array(shape: [1, 2, 3], from: [10, 20, 30, 40, 50, 60])
        let (en, asr) = try KokoroAneV3Alignment.expand(d: d, tEn: tEn, durations: [2, 0, 3])
        XCTAssertEqual(en.shape, [1, 2, 5])
        XCTAssertEqual(asr.shape, [1, 2, 5])
        XCTAssertEqual(KokoroAneV3Arrays.readFloats(en), [1, 1, 5, 5, 5, 2, 2, 6, 6, 6])
        XCTAssertEqual(KokoroAneV3Arrays.readFloats(asr), [10, 10, 30, 30, 30, 40, 40, 60, 60, 60])
    }

    func testReadsStridedInputs() throws {
        // `d` laid out channel-major, as a Core ML output may be: tokens [1, 2, 3] and [4, 5, 6], strides [6, 1, 2].
        let storage = try KokoroAneV3Arrays.float16Array(shape: [6], from: [1, 4, 2, 5, 3, 6])
        let d = try MLMultiArray(
            dataPointer: storage.dataPointer, shape: [1, 2, 3], dataType: .float16, strides: [6, 1, 2])
        let tEn = try KokoroAneV3Arrays.float16Array(shape: [1, 1, 2], from: [7, 8])
        let (en, asr) = try KokoroAneV3Alignment.expand(d: d, tEn: tEn, durations: [1, 2])
        withExtendedLifetime(storage) {
            XCTAssertEqual(KokoroAneV3Arrays.readFloats(en), [1, 4, 4, 2, 5, 5, 3, 6, 6])
            XCTAssertEqual(KokoroAneV3Arrays.readFloats(asr), [7, 8, 8])
        }
    }

    func testRejectsMismatchedInputs() throws {
        let d = try KokoroAneV3Arrays.float16Array(shape: [1, 2, 2], from: [1, 2, 3, 4])
        let tEn = try KokoroAneV3Arrays.float16Array(shape: [1, 2, 2], from: [1, 2, 3, 4])
        XCTAssertThrowsError(try KokoroAneV3Alignment.expand(d: d, tEn: tEn, durations: [1, 1, 1]))
        XCTAssertThrowsError(try KokoroAneV3Alignment.expand(d: d, tEn: tEn, durations: [1, -1]))
        let single = try KokoroAneV3Arrays.float32Array(shape: [1, 2, 2], from: [1, 2, 3, 4])
        XCTAssertThrowsError(try KokoroAneV3Alignment.expand(d: single, tEn: tEn, durations: [1, 1]))
    }

    /// The native expansion gives the published `KokoroAlignment` graph's output bit for bit,
    /// including at its default shape of 48 tokens. The graph runs on `.cpuOnly` here: on
    /// `.cpuAndNeuralEngine` it traps in BNNS at 48 tokens (#1172).
    func testMatchesTheAlignmentGraph() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKORO_V3"] == "1",
            "Explicitly enable the bounded real-model integration check")
        let directory = ProcessInfo.processInfo.environment["KOKORO_V3_TEST_CACHE"].map { URL(fileURLWithPath: $0) }
        let root = try await KokoroAneV3Assets.ensure(variant: .english, directory: directory, fastVocoder: false)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        // The pipeline no longer loads this graph, so compile its published source when needed.
        var compiled = root.appendingPathComponent("base/KokoroAlignment.mlmodelc")
        if !KokoroAneV3Assets.isCompiledBundle(at: compiled) {
            compiled = try await MLModel.compileModel(at: root.appendingPathComponent("base/KokoroAlignment.mlpackage"))
        }
        let graph = try MLModel(contentsOf: compiled, configuration: configuration)
        for tokens in [2, 3, 47, 48, 49, 120, 512] {
            // Durations of 1 to 3 frames, as PostAlbert predicts for most tokens, within the 2000-frame cap.
            let durations = (0..<tokens).map { Int32(1 + ($0 * 7) % 3) }
            let d = try KokoroAneV3Arrays.float16Array(
                shape: [1, tokens, 640], from: (0..<tokens * 640).map { Float(($0 * 37) % 401) / 97 - 2 })
            let tEn = try KokoroAneV3Arrays.float16Array(
                shape: [1, 512, tokens], from: (0..<512 * tokens).map { Float(($0 * 53) % 509) / 101 - 2.5 })
            let output = try await graph.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "pred_dur": KokoroAneV3Arrays.int32Array(shape: [1, tokens], from: durations),
                    "d": d, "t_en": tEn,
                ]))
            let (en, asr) = try KokoroAneV3Alignment.expand(d: d, tEn: tEn, durations: durations)
            for (name, native) in [("en", en), ("asr", asr)] {
                let expected = try XCTUnwrap(output.featureValue(for: name)?.multiArrayValue)
                XCTAssertEqual(native.shape, expected.shape, "\(name) at \(tokens) tokens")
                XCTAssertEqual(
                    KokoroAneV3Arrays.readFloats(native), KokoroAneV3Arrays.readFloats(expected),
                    "\(name) at \(tokens) tokens")
            }
        }
    }

    /// #1172: these sentences come to 46 phonemes, 48 tokens with the boundary tokens, and
    /// trapped the process under the default routing before Alignment ran natively.
    func testSynthesizesSentencesOfFortyEightTokens() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_KOKORO_V3"] == "1",
            "Explicitly enable the bounded real-model integration check")
        let directory = ProcessInfo.processInfo.environment["KOKORO_V3_TEST_CACHE"].map { URL(fileURLWithPath: $0) }
        let manager = KokoroAneManager(directory: directory, version: .v3)
        for text in [
            "Your next meeting is the design review at four.", "Lunch with Sam moves to three, so plan ahead.",
        ] {
            let result = try await manager.synthesizeDetailed(text: text)
            XCTAssertEqual(result.encoderTokens, 48, text)
            XCTAssertFalse(result.samples.isEmpty, text)
        }
    }
}
