import CoreML
import XCTest

@testable import FluidAudio

/// CAM++ embeddings against the PyTorch reference (#985).
///
/// The reference is 3D-Speaker's `CAMPPlus` with `campplus_cn_common.bin`
/// (iic/speech_campplus_sv_zh-cn_16k-common), fed FunASR's `extract_feature` fbank:
/// Kaldi fbank, then the per-utterance mean subtracted.
final class CampPlusEmbedderTests: XCTestCase {

    func testSubtractUtteranceMeanCentersEveryBandWithStrides() throws {
        // [1, 3 frames, 2 bands] stored with a padded frame stride of 4.
        var storage: [Float32] = [
            1, 10, -99, -99,
            2, 20, -99, -99,
            3, 30, -99, -99,
        ]
        try storage.withUnsafeMutableBytes { raw in
            let fbank = try MLMultiArray(
                dataPointer: raw.baseAddress!, shape: [1, 3, 2], dataType: .float32, strides: [12, 4, 1])
            try CampPlusEmbedder.subtractUtteranceMean(fbank)
        }
        XCTAssertEqual(storage, [-1, -10, -99, -99, 0, 0, -99, -99, 1, 10, -99, -99])
    }

    func testSubtractUtteranceMeanRejectsAnUnexpectedLayout() throws {
        let fbank = try MLMultiArray(shape: [3, 80], dataType: .float32)
        XCTAssertThrowsError(try CampPlusEmbedder.subtractUtteranceMean(fbank))
    }

    /// A real 16.9 s recording from the ASR test fixtures. Without the mean subtraction this clip
    /// scored a cosine of 0.67 to the reference; with it, 0.996.
    func testEmbeddingMatchesTheReferenceOnRealSpeech() async throws {
        let name = "03-diff-explanation-16.9s.wav"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else {
            throw XCTSkip("fixture \(name) missing from test bundle")
        }
        let embedder: CampPlusEmbedder
        do {
            embedder = try await CampPlusEmbedder.load()
        } catch {
            throw XCTSkip("CAM++ models unavailable in this environment: \(error)")
        }
        let embedding = try await embedder.embed(audioURL: url)
        XCTAssertEqual(embedding.count, CampPlusEmbedder.embeddingDim)
        let cosine = CampPlusEmbedder.cosine(embedding, Self.referenceEmbedding03)
        XCTAssertGreaterThan(cosine, 0.99, "cosine to the PyTorch reference: \(cosine)")
    }

    /// The reference embedding of `03-diff-explanation-16.9s.wav`, L2-normalized.
    private static let referenceEmbedding03: [Float] = [
        0.06401, 0.06399, 0.16433, -0.14174, 0.00074, -0.04400, -0.04568, 0.01061,
        0.02971, -0.01334, -0.03438, 0.01674, -0.06457, -0.10395, -0.03064, -0.03028,
        0.04982, -0.00722, -0.02419, 0.01253, 0.04480, 0.00214, -0.01389, -0.05800,
        0.12038, 0.04152, -0.02646, 0.09497, -0.04773, 0.09279, -0.00958, -0.12299,
        -0.02580, 0.07764, -0.10720, -0.00564, 0.02293, -0.07939, -0.04890, 0.09210,
        0.03539, -0.00749, -0.06467, 0.05902, -0.03969, -0.02243, 0.03502, 0.00828,
        0.08323, 0.01685, 0.04469, 0.04127, 0.12493, 0.06788, 0.01950, -0.09216,
        0.03677, -0.13885, -0.04442, 0.02034, -0.01370, 0.02518, -0.06177, -0.02019,
        -0.17661, 0.16729, 0.01022, -0.05125, 0.02963, 0.01495, 0.00018, 0.12297,
        0.03786, -0.04195, -0.00174, -0.09107, -0.06466, 0.03450, 0.01815, 0.02734,
        0.04900, -0.04076, 0.01743, 0.00558, -0.01115, -0.02493, 0.15080, -0.17669,
        0.00847, 0.00721, -0.03739, 0.02059, -0.05682, -0.05383, 0.00370, -0.04306,
        -0.02142, 0.02752, -0.04769, -0.07747, 0.09353, -0.08987, -0.00141, -0.03930,
        -0.02780, -0.08065, 0.00856, -0.08571, 0.08289, 0.00216, -0.04766, -0.02105,
        -0.01870, -0.08324, -0.02802, 0.07538, 0.00455, -0.07831, 0.04302, 0.15250,
        -0.06955, -0.00022, -0.00174, -0.12875, -0.20018, -0.10299, -0.16209, -0.10074,
        -0.02637, -0.00999, 0.05858, -0.04777, -0.08264, 0.18719, 0.00221, 0.00079,
        0.04731, -0.03487, -0.02188, 0.00882, 0.21329, 0.04098, 0.00852, -0.08918,
        -0.05353, 0.02062, -0.09261, 0.01918, -0.09978, -0.06912, -0.09628, -0.03868,
        0.09402, 0.08269, -0.09796, -0.00330, -0.14189, -0.08816, 0.06196, -0.14228,
        -0.11511, 0.01401, -0.11708, -0.08712, 0.13352, 0.08505, -0.01739, -0.07736,
        -0.08167, -0.03337, 0.01242, 0.09821, 0.06028, 0.03863, -0.04266, -0.00260,
        -0.00634, -0.01301, 0.06045, 0.06733, -0.02630, -0.07665, -0.02752, 0.10946,
        0.05232, 0.04936, 0.03680, 0.05166, 0.00569, -0.02157, -0.15029, -0.00175,
    ]
}
