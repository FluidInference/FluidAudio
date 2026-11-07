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

    /// A real 16.9 s recording from the ASR test fixtures. The v1 models scored a cosine of 0.67
    /// to the reference (0.996 with the mean subtraction); the `_v2` models match it.
    func testEmbeddingMatchesTheReferenceOnRealSpeech() async throws {
        try await assertMatchesReference("03-diff-explanation-16.9s.wav", Self.referenceEmbedding03)
    }

    /// A real 19.8 s recording: 989 frames after the stride-2 TDNN, inside the lengths where the
    /// fp16 v1 CamPlusPlus gave an unrelated embedding (cosine 0.06 to the reference, #985).
    func testEmbeddingMatchesTheReferenceInAFormerlyBrokenLength() async throws {
        try await assertMatchesReference("02-release-readiness-19.8s.wav", Self.referenceEmbedding02)
    }

    private func assertMatchesReference(_ name: String, _ reference: [Float]) async throws {
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
        let cosine = CampPlusEmbedder.cosine(embedding, reference)
        XCTAssertGreaterThan(cosine, 0.999, "\(name): cosine to the PyTorch reference: \(cosine)")
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

    /// The reference embedding of `02-release-readiness-19.8s.wav`, L2-normalized.
    private static let referenceEmbedding02: [Float] = [
        0.03670, -0.06283, 0.08727, -0.03146, 0.07039, -0.02975, -0.10361, 0.02723,
        0.04544, 0.01531, -0.02666, 0.00048, -0.08673, -0.11098, 0.09987, -0.00004,
        0.04156, 0.03638, 0.00820, -0.01215, -0.00703, -0.04664, -0.02740, -0.04428,
        0.01813, 0.06999, 0.01000, 0.16985, -0.06233, 0.04137, 0.00339, -0.11275,
        -0.12655, 0.03146, -0.03477, -0.05904, -0.05062, -0.04350, -0.01530, 0.08947,
        0.04202, 0.03611, 0.01056, 0.02510, -0.07539, -0.00622, 0.05364, -0.04348,
        0.04671, 0.00948, 0.08431, -0.01762, 0.08555, 0.04939, 0.03560, -0.09126,
        -0.02796, -0.07610, -0.06706, 0.02516, -0.03320, 0.07217, 0.03359, 0.01020,
        -0.09823, 0.12125, -0.01113, -0.00952, 0.03151, 0.07742, 0.05747, 0.13879,
        -0.02729, -0.00670, -0.03912, -0.15926, -0.02687, 0.07147, -0.00875, 0.06824,
        0.05321, 0.01177, -0.07769, 0.08835, -0.01993, 0.00457, 0.14418, -0.14486,
        0.01521, -0.10318, -0.01274, 0.04021, -0.02702, 0.03984, -0.00272, 0.00842,
        -0.02190, 0.05264, -0.11008, -0.02716, 0.09248, -0.13584, 0.01380, -0.02089,
        0.03869, -0.09048, 0.07604, -0.05072, 0.06989, -0.07446, -0.05942, -0.03576,
        -0.09613, -0.18851, 0.00439, 0.07397, 0.01619, -0.10042, -0.02105, 0.10889,
        -0.01465, -0.01091, 0.03899, -0.12024, -0.09505, -0.11311, -0.14330, -0.17467,
        -0.11450, -0.09470, 0.06303, -0.07008, -0.07806, 0.09103, -0.05083, 0.07350,
        0.04010, -0.05045, 0.06453, 0.02105, 0.17930, 0.05267, 0.00917, -0.11649,
        0.00463, 0.08434, -0.02910, -0.02009, -0.16850, -0.11586, -0.16218, -0.07839,
        0.04109, 0.08715, -0.04865, 0.05158, -0.11934, -0.12402, 0.03195, -0.13136,
        -0.09344, 0.01473, -0.09145, -0.05736, 0.13482, 0.06400, -0.01388, -0.11194,
        -0.08062, -0.09605, 0.01991, 0.09843, 0.04947, 0.01369, -0.02316, -0.00877,
        0.04565, -0.00290, -0.04340, -0.01068, -0.02695, -0.05790, -0.10437, 0.02970,
        0.05276, -0.02784, 0.02026, 0.03368, 0.04097, 0.05806, -0.02798, -0.04093,
    ]
}
