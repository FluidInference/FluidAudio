import Foundation
import XCTest

/// Typed access to the Python-generated LuxTTS parity fixtures
/// (`Resources/luxtts_fixtures.json` + binary companions), produced by
/// `mobius/models/tts/zipvoice/coreml/dump_swift_fixtures.py`.
struct LuxTtsFixtures: Decodable {

    struct Expansion: Decodable {
        let tokensLen: Int
        let featuresLen: Int
        let avgTokenDuration: Int
        let tokensIndexLen: Int
        let tokensIndex: [Int]

        enum CodingKeys: String, CodingKey {
            case tokensLen = "tokens_len"
            case featuresLen = "features_len"
            case avgTokenDuration = "avg_token_duration"
            case tokensIndexLen = "tokens_index_len"
            case tokensIndex = "tokens_index"
        }
    }

    struct Prompt: Decodable {
        let transcript: String
        let rmsPreNorm: Double
        let targetRms: Double
        let wav24kSamples: Int
        let melFrames: Int
        let melDim: Int
        let melFirst3Frames: [[Double]]
        let phonemeString: String
        let tokenIds: [Int]

        enum CodingKeys: String, CodingKey {
            case transcript
            case rmsPreNorm = "rms_pre_norm"
            case targetRms = "target_rms"
            case wav24kSamples = "wav_24k_samples"
            case melFrames = "mel_frames"
            case melDim = "mel_dim"
            case melFirst3Frames = "mel_first3_frames"
            case phonemeString = "phoneme_string"
            case tokenIds = "token_ids"
        }
    }

    struct Text: Decodable {
        let text: String
        let phonemeString: String
        let tokenIds: [Int]
        let catTokensLen: Int
        let featuresLenSpeed1: Int
        let expansion: Expansion

        enum CodingKeys: String, CodingKey {
            case text
            case phonemeString = "phoneme_string"
            case tokenIds = "token_ids"
            case catTokensLen = "cat_tokens_len"
            case featuresLenSpeed1 = "features_len_speed1"
            case expansion
        }
    }

    struct MiniTrajectory: Decodable {
        let x0: [Double]
        let vSteps: [[Double]]
        let xFinal: [Double]

        enum CodingKeys: String, CodingKey {
            case x0
            case vSteps = "v_steps"
            case xFinal = "x_final"
        }
    }

    struct Solver: Decodable {
        let numSteps: Int
        let tShift: Double
        let timesteps: [Double]
        let miniTrajectory: MiniTrajectory

        enum CodingKeys: String, CodingKey {
            case numSteps = "num_steps"
            case tShift = "t_shift"
            case timesteps
            case miniTrajectory = "mini_trajectory"
        }
    }

    struct E2E: Decodable {
        let textIndex: Int
        let seed: Int
        let speed: Double
        let featuresLen: Int
        let genFrames: Int
        let wavSamples: Int
        let wavSeconds: Double
        let rms: Double

        enum CodingKeys: String, CodingKey {
            case textIndex = "text_index"
            case seed
            case speed
            case featuresLen = "features_len"
            case genFrames = "gen_frames"
            case wavSamples = "wav_samples"
            case wavSeconds = "wav_seconds"
            case rms
        }
    }

    let prompt: Prompt
    let texts: [Text]
    let solver: Solver
    let e2e: E2E

    // MARK: - Loading

    static func resourceURL(_ name: String) throws -> URL {
        guard
            let url = Bundle.module.url(
                forResource: name, withExtension: nil, subdirectory: "Resources")
        else {
            throw XCTSkip("LuxTts fixture resource missing: \(name)")
        }
        return url
    }

    static func load() throws -> LuxTtsFixtures {
        let data = try Data(contentsOf: resourceURL("luxtts_fixtures.json"))
        return try JSONDecoder().decode(LuxTtsFixtures.self, from: data)
    }

    /// Read a little-endian Float32 binary fixture.
    static func loadFloats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: resourceURL(name))
        precondition(data.count % 4 == 0, "\(name) size not a multiple of 4")
        return data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }
}
