import Accelerate
import Foundation

/// Pre-loaded binary constants for VoxCPM inference.
public struct VoxCpmConstantsBundle: Sendable {
    /// Text embedding table: [73448 × 1024] flattened, loaded from Float16.
    public let embedTokens: [Float]
    /// Encoder-to-LM projection weight: [1024 × 1024] flattened.
    public let encToLmProjW: [Float]
    /// Encoder-to-LM projection bias: [1024].
    public let encToLmProjB: [Float]
    /// LM-to-DiT projection weight: [1024 × 1024] flattened.
    public let lmToDitProjW: [Float]
    /// LM-to-DiT projection bias: [1024].
    public let lmToDitProjB: [Float]
    /// Residual-to-DiT projection weight: [1024 × 1024] flattened.
    public let resToDitProjW: [Float]
    /// Residual-to-DiT projection bias: [1024].
    public let resToDitProjB: [Float]
    /// HuggingFace BPE tokenizer for text encoding.
    public let tokenizer: VoxCpmTokenizer
}

/// Loads VoxCPM constants from raw `.bin` Float16 files on disk.
public enum VoxCpmConstantsLoader {

    private static let logger = AppLogger(category: "VoxCpmConstantsLoader")

    public enum LoadError: Error {
        case fileNotFound(String)
        case invalidSize(String, expected: Int, actual: Int)
        case tokenizerLoadFailed(String)
    }

    /// Load all constants from the constants_bin directory within the repo.
    public static func load(from directory: URL) async throws -> VoxCpmConstantsBundle {
        let constantsDir = directory.appendingPathComponent(ModelNames.VoxCPM.constantsBinDir)

        let dim = VoxCpmConstants.hiddenSize
        let vocab = VoxCpmConstants.vocabSize

        let embedTokens = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("embed_tokens.bin"),
            expectedCount: vocab * dim,
            name: "embed_tokens"
        )
        let encToLmProjW = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("enc_to_lm_proj_w.bin"),
            expectedCount: dim * dim,
            name: "enc_to_lm_proj_w"
        )
        let encToLmProjB = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("enc_to_lm_proj_b.bin"),
            expectedCount: dim,
            name: "enc_to_lm_proj_b"
        )
        let lmToDitProjW = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("lm_to_dit_proj_w.bin"),
            expectedCount: dim * dim,
            name: "lm_to_dit_proj_w"
        )
        let lmToDitProjB = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("lm_to_dit_proj_b.bin"),
            expectedCount: dim,
            name: "lm_to_dit_proj_b"
        )
        let resToDitProjW = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("res_to_dit_proj_w.bin"),
            expectedCount: dim * dim,
            name: "res_to_dit_proj_w"
        )
        let resToDitProjB = try loadFloat16Array(
            from: constantsDir.appendingPathComponent("res_to_dit_proj_b.bin"),
            expectedCount: dim,
            name: "res_to_dit_proj_b"
        )

        let tokenizer = try await VoxCpmTokenizer.load(from: constantsDir)

        logger.info("Loaded VoxCPM constants from \(directory.lastPathComponent)")

        return VoxCpmConstantsBundle(
            embedTokens: embedTokens,
            encToLmProjW: encToLmProjW,
            encToLmProjB: encToLmProjB,
            lmToDitProjW: lmToDitProjW,
            lmToDitProjB: lmToDitProjB,
            resToDitProjW: resToDitProjW,
            resToDitProjB: resToDitProjB,
            tokenizer: tokenizer
        )
    }

    // MARK: - Private

    /// Load a raw Float16 binary file and convert to [Float] using vDSP.
    private static func loadFloat16Array(
        from url: URL, expectedCount: Int, name: String
    ) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoadError.fileNotFound(name)
        }

        let data = try Data(contentsOf: url)
        let actualCount = data.count / MemoryLayout<UInt16>.size

        guard actualCount == expectedCount else {
            throw LoadError.invalidSize(name, expected: expectedCount, actual: actualCount)
        }

        var result = [Float](repeating: 0, count: actualCount)
        data.withUnsafeBytes { rawBuffer in
            let f16Ptr = rawBuffer.baseAddress!
            // vImageConvert_Planar16FtoPlanarF converts Float16 → Float32 via Accelerate
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: f16Ptr),
                height: 1,
                width: vImagePixelCount(actualCount),
                rowBytes: actualCount * MemoryLayout<UInt16>.size
            )
            result.withUnsafeMutableBufferPointer { dstBuf in
                var dst = vImage_Buffer(
                    data: dstBuf.baseAddress!,
                    height: 1,
                    width: vImagePixelCount(actualCount),
                    rowBytes: actualCount * MemoryLayout<Float>.size
                )
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
            }
        }
        return result
    }
}
