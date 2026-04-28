@preconcurrency import CoreML
import Foundation

/// mmap'd [6761, 896] fp16 speech-embedding lookup table.
///
/// Python-side, this table is `self.llm.speech_embedding.weight` fetched per
/// decoded token id and fed into the LLM decode step as `inputs_embeds`.
/// Swift-side we mmap the exported safetensors and convert one row from fp16
/// to fp32 per decode step into a freshly allocated `[1, 1, 896]` fp32
/// MLMultiArray (the decode mlpackage declares fp32 at its I/O boundary).
public final class CosyVoice3SpeechEmbeddings {

    private let file: SafetensorsFile
    private let tableBytes: Data
    private let rowByteSize: Int
    public let numTokens: Int
    public let embedDim: Int

    public init(url: URL) throws {
        let file = try SafetensorsFile(url: url)
        guard let info = file.tensors["speech_embedding"] else {
            throw CosyVoice3Error.embeddingTableMissing("speech_embedding")
        }
        guard info.dtype == .f16, info.shape.count == 2 else {
            throw CosyVoice3Error.invalidShape(
                "speech_embedding expects [vocab, 896] fp16, got shape=\(info.shape) dtype=\(info.dtype.rawValue)"
            )
        }
        self.file = file
        self.tableBytes = try file.rawBytes("speech_embedding")
        self.numTokens = info.shape[0]
        self.embedDim = info.shape[1]
        self.rowByteSize = info.shape[1] * 2  // fp16
        guard self.embedDim == CosyVoice3Constants.embedDim else {
            throw CosyVoice3Error.invalidShape(
                "speech_embedding dim=\(embedDim) does not match CosyVoice3 embedDim=\(CosyVoice3Constants.embedDim)"
            )
        }
    }

    /// Returns a `[1, 1, 896]` fp32 MLMultiArray containing the embedding row
    /// for `tokenId`, converted from fp16. Allocates fresh each call — the
    /// LLM decode step owns the tensor for exactly one prediction.
    public func embedding(tokenId: Int32) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: embedDim)],
            dataType: .float32)
        try copyEmbedding(tokenId: tokenId, into: array)
        return array
    }

    /// Copy the fp16 embedding row for `tokenId` into an existing
    /// `[1, 1, embedDim]` fp32 MLMultiArray. Avoids the per-step allocation
    /// of `embedding(tokenId:)` in the hot decode loop.
    public func copyEmbedding(tokenId: Int32, into array: MLMultiArray) throws {
        guard tokenId >= 0 && Int(tokenId) < numTokens else {
            throw CosyVoice3Error.invalidShape(
                "speech token id \(tokenId) out of range [0, \(numTokens))")
        }
        let rowStart = Int(tokenId) * rowByteSize
        let dim = embedDim
        let lastStride = array.strides.last?.intValue ?? 1
        tableBytes.withUnsafeBytes { src in
            let basePtr = src.baseAddress!.advanced(by: rowStart)
            let fp16Ptr = basePtr.assumingMemoryBound(to: Float16.self)
            let dstPtr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            for i in 0..<dim {
                dstPtr[i * lastStride] = Float(fp16Ptr[i])
            }
        }
    }
}
