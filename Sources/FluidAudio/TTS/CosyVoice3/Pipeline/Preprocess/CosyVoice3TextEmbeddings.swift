@preconcurrency import CoreML
import Foundation

/// mmap'd reader for Qwen2 `text_embedding` [151936, 896] and CosyVoice3
/// `speech_embedding` [6761, 896] tables (both fp32). Used by the Phase 2
/// text frontend to assemble `lm_input_embeds` natively in Swift.
///
/// The Phase 1 per-step decode embedding path still uses
/// `CosyVoice3SpeechEmbeddings` (fp16 table) to save memory during long
/// autoregressive loops; that code remains unchanged.
public final class CosyVoice3TextEmbeddings {

    private let file: SafetensorsFile
    private let textBytes: Data
    private let speechBytes: Data
    public let textVocab: Int
    public let speechVocab: Int
    public let embedDim: Int

    public init(url: URL) throws {
        let file = try SafetensorsFile(url: url)
        guard let text = file.tensors["text_embedding"] else {
            throw CosyVoice3Error.embeddingTableMissing("text_embedding")
        }
        guard let speech = file.tensors["speech_embedding"] else {
            throw CosyVoice3Error.embeddingTableMissing("speech_embedding")
        }
        guard text.dtype == .f32, text.shape.count == 2 else {
            throw CosyVoice3Error.invalidShape(
                "text_embedding expects [vocab, 896] fp32, got shape=\(text.shape) dtype=\(text.dtype.rawValue)"
            )
        }
        guard speech.dtype == .f32, speech.shape.count == 2 else {
            throw CosyVoice3Error.invalidShape(
                "speech_embedding expects [vocab, 896] fp32, got shape=\(speech.shape) dtype=\(speech.dtype.rawValue)"
            )
        }
        guard text.shape[1] == speech.shape[1] else {
            throw CosyVoice3Error.invalidShape(
                "text_embedding dim=\(text.shape[1]) != speech_embedding dim=\(speech.shape[1])"
            )
        }
        self.file = file
        self.textBytes = try file.rawBytes("text_embedding")
        self.speechBytes = try file.rawBytes("speech_embedding")
        self.textVocab = text.shape[0]
        self.speechVocab = speech.shape[0]
        self.embedDim = text.shape[1]
        guard self.embedDim == CosyVoice3Constants.embedDim else {
            throw CosyVoice3Error.invalidShape(
                "embed_dim=\(embedDim) does not match CosyVoice3Constants.embedDim=\(CosyVoice3Constants.embedDim)"
            )
        }
    }

    /// Assemble LLM-Prefill input:
    /// `lm_input = concat([sos, text_embedding[text_ids], task_id, speech_embedding[prompt_speech_ids]], dim=1)`
    ///
    /// Returns a `[1, T_pre, 896]` fp32 MLMultiArray and `T_pre = 1 + N_text + 1 + N_speech`.
    /// The LLM-Prefill model expects T padded to 256; this method returns the
    /// unpadded tensor — callers must pad or pass `T_pre` separately.
    public func assembleLmInput(
        textTokenIds: [Int32],
        promptSpeechIds: [Int32],
        sos: Int32 = CosyVoice3Constants.sosId,
        taskId: Int32 = CosyVoice3Constants.taskId
    ) throws -> (embeds: MLMultiArray, tPre: Int) {
        let nText = textTokenIds.count
        let nSpeech = promptSpeechIds.count
        let tPre = 1 + nText + 1 + nSpeech
        let dim = embedDim
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: tPre), NSNumber(value: dim)],
            dataType: .float32)
        let strides = array.strides.map { $0.intValue }
        let dst = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)

        // Row t (within the T_pre axis) → destination pointer.
        func row(_ t: Int) -> UnsafeMutablePointer<Float> {
            dst.advanced(by: t * strides[1])
        }

        // 1) sos
        try copySpeechRow(sos, into: row(0), stride: strides[2])
        // 2) text_embedding[text_ids]
        for (i, id) in textTokenIds.enumerated() {
            try copyTextRow(id, into: row(1 + i), stride: strides[2])
        }
        // 3) task_id
        try copySpeechRow(taskId, into: row(1 + nText), stride: strides[2])
        // 4) speech_embedding[prompt_speech_ids]
        for (i, id) in promptSpeechIds.enumerated() {
            try copySpeechRow(id, into: row(1 + nText + 1 + i), stride: strides[2])
        }

        return (array, tPre)
    }

    // MARK: - Row copy

    private func copyTextRow(
        _ id: Int32, into dst: UnsafeMutablePointer<Float>, stride: Int
    ) throws {
        guard id >= 0 && Int(id) < textVocab else {
            throw CosyVoice3Error.invalidShape(
                "text token id \(id) out of range [0, \(textVocab))")
        }
        let rowStart = Int(id) * embedDim * 4
        textBytes.withUnsafeBytes { src in
            let basePtr = src.baseAddress!.advanced(by: rowStart)
                .assumingMemoryBound(to: Float.self)
            if stride == 1 {
                memcpy(dst, basePtr, embedDim * 4)
            } else {
                for i in 0..<embedDim {
                    dst[i * stride] = basePtr[i]
                }
            }
        }
    }

    private func copySpeechRow(
        _ id: Int32, into dst: UnsafeMutablePointer<Float>, stride: Int
    ) throws {
        guard id >= 0 && Int(id) < speechVocab else {
            throw CosyVoice3Error.invalidShape(
                "speech token id \(id) out of range [0, \(speechVocab))")
        }
        let rowStart = Int(id) * embedDim * 4
        speechBytes.withUnsafeBytes { src in
            let basePtr = src.baseAddress!.advanced(by: rowStart)
                .assumingMemoryBound(to: Float.self)
            if stride == 1 {
                memcpy(dst, basePtr, embedDim * 4)
            } else {
                for i in 0..<embedDim {
                    dst[i * stride] = basePtr[i]
                }
            }
        }
    }
}
