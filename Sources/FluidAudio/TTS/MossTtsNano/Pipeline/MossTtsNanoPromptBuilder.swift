import Foundation

/// Builds the `[rows][17]` token matrix the LM consumes.
///
/// Every row is `[text_id, code_0 … code_15]`. Text-only rows carry the audio pad
/// id in the 16 code slots; reference-audio rows carry the user slot id as text
/// and the clip's codec codes. Layout (upstream `build_inference_input_ids`,
/// voice-clone mode):
///
/// ```
/// voice_clone_prefix (… <audio_start>)
/// [audio_user_slot, codes…]  × reference frames
/// voice_clone_after_reference (<audio_end> … Text:)
/// text ids
/// assistant_suffix (… assistant <audio_start>)
/// ```
struct MossTtsNanoPromptBuilder: Sendable {
    let config: MossTtsNanoConfig

    var rowWidth: Int { config.model.rowWidth }

    func textRow(_ tokenId: Int) -> [Int32] {
        var row = [Int32](repeating: Int32(config.tokens.audioPad), count: rowWidth)
        row[0] = Int32(tokenId)
        return row
    }

    func generationRow(codes: [Int32]) -> [Int32] {
        var row = [Int32](repeating: Int32(config.tokens.audioPad), count: rowWidth)
        row[0] = Int32(config.tokens.audioAssistantSlot)
        for (i, c) in codes.prefix(rowWidth - 1).enumerated() {
            row[i + 1] = c
        }
        return row
    }

    func voiceCloneRows(textIds: [Int], voice: MossTtsNanoVoice) -> [[Int32]] {
        var rows: [[Int32]] = []
        rows.reserveCapacity(
            config.prompt.voiceClonePrefix.count + voice.frames + config.prompt.voiceCloneAfterReference.count
                + textIds.count + config.prompt.assistantSuffix.count)
        for id in config.prompt.voiceClonePrefix { rows.append(textRow(id)) }
        for codes in voice.codes {
            var row = [Int32](repeating: Int32(config.tokens.audioPad), count: rowWidth)
            row[0] = Int32(config.tokens.audioUserSlot)
            for (i, c) in codes.prefix(rowWidth - 1).enumerated() {
                row[i + 1] = c
            }
            rows.append(row)
        }
        for id in config.prompt.voiceCloneAfterReference { rows.append(textRow(id)) }
        for id in textIds { rows.append(textRow(id)) }
        for id in config.prompt.assistantSuffix { rows.append(textRow(id)) }
        return rows
    }

    /// Rows that are not text: fixed template overhead for a given voice.
    func voiceCloneOverheadRows(voice: MossTtsNanoVoice) -> Int {
        config.prompt.voiceClonePrefix.count + voice.frames + config.prompt.voiceCloneAfterReference.count
            + config.prompt.assistantSuffix.count
    }
}
