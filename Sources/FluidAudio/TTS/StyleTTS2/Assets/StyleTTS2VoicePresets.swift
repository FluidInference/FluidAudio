import Foundation

/// Catalog of StyleTTS2 voice references shipped with the model bundle.
///
/// These are the 17 reference clips published by the StyleTTS2 author
/// (`yl4579/StyleTTS2-LibriTTS/reference_audio.zip`), pre-extracted offline by
/// `mobius/models/tts/styletts2/scripts/06_dump_ref_s.py` into the 256-fp32
/// `ref_s.bin` format that `StyleTTS2VoiceStyle.load(from:)` consumes.
///
/// On HuggingFace they live alongside the CoreML bundles at
/// `FluidInference/StyleTTS-2-coreml/voices/ref_s_<name>.bin`. The downloader
/// (`StyleTTS2ResourceDownloader`) fetches the whole `voices/` directory.
public enum StyleTTS2VoicePresets {

    /// Subdirectory (relative to the resolved bundle root) where `ref_s_*.bin`
    /// files are staged on disk. Mirrors the HuggingFace repo layout.
    public static let directoryName = "voices"

    /// One catalog entry. `id` is the canonical lookup key (lowercase,
    /// dot-free); `filename` is the on-disk artifact under `voices/`.
    public struct Voice: Sendable, Equatable {
        public let id: String
        public let filename: String
        public let cohort: Cohort
        public let description: String

        public init(id: String, filename: String, cohort: Cohort, description: String) {
            self.id = id
            self.filename = filename
            self.cohort = cohort
            self.description = description
        }
    }

    /// Source of the reference clip — useful for callers picking a voice by
    /// quality expectations (cloned > zero-shot > emotion > LibriTTS-seen).
    public enum Cohort: String, Sendable, CaseIterable {
        /// LibriTTS speakers the model was trained on (24 kHz studio).
        case libriTTSSeen = "libritts_seen"
        /// LibriTTS speakers held out from training (16 kHz, harder).
        case libriTTSUnseen = "libritts_unseen"
        /// Author voice-cloning targets (Yinghao, Gavin, Vinay, Nima — 44/48 kHz).
        case authorCloned = "author_cloned"
        /// Zero-shot real-world clips (`3.wav`, `4.wav`, `5.wav` — 16 kHz).
        case zeroShot = "zero_shot"
        /// Emotion samples (amused, anger, disgusted, sleepy — 16 kHz).
        case emotion = "emotion"
    }

    /// Default voice when no `--voice` / `--voice-name` is supplied.
    /// Picked because it's the cleanest CoreML-side intelligible voice in the
    /// internal A/B over the 17-voice set — author-cloned, well-modelled.
    public static let defaultVoiceID = "vinay"

    /// Full catalog. Keep in sync with the `voices/` directory shipped to HF.
    public static let all: [Voice] = [
        // LibriTTS — seen at training time
        .init(
            id: "lj_696", filename: "ref_s_696_92939_000016_000006.bin",
            cohort: .libriTTSSeen,
            description: "LibriTTS speaker 696 (clip 92939_000016_000006)."),
        .init(
            id: "lj_1789", filename: "ref_s_1789_142896_000022_000005.bin",
            cohort: .libriTTSSeen,
            description: "LibriTTS speaker 1789 (clip 142896_000022_000005)."),

        // LibriTTS — unseen / OOD
        .init(
            id: "lj_1221", filename: "ref_s_1221-135767-0014.bin",
            cohort: .libriTTSUnseen,
            description: "LibriTTS held-out speaker 1221."),
        .init(
            id: "lj_5639", filename: "ref_s_5639-40744-0020.bin",
            cohort: .libriTTSUnseen,
            description: "LibriTTS held-out speaker 5639."),
        .init(
            id: "lj_908", filename: "ref_s_908-157963-0027.bin",
            cohort: .libriTTSUnseen,
            description: "LibriTTS held-out speaker 908."),
        .init(
            id: "lj_4077", filename: "ref_s_4077-13754-0000.bin",
            cohort: .libriTTSUnseen,
            description: "LibriTTS held-out speaker 4077."),

        // Author-cloned (named voices from the StyleTTS2 demo page)
        .init(
            id: "yinghao", filename: "ref_s_Yinghao.bin",
            cohort: .authorCloned,
            description: "Yinghao (author voice clone)."),
        .init(
            id: "gavin", filename: "ref_s_Gavin.bin",
            cohort: .authorCloned,
            description: "Gavin (author voice clone)."),
        .init(
            id: "vinay", filename: "ref_s_Vinay.bin",
            cohort: .authorCloned,
            description: "Vinay (author voice clone). Default."),
        .init(
            id: "nima", filename: "ref_s_Nima.bin",
            cohort: .authorCloned,
            description: "Nima (author voice clone)."),

        // Zero-shot real-world clips
        .init(
            id: "zs_3", filename: "ref_s_3.bin",
            cohort: .zeroShot,
            description: "Zero-shot reference clip #3."),
        .init(
            id: "zs_4", filename: "ref_s_4.bin",
            cohort: .zeroShot,
            description: "Zero-shot reference clip #4."),
        .init(
            id: "zs_5", filename: "ref_s_5.bin",
            cohort: .zeroShot,
            description: "Zero-shot reference clip #5."),

        // Emotion
        .init(
            id: "emotion_amused", filename: "ref_s_amused.bin",
            cohort: .emotion, description: "Emotion: amused."),
        .init(
            id: "emotion_anger", filename: "ref_s_anger.bin",
            cohort: .emotion, description: "Emotion: angry."),
        .init(
            id: "emotion_disgusted", filename: "ref_s_disgusted.bin",
            cohort: .emotion, description: "Emotion: disgusted."),
        .init(
            id: "emotion_sleepy", filename: "ref_s_sleepy.bin",
            cohort: .emotion, description: "Emotion: sleepy."),
    ]

    /// Set of every artifact filename, used by the downloader to verify the
    /// `voices/` directory was fully fetched.
    public static var requiredFilenames: Set<String> {
        Set(all.map { $0.filename })
    }

    /// Resolve a voice by id (case-insensitive). Returns `nil` for unknown ids.
    public static func voice(forID id: String) -> Voice? {
        let needle = id.lowercased()
        return all.first { $0.id == needle }
    }

    /// Sorted list of public voice ids (handy for CLI usage messages).
    public static var allIDs: [String] {
        all.map { $0.id }.sorted()
    }
}
