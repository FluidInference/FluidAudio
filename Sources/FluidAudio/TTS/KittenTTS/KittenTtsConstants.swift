import Foundation

/// Constants for the KittenTTS StyleTTS2-based TTS backend.
public enum KittenTtsConstants {

    // MARK: - Audio

    /// Output sample rate in Hz.
    public static let audioSampleRate: Int = 24_000

    // MARK: - Vocabulary

    /// The 178-token IPA vocabulary as Unicode scalars.
    /// Index 0 (`$`) is the BOS/EOS/padding token.
    /// Each scalar's position in this array is its token ID.
    ///
    /// Note: stored as `[Unicode.Scalar]` rather than `String` because
    /// U+0329 (COMBINING VERTICAL LINE BELOW) at index 175 merges with
    /// the preceding U+2018 into a single Swift `Character`, making
    /// `String.count` return 177 instead of 178.
    // swiftlint:disable:next line_length
    public static let vocabScalars: [Unicode.Scalar] = Array(
        "$;:,.!?¡¿—…\"«»\u{201C}\u{201D} ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u{2018}\u{0329}\u{2019}ᵻ"
            .unicodeScalars)

    /// Vocabulary size (number of tokens including padding).
    public static let vocabSize: Int = 178

    /// BOS/EOS/padding token ID.
    public static let padTokenId: Int32 = 0

    // MARK: - Model dimensions

    /// Nano voice embedding dimension (single 256-float vector per voice).
    public static let nanoVoiceDim: Int = 256

    /// Mini voice matrix rows (one row per token count, 0-399).
    public static let miniVoiceRows: Int = 400

    /// Mini voice embedding dimension per row.
    public static let miniVoiceDim: Int = 256

    /// Number of harmonic channels for Nano source noise and random phases.
    public static let nanoHarmonics: Int = 9

    // MARK: - Nano model sizes

    /// Maximum audio samples for 5-second Nano model.
    public static let nano5sMaxSamples: Int = 120_000

    /// Maximum audio samples for 10-second Nano model.
    public static let nano10sMaxSamples: Int = 240_000

    // MARK: - Voices

    /// Default voice identifier.
    public static let defaultVoice: String = "expr-voice-3-f"

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
}
