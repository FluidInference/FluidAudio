import Foundation

/// Constants for ASR audio processing and frame calculations
public enum ASRConstants {
    /// Audio sample rate expected by ASR models
    public static let sampleRate: Int = 16_000

    /// Maximum audio duration supported by CoreML encoder (seconds)
    public static let maxDurationSeconds: Double = 15.0

    /// Maximum audio samples supported by CoreML encoder (sampleRate × maxDurationSeconds)
    public static let maxModelSamples: Int = 240_000

    /// Minimum audio duration accepted by the ASR guard (seconds).
    public static let minimumAudioDurationSeconds: Double = 0.3

    /// Mel-spectrogram hop size in samples (10ms at 16kHz)
    public static let melHopSize: Int = 160

    /// Encoder subsampling factor (8x downsampling from mel frames to encoder frames)
    public static let encoderSubsampling: Int = 8

    /// Size of encoder hidden representation for Parakeet-TDT models
    public static let encoderHiddenSize: Int = 1024

    /// Size of decoder hidden state for Parakeet-TDT models
    public static let decoderHiddenSize: Int = 640

    /// Samples per encoder frame (melHopSize * encoderSubsampling)
    /// Each encoder frame represents ~80ms of audio at 16kHz
    public static let samplesPerEncoderFrame: Int = melHopSize * encoderSubsampling  // 1280

    /// Duration of one encoder frame in seconds (80ms)
    public static let secondsPerEncoderFrame: Double = Double(samplesPerEncoderFrame) / Double(sampleRate)  // 0.08

    /// WER threshold for detailed error analysis in benchmarks
    public static let highWERThreshold: Double = 0.15

    /// Sentence-final punctuation token IDs (`.` `?` `!`) in the
    /// parakeet-tdt-0.6b-v3 vocabulary. Every other shipped vocabulary uses
    /// different ids (v2: 841/854/885, 110m: 986/1002/1016), so runtime code
    /// resolves the set from the loaded vocabulary via
    /// ``punctuationTokenIds(in:)`` and only falls back to this when no
    /// vocabulary is available. See issue #905.
    public static let punctuationTokens: [Int] = [7883, 7956, 8020]

    /// Sentence-final punctuation pieces resolved by text.
    public static let sentenceFinalPunctuation: Set<String> = [".", "?", "!"]

    /// Resolve the sentence-final punctuation token ids (`.` `?` `!`) from a
    /// loaded vocabulary. A piece matches with or without a leading word
    /// boundary (`▁` or the space it is normalized to).
    public static func punctuationTokenIds(in vocabulary: [Int: String]) -> Set<Int> {
        var ids: Set<Int> = []
        for (id, piece) in vocabulary {
            var core = Substring(piece)
            if core.hasPrefix(sentencePieceWordBoundary) {
                core = core.dropFirst(sentencePieceWordBoundary.count)
            } else if core.hasPrefix(" ") {
                core = core.dropFirst()
            }
            if sentenceFinalPunctuation.contains(String(core)) {
                ids.insert(id)
            }
        }
        return ids
    }

    /// SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
    /// Prefixes tokens that begin a new word in BPE/Unigram tokenization.
    /// Used by Parakeet's tokenizer (TDT vocab, CTC vocab, etc.) and the
    /// rescorer's word-boundary detection.
    public static let sentencePieceWordBoundary: String = "▁"

    /// Standard overlap in encoder frames (2.0s = 25 frames at 0.08s per frame)
    public static let standardOverlapFrames: Int = 25

    /// Maximum global-frame gap between two token occurrences for them to be
    /// treated as the *same* acoustic event during sliding-window token dedup.
    ///
    /// A genuine chunk-boundary duplicate lands at nearly identical global audio
    /// time in both overlapping windows (difference ~0, plus a few frames of
    /// cross-window emission jitter). A coincidental subword-prefix match between
    /// two *different* words spoken seconds apart is far outside this bound, so
    /// gating on it prevents false-positive dedup that drops real tokens.
    /// See issue #787.
    public static let duplicateFrameTolerance: Int = standardOverlapFrames  // 25 frames = 2.0s

    /// Minimum confidence score (for empty or very uncertain transcriptions)
    public static let minConfidence: Float = 0.1

    /// Maximum confidence score (perfect confidence)
    public static let maxConfidence: Float = 1.0

    /// Calculate encoder frames from audio samples using proper ceiling division
    /// - Parameter samples: Number of audio samples
    /// - Returns: Number of encoder frames
    public static func calculateEncoderFrames(from samples: Int) -> Int {
        return Int(ceil(Double(samples) / Double(samplesPerEncoderFrame)))
    }

    /// Minimum number of samples required by the ASR guard for a given sample rate.
    /// - Parameter sampleRate: Audio sample rate in Hz
    /// - Returns: Sample count corresponding to `minimumAudioDurationSeconds`
    public static func minimumRequiredSamples(forSampleRate sampleRate: Int) -> Int {
        return Int(Double(sampleRate) * minimumAudioDurationSeconds)
    }
}
