import Foundation

/// Centralized model names for all FluidAudio components
public enum ModelNames {

    /// Diarizer model names
    public enum Diarizer {
        public static let segmentation = "pyannote_segmentation"
        public static let embedding = "wespeaker_v2"

        public static let segmentationFile = segmentation + ".mlmodelc"
        public static let embeddingFile = embedding + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            segmentationFile,
            embeddingFile,
        ]
    }

    /// ASR model names
    public enum ASR {
        public static let melspectrogram = "Melspectrogram_15s"
        public static let encoder = "ParakeetEncoder_15s"
        public static let decoder = "ParakeetDecoder"
        public static let joint = "RNNTJoint"
        public static let vocabulary = "parakeet_v3_vocab.json"

        public static let melspectrogramFile = melspectrogram + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            melspectrogramFile,
            encoderFile,
            decoderFile,
            jointFile,
        ]
    }

    /// VAD model names
    public enum VAD {
        public static let stft = "silero_stft"
        public static let encoder = "silero_encoder"
        public static let rnnDecoder = "silero_rnn_decoder"

        public static let stftFile = stft + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let rnnDecoderFile = rnnDecoder + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            stftFile,
            encoderFile,
            rnnDecoderFile,
        ]
    }
}
