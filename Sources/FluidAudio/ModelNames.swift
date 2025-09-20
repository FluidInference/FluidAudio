import Foundation

/// Model repositories on HuggingFace
public enum Repo: String, CaseIterable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeet = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    case diarizer = "FluidInference/speaker-diarization-coreml"
    case kokoro = "FluidInference/kokoro-82m-coreml"

    var folderName: String {
        rawValue.split(separator: "/").last?.description ?? rawValue
    }

}

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
        public static let sileroVad = "silero-vad-unified-256ms-v6.0.0"

        public static let sileroVadFile = sileroVad + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            sileroVadFile
        ]
    }

    /// TTS model names
    public enum TTS {
        public static let kokoroBundle = "kokoro_completev22.mlmodelc"

        public static let requiredModels: Set<String> = [
            kokoroBundle
        ]
    }

    @available(macOS 13.0, iOS 16.0, *)
    static func getRequiredModelNames(for repo: Repo) -> Set<String> {
        switch repo {
        case .vad:
            return ModelNames.VAD.requiredModels
        case .parakeet:
            return ModelNames.ASR.requiredModels
        case .diarizer:
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            return ModelNames.TTS.requiredModels
        }
    }

}
