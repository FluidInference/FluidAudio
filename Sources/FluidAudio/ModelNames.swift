import Foundation

/// Model repositories on HuggingFace
public enum Repo: String, CaseIterable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeet = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    case parakeetV2 = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    case parakeetCtc110m = "argmaxinc/ctckit-pro"
    case diarizer = "FluidInference/speaker-diarization-coreml"
    case kokoro = "FluidInference/kokoro-82m-coreml"

    /// Repository slug (without owner)
    public var name: String {
        switch self {
        case .vad:
            return "silero-vad-coreml"
        case .parakeet:
            return "parakeet-tdt-0.6b-v3-coreml"
        case .parakeetV2:
            return "parakeet-tdt-0.6b-v2-coreml"
        case .parakeetCtc110m:
            return "ctckit-pro"
        case .diarizer:
            return "speaker-diarization-coreml"
        case .kokoro:
            return "kokoro-82m-coreml"
        }
    }

    /// Fully qualified HuggingFace repo path (owner/name)
    public var remotePath: String {
        switch self {
        case .parakeetCtc110m:
            // Uses Argmax CoreML export for Parakeet CTC 110M
            return rawValue
        default:
            return "FluidInference/\(name)"
        }
    }

    /// Local folder name used for caching
    public var folderName: String {
        switch self {
        case .kokoro:
            return "kokoro"
        default:
            return name
        }
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

    /// Offline diarizer model names (VBx-based clustering)
    public enum OfflineDiarizer {
        public static let subfolder = "speaker-diarization-offline"
        public static let segmentation = "Segmentation"
        public static let fbank = "FBank"
        public static let embedding = "Embedding"
        public static let pldaRho = "PldaRho"

        public static let segmentationFile = segmentation + ".mlmodelc"
        public static let fbankFile = fbank + ".mlmodelc"
        public static let embeddingFile = embedding + ".mlmodelc"
        public static let pldaRhoFile = pldaRho + ".mlmodelc"

        // Full paths including subfolder (for DownloadUtils)
        public static let segmentationPath = subfolder + "/" + segmentationFile
        public static let fbankPath = subfolder + "/" + fbankFile
        public static let embeddingPath = subfolder + "/" + embeddingFile
        public static let pldaRhoPath = subfolder + "/" + pldaRhoFile

        public static let requiredModels: Set<String> = [
            segmentationPath,
            fbankPath,
            embeddingPath,
            pldaRhoPath,
        ]
    }

    /// ASR model names
    public enum ASR {
        public static let preprocessor = "Preprocessor"
        public static let encoder = "Encoder"
        public static let decoder = "Decoder"

        // Joint model names differ between versions
        public static let jointV2 = "JointDecision"  // v2 uses JointDecision
        public static let jointV3 = "JointDecisionv2"  // v3 uses JointDecisionv2

        // Shared vocabulary file across all model versions
        public static let vocabularyFile = "parakeet_vocab.json"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"

        // Get joint file name based on repo/version
        public static func jointFile(for repo: Repo) -> String {
            switch repo {
            case .parakeetV2:
                return jointV2 + ".mlmodelc"
            case .parakeet:
                return jointV3 + ".mlmodelc"
            default:
                return jointV3 + ".mlmodelc"  // Default to v3
            }
        }

        // Get required models based on repo/version
        public static func requiredModels(for repo: Repo) -> Set<String> {
            return [
                preprocessorFile,
                encoderFile,
                decoderFile,
                jointFile(for: repo),
            ]
        }

        /// Get vocabulary filename for specific model version
        public static func vocabulary(for repo: Repo) -> String {
            return vocabularyFile
        }
    }

    /// CTC keyword spotting model names (Parakeet-TDT CTC 110M).
    public enum CTC {
        public static let subfolder = "parakeet-tdt_ctc-110m"

        public static let melSpectrogram = "MelSpectrogram"
        public static let audioEncoder = "AudioEncoder"

        public static let melSpectrogramPath = subfolder + "/" + melSpectrogram + ".mlmodelc"
        public static let audioEncoderPath = subfolder + "/" + audioEncoder + ".mlmodelc"

        // Vocabulary JSON path (shared by Python/Nemo and CoreML exports).
        public static let vocabularyPath = subfolder + "/vocab.json"

        public static let requiredModels: Set<String> = [
            melSpectrogramPath,
            audioEncoderPath,
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

        /// Available Kokoro variants shipped with the library.
        public enum Variant: CaseIterable, Sendable {
            case fiveSecond
            case fifteenSecond

            /// Underlying model bundle filename.
            public var fileName: String {
                switch self {
                case .fiveSecond:
                    return "kokoro_21_5s.mlmodelc"
                case .fifteenSecond:
                    return "kokoro_21_15s.mlmodelc"
                }
            }

            /// Approximate maximum duration in seconds handled by the variant.
            public var maxDurationSeconds: Int {
                switch self {
                case .fiveSecond:
                    return 5
                case .fifteenSecond:
                    return 15
                }
            }
        }

        /// Preferred variant for general-purpose synthesis.
        public static let defaultVariant: Variant = .fifteenSecond

        /// Convenience accessor for bundle name lookup.
        public static func bundle(for variant: Variant) -> String {
            variant.fileName
        }

        /// Default bundle filename (legacy accessor).
        public static var defaultBundle: String {
            defaultVariant.fileName
        }

        /// All Kokoro model bundles required by the downloader.
        public static var requiredModels: Set<String> {
            Set(Variant.allCases.map { $0.fileName })
        }
    }

    static func getRequiredModelNames(for repo: Repo, variant: String?) -> Set<String> {
        switch repo {
        case .vad:
            return ModelNames.VAD.requiredModels
        case .parakeet, .parakeetV2:
            return ModelNames.ASR.requiredModels(for: repo)
        case .parakeetCtc110m:
            return ModelNames.CTC.requiredModels
        case .diarizer:
            if variant == "offline" {
                return ModelNames.OfflineDiarizer.requiredModels
            }
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            return ModelNames.TTS.requiredModels
        }
    }

}
