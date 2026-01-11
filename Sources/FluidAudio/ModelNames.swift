import Foundation

/// Model repositories on HuggingFace
public enum Repo: String, CaseIterable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeet = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    case parakeetV2 = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    case parakeetEou160 = "FluidInference/parakeet-realtime-eou-120m-coreml/160ms"
    case parakeetEou320 = "FluidInference/parakeet-realtime-eou-120m-coreml/320ms"
    case diarizer = "FluidInference/speaker-diarization-coreml"
    case kokoro = "FluidInference/kokoro-82m-coreml"
    case sortformer = "FluidInference/diar-streaming-sortformer-coreml"

    /// Repository slug (without owner)
    public var name: String {
        switch self {
        case .vad:
            return "silero-vad-coreml"
        case .parakeet:
            return "parakeet-tdt-0.6b-v3-coreml"
        case .parakeetV2:
            return "parakeet-tdt-0.6b-v2-coreml"
        case .parakeetEou160:
            return "parakeet-realtime-eou-120m-coreml/160ms"
        case .parakeetEou320:
            return "parakeet-realtime-eou-120m-coreml/320ms"
        case .diarizer:
            return "speaker-diarization-coreml"
        case .kokoro:
            return "kokoro-82m-coreml"
        case .sortformer:
            return "diar-streaming-sortformer-coreml"
        }
    }

    /// Fully qualified HuggingFace repo path (owner/name)
    public var remotePath: String {
        switch self {
        case .parakeetEou160, .parakeetEou320:
            return "FluidInference/parakeet-realtime-eou-120m-coreml"
        case .sortformer:
            return "FluidInference/diar-streaming-sortformer-coreml"
        default:
            return "FluidInference/\(name)"
        }
    }

    /// Subdirectory within repo (for repos with multiple model variants)
    public var subPath: String? {
        switch self {
        case .parakeetEou160:
            return "160ms"
        case .parakeetEou320:
            return "320ms"
        default:
            return nil
        }
    }

    /// Local folder name used for caching
    public var folderName: String {
        switch self {
        case .kokoro:
            return "kokoro"
        case .parakeetEou160:
            return "parakeet-eou-streaming/160ms"
        case .parakeetEou320:
            return "parakeet-eou-streaming/320ms"
        case .sortformer:
            return "sortformer"
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
        public static let segmentation = "Segmentation"
        public static let fbank = "FBank"
        public static let embedding = "Embedding"
        public static let pldaRho = "PldaRho"

        public static let segmentationFile = segmentation + ".mlmodelc"
        public static let fbankFile = fbank + ".mlmodelc"
        public static let embeddingFile = embedding + ".mlmodelc"
        public static let pldaRhoFile = pldaRho + ".mlmodelc"

        public static let segmentationPath = segmentationFile
        public static let fbankPath = fbankFile
        public static let embeddingPath = embeddingFile
        public static let pldaRhoPath = pldaRhoFile

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
        public static let joint = "JointDecision"

        // Shared vocabulary file across all model versions
        public static let vocabularyFile = "parakeet_vocab.json"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderFile,
            decoderFile,
            jointFile,
        ]

        /// Get vocabulary filename for specific model version
        public static func vocabulary(for repo: Repo) -> String {
            return vocabularyFile
        }
    }

    /// VAD model names
    public enum VAD {
        public static let sileroVad = "silero-vad-unified-256ms-v6.0.0"

        public static let sileroVadFile = sileroVad + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            sileroVadFile
        ]
    }

    /// Parakeet EOU streaming model names
    public enum ParakeetEOU {
        public static let encoder = "streaming_encoder"
        public static let decoder = "decoder"
        public static let joint = "joint_decision"
        public static let vocab = "vocab.json"

        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            encoderFile,
            decoderFile,
            jointFile,
            vocab,
        ]
    }

    /// Sortformer streaming diarization model names
    public enum Sortformer {
        public enum Variant: CaseIterable, Sendable {
            case `default`
            case nvidiaLowLatency
            case nvidiaHighLatency

            public var name: String {
                switch self {
                case .default:
                    return "Sortformer"
                case .nvidiaLowLatency:
                    return "SortformerNvidiaLow"
                case .nvidiaHighLatency:
                    return "SortformerNvidiaHigh"
                }
            }

            public var defaultConfiguration: SortformerConfig {
                switch self {
                case .default:
                    return .default
                case .nvidiaLowLatency:
                    return .nvidiaLowLatency
                case .nvidiaHighLatency:
                    return .nvidiaHighLatency
                }
            }

            public var fileName: String {
                return "\(name).mlmodelc"
            }

            public func isCompatible(with config: SortformerConfig) -> Bool {
                defaultConfiguration.isCompatible(with: config)
            }
        }

        /// Lowest latency for streaming
        public static let defaultVariant: Variant = .default

        /// Bundle name for a specific variant
        public static func bundle(for variant: Variant) -> String {
            return variant.fileName
        }

        /// Bundle name for a given configuration
        public static func bundle(for config: SortformerConfig) -> String? {
            return Variant.allCases.first { $0.isCompatible(with: config) }?.fileName
        }

        /// Default bundle name
        public static var defaultBundle: String {
            return defaultVariant.fileName
        }

        /// All Sortformer bundle models required by the downloader
        public static var requiredModels: Set<String> {
            Set(Variant.allCases.map(\.fileName))
        }
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
            Set(Variant.allCases.map(\.fileName))
        }
    }

    static func getRequiredModelNames(for repo: Repo, variant: String?) -> Set<String> {
        switch repo {
        case .vad:
            return ModelNames.VAD.requiredModels
        case .parakeet, .parakeetV2:
            return ModelNames.ASR.requiredModels
        case .parakeetEou160, .parakeetEou320:
            return ModelNames.ParakeetEOU.requiredModels
        case .diarizer:
            if variant == "offline" {
                return ModelNames.OfflineDiarizer.requiredModels
            }
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            return ModelNames.TTS.requiredModels
        case .sortformer:
            return ModelNames.Sortformer.requiredModels
        }
    }
}
