import Foundation

/// Model repositories on HuggingFace
public enum Repo: String, CaseIterable, Sendable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeet = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    case parakeetV2 = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    case parakeetCtc110m = "FluidInference/parakeet-ctc-110m-coreml"
    case parakeetCtc06b = "FluidInference/parakeet-ctc-0.6b-coreml"
    case parakeetCtcZhCn = "FluidInference/parakeet-ctc-0.6b-zh-cn-coreml"
    case parakeetJa = "FluidInference/parakeet-0.6b-ja-coreml"  // Contains both CTC and TDT models (INT8 quantized encoder)
    case parakeetEou160 = "FluidInference/parakeet-realtime-eou-120m-coreml/160ms"
    case parakeetEou320 = "FluidInference/parakeet-realtime-eou-120m-coreml/320ms"
    case parakeetEou1280 = "FluidInference/parakeet-realtime-eou-120m-coreml/1280ms"
    case nemotronStreaming1120 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/1120ms"
    case nemotronStreaming560 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/560ms"
    case nemotronStreaming160 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/160ms"
    case nemotronStreaming80 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/80ms"
    case diarizer = "FluidInference/speaker-diarization-coreml"
    case kokoro = "FluidInference/kokoro-82m-coreml"
    case sortformer = "FluidInference/diar-streaming-sortformer-coreml"
    case lseend = "FluidInference/ls-eend-coreml"
    case pocketTts = "FluidInference/pocket-tts-coreml"
    case qwen3Asr = "FluidInference/qwen3-asr-0.6b-coreml/f32"
    case qwen3AsrInt8 = "FluidInference/qwen3-asr-0.6b-coreml/int8"
    case multilingualG2p = "FluidInference/charsiu-g2p-byt5-coreml"
    case parakeetTdtCtc110m = "FluidInference/parakeet-tdt-ctc-110m-coreml"

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
            return "parakeet-ctc-110m-coreml"
        case .parakeetCtc06b:
            return "parakeet-ctc-0.6b-coreml"
        case .parakeetCtcZhCn:
            return "parakeet-ctc-0.6b-zh-cn-coreml"
        case .parakeetJa:
            return "parakeet-0.6b-ja-coreml"
        case .parakeetEou160:
            return "parakeet-realtime-eou-120m-coreml/160ms"
        case .parakeetEou320:
            return "parakeet-realtime-eou-120m-coreml/320ms"
        case .parakeetEou1280:
            return "parakeet-realtime-eou-120m-coreml/1280ms"
        case .nemotronStreaming1120:
            return "nemotron-speech-streaming-en-0.6b-coreml/1120ms"
        case .nemotronStreaming560:
            return "nemotron-speech-streaming-en-0.6b-coreml/560ms"
        case .nemotronStreaming160:
            return "nemotron-speech-streaming-en-0.6b-coreml/160ms"
        case .nemotronStreaming80:
            return "nemotron-speech-streaming-en-0.6b-coreml/80ms"
        case .diarizer:
            return "speaker-diarization-coreml"
        case .kokoro:
            return "kokoro-82m-coreml"
        case .sortformer:
            return "diar-streaming-sortformer-coreml"
        case .lseend:
            return "ls-eend-coreml"
        case .pocketTts:
            return "pocket-tts-coreml"
        case .qwen3Asr:
            return "qwen3-asr-0.6b-coreml/f32"
        case .qwen3AsrInt8:
            return "qwen3-asr-0.6b-coreml/int8"
        case .multilingualG2p:
            return "charsiu-g2p-byt5-coreml"
        case .parakeetTdtCtc110m:
            return "parakeet-tdt-ctc-110m-coreml"
        }
    }

    /// Fully qualified HuggingFace repo path (owner/name)
    public var remotePath: String {
        switch self {
        case .parakeetCtc110m:
            return "FluidInference/parakeet-ctc-110m-coreml"
        case .parakeetCtc06b:
            return "FluidInference/parakeet-ctc-0.6b-coreml"
        case .parakeetEou160, .parakeetEou320, .parakeetEou1280:
            return "FluidInference/parakeet-realtime-eou-120m-coreml"
        case .nemotronStreaming1120, .nemotronStreaming560, .nemotronStreaming160, .nemotronStreaming80:
            return "FluidInference/nemotron-speech-streaming-en-0.6b-coreml"
        case .sortformer:
            return "FluidInference/diar-streaming-sortformer-coreml"
        case .lseend:
            return "FluidInference/ls-eend-coreml"
        case .qwen3Asr, .qwen3AsrInt8:
            return "FluidInference/qwen3-asr-0.6b-coreml"
        case .parakeetTdtCtc110m:
            return "FluidInference/parakeet-tdt-ctc-110m-coreml"
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
        case .parakeetEou1280:
            return "1280ms"
        case .qwen3Asr:
            return "f32"
        case .qwen3AsrInt8:
            return "int8"
        case .nemotronStreaming1120:
            return "nemotron_coreml_1120ms"
        case .nemotronStreaming560:
            return "nemotron_coreml_560ms"
        case .nemotronStreaming160:
            return "nemotron_coreml_160ms"
        case .nemotronStreaming80:
            return "nemotron_coreml_80ms"
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
        case .parakeetEou1280:
            return "parakeet-eou-streaming/1280ms"
        case .nemotronStreaming1120:
            return "nemotron-streaming/1120ms"
        case .nemotronStreaming560:
            return "nemotron-streaming/560ms"
        case .nemotronStreaming160:
            return "nemotron-streaming/160ms"
        case .nemotronStreaming80:
            return "nemotron-streaming/80ms"
        case .sortformer:
            return "sortformer"
        case .parakeetCtc110m:
            return "parakeet-ctc-110m-coreml"
        case .parakeetCtc06b:
            return "parakeet-ctc-0.6b-coreml"
        case .parakeetCtcZhCn:
            return "parakeet-ctc-zh-cn"
        case .parakeetJa:
            return "parakeet-ja"
        case .parakeetTdtCtc110m:
            return "parakeet-tdt-ctc-110m"
        default:
            return name.replacingOccurrences(of: "-coreml", with: "")
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
        public static let pldaParameters = "plda-parameters.json"

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
            pldaParameters,
        ]
    }

    /// ASR model names
    public enum ASR {
        public static let preprocessor = "Preprocessor"
        public static let encoder = "Encoder"
        public static let decoder = "Decoder"
        public static let joint = "JointDecision"
        public static let ctcHead = "CtcHead"

        // Shared vocabulary file across all model versions
        public static let vocabularyFile = "parakeet_vocab.json"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"
        public static let ctcHeadFile = ctcHead + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderFile,
            decoderFile,
            jointFile,
        ]

        /// Required models for fused frontend (110m hybrid: preprocessor contains encoder)
        public static let requiredModelsFused: Set<String> = [
            preprocessorFile,
            decoderFile,
            jointFile,
        ]

        /// Get vocabulary filename for specific model version
        public static func vocabulary(for repo: Repo) -> String {
            // All Parakeet models use the same vocabulary file (format varies: dict for v2/v3, array for 110m)
            return vocabularyFile
        }
    }

    /// CTC model names
    public enum CTC {
        public static let melSpectrogram = "MelSpectrogram"
        public static let audioEncoder = "AudioEncoder"

        public static let melSpectrogramPath = melSpectrogram + ".mlmodelc"
        public static let audioEncoderPath = audioEncoder + ".mlmodelc"

        // Vocabulary JSON path (shared by Python/Nemo and CoreML exports).
        public static let vocabularyPath = "vocab.json"

        public static let requiredModels: Set<String> = [
            melSpectrogramPath,
            audioEncoderPath,
        ]
    }

    /// CTC zh-CN model names (full pipeline: Preprocessor + Encoder + CTC Decoder)
    public enum CTCZhCn {
        public static let preprocessor = "Preprocessor"
        public static let encoder = "Encoder-v2-int8"  // Default to int8 quantized version
        public static let encoderFp32 = "Encoder-v1-fp32"
        public static let decoder = "Decoder"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let encoderFp32File = encoderFp32 + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"

        // Vocabulary JSON path
        public static let vocabularyFile = "vocab.json"

        // Download both encoder variants (int8 and fp32) so users can choose at runtime
        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderFile,  // int8 encoder
            encoderFp32File,  // fp32 encoder
            decoderFile,
        ]
    }

    /// TDT ja (Japanese) model names (hybrid model: CTC preprocessor/encoder + TDT decoder/joint v2)
    /// NOTE: Uses parakeetJa repo where v2 models are uploaded
    public enum TDTJa {
        public static let preprocessor = "Preprocessor"
        public static let encoder = "Encoder"
        public static let decoder = "Decoderv2"  // v2: newly converted TDT decoder (uploaded to CTC repo)
        public static let joint = "Jointerv2"  // v2: newly converted TDT joint (uploaded to CTC repo)

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        public static let vocabularyFile = "vocab.json"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
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

    /// Nemotron Speech Streaming 0.6B model names
    /// NVIDIA's streaming FastConformer RNNT with encoder cache
    public enum NemotronStreaming {
        public static let preprocessor = "preprocessor"
        public static let encoder = "encoder"
        public static let decoder = "decoder"
        public static let joint = "joint"
        public static let tokenizer = "tokenizer.json"
        public static let metadata = "metadata.json"

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let jointFile = joint + ".mlmodelc"

        // Encoder in subdirectory (int8 quantized only)
        public static let encoderInt8File = "encoder/encoder_int8.mlmodelc"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderInt8File,
            decoderFile,
            jointFile,
            tokenizer,
            metadata,
        ]
    }

    /// Sortformer streaming diarization model names
    public enum Sortformer {
        public enum Variant: CaseIterable, Sendable {
            case fastV2
            case fastV2_1
            case balancedV2
            case balancedV2_1
            case highContextV2
            case highContextV2_1

            public var name: String {
                switch self {
                case .fastV2:
                    return "Sortformer_v2"
                case .fastV2_1:
                    return "Sortformer_v2.1"
                case .balancedV2:
                    return "SortformerNvidiaLow_v2"
                case .balancedV2_1:
                    return "SortformerNvidiaLow_v2.1"
                case .highContextV2:
                    return "SortformerNvidiaHigh_v2"
                case .highContextV2_1:
                    return "SortformerNvidiaHigh_v2.1"
                }
            }

            public var defaultConfiguration: SortformerConfig {
                switch self {
                case .fastV2:
                    return .fastV2
                case .fastV2_1:
                    return .fastV2_1
                case .balancedV2:
                    return .balancedV2
                case .balancedV2_1:
                    return .balancedV2_1
                case .highContextV2:
                    return .highContextV2
                case .highContextV2_1:
                    return .highContextV2_1
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
        public static let defaultVariant: Variant = .fastV2_1

        /// Bundle name for a specific variant
        public static func bundle(for variant: Variant) -> String {
            return variant.fileName
        }

        /// Bundle name for a given configuration
        public static func bundle(for config: SortformerConfig) -> String? {
            guard let variant = config.modelVariant else {
                return nil
            }
            assert(variant.isCompatible(with: config), "ERROR: Model variant and configuration are not compatible.")
            return variant.fileName
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

    /// LS-EEND streaming diarization model names
    public enum LSEEND {
        public enum Variant: String, CaseIterable, Sendable, CustomStringConvertible {
            case ami = "AMI"
            case callhome = "CALLHOME"
            case dihard2 = "DIHARD II"
            case dihard3 = "DIHARD III"

            public var name: String {
                switch self {
                case .ami:
                    return "ls_eend_ami_step"
                case .callhome:
                    return "ls_eend_callhome_step"
                case .dihard2:
                    return "ls_eend_dih2_step"
                case .dihard3:
                    return "ls_eend_dih3_step"
                }
            }

            public var description: String { rawValue }

            public var stem: String { "\(rawValue)/\(name)" }

            public var modelFile: String { "\(stem).mlmodelc" }

            public var configFile: String { "\(stem).json" }

            public var fileNames: [String] { [modelFile, configFile] }
        }

        /// Lowest latency for streaming
        public static let defaultVariant: Variant = .dihard3

        /// Bundle name for a specific variant
        public static func bundle(for variant: Variant) -> [String] {
            return variant.fileNames
        }

        /// Default bundle name
        public static var defaultBundle: [String] {
            return defaultVariant.fileNames
        }

        /// All Sortformer bundle models required by the downloader
        public static var requiredModels: Set<String> {
            Set(Variant.allCases.flatMap(\.fileNames))
        }
    }

    /// Qwen3-ASR model names
    public enum Qwen3ASR {
        public static let audioEncoderFile = "qwen3_asr_audio_encoder_v2.mlmodelc"
        public static let embeddingFile = "qwen3_asr_embedding.mlmodelc"
        public static let decoderStatefulFile = "qwen3_asr_decoder_stateful.mlmodelc"
        public static let decoderFullFile = "qwen3_asr_decoder_full.mlmodelc"
        public static let embeddingsFile = "qwen3_asr_embeddings.bin"

        /// Legacy model names (lmHead is now fused into decoder_stateful)
        public static let lmHeadFile = "qwen3_asr_lm_head.mlmodelc"
        public static let decoderStackFile = "qwen3_asr_decoder_stack.mlmodelc"
        public static let decoderPrefillFile = "qwen3_asr_decoder_prefill.mlmodelc"

        /// Required models for 3-model pipeline (with embedding CoreML model)
        public static let requiredModels: Set<String> = [
            audioEncoderFile,
            embeddingFile,
            decoderStatefulFile,
        ]

        /// Required files for 2-model pipeline (with Swift-side embedding)
        public static let requiredModelsFull: Set<String> = [
            audioEncoderFile,
            decoderStatefulFile,
            embeddingsFile,
        ]
    }

    /// PocketTTS model names (flow-matching language model TTS)
    public enum PocketTTS {
        public static let condStep = "cond_step"
        public static let flowlmStep = "flowlm_step"
        public static let flowDecoder = "flow_decoder"
        public static let mimiDecoder = "mimi_decoder_v2"
        public static let mimiEncoder = "mimi_encoder"

        public static let condStepFile = condStep + ".mlmodelc"
        public static let flowlmStepFile = flowlmStep + ".mlmodelc"
        public static let flowDecoderFile = flowDecoder + ".mlmodelc"
        public static let mimiDecoderFile = mimiDecoder + ".mlmodelc"
        public static let mimiEncoderFile = mimiEncoder + ".mlmodelc"

        /// Directory containing binary constants, tokenizer, and voice data.
        public static let constantsBinDir = "constants_bin"

        public static let requiredModels: Set<String> = [
            condStepFile,
            flowlmStepFile,
            flowDecoderFile,
            mimiDecoderFile,
            constantsBinDir,
        ]

        /// Models required for voice cloning (optional feature).
        public static let voiceCloningModels: Set<String> = [
            mimiEncoderFile
        ]
    }

    /// Multilingual G2P (CharsiuG2P ByT5) model names
    public enum MultilingualG2P {
        public static let encoder = "MultilingualG2PEncoder"
        public static let decoder = "MultilingualG2PDecoder"

        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"

        public static let requiredModels: Set<String> = [
            encoderFile,
            decoderFile,
        ]
    }

    /// G2P (grapheme-to-phoneme) model names
    public enum G2P {
        public static let encoder = "G2PEncoder"
        public static let decoder = "G2PDecoder"
        public static let vocabulary = "g2p_vocab"

        public static let encoderFile = encoder + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let vocabularyFile = vocabulary + ".json"

        public static let requiredModels: Set<String> = [
            encoderFile,
            decoderFile,
            vocabularyFile,
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
                // Use v1 models on all platforms - v2 has source_noise issues
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
        case .parakeetTdtCtc110m:
            return ModelNames.ASR.requiredModelsFused
        case .parakeetCtc110m, .parakeetCtc06b:
            return ModelNames.CTC.requiredModels
        case .parakeetCtcZhCn:
            return ModelNames.CTCZhCn.requiredModels
        case .parakeetJa:
            return ModelNames.TDTJa.requiredModels
        case .parakeetEou160, .parakeetEou320, .parakeetEou1280:
            return ModelNames.ParakeetEOU.requiredModels
        case .nemotronStreaming1120, .nemotronStreaming560, .nemotronStreaming160, .nemotronStreaming80:
            return ModelNames.NemotronStreaming.requiredModels
        case .diarizer:
            if variant == "offline" {
                return ModelNames.OfflineDiarizer.requiredModels
            }
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            let ttsModels: Set<String>
            if let variant = variant {
                ttsModels = [variant]
            } else {
                ttsModels = ModelNames.TTS.requiredModels
            }
            return ttsModels.union(ModelNames.G2P.requiredModels)
                .union(ModelNames.MultilingualG2P.requiredModels)
        case .pocketTts:
            return ModelNames.PocketTTS.requiredModels
        case .sortformer:
            if let variant = variant {
                return [variant]
            }
            return ModelNames.Sortformer.requiredModels
        case .lseend:
            if let variant = variant {
                return [variant + ".mlmodelc", variant + ".json"]
            }
            return ModelNames.LSEEND.requiredModels
        case .qwen3Asr, .qwen3AsrInt8:
            return ModelNames.Qwen3ASR.requiredModelsFull
        case .multilingualG2p:
            return ModelNames.MultilingualG2P.requiredModels
        }
    }
}
