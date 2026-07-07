import Foundation

/// Model repositories on HuggingFace
public enum Repo: String, CaseIterable, Sendable {
    case vad = "FluidInference/silero-vad-coreml"
    case parakeetV3 = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
    case parakeetV2 = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    case parakeetCtc110m = "FluidInference/parakeet-ctc-110m-coreml"
    case parakeetCtc06b = "FluidInference/parakeet-ctc-0.6b-coreml"
    /// SenseVoiceSmall (FunASR) — non-autoregressive multilingual ASR (50+ langs).
    /// 3-stage: fp32 CPU preprocessor (waveform→560-d LFR feats) + fp16 ANE
    /// encoder+CTC (+ fp32 fallback) + host greedy-CTC decode. See ASR/SenseVoice.
    case senseVoiceSmall = "FluidInference/sensevoice-small-coreml"
    /// Paraformer-large (zh) — non-autoregressive ASR: SANM encoder + CIF
    /// predictor (host-side integrate-and-fire) + parallel decoder. See ASR/Paraformer.
    case paraformerLargeZh = "FluidInference/paraformer-large-zh-coreml"
    // Japanese hybrid TDT: INT8 CTC-trained preprocessor+encoder paired with a
    // TDT decoder+joint. CTC-only inference for Japanese was removed in
    // 846924a1d; only the preprocessor+encoder files from this repo are reused.
    case parakeetJa = "FluidInference/parakeet-0.6b-ja-coreml"
    case parakeetEou160 = "FluidInference/parakeet-realtime-eou-120m-coreml/160ms"
    case parakeetEou320 = "FluidInference/parakeet-realtime-eou-120m-coreml/320ms"
    case parakeetEou1280 = "FluidInference/parakeet-realtime-eou-120m-coreml/1280ms"
    case nemotronStreaming2240 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/2240ms"
    case nemotronStreaming1120 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/1120ms"
    case nemotronStreaming560 = "FluidInference/nemotron-speech-streaming-en-0.6b-coreml/560ms"
    /// Parakeet Unified 0.6B (FastConformer-RNNT). One checkpoint serves both
    /// offline (15 s window) and streaming inference; streaming uses a
    /// chunked-attention encoder re-run over a [left|chunk|right] window
    /// (stateless — no encoder caches). See ASR/Parakeet/Unified.
    case parakeetUnified = "FluidInference/parakeet-unified-en-0.6b-coreml"
    /// Multilingual streaming model. The HF repo is organized as
    /// `<lang>/<tier>ms/` subfolders (9 languages x 4 chunk tiers); the
    /// specific variant subdirectory is supplied dynamically at download
    /// time (see `StreamingNemotronMultilingualAsrManager.downloadAndPreloadShared`),
    /// so this case carries no static subPath.
    case nemotronMultilingual = "FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML"
    case diarizer = "FluidInference/speaker-diarization-coreml"
    /// Root of the kokoro HF repo. The mono Kokoro TTS backend was removed in
    /// favor of `kokoroAne`/`kokoroAneZh`, but this case is kept because the
    /// shared G2P CoreML assets (`G2PEncoder.mlmodelc`, `G2PDecoder.mlmodelc`,
    /// `g2p_vocab.json`) used by KokoroAne for text→IPA still live at the
    /// repository root and are pulled via `variant: "g2p-only"`.
    case kokoro = "FluidInference/kokoro-82m-coreml"
    case kokoroAne = "FluidInference/kokoro-82m-coreml/ANE"
    case kokoroAneZh = "FluidInference/kokoro-82m-coreml/ANE-zh"
    case kokoroAneJa = "FluidInference/kokoro-82m-coreml/ANE-ja"
    case sortformer = "FluidInference/diar-streaming-sortformer-coreml"
    case lseendAmi = "FluidInference/ls-eend-coreml/optimized/ami"
    case lseendCallHome = "FluidInference/ls-eend-coreml/optimized/ch"
    case lseendDihard2 = "FluidInference/ls-eend-coreml/optimized/dih2"
    case lseendDihard3 = "FluidInference/ls-eend-coreml/optimized/dih3"
    case pocketTts = "FluidInference/pocket-tts-coreml"
    case multilingualG2p = "FluidInference/charsiu-g2p-byt5-coreml"
    case parakeetTdtCtc110m = "FluidInference/parakeet-tdt-ctc-110m-coreml"
    case cohereTranscribeCoreml = "FluidInference/cohere-transcribe-03-2026-coreml/q8"
    /// StyleTTS2 LibriTTS — `iteration_3/compiled/` is the only directory
    /// with `.mlmodelc` artifacts; the parent repo also ships `packages/`
    /// (`.mlpackage` source) and `swift/` (a debug harness) that the Swift
    /// loader never touches.
    case styletts2 = "FluidInference/StyleTTS-2-coreml/iteration_3/compiled"
    /// Supertonic-3 multilingual TTS (31 langs). Republished CoreML
    /// conversion of the upstream `Supertone/supertonic-3` ONNX checkpoint;
    /// see `Scripts/convert_supertonic3_to_coreml.py` for the conversion
    /// recipe. Ships four `.mlmodelc` bundles + `tts.json` +
    /// `unicode_indexer.json` at the repo root.
    case supertonic3 = "FluidInference/supertonic-3-coreml"

    /// Repository slug (without owner)
    public var name: String {
        switch self {
        case .nemotronMultilingual:
            return "Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML"
        case .vad:
            return "silero-vad-coreml"
        case .parakeetV3:
            return "parakeet-tdt-0.6b-v3-coreml"
        case .parakeetV2:
            return "parakeet-tdt-0.6b-v2-coreml"
        case .parakeetCtc110m:
            return "parakeet-ctc-110m-coreml"
        case .parakeetCtc06b:
            return "parakeet-ctc-0.6b-coreml"
        case .senseVoiceSmall:
            return "sensevoice-small-coreml"
        case .paraformerLargeZh:
            return "paraformer-large-zh-coreml"
        case .parakeetJa:
            return "parakeet-0.6b-ja-coreml"
        case .parakeetEou160:
            return "parakeet-realtime-eou-120m-coreml/160ms"
        case .parakeetEou320:
            return "parakeet-realtime-eou-120m-coreml/320ms"
        case .parakeetEou1280:
            return "parakeet-realtime-eou-120m-coreml/1280ms"
        case .nemotronStreaming2240:
            return "nemotron-speech-streaming-en-0.6b-coreml/2240ms"
        case .nemotronStreaming1120:
            return "nemotron-speech-streaming-en-0.6b-coreml/1120ms"
        case .nemotronStreaming560:
            return "nemotron-speech-streaming-en-0.6b-coreml/560ms"
        case .parakeetUnified:
            return "parakeet-unified-en-0.6b-coreml"
        case .diarizer:
            return "speaker-diarization-coreml"
        case .kokoro:
            return "kokoro-82m-coreml"
        case .kokoroAne:
            return "kokoro-82m-coreml/ANE"
        case .kokoroAneZh:
            return "kokoro-82m-coreml/ANE-zh"
        case .kokoroAneJa:
            return "kokoro-82m-coreml/ANE-ja"
        case .sortformer:
            return "diar-streaming-sortformer-coreml"
        case .lseendAmi:
            return "ls-eend-coreml/optimized/ami"
        case .lseendCallHome:
            return "ls-eend-coreml/optimized/ch"
        case .lseendDihard2:
            return "ls-eend-coreml/optimized/dih2"
        case .lseendDihard3:
            return "ls-eend-coreml/optimized/dih3"
        case .pocketTts:
            return "pocket-tts-coreml"
        case .multilingualG2p:
            return "charsiu-g2p-byt5-coreml"
        case .parakeetTdtCtc110m:
            return "parakeet-tdt-ctc-110m-coreml"
        case .cohereTranscribeCoreml:
            return "cohere-transcribe-03-2026-coreml/q8"
        case .styletts2:
            return "StyleTTS-2-coreml/iteration_3/compiled"
        case .supertonic3:
            return "supertonic-3-coreml"
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
        case .kokoroAne, .kokoroAneZh, .kokoroAneJa:
            return "FluidInference/kokoro-82m-coreml"
        case .nemotronStreaming2240, .nemotronStreaming1120, .nemotronStreaming560:
            return "FluidInference/nemotron-speech-streaming-en-0.6b-coreml"
        case .nemotronMultilingual:
            return "FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML"
        case .sortformer:
            return "FluidInference/diar-streaming-sortformer-coreml"
        case .lseendAmi, .lseendCallHome, .lseendDihard2, .lseendDihard3:
            return "FluidInference/ls-eend-coreml"
        case .parakeetTdtCtc110m:
            return "FluidInference/parakeet-tdt-ctc-110m-coreml"
        case .cohereTranscribeCoreml:
            return "FluidInference/cohere-transcribe-03-2026-coreml"
        case .styletts2:
            return "FluidInference/StyleTTS-2-coreml"
        default:
            return "FluidInference/\(name)"
        }
    }

    /// Subdirectory within repo (for repos with multiple model variants)
    public var subPath: String? {
        switch self {
        case .kokoroAne:
            return "ANE"
        case .kokoroAneZh:
            return "ANE-zh"
        case .kokoroAneJa:
            return "ANE-ja"
        case .parakeetEou160:
            return "160ms"
        case .parakeetEou320:
            return "320ms"
        case .parakeetEou1280:
            return "1280ms"
        case .nemotronStreaming2240:
            return "nemotron_coreml_2240ms"
        case .nemotronStreaming1120:
            return "nemotron_coreml_1120ms"
        case .nemotronStreaming560:
            return "nemotron_coreml_560ms"
        case .lseendAmi:
            return "optimized/ami"
        case .lseendCallHome:
            return "optimized/ch"
        case .lseendDihard2:
            return "optimized/dih2"
        case .lseendDihard3:
            return "optimized/dih3"
        case .cohereTranscribeCoreml:
            return "q8"
        case .styletts2:
            return "iteration_3/compiled"
        default:
            return nil
        }
    }

    /// Local folder name used for caching
    public var folderName: String {
        switch self {
        case .kokoro:
            return "kokoro"
        case .kokoroAne:
            return "kokoro-82m-coreml/ANE"
        case .kokoroAneZh:
            return "kokoro-82m-coreml/ANE-zh"
        case .kokoroAneJa:
            return "kokoro-82m-coreml/ANE-ja"
        case .parakeetEou160:
            return "parakeet-eou-streaming/160ms"
        case .parakeetEou320:
            return "parakeet-eou-streaming/320ms"
        case .parakeetEou1280:
            return "parakeet-eou-streaming/1280ms"
        case .nemotronMultilingual:
            return "nemotron-multilingual"
        case .nemotronStreaming2240:
            return "nemotron-streaming/2240ms"
        case .nemotronStreaming1120:
            return "nemotron-streaming/1120ms"
        case .nemotronStreaming560:
            return "nemotron-streaming/560ms"
        case .sortformer:
            return "sortformer"
        case .parakeetCtc110m:
            return "parakeet-ctc-110m-coreml"
        case .parakeetCtc06b:
            return "parakeet-ctc-0.6b-coreml"
        case .parakeetJa:
            return "parakeet-ja"
        case .parakeetTdtCtc110m:
            return "parakeet-tdt-ctc-110m"
        case .lseendAmi:
            return "ls-eend/ami"
        case .lseendCallHome:
            return "ls-eend/ch"
        case .lseendDihard2:
            return "ls-eend/dih2"
        case .lseendDihard3:
            return "ls-eend/dih3"
        case .cohereTranscribeCoreml:
            return "cohere-transcribe/q8"
        case .styletts2:
            return "styletts2"
        case .supertonic3:
            return "supertonic-3"
        default:
            return name.replacingOccurrences(of: "-coreml", with: "")
        }
    }
}

/// Encoder precision for the v3 Parakeet TDT 0.6B encoder.
public enum ParakeetEncoderPrecision: String, Sendable, CaseIterable {
    case int8
    case int4

    public var encoderFileName: String {
        switch self {
        case .int8:
            return ModelNames.ASR.encoderFile
        case .int4:
            return ModelNames.ASR.encoderInt4File
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
        /// Joint decoder variant for v3 that exposes top-K outputs
        /// (`top_k_ids`, `top_k_logits`) used for language-aware script filtering.
        public static let jointV3File = "JointDecisionv3.mlmodelc"
        public static let encoderInt4File = "EncoderInt4.mlmodelc"
        public static let ctcHeadFile = ctcHead + ".mlmodelc"

        /// Required models for v2 / legacy split-frontend loaders.
        /// v3 uses `requiredModelsV3(precision:)` (with `jointV3File`).
        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderFile,
            decoderFile,
            jointFile,
        ]

        public static func requiredModelsV3(
            precision: ParakeetEncoderPrecision = .int8
        ) -> Set<String> {
            [
                preprocessorFile,
                precision.encoderFileName,
                decoderFile,
                jointV3File,
            ]
        }

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

    /// SenseVoiceSmall (FunASR) model names. 3-stage pipeline:
    ///   Preprocessor (fp32, CPU): waveform → 560-d LFR features
    ///   SenseVoiceSmall (fp16, ANE): features + lang/textnorm → CTC logits
    ///   SenseVoiceSmall_fp32 (fp32): encoder fallback for non-ANE hardware
    /// Plus `vocab.json` (25055 SentencePiece tokens, auto-fetched as a root file).
    public enum SenseVoice {
        public static let preprocessor = "SenseVoicePreprocessor"
        public static let encoder = "SenseVoiceSmall"  // fp16, runs on ANE (default)
        public static let encoderInt8 = "SenseVoiceSmall_int8"  // int8 weights, ANE, ~half size
        public static let encoderFp32 = "SenseVoiceSmall_fp32"  // fp32 fallback (non-ANE)

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let encoderInt8File = encoderInt8 + ".mlmodelc"
        public static let encoderFp32File = encoderFp32 + ".mlmodelc"
        public static let vocabularyFile = "vocab.json"

        public static func requiredModels(precision: String? = nil) -> Set<String> {
            let encoder: String
            switch precision {
            case "int8":
                encoder = encoderInt8File
            case "fp32":
                encoder = encoderFp32File
            default:
                encoder = encoderFile
            }
            return [
                preprocessorFile,
                encoder,
            ]
        }

        public static let requiredModels: Set<String> = requiredModels()
    }

    /// Paraformer-large (zh) model names. 4 CoreML stages + host CIF:
    ///   Preprocessor (fp32/CPU): waveform -> 560-d LFR features
    ///   Encoder (fp16/ANE): SANM encoder (enumerated buckets)
    ///   CifAlphas (fp16/ANE): enc_out -> per-frame alphas (host does integrate-and-fire)
    ///   Decoder (fp16/ANE): parallel decoder -> token logits
    /// Plus `vocab.json` (8404 CharTokenizer tokens, auto-fetched as a root file).
    public enum ParaformerZh {
        public static let preprocessor = "ParaformerPreprocessor"
        public static let encoder = "ParaformerEncoder"
        public static let encoderInt8 = "ParaformerEncoder_int8"  // ~half size, ANE
        public static let cifAlphas = "ParaformerCifAlphas"
        public static let decoder = "ParaformerDecoder"
        public static let decoderInt8 = "ParaformerDecoder_int8"  // ~half size, ANE

        public static let preprocessorFile = preprocessor + ".mlmodelc"
        public static let encoderFile = encoder + ".mlmodelc"
        public static let encoderInt8File = encoderInt8 + ".mlmodelc"
        public static let cifAlphasFile = cifAlphas + ".mlmodelc"
        public static let decoderFile = decoder + ".mlmodelc"
        public static let decoderInt8File = decoderInt8 + ".mlmodelc"
        public static let vocabularyFile = "vocab.json"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderFile,
            encoderInt8File,
            cifAlphasFile,
            decoderFile,
            decoderInt8File,
        ]
    }

    /// TDT ja (Japanese) model names.
    ///
    /// Hybrid layout: the CTC-trained preprocessor + encoder from the
    /// `parakeetJa` repo are reused as the acoustic frontend, paired with a TDT
    /// decoder + joint (filenames `Decoderv2.mlmodelc` / `Jointerv2.mlmodelc`
    /// from the same repo). CTC-only inference for Japanese was removed in
    /// 846924a1d.
    public enum TDTJa {
        public static let preprocessor = "Preprocessor"
        public static let encoder = "Encoder"
        public static let decoder = "Decoderv2"
        public static let joint = "Jointerv2"

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
        public static let sileroVad = "silero-vad-unified-256ms-v6.2.1"

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
        /// Optional fused decoder+joint (B1). Present in tiers that ship the
        /// merged inner-loop model (e.g. 2240ms); loaded only if the file exists.
        public static let decoderJointFile = "decoder_joint.mlmodelc"

        // Encoder in subdirectory (int8 quantized only)
        public static let encoderInt8File = "encoder/encoder_int8.mlmodelc"

        public static let requiredModels: Set<String> = [
            preprocessorFile,
            encoderInt8File,
            decoderFile,
            jointFile,
            decoderJointFile,
            tokenizer,
            metadata,
        ]
    }

    /// Parakeet Unified 0.6B (FastConformer-RNNT) model names.
    /// One checkpoint, two encoder exports: chunked-attention streaming
    /// (default download) and full-attention offline 15 s window
    /// (variant "offline" — used for overlapping-batch long-form transcription).
    public enum ParakeetUnified {
        // Mel features are computed natively in Swift (`UnifiedMelExtractor` /
        // `AudioMelSpectrogram` + NeMo per_feature normalization), so the CoreML
        // preprocessor bundle is neither downloaded nor loaded.
        /// Encoders ship in two precisions. int8 (per-channel linear symmetric
        /// weights) is the default: identical test-clean WER to fp16
        /// (1.83%/2.14% vs 1.82%/2.15%), same ANE latency, half the download.
        public static let streamingEncoderInt8File = "parakeet_unified_encoder_streaming_70_13_13_int8.mlmodelc"
        public static let streamingEncoderFp16File = "parakeet_unified_encoder_streaming_70_13_13.mlmodelc"
        public static let offlineEncoderInt8File = "parakeet_unified_encoder_int8.mlmodelc"
        public static let offlineEncoderFp16File = "parakeet_unified_encoder.mlmodelc"
        public static let decoderFile = "parakeet_unified_decoder.mlmodelc"
        public static let jointDecisionFile = "parakeet_unified_joint_decision_single_step.mlmodelc"
        public static let vocab = "vocab.json"
        public static let metadata = "metadata.json"

        /// Streaming encoder bundle name. The `[left, chunk, right]` attention
        /// context is baked into the encoder at conversion time and encoded in
        /// the filename suffix (e.g. `70_13_13`), so each latency tier is a
        /// distinct file. Defaults to the shipped `70_13_13` (2.08 s) config.
        public static func streamingEncoderFile(
            precision: UnifiedEncoderPrecision, contextSuffix: String = "70_13_13"
        ) -> String {
            let base = "parakeet_unified_encoder_streaming_\(contextSuffix)"
            return precision == .int8 ? "\(base)_int8.mlmodelc" : "\(base).mlmodelc"
        }

        public static func offlineEncoderFile(precision: UnifiedEncoderPrecision) -> String {
            precision == .int8 ? offlineEncoderInt8File : offlineEncoderFp16File
        }

        public static func requiredModels(variant: String?) -> Set<String> {
            let isOffline = variant?.hasPrefix("offline") == true
            let isFp16 = variant?.hasSuffix("fp16") == true
            let precision: UnifiedEncoderPrecision = isFp16 ? .fp16 : .int8
            // Context-independent shared files.
            var models: Set<String> = [decoderFile, jointDecisionFile, vocab, metadata]
            // The streaming encoder is context-specific — each [L,C,R] latency
            // tier bakes its attention mask into a distinct bundle — so the
            // streaming manager passes its exact encoder via ModelHub.download's
            // `additionalModelNames` instead of pulling a fixed default here
            // (which would over-fetch the 70_13_13 encoder for every tier). The
            // offline encoder is fixed, so it stays in the base set.
            if isOffline {
                models.insert(offlineEncoderFile(precision: precision))
            }
            return models
        }
    }

    /// Nemotron Speech Streaming Multilingual 0.6B model names.
    ///
    /// Unlike the English variant, the multilingual build keeps all four
    /// CoreML artifacts at the top level (no `encoder/` subdirectory), and
    /// the encoder takes an extra `prompt_id` int32 input per chunk.
    /// The model is local-path-only at the moment (no HuggingFace repo yet),
    /// so there is no `Repo` enum case and no `requiredModels` set wired into
    /// `getRequiredModelNames`.
    public enum NemotronMultilingualStreaming {
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

        /// Same model names with the uncompiled `.mlpackage` extension that
        /// `mobius/.../coreml/build_int8/` ships before CoreML compilation.
        /// `StreamingNemotronMultilingualAsrManager` accepts either layout
        /// (compiled `.mlmodelc` is preferred when both are present).
        public static let preprocessorPackage = preprocessor + ".mlpackage"
        public static let encoderPackage = encoder + ".mlpackage"
        public static let decoderPackage = decoder + ".mlpackage"
        public static let jointPackage = joint + ".mlpackage"
    }

    /// Sortformer streaming diarization model names
    public enum Sortformer {
        /// Selects which weight-precision build of the model set to download.
        ///
        /// Both sets are the BNNS-fixed v3 rebuild (issue #726); they differ only in head weights:
        /// - `.fp16`: full-precision head. Default. Best DER.
        /// - `.palettized`: 6-bit k-means LUT head, ~2.5x smaller on disk and ~330MB RAM
        ///   (vs ~2.4GB for `highContextV2_1` fp16), at ~+0.9pp DER. Use on RAM-constrained
        ///   devices where the fp16 set crashes (older iPhones).
        public enum ModelPrecision: String, Sendable, CaseIterable {
            case fp16
            case palettized

            /// Repo subdirectory holding this precision's model set.
            public var subdirectory: String {
                switch self {
                case .fp16: return "v3/fp16"
                case .palettized: return "v3/palettized"
                }
            }
        }

        public enum Variant: CaseIterable, Sendable {
            case fastV2
            case fastV2_1
            case balancedV2
            case balancedV2_1
            case highContextV2
            case highContextV2_1
            /// Higher-throughput streaming: larger chunk (~2s output latency) for ~4x the
            /// real-time factor of `fastV2_1` at near-identical per-inference cost.
            case efficientV2_1

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
                case .efficientV2_1:
                    return "SortformerEfficient_v2.1"
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
                case .efficientV2_1:
                    return .efficientV2_1
                }
            }

            /// Compiled-model path for this variant at the default (`.fp16`) precision.
            public var fileName: String {
                return fileName(precision: .fp16)
            }

            /// Compiled-model path for this variant at the given weight precision.
            public func fileName(precision: ModelPrecision) -> String {
                return "\(precision.subdirectory)/\(name).mlmodelc"
            }

            public func isCompatible(with config: SortformerConfig) -> Bool {
                defaultConfiguration.isCompatible(with: config)
            }
        }

        /// Repo subdirectory holding the default (`.fp16`) model set. Both v3 sets are the
        /// BNNS-fixed rebuild (the older root-level models hit a "tensor as both input and output"
        /// graph-compile crash on newer BNNS — issue #726). Select the 6-bit, ~2.5x-smaller set
        /// via `SortformerConfig.precision = .palettized` (fixes RAM-driven crashes on older
        /// devices at ~+0.9pp DER).
        public static let modelsSubdirectory = ModelPrecision.fp16.subdirectory

        /// Lowest latency for streaming
        public static let defaultVariant: Variant = .fastV2_1

        /// Bundle name for a specific variant at the default (`.fp16`) precision
        public static func bundle(for variant: Variant) -> String {
            return variant.fileName
        }

        /// Bundle name for a specific variant at the given precision
        public static func bundle(for variant: Variant, precision: ModelPrecision) -> String {
            return variant.fileName(precision: precision)
        }

        /// Bundle name for a given configuration (honors `config.precision`)
        public static func bundle(for config: SortformerConfig) -> String? {
            guard let variant = config.modelVariant else {
                return nil
            }
            assert(variant.isCompatible(with: config), "ERROR: Model variant and configuration are not compatible.")
            return variant.fileName(precision: config.precision)
        }

        /// Default bundle name
        public static var defaultBundle: String {
            return defaultVariant.fileName
        }

        /// All Sortformer bundle models required by the downloader
        public static var requiredModels: Set<String> {
            Set(Variant.allCases.map(\.fileName))
        }

        // MARK: - Offline (whole-window) model

        /// Fused offline model name (v2.1 weights). Unlike the streaming variants this is a single
        /// graph `mel -> speaker_preds` over a fixed 30.72s window (3072 mel -> 384 output frames),
        /// with no spkcache/FIFO state — see ``OfflineSortformerDiarizer``.
        public static let offlineModelName = "SortformerOffline_v2.1"

        /// Compiled-model path for the offline model at the given precision.
        public static func offlineBundle(precision: ModelPrecision = .fp16) -> String {
            return "\(precision.subdirectory)/\(offlineModelName).mlmodelc"
        }
    }

    /// LS-EEND streaming diarization model names
    public enum LSEEND {
        public enum Variant: CaseIterable, Sendable, CustomStringConvertible {
            case ami
            case callhome
            case dihard2
            case dihard3

            public var repo: Repo {
                switch self {
                case .ami: return .lseendAmi
                case .callhome: return .lseendCallHome
                case .dihard2: return .lseendDihard2
                case .dihard3: return .lseendDihard3
                }
            }

            public var name: String {
                switch self {
                case .ami:
                    return "ls_eend_ami"
                case .callhome:
                    return "ls_eend_ch"
                case .dihard2:
                    return "ls_eend_dih2"
                case .dihard3:
                    return "ls_eend_dih3"
                }
            }

            public var description: String { name }

            public func name(forStep step: StepSize) -> String {
                "\(name)_\(step)"
            }

            public func fileName(forStep step: StepSize) -> String {
                "\(step)/\(name)_\(step).mlmodelc"
            }
        }

        public enum StepSize: Int, CaseIterable, Sendable, CustomStringConvertible {
            case step100ms = 1
            case step200ms = 2
            case step300ms = 3
            case step400ms = 4
            case step500ms = 5

            public var description: String {
                switch self {
                case .step100ms: return "100ms"
                case .step200ms: return "200ms"
                case .step300ms: return "300ms"
                case .step400ms: return "400ms"
                case .step500ms: return "500ms"
                }
            }
        }

        /// Lowest latency for streaming
        public static let defaultVariant: Variant = .dihard3
        public static let defaultStep: StepSize = .step100ms

        /// Bundle name for a specific variant
        public static func bundle(for variant: Variant, withStep step: StepSize) -> [String] {
            return [variant.fileName(forStep: step)]
        }

        /// Default bundle name
        public static var defaultBundle: [String] {
            return [defaultVariant.fileName(forStep: defaultStep)]
        }

        /// All LS-EEND bundle models required by the downloader
        public static var requiredModels: Set<String> {
            Set(Variant.allCases.flatMap { StepSize.allCases.map($0.fileName) })
        }
    }

    /// PocketTTS model names (flow-matching language model TTS)
    public enum PocketTTS {
        public static let condStep = "cond_step"
        /// Optional one-shot conditioning prefill. Fills the whole voice+text
        /// KV block in a single predict; when the file is absent the prefill
        /// runs token-by-token through `cond_step` (same output schema).
        public static let condPrefill = "cond_prefill"
        public static let flowlmStep = "flowlm_step"
        /// int8 variant of the FlowLM transformer published upstream alongside
        /// the default `flowlm_step`. Lives in the same `v2/<lang>/` directory
        /// and gets selected when the caller asks for `.int8` precision.
        public static let flowlmStepV2 = "flowlm_stepv2"
        public static let flowDecoder = "flow_decoder"
        /// v2.1 fused flow decoder (8-step LSD unrolled, 100% ANE). Replaces
        /// the per-step `flow_decoder` in v2.1 packs.
        public static let flowDecoderFused = "flow_decoder_fused"
        /// Rank-4 ANE-eligible FlowLM step (mobius Trial 19): 100% ANE under
        /// `.cpuAndNeuralEngine`; split k/v cache I/O with explicit output
        /// names. Selected by `PocketTtsModelPlacement.ane`.
        public static let flowlmStepAne = "flowlm_step_ane"
        /// Rank-4 ANE-eligible conditioning prefill (mobius Trial 20).
        public static let condPrefillAne = "cond_prefill_ane"
        /// MLState multifunction package (mobius Trial 23): `write_state` /
        /// `prefill` / `generate` functions over a shared fp16 KV state.
        /// Replaces BOTH the conditioner and the FlowLM+flow-decoder pair
        /// for `PocketTtsModelPlacement.aneState`. Requires macOS 15+/iOS 18+.
        public static let pocketState = "pocket_state"
        public static let mimiDecoder = "mimi_decoder"
        public static let mimiEncoder = "mimi_encoderv2"

        /// Function names inside the `pocket_state` multifunction package.
        public enum StateFunction {
            public static let writeState = "write_state"
            public static let prefill = "prefill"
            public static let generate = "generate"
        }

        public static let condStepFile = condStep + ".mlmodelc"
        public static let condPrefillFile = condPrefill + ".mlmodelc"
        public static let flowlmStepFile = flowlmStep + ".mlmodelc"
        public static let flowlmStepV2File = flowlmStepV2 + ".mlmodelc"
        public static let flowDecoderFile = flowDecoder + ".mlmodelc"
        public static let flowDecoderFusedFile = flowDecoderFused + ".mlmodelc"
        public static let flowlmStepAneFile = flowlmStepAne + ".mlmodelc"
        public static let condPrefillAneFile = condPrefillAne + ".mlmodelc"
        public static let pocketStateFile = pocketState + ".mlmodelc"
        public static let mimiDecoderFile = mimiDecoder + ".mlmodelc"
        public static let mimiEncoderFile = mimiEncoder + ".mlmodelc"

        /// Directory containing binary constants, tokenizer, and voice data.
        public static let constantsBinDir = "constants_bin"

        /// FlowLM filename for a given precision. Both variants ship in the
        /// same `v2/<lang>/` directory upstream; only the FlowLM transformer
        /// has an int8 variant — `cond_step`, `flow_decoder`, and
        /// `mimi_decoder` always load the default file.
        public static func flowlmStepFile(precision: PocketTtsPrecision) -> String {
            switch precision {
            case .fp16: return flowlmStepFile
            case .int8: return flowlmStepV2File
            }
        }

        /// Required files inside any language's `v2/<lang>/` pack for the
        /// given precision. The set differs only in the FlowLM filename.
        public static func requiredModels(precision: PocketTtsPrecision) -> Set<String> {
            requiredModels(precision: precision, placement: .gpu)
        }

        /// Required files for a precision + placement combination. `.ane`
        /// swaps the rank-5 conditioner/FlowLM for the rank-4 ANE-eligible
        /// variants (fp16 only — `precision` does not affect the ANE set).
        public static func requiredModels(
            precision: PocketTtsPrecision,
            placement: PocketTtsModelPlacement
        ) -> Set<String> {
            switch placement {
            case .gpu:
                return [
                    condPrefillFile,
                    flowlmStepFile(precision: precision),
                    flowDecoderFusedFile,
                    mimiDecoderFile,
                    constantsBinDir,
                ]
            case .ane:
                return [
                    condPrefillAneFile,
                    flowlmStepAneFile,
                    flowDecoderFusedFile,
                    mimiDecoderFile,
                    constantsBinDir,
                ]
            case .aneState:
                // The multifunction package fuses conditioner + FlowLM +
                // flow decoder; only mimi stays a separate model.
                return [
                    pocketStateFile,
                    mimiDecoderFile,
                    constantsBinDir,
                ]
            }
        }

        /// Required files for the default precision. Kept for callers that
        /// haven't been updated to pass a precision argument.
        public static let requiredModels: Set<String> = [
            condPrefillFile,
            flowlmStepFile,
            flowDecoderFusedFile,
            mimiDecoderFile,
            constantsBinDir,
        ]
    }

    /// StyleTTS2 LibriTTS (iteration_3) — 8-stage CoreML pipeline + 6 bucket
    /// variants (T = 64 / 128 / 256) for the two stages that can't accept
    /// `RangeDim` on the token axis (`bert`, `fused_diffusion_sampler`).
    /// File names match the HuggingFace tree at
    /// `FluidInference/StyleTTS-2-coreml/iteration_3/compiled/`.
    public enum StyleTTS2 {
        // ---- Stage 1: text encoder (CPU_ONLY, fp16, RangeDim T) ----
        public static let textEncoder = "text_encoder_fp16"
        public static let textEncoderFile = textEncoder + ".mlmodelc"

        // ---- Stage 2: bert + bert_encoder (ALL, fp16, fixed T axis) ----
        // Default T = 57 (capped at ~37 chars). Buckets cover longer prompts.
        public static let bert = "bert_fp16"
        public static let bertFile = bert + ".mlmodelc"
        public static let bertT64File = "bert_fp16_t64.mlmodelc"
        public static let bertT128File = "bert_fp16_t128.mlmodelc"
        public static let bertT256File = "bert_fp16_t256.mlmodelc"

        // ---- Stage 3: ref encoder (CPU_AND_GPU, fp16, mel-driven) ----
        public static let refEncoder = "ref_encoder_fp16"
        public static let refEncoderFile = refEncoder + ".mlmodelc"

        // ---- Stage 4: fused 5-step ADPM2 sampler (ALL, fp16, fixed T axis) ----
        public static let fusedDiffusionSampler = "fused_diffusion_sampler_fp16"
        public static let fusedDiffusionSamplerFile = fusedDiffusionSampler + ".mlmodelc"
        public static let fusedDiffusionSamplerT64File = "fused_diffusion_sampler_fp16_t64.mlmodelc"
        public static let fusedDiffusionSamplerT128File = "fused_diffusion_sampler_fp16_t128.mlmodelc"
        public static let fusedDiffusionSamplerT256File = "fused_diffusion_sampler_fp16_t256.mlmodelc"

        // ---- Stage 5: duration predictor (CPU_ONLY, fp16, RangeDim T) ----
        public static let durationPredictor = "duration_predictor_fp16"
        public static let durationPredictorFile = durationPredictor + ".mlmodelc"

        // ---- Stage 6: fused f0n + harmonic source (CPU_ONLY, **fp32**) ----
        // Kept fp32 — har computes sin(2π × cumsum(f0)) over 88 200 samples
        // and fp16 cumsum drifts ~10 bits, causing audible phase distortion
        // in the second half of the clip.
        public static let fusedF0nHarSource = "fused_f0n_har_source"
        public static let fusedF0nHarSourceFile = fusedF0nHarSource + ".mlmodelc"

        // ---- Stage 7: decoder pre (CPU_AND_NE, fp16, AdaIN encode/decode) ----
        public static let decoderPre = "decoder_pre_fp16"
        public static let decoderPreFile = decoderPre + ".mlmodelc"

        // ---- Stage 8: decoder upsample (CPU_ONLY, fp16, HiFi-GAN ups) ----
        public static let decoderUpsample = "decoder_upsample_fp16"
        public static let decoderUpsampleFile = decoderUpsample + ".mlmodelc"

        /// The 8 default-path mlmodelc bundles (T = 57). Bucketed variants
        /// are downloaded on demand by the synthesizer when a prompt's
        /// token count exceeds the bucket below it.
        public static let requiredModels: Set<String> = [
            textEncoderFile,
            bertFile,
            refEncoderFile,
            fusedDiffusionSamplerFile,
            durationPredictorFile,
            fusedF0nHarSourceFile,
            decoderPreFile,
            decoderUpsampleFile,
        ]

        /// All 14 mlmodelc bundles (8 defaults + 6 bucket variants). Used
        /// when the caller wants to pre-stage every artefact at install
        /// time (e.g. CLI download command).
        public static let allModels: Set<String> = requiredModels.union([
            bertT64File, bertT128File, bertT256File,
            fusedDiffusionSamplerT64File, fusedDiffusionSamplerT128File, fusedDiffusionSamplerT256File,
        ])

        /// Sentinel used by the downloader to fetch only the bucket variants
        /// for a specific T. Returned set holds both the bert + sampler files
        /// for that bucket.
        public static func bucketModels(forT t: Int) -> Set<String> {
            switch t {
            case 64: return [bertT64File, fusedDiffusionSamplerT64File]
            case 128: return [bertT128File, fusedDiffusionSamplerT128File]
            case 256: return [bertT256File, fusedDiffusionSamplerT256File]
            default: return []
            }
        }
    }

    /// Supertonic-3 multilingual TTS — 4 `.mlmodelc` bundles + 2 companion
    /// JSON files. File names match the HuggingFace tree at
    /// `FluidInference/supertonic-3-coreml/`.
    public enum Supertonic3 {
        public static let textEncoder = "TextEncoder"
        public static let durationPredictor = "DurationPredictor"
        public static let vectorEstimator = "VectorEstimator"
        public static let vocoder = "Vocoder"

        public static let textEncoderFile = textEncoder + ".mlmodelc"
        public static let durationPredictorFile = durationPredictor + ".mlmodelc"
        public static let vectorEstimatorFile = vectorEstimator + ".mlmodelc"
        public static let vocoderFile = vocoder + ".mlmodelc"

        public static let configFile = "tts.json"
        public static let unicodeIndexerFile = "unicode_indexer.json"

        /// The four CoreML bundles required by `Supertonic3Synthesizer`.
        public static let requiredModels: Set<String> = [
            textEncoderFile,
            durationPredictorFile,
            vectorEstimatorFile,
            vocoderFile,
        ]

        /// Models + companion JSON files the downloader must fetch.
        public static let requiredFiles: Set<String> =
            requiredModels.union([configFile, unicodeIndexerFile])

        // MARK: VectorEstimator variants

        /// The three modules shared by every VectorEstimator variant.
        public static let sharedModelFiles: Set<String> = [
            textEncoderFile, durationPredictorFile, vocoderFile,
        ]
        public static let companionFiles: Set<String> = [configFile, unicodeIndexerFile]

        /// Fixed latent buckets published for ANE residency (smallest ≥ chunk
        /// length is selected at synthesis time).
        public static let aneBuckets: [Int] = [128, 256, 512]

        /// Quantized / fixed-length VectorEstimator builds live under this repo
        /// subdirectory; the FP16 dynamic model + the 3 shared modules sit at
        /// the repo root.
        public static let variantsSubdir = "VectorEstimatorVariants"

        /// Bundle (dir) name of a VectorEstimator build, e.g.
        /// `VectorEstimator`, `VectorEstimator_int4`, `VectorEstimator_L256_int8`.
        public static func vectorEstimatorName(precisionSuffix: String?, bucket: Int?) -> String {
            var name = vectorEstimator
            if let bucket { name += "_L\(bucket)" }
            if let precisionSuffix { name += "_\(precisionSuffix)" }
            return name
        }

        /// Repo-relative `.mlmodelc` path for a VectorEstimator build. FP16
        /// dynamic stays at the root; every quantized/bucketed variant is under
        /// `variantsSubdir`.
        public static func vectorEstimatorFile(precisionSuffix: String?, bucket: Int?) -> String {
            let base = vectorEstimatorName(precisionSuffix: precisionSuffix, bucket: bucket) + ".mlmodelc"
            if precisionSuffix == nil && bucket == nil { return base }
            return "\(variantsSubdir)/\(base)"
        }

        /// Files to fetch for a given VectorEstimator download variant token
        /// (see `Supertonic3VectorEstimator.downloadVariant`). `nil` ⇒ the
        /// historical FP16 dynamic model.
        public static func requiredFiles(veVariant: String?) -> Set<String> {
            var set = sharedModelFiles.union(companionFiles)
            switch veVariant {
            case .some(let v) where v.hasPrefix("dyn-"):
                set.insert(vectorEstimatorFile(precisionSuffix: String(v.dropFirst(4)), bucket: nil))
            case .some(let v) where v.hasPrefix("ane-"):
                let q = String(v.dropFirst(4))
                for b in aneBuckets {
                    set.insert(vectorEstimatorFile(precisionSuffix: q, bucket: b))
                }
            default:
                set.insert(vectorEstimatorFile)  // fp16 dynamic
            }
            return set
        }
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

    /// Cohere Transcribe model names
    /// Encoder-decoder ASR with 14-language support (35-second window architecture).
    ///
    /// Two decoder variants are published:
    ///   - `decoderCacheExternal` (v1) — FP16, dynamic `attention_mask`
    ///     (`RangeDim(1, 108)`). CPU/GPU only — dynamic shapes block ANE.
    ///   - `decoderCacheExternalV2` — FP32, fixed `attention_mask` shape
    ///     `[1, 1, 1, 108]`. ANE-resident, ~1.6× faster decoder end-to-end
    ///     on Apple Silicon. Drop-in replacement; `CoherePipeline`
    ///     auto-detects the variant by inspecting the `attention_mask`
    ///     input shape.
    public enum CohereTranscribe {
        public static let encoder = "cohere_encoder"
        public static let decoderCacheExternal = "cohere_decoder_cache_external"
        public static let decoderCacheExternalV2 = "cohere_decoder_cache_external_v2"
        public static let vocab = "vocab.json"

        public static let encoderCompiledFile = encoder + ".mlmodelc"
        public static let decoderCacheExternalCompiledFile = decoderCacheExternal + ".mlmodelc"
        public static let decoderCacheExternalV2CompiledFile = decoderCacheExternalV2 + ".mlmodelc"

        /// Default required set — ships the ANE-friendly v2 decoder.
        public static let requiredModels: Set<String> = [
            encoderCompiledFile,
            decoderCacheExternalV2CompiledFile,
            vocab,
        ]

        /// Legacy set using the FP16 dynamic decoder (pre-v2). Retained so
        /// callers that want the older decoder can opt in explicitly.
        public static let requiredModelsLegacy: Set<String> = [
            encoderCompiledFile,
            decoderCacheExternalCompiledFile,
            vocab,
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

    /// laishere/kokoro-coreml — 7-stage CoreML chain (fp16+int8pal, ANE-optimized)
    /// vendored from https://github.com/laishere/kokoro-coreml.
    public enum KokoroAne {
        public static let albert = "KokoroAlbert.mlmodelc"
        public static let postAlbert = "KokoroPostAlbert.mlmodelc"
        public static let alignment = "KokoroAlignment.mlmodelc"
        public static let prosody = "KokoroProsody.mlmodelc"
        // v2: atan2 phase-correction in the noise-source STFT (removes broad-spectrum
        // HF noise / "sharpness"). Renamed (not overwritten) so cached clients
        // re-download. See mobius laishere-coreml docs/trials-and-errors.md.
        public static let noise = "KokoroNoise_v2.mlmodelc"
        public static let vocoder = "KokoroVocoder.mlmodelc"
        public static let tail = "KokoroTail.mlmodelc"

        /// Auxiliary (non-CoreML) files that must accompany the mlmodelc bundles.
        public static let vocab = "vocab.json"
        public static let defaultVoiceFile = "af_heart.bin"

        /// Mandarin (`ANE-zh/`) default voice. Voice packs in the Mandarin
        /// bundle live under a `voices/` subdirectory; the path is kept in
        /// the constant so the existing "all-required-files-present" check
        /// still resolves correctly when the file lands at
        /// `<repoDir>/voices/zf_001.bin`.
        public static let defaultVoiceFileZh = "voices/zf_001.bin"

        /// Japanese (`ANE-ja/`) default voice. Mirrors the Mandarin layout —
        /// voice packs live under a `voices/` subdirectory, so the path is
        /// kept in the constant for the "all-required-files-present" check to
        /// resolve against `<repoDir>/voices/jf_alpha.bin`.
        public static let defaultVoiceFileJa = "voices/jf_alpha.bin"

        /// Mandarin g2pW polyphone-disambiguator CoreML bundle. Lives under
        /// `<repoDir>/g2pw/` — included in `requiredModelsZh` so the bulk
        /// `ensureModels(.mandarin)` grab pulls it without an extra round
        /// trip. The two auxiliary text files (`vocab.txt`,
        /// `POLYPHONIC_CHARS.txt`) ship via the lazy
        /// `KokoroAneResourceDownloader.ensureMandarinG2pw` helper because
        /// `ModelHub.download` does not whitelist `.txt` for
        /// subPath repos and a manual fetch keeps the bulk-grab matcher
        /// idempotent.
        public static let g2pwModelZh = "g2pw/g2pw.mlmodelc"

        /// All seven .mlmodelc bundles.
        public static let requiredCoreMLModels: Set<String> = [
            albert, postAlbert, alignment, prosody, noise, vocoder, tail,
        ]

        /// CoreML bundles + the vocab JSON + the English default voice .bin.
        public static var requiredModels: Set<String> {
            requiredCoreMLModels.union([vocab, defaultVoiceFile])
        }

        /// CoreML bundles + the vocab JSON + the Mandarin default voice .bin
        /// (under `voices/`) + the g2pW CoreML bundle (under `g2pw/`).
        public static var requiredModelsZh: Set<String> {
            requiredCoreMLModels.union([
                vocab, defaultVoiceFileZh, g2pwModelZh,
            ])
        }

        /// CoreML bundles + the vocab JSON + the Japanese default voice .bin
        /// (under `voices/`). No G2P assets — the Japanese variant ships no
        /// in-process text frontend; callers feed pre-computed IPA via
        /// `synthesizeFromPhonemes(_:voice:)`.
        public static var requiredModelsJa: Set<String> {
            requiredCoreMLModels.union([vocab, defaultVoiceFileJa])
        }
    }

    static func getRequiredModelNames(for repo: Repo, variant: String?) -> Set<String> {
        switch repo {
        case .nemotronMultilingual:
            // Compiled .mlmodelc component dirs. Download is normally driven by
            // `download(subdirectory:)` (dynamic <lang>/<tier>ms path), which does
            // not consult this set; provided for completeness / exhaustiveness.
            return [
                NemotronMultilingualStreaming.preprocessorFile,
                NemotronMultilingualStreaming.encoderFile,
                NemotronMultilingualStreaming.decoderFile,
                NemotronMultilingualStreaming.jointFile,
                "decoder_joint.mlmodelc",
                "decoder_joint_noencproj.mlmodelc",
                "joint_noencproj_batched.mlmodelc",
            ]
        case .vad:
            return ModelNames.VAD.requiredModels
        case .parakeetV3:
            let precision = ParakeetEncoderPrecision(rawValue: variant ?? "") ?? .int8
            return ModelNames.ASR.requiredModelsV3(precision: precision)
        case .parakeetV2:
            return ModelNames.ASR.requiredModels
        case .parakeetTdtCtc110m:
            return ModelNames.ASR.requiredModelsFused
        case .parakeetCtc110m, .parakeetCtc06b:
            return ModelNames.CTC.requiredModels
        case .senseVoiceSmall:
            return ModelNames.SenseVoice.requiredModels(precision: variant)
        case .paraformerLargeZh:
            return ModelNames.ParaformerZh.requiredModels
        case .parakeetJa:
            return ModelNames.TDTJa.requiredModels
        case .parakeetEou160, .parakeetEou320, .parakeetEou1280:
            return ModelNames.ParakeetEOU.requiredModels
        case .nemotronStreaming2240, .nemotronStreaming1120, .nemotronStreaming560:
            return ModelNames.NemotronStreaming.requiredModels
        case .parakeetUnified:
            // Variants: nil/"fp16" (streaming), "offline"/"offline-fp16" (batch).
            return ModelNames.ParakeetUnified.requiredModels(variant: variant)
        case .diarizer:
            if variant == "offline" {
                return ModelNames.OfflineDiarizer.requiredModels
            }
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            // The mono Kokoro TTS backend was removed; this repo is now only
            // used by KokoroAne to fetch the shared G2P CoreML assets out of
            // the repo root for text -> IPA conversion.
            return ModelNames.G2P.requiredModels
        case .pocketTts:
            return ModelNames.PocketTTS.requiredModels
        case .kokoroAne:
            return ModelNames.KokoroAne.requiredModels
        case .kokoroAneZh:
            return ModelNames.KokoroAne.requiredModelsZh
        case .kokoroAneJa:
            return ModelNames.KokoroAne.requiredModelsJa
        case .sortformer:
            if let variant = variant {
                return [variant]
            }
            return ModelNames.Sortformer.requiredModels
        case .lseendAmi, .lseendCallHome, .lseendDihard2, .lseendDihard3:
            if let variant = variant {
                return [variant + ".mlmodelc"]
            }
            return ModelNames.LSEEND.requiredModels
        case .multilingualG2p:
            return ModelNames.MultilingualG2P.requiredModels
        case .cohereTranscribeCoreml:
            return ModelNames.CohereTranscribe.requiredModels
        case .styletts2:
            // Sentinel variants:
            //   "all"     → 14 bundles (8 defaults + 6 buckets)
            //   "t64" / "t128" / "t256" → just that bucket pair
            //   nil       → 8 default mlmodelc bundles
            switch variant {
            case "all":
                return ModelNames.StyleTTS2.allModels
            case "t64":
                return ModelNames.StyleTTS2.bucketModels(forT: 64)
            case "t128":
                return ModelNames.StyleTTS2.bucketModels(forT: 128)
            case "t256":
                return ModelNames.StyleTTS2.bucketModels(forT: 256)
            default:
                return ModelNames.StyleTTS2.requiredModels
            }
        case .supertonic3:
            return ModelNames.Supertonic3.requiredFiles(veVariant: variant)
        }
    }
}
