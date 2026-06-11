import AVFoundation
@preconcurrency import CoreML
import Foundation

/// Callback invoked when new tokens are decoded (for live transcription updates).
/// Fires with the running transcript text only — the language tag, if any,
/// is surfaced via `detectedLanguage()`.
public typealias NemotronMultilingualPartialCallback = @Sendable (String) -> Void

/// High-level manager for the Nemotron Speech Streaming Multilingual 0.6B pipeline.
///
/// Distinct from the English `StreamingNemotronAsrManager` because:
///   1. The encoder takes an extra `prompt_id` int32 [1] input per chunk.
///   2. The vocab is ~13k tokens and includes language-tag pieces like
///      `<en-US>` which are filtered from the transcript.
///   3. The channel cache shape is `[1, 24, 56, 1024]` (att_context_size=[56,0]).
///
/// **Models** are published at
/// `FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML`. Use
/// `downloadAndPreloadShared(...)` to fetch + load a `latin/` or `multilingual/`
/// `<chunkMs>ms` variant from HuggingFace, or point at a local directory
/// containing the compiled `.mlmodelc` bundles (or `.mlpackage` archives) plus
/// `metadata.json` and `tokenizer.json`.
public actor StreamingNemotronMultilingualAsrManager {
    internal let logger = AppLogger(category: "NemotronMultilingualStreaming")

    // Models
    internal var preprocessor: MLModel?
    internal var encoder: MLModel?
    internal var decoder: MLModel?
    internal var joint: MLModel?
    /// Optional fused decoder+joint mlpackage. If non-nil, the RNN-T inner
    /// loop uses this single call instead of separate decoder + joint calls
    /// (Tier B1 optimization, expected +5-12% RTFx by halving per-token CoreML
    /// invocations).
    internal var decoderJoint: MLModel?
    /// Optional triple-fused decoder+joint+argmax mlpackage. If non-nil,
    /// supersedes decoderJoint AND eliminates the logit-tensor transfer + Swift
    /// argmax (Tier B2 extension, additional +1-3% RTFx).
    /// Returns `token_id (int32)` directly instead of full vocab logits.
    internal var decoderJointArgmax: MLModel?
    /// Optional B3+B1 fused decoder+joint-without-encproj mlpackage. Pairs with
    /// the B3 encoder (encprojsplit converter) which emits `encoder_proj` as a
    /// 6th output. Takes precedence over decoderJoint when present.
    internal var decoderJointNoEncProj: MLModel?

    /// B3 speculative-blank batched joint: takes pre-projected
    /// `encoder_proj` (640-d, K frames) + decoder (640-d, 1 frame) and
    /// produces logits over K frames. Unlocked when (a) the encoder
    /// mlpackage emits `encoder_proj` as a separate output (B3 split)
    /// and (b) `joint_noencproj_batched.mlpackage` is present.
    /// Used by the smarter speculative-blank decode path that fast-skips
    /// blank-streaks K-at-a-time.
    internal var jointNoEncProjBatched: MLModel?
    /// K (speculative window width) read from jointNoEncProjBatched's
    /// encoder_proj input shape at load time. The Swift inner loop must
    /// always match the loaded mlpackage's K — hardcoding here would
    /// shape-mismatch any non-K=8 build. Defaults to 8 (no-op fallback)
    /// when the model isn't present.
    internal var jointNoEncProjBatchedK: Int = 8

    // Components
    private let audioConverter = AudioConverter()
    internal var tokenizer: NemotronMultilingualTokenizer?

    // Configuration (loaded from metadata.json)
    public internal(set) var config: NemotronMultilingualStreamingConfig

    /// When `true`, `finish()` tidies the final transcript for display:
    /// capitalizes the first character and appends a sentence-terminal mark
    /// (`.`, or `。` for zh/ja) when the text doesn't already end in one.
    ///
    /// Heuristic only — it cannot recover a missing mid-clause comma and always
    /// guesses a period over `?`/`!`. Off by default; opt in for UI polish.
    public var appendTerminalPunctuation: Bool = false

    // Audio Buffer + read offset. Using an offset instead of removeFirst()
    // avoids O(N²) memmove cost when processing very long files (the buffer
    // holds the whole input until compacted; removeFirst would shift
    // gigabytes per chunk). The offset is reset by periodic compaction.
    private var audioBuffer: [Float] = []
    private var audioBufferOffset: Int = 0

    // Accumulated token IDs (raw, including any lang-tag tokens)
    internal var accumulatedTokenIds: [Int] = []

    // Per-token absolute timings captured during the RNNT decode loop, parallel
    // to the user-visible (lang-tag-stripped) token stream. Each token's
    // startTime is its absolute encoder-frame index * secondsPerEncoderFrame.
    // Exposed via finishWithTokenTimings() so callers can derive word-level
    // timestamps (e.g. for speaker attribution). Lang-tag tokens are excluded
    // (they are stripped from the decoded transcript), see appendTokenTiming.
    internal var accumulatedTokenTimings: [TokenTiming] = []
    // Running encoder-frame base across processed chunks (advances by the
    // chunk's encoder-frame count after each decode loop, including VAD-skipped
    // chunks so the timeline stays aligned to real audio). Reset with the session.
    internal var absoluteFrameBase: Int = 0
    // Snapshot of token timings taken in finish() before the working buffers are
    // cleared, so finishWithTokenTimings() can return them after finish() runs.
    internal var lastFinishTokenTimings: [TokenTiming] = []

    // First lang-tag piece encountered this session (without angle brackets).
    private var firstDetectedLanguage: String?

    // Encoder cache states
    internal var cacheChannel: MLMultiArray?
    internal var cacheTime: MLMultiArray?
    internal var cacheLen: MLMultiArray?

    /// MLState for stateful encoder (cache_channel + cache_time live in ANE
    /// memory across calls). Stored as `Any?` because `MLState` is only
    /// available on macOS 15+/iOS 18+; cast at use sites guarded by
    /// `#available`. Non-nil only if the loaded encoder was traced with
    /// ct.StateType for those tensors (see
    /// `convert_nemotron_multilingual_mlstate.py`).
    internal var encoderState: Any?

    // Mel cache (last 9 frames from previous chunk)
    internal var melCache: MLMultiArray?

    /// Pipelining: mel for the next chunk, pre-computed by the previous
    /// processChunk while the encoder was running on ANE. Hides preprocessor
    /// latency from every chunk after the first.
    internal var prefetchedMel: MLMultiArray?

    /// Pre-allocated output-backing buffers passed via
    /// `MLPredictionOptions.outputBackings` so each CoreML prediction writes
    /// in-place into a stable buffer instead of allocating a fresh
    /// MLMultiArray every call. Saves ~5 MB of allocation per encoder call
    /// and ~4 KB per inner-loop joint/decoder_joint call.
    internal var encoderPredictionOptions: MLPredictionOptions?
    internal var decoderPredictionOptions: MLPredictionOptions?
    internal var jointPredictionOptions: MLPredictionOptions?
    internal var decoderJointPredictionOptions: MLPredictionOptions?
    internal var decoderJointArgmaxPredictionOptions: MLPredictionOptions?
    internal var decoderJointNoEncProjPredictionOptions: MLPredictionOptions?
    /// Output backings for the smart-spec batched joint hot path.
    /// runSpeculativeBlankDecodeV2 calls jointBatched.prediction() once per
    /// K-frame window; pre-allocating the [1, K, 1, V] logits backing once
    /// avoids per-call MLMultiArray allocation. Only populated when
    /// jointNoEncProjBatched is loaded.
    internal var jointNoEncProjBatchedPredictionOptions: MLPredictionOptions?
    /// Reusable per-frame encoder step buffer. Refilled in-place inside the
    /// inner RNN-T greedy loop instead of allocating a fresh [1, 1024, 1]
    /// every emitted token.
    internal var encoderStepBuf: MLMultiArray?
    internal var encoderProjStepBuf: MLMultiArray?

    /// L8 outputBackings tightening: reusable buffers for the Swift-side
    /// encoder_proj computation in the smart-spec path.
    ///
    /// `encProjReusable`: [1, T_enc, 640] fp32 — holds the full-chunk
    ///   encoder_proj computed by computeEncoderProjSwift each chunk. Was
    ///   allocated fresh per chunk (~143 KB × 7860 chunks ≈ 1.1 GB churn
    ///   across a full test-clean run).
    /// `encProjBatchReusable`: [1, K, 640] fp32 — sliced batch of K frames
    ///   from encProjReusable fed to jointBatched. Was allocated fresh per
    ///   call (~30 KB × N calls/chunk × 7860 chunks).
    /// Reuse is safe because processChunk is sequential per actor — no
    /// overlapping consumers of these buffers within a single stream.
    internal var encProjReusable: MLMultiArray?
    internal var encProjBatchReusable: MLMultiArray?

    /// Reusable per-token decoder inputs. The inner RNN-T loop previously
    /// allocated fresh [1,1] int32 token + [1] int32 length arrays on every
    /// emitted token (~25k allocs/h on test-clean). Pre-allocated once at
    /// loadModels, refilled in place by the caller.
    internal var tokenInputBuf: MLMultiArray?
    internal var tokenLenBuf: MLMultiArray?

    /// Reusable preprocessor input buffers ([1, chunkSamples] float32 audio +
    /// [1] int32 length). Triple-stage helper allocates one set per chunk
    /// (~25-800 allocs/h depending on chunk size). Pre-allocated once,
    /// refilled in place. Triple-stage helpers are sequential (await before
    /// next dispatch) so a single shared buffer is safe.
    internal var audioInputBuf: MLMultiArray?
    internal var audioLenBuf: MLMultiArray?

    // Triple-stage pipelining: encoder[t+1] dispatched concurrent with
    // decode[t]. These are the prefetched encoder outputs (and the caches
    // it produced) — set during processChunk(t), consumed by processChunk(t+1).
    internal var prefetchedEncoded: MLMultiArray?
    internal var prefetchedEncoderProj: MLMultiArray?
    internal var prefetchedCacheChannel: MLMultiArray?
    internal var prefetchedCacheTime: MLMultiArray?
    internal var prefetchedCacheLen: MLMultiArray?

    // Per-stage timing accumulators (seconds). Reset on `reset()`.
    public internal(set) var prepNanos: UInt64 = 0
    public internal(set) var encNanos: UInt64 = 0
    public internal(set) var decNanos: UInt64 = 0
    public internal(set) var chunkCount: Int = 0
    public internal(set) var vadSkipCount: Int = 0

    // E4: smart-spec acceptance-rate counters. Track how often the K=4
    // batched speculation finds an all-blank window (fast-skip path) vs
    // hits a non-blank (must fall back to per-frame inner loop). High
    // all-blank rate means K could potentially be larger; low rate means
    // smaller K might be better (less wasted speculation).
    public internal(set) var specWindowsTotal: Int = 0
    public internal(set) var specWindowsAllBlank: Int = 0
    public internal(set) var specWindowsHitNonBlank: Int = 0

    /// Smarter-VAD hangover state: number of consecutive low-RMS chunks
    /// seen. Skip fires only after `vadHangoverChunks` consecutive low-RMS
    /// chunks — preserves the first low chunk after speech (consonant
    /// tails) and only skips true sustained silence.
    internal var vadConsecutiveLowChunks: Int = 0

    // Decoder LSTM states
    internal var hState: MLMultiArray?
    internal var cState: MLMultiArray?
    internal var lastToken: Int32

    // Current prompt id (the language hint). Defaults to `defaultPromptId`
    // ("auto" mode) until the caller sets a specific language.
    internal var currentPromptId: Int32

    // Current language code requested by the caller (e.g. "en-US"). Used
    // to look up the matching lang-tag token id when forced-prefix decoding
    // is enabled.
    private var currentLanguageCode: String?

    // When true, after each reset / language change, run the decoder once
    // with the lang-tag token id for `currentLanguageCode` to seed the
    // LSTM state. This is the Whisper-style hard language lock; the
    // encoder still receives `prompt_id` as usual.
    private var useForcedPrefix: Bool = false

    // Callbacks
    internal var partialCallback: NemotronMultilingualPartialCallback?

    // Stats
    internal var processedChunks: Int = 0

    // Diagnostic stats from last `finish()` call (token count + detected language
    // captured before state is cleared). Used by benchmark `--dump-samples`.
    private var lastFinishTokenCount: Int = 0
    private var lastFinishDetectedLanguage: String?
    private var lastFinishProcessedChunks: Int = 0

    public internal(set) var mlConfiguration: MLModelConfiguration

    public init(configuration: MLModelConfiguration? = nil) {
        // Default to `.cpuAndNeuralEngine`: the int8 encoder is ANE-targeted.
        // `MLModelConfiguration()`'s default `.all` routes int8 ops to GPU
        // and runs ~10× slower than the ANE path.
        self.mlConfiguration = configuration ?? MLModelConfigurationUtils.defaultConfiguration()
        self.config = NemotronMultilingualStreamingConfig()
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)
    }

    /// Set callback for partial transcription updates
    public func setPartialCallback(_ callback: @escaping NemotronMultilingualPartialCallback) {
        self.partialCallback = callback
    }

    /// Set the language hint by code (e.g. `"en-US"`, `"zh-CN"`, `"auto"`).
    /// Falls back to the model's `default_prompt_id` if the code is unknown.
    public func setLanguage(_ language: String?) async {
        let id = config.promptId(forLanguage: language)
        self.currentPromptId = Int32(id)
        self.currentLanguageCode = language
        logger.info("Prompt id set to \(id) for language \(language ?? "auto")")
        if useForcedPrefix {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Enable or disable Whisper-style forced-prefix decoding. When enabled,
    /// after each `reset()` or `setLanguage(_:)` call we run the decoder once
    /// with the lang-tag token id matching `currentLanguageCode`, threading
    /// its output `h_out`/`c_out` into the LSTM state and setting
    /// `lastToken` to the lang-tag id. The encoder still gets `prompt_id`.
    public func setForcedPrefix(_ enabled: Bool) async {
        self.useForcedPrefix = enabled
        logger.info("Forced prefix \(enabled ? "enabled" : "disabled")")
        if enabled {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Whether forced-prefix decoding is currently enabled.
    public func forcedPrefixEnabled() -> Bool { useForcedPrefix }

    /// Set the language hint by raw prompt id (advanced users).
    /// The caller is responsible for ensuring the id is in `[0, numPrompts)`.
    public func setPromptId(_ promptId: Int) {
        self.currentPromptId = Int32(promptId)
    }

    /// Current prompt id (the language hint fed to the encoder).
    public func promptId() -> Int { Int(currentPromptId) }

    /// First language-tag piece (e.g. `"en-US"`) emitted by the decoder this
    /// session, or `nil` if no tag has been seen yet.
    public func detectedLanguage() -> String? { firstDetectedLanguage }

    /// Load models from a directory containing preprocessor, encoder, decoder,
    /// joint, plus `metadata.json` and `tokenizer.json`. Accepts either
    /// `.mlmodelc` (preferred) or uncompiled `.mlpackage` bundles.
    public func loadModels(from directory: URL) async throws {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform(
                "Nemotron multilingual int8 streaming models require Apple Silicon (ANE). Intel Macs are not supported."
            )
        }

        logger.info("Loading Nemotron multilingual CoreML models from \(directory.path)...")

        // Load config from metadata.json (required — the prompt dictionary lives here)
        let metadataPath = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.metadata)
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw ASRError.processingFailed(
                "metadata.json not found at \(metadataPath.path). The multilingual variant requires it for prompt_dictionary and lang_tag_token_ids."
            )
        }
        self.config = try NemotronMultilingualStreamingConfig(from: metadataPath)
        self.lastToken = Int32(config.blankIdx)
        self.currentPromptId = Int32(config.defaultPromptId)
        logger.info(
            "Loaded multilingual config: \(config.chunkMs)ms chunks, vocab=\(config.vocabSize), \(config.numPrompts) prompts, default=\(config.defaultPromptId)"
        )

        // Load model bundles (prefer .mlmodelc, fall back to .mlpackage with on-demand compile)
        let preprocessorURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.preprocessorFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.preprocessorPackage
        )
        self.preprocessor = try await MLModel.load(
            contentsOf: preprocessorURL,
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_PREPROCESSOR_CU", base: mlConfiguration, logger: logger)
        )

        let encoderURL = try await locateModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.encoderFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.encoderPackage
        )
        self.encoder = try await MLModel.load(
            contentsOf: encoderURL,
            configuration: Self.computeUnitOverride(
                name: "FLUIDAUDIO_ENCODER_CU", base: mlConfiguration, logger: logger)
        )
        // Detect stateful encoder (cache_channel/cache_time as MLState rather
        // than I/O tensors). makeState() returns a fresh zero-initialized state.
        if #available(macOS 15, iOS 18, *) {
            if let enc = self.encoder, !enc.modelDescription.stateDescriptionsByName.isEmpty {
                self.encoderState = enc.makeState()
                logger.info("Loaded stateful encoder (MLState cache) — using ANE-resident cache path")
            }
        }

        // Bare decoder + joint are optional (lean B1 ships omit them).
        // Loaded only if present; a valid decode path is enforced below.
        if let decoderURL = try await locateOptionalModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.decoderFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.decoderPackage
        ) {
            self.decoder = try await MLModel.load(
                contentsOf: decoderURL,
                configuration: Self.computeUnitOverride(
                    name: "FLUIDAUDIO_DECODER_CU", base: mlConfiguration, logger: logger)
            )
        }

        if let jointURL = try await locateOptionalModelBundle(
            in: directory,
            compiled: ModelNames.NemotronMultilingualStreaming.jointFile,
            uncompiled: ModelNames.NemotronMultilingualStreaming.jointPackage
        ) {
            self.joint = try await MLModel.load(
                contentsOf: jointURL,
                configuration: Self.computeUnitOverride(
                    name: "FLUIDAUDIO_JOINT_CU", base: mlConfiguration, logger: logger)
            )
        }

        // Optional triple-fused decoder+joint+argmax mlpackage (Tier B2).
        // Takes precedence over the dec+joint fusion below if present.
        if let argmaxURL = try await locateOptionalModelBundle(
            in: directory, compiled: "decoder_joint_argmax.mlmodelc",
            uncompiled: "decoder_joint_argmax.mlpackage"
        ) {
            self.decoderJointArgmax = try await MLModel.load(
                contentsOf: argmaxURL,
                configuration: Self.computeUnitOverride(
                    name: "FLUIDAUDIO_DECODERJOINT_CU", base: mlConfiguration, logger: logger)
            )
            logger.info("Loaded decoder_joint_argmax — using triple-fusion inner-loop path")
        }

        // Optional B3+B1 fused decoder+joint-without-encproj mlpackage. Takes
        // precedence over plain B1 decoder_joint when present (requires the
        // encoder to also emit `encoder_proj`).
        if self.decoderJointArgmax == nil,
            let noEncProjURL = try await locateOptionalModelBundle(
                in: directory, compiled: "decoder_joint_noencproj.mlmodelc",
                uncompiled: "decoder_joint_noencproj.mlpackage"
            )
        {
            self.decoderJointNoEncProj = try await MLModel.load(
                contentsOf: noEncProjURL, configuration: mlConfiguration)
            logger.info("Loaded decoder_joint_noencproj — using B3+B1 inner-loop path")
        }

        // Optional fused decoder+joint mlpackage (Tier B1 optimization).
        // Used if triple-fusion and B3+B1 are not present.
        if self.decoderJointArgmax == nil && self.decoderJointNoEncProj == nil,
            let fusedURL = try await locateOptionalModelBundle(
                in: directory, compiled: "decoder_joint.mlmodelc",
                uncompiled: "decoder_joint.mlpackage"
            )
        {
            self.decoderJoint = try await MLModel.load(contentsOf: fusedURL, configuration: mlConfiguration)
            logger.info("Loaded decoder_joint — using merged inner-loop path")
        }

        // Optional smart speculative-blank batched joint (V2, default-on).
        // Takes pre-projected encoder_proj [1, K, 640] + decoder
        // [1, 640, 1] → logits [1, K, 1, V]. K is auto-detected from the
        // input shape further down.
        if let specBatchedURL = try await locateOptionalModelBundle(
            in: directory, compiled: "joint_noencproj_batched.mlmodelc",
            uncompiled: "joint_noencproj_batched.mlpackage"
        ) {
            let specConfig = Self.computeUnitOverride(
                name: "FLUIDAUDIO_JOINT_BATCHED_CU", base: mlConfiguration, logger: logger)
            self.jointNoEncProjBatched = try await MLModel.load(contentsOf: specBatchedURL, configuration: specConfig)
            logger.info("Loaded joint_noencproj_batched — smart speculative-blank path available")
        }
        // Read K from the loaded model's encoder_proj input shape so the
        // Swift hot loop always matches the asset (K=8 historically; K=4
        // build under evaluation at 1120ms).
        if let m = self.jointNoEncProjBatched,
            let constraint = m.modelDescription.inputDescriptionsByName["encoder_proj"]?.multiArrayConstraint,
            constraint.shape.count >= 2
        {
            let kFromModel = constraint.shape[1].intValue
            if kFromModel > 0 {
                self.jointNoEncProjBatchedK = kFromModel
                logger.info("Smart-spec K = \(kFromModel) (from joint_noencproj_batched encoder_proj input shape)")
            }
        }

        // Smart-speculative-blank load-time state report. Smart-spec is
        // default-on as of May 2026 (T3 K=4 at 1120ms = +2.0%, K=8 at
        // 4480ms = +1.7%, both A/B/A/B non-overlapping, WER-neutral).
        // Honor explicit opt-out: env-var = "0"/"false"/"no" disables.
        // Missing assets → transparent fallback to legacy inner loop.
        let smartSpecEnvVar = ProcessInfo.processInfo.environment["FLUIDAUDIO_ENABLE_SMART_SPECULATIVE"]
        let smartSpecExplicitlyDisabled: Bool = {
            guard let v = smartSpecEnvVar?.lowercased() else { return false }
            return v == "0" || v == "false" || v == "no"
        }()
        var smartSpecMissing: [String] = []
        if self.jointNoEncProjBatched == nil {
            smartSpecMissing.append("joint_noencproj_batched.mlpackage")
        }
        if smartSpecExplicitlyDisabled {
            logger.info(
                "Smart-spec: explicitly disabled via FLUIDAUDIO_ENABLE_SMART_SPECULATIVE=\(smartSpecEnvVar ?? "")")
        } else if smartSpecMissing.isEmpty {
            logger.info("Smart-spec: enabled (default-on; assets present; K=\(self.jointNoEncProjBatchedK))")
        } else {
            // Default-on intent, but assets missing → legacy fallback.
            // Warn only if the user explicitly opted IN with the env-var
            // (they probably expected smart-spec to run); otherwise emit
            // info because the operator may have intentionally trimmed
            // the bundle.
            let msg =
                "Smart-spec: assets missing (\(smartSpecMissing.joined(separator: ", "))); falling back to legacy inner loop"
            if smartSpecEnvVar != nil {
                logger.warning(msg)
            } else {
                logger.info(msg)
            }
        }
        // Load tokenizer with lang-tag filter set
        let tokenizerURL = directory.appendingPathComponent(ModelNames.NemotronMultilingualStreaming.tokenizer)
        self.tokenizer = try NemotronMultilingualTokenizer(
            vocabPath: tokenizerURL,
            langTagTokenIds: config.langTagTokenIds
        )

        // Initialize states
        try resetStates()

        // Build output-backing prediction options for each model. All output
        // shapes are static, so we can allocate once and pass via
        // MLPredictionOptions.outputBackings to skip per-call allocation.
        self.encoderPredictionOptions = Self.makePredictionOptions(for: self.encoder)
        self.decoderPredictionOptions = Self.makePredictionOptions(for: self.decoder)
        self.jointPredictionOptions = Self.makePredictionOptions(for: self.joint)
        self.decoderJointPredictionOptions = Self.makePredictionOptions(for: self.decoderJoint)
        self.decoderJointArgmaxPredictionOptions = Self.makePredictionOptions(for: self.decoderJointArgmax)
        self.decoderJointNoEncProjPredictionOptions = Self.makePredictionOptions(for: self.decoderJointNoEncProj)
        self.jointNoEncProjBatchedPredictionOptions = Self.makePredictionOptions(for: self.jointNoEncProjBatched)
        // Reusable inner-loop step buffers ([1, encoder_dim, 1] and
        // [1, 1, joint_dim] for the B3 path).
        self.encoderStepBuf = try? MLMultiArray(shape: [1, NSNumber(value: config.encoderDim), 1], dataType: .float32)
        self.encoderProjStepBuf = try? MLMultiArray(shape: [1, 1, NSNumber(value: 640)], dataType: .float32)

        // Reusable per-token decoder input buffers. tokenLen is a constant
        // 1 written once at allocation; tokenInput is refilled per iteration
        // with the current token ID. Saves ~25k MLMultiArray allocs/h on
        // test-clean.
        if let tokInput = try? MLMultiArray(shape: [1, 1], dataType: .int32) {
            self.tokenInputBuf = tokInput
        }
        if let tokLen = try? MLMultiArray(shape: [1], dataType: .int32) {
            tokLen[0] = 1
            self.tokenLenBuf = tokLen
        }

        // Reusable preprocessor input buffers — [1, chunkSamples] float32
        // audio + [1] int32 length. Refilled by triple-stage helper.
        if let audBuf = try? MLMultiArray(shape: [1, NSNumber(value: config.chunkSamples)], dataType: .float32) {
            self.audioInputBuf = audBuf
        }
        if let audLen = try? MLMultiArray(shape: [1], dataType: .int32) {
            audLen[0] = NSNumber(value: config.chunkSamples)
            self.audioLenBuf = audLen
        }

        // First-chunk warm-up: dispatch one zero-input prediction per model so
        // the ANE program is compiled + resident before the first real
        // chunk. Cuts ~10-20ms off every clip's first chunk (which can't
        // benefit from triple-stage prefetch since there's no prior chunk).
        // Streaming-tier per-clip RTFx improvement: ~+15-30% expected.
        await warmupModels()

        logger.info(
            "Nemotron multilingual models loaded successfully (\(config.chunkMs)ms chunks)."
        )
    }

    /// Warm up each loaded CoreML model by issuing one zero-input prediction.
    /// Ensures ANE programs are compiled + resident BEFORE the first real
    /// audio chunk, so the per-clip first-chunk cold start is gone.
    private func warmupModels() async {
        // preprocessor + encoder are always present and are the bulk of cold
        // start — warm them unconditionally. Bare decoder/joint are optional on
        // lean B1 ships; requiring them here skipped ALL warmup (incl. encoder)
        // on those ships. They're bound only in the unfused branch below.
        guard let preprocessor = preprocessor,
            let encoder = encoder,
            let cacheChannel = cacheChannel,
            let cacheTime = cacheTime,
            let cacheLen = cacheLen,
            let hState = hState,
            let cState = cState
        else { return }

        // Preprocessor: 1s of silence
        if let audio = try? MLMultiArray(shape: [1, 16000], dataType: .float32),
            let audioLen = try? MLMultiArray(shape: [1], dataType: .int32)
        {
            audio.reset(to: 0)
            audioLen[0] = 16000
            let input = try? MLDictionaryFeatureProvider(dictionary: [
                "audio": MLFeatureValue(multiArray: audio),
                "audio_length": MLFeatureValue(multiArray: audioLen),
            ])
            if let input = input {
                _ = try? await preprocessor.prediction(from: input)
            }
        }

        // Encoder: zero mel + zeros caches
        if let mel = try? MLMultiArray(
            shape: [1, NSNumber(value: config.melFeatures), NSNumber(value: config.totalMelFrames)],
            dataType: .float32),
            let melLen = try? MLMultiArray(shape: [1], dataType: .int32),
            let promptId = try? MLMultiArray(shape: [1], dataType: .int32)
        {
            mel.reset(to: 0)
            melLen[0] = NSNumber(value: config.totalMelFrames)
            promptId[0] = NSNumber(value: currentPromptIdValue())
            let input = try? MLDictionaryFeatureProvider(dictionary: [
                "mel": MLFeatureValue(multiArray: mel),
                "mel_length": MLFeatureValue(multiArray: melLen),
                "cache_channel": MLFeatureValue(multiArray: cacheChannel),
                "cache_time": MLFeatureValue(multiArray: cacheTime),
                "cache_len": MLFeatureValue(multiArray: cacheLen),
                "prompt_id": MLFeatureValue(multiArray: promptId),
            ])
            if let input = input {
                _ = try? await encoder.prediction(from: input)
            }
        }

        // Decoder + joint warm-up (or B1/B2 if loaded). Easiest to just hit
        // the fully-fused path if available, falling back to individual
        // decoder + joint calls.
        let tokenInput = try? MLMultiArray(shape: [1, 1], dataType: .int32)
        let tokenLen = try? MLMultiArray(shape: [1], dataType: .int32)
        let encStep = try? MLMultiArray(
            shape: [1, NSNumber(value: config.encoderDim), 1], dataType: .float32)
        guard let tokenInput = tokenInput, let tokenLen = tokenLen, let encStep = encStep else {
            return
        }
        tokenInput[0] = NSNumber(value: config.blankIdx)
        tokenLen[0] = 1
        encStep.reset(to: 0)

        if let dja = decoderJointArgmax {
            let input = try? MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: hState),
                "c_in": MLFeatureValue(multiArray: cState),
                "encoder": MLFeatureValue(multiArray: encStep),
            ])
            if let input = input { _ = try? await dja.prediction(from: input) }
        } else if let dj = decoderJoint {
            let input = try? MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: hState),
                "c_in": MLFeatureValue(multiArray: cState),
                "encoder": MLFeatureValue(multiArray: encStep),
            ])
            if let input = input { _ = try? await dj.prediction(from: input) }
        } else if let decoder = decoder, let joint = joint {
            let decInput = try? MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: hState),
                "c_in": MLFeatureValue(multiArray: cState),
            ])
            if let decInput = decInput {
                _ = try? await decoder.prediction(from: decInput)
            }
            let decStep = try? MLMultiArray(shape: [1, 640, 1], dataType: .float32)
            if let decStep = decStep {
                decStep.reset(to: 0)
                let jntInput = try? MLDictionaryFeatureProvider(dictionary: [
                    "encoder": MLFeatureValue(multiArray: encStep),
                    "decoder": MLFeatureValue(multiArray: decStep),
                ])
                if let jntInput = jntInput {
                    _ = try? await joint.prediction(from: jntInput)
                }
            }
        }

        // Smart-spec batched joint warm-up. Skipped until now — production
        // hot path was paying compile/dispatch cost on the first chunk.
        // Uses encoder_proj [1, K, 640] + decoder [1, 640, 1] → logits.
        if let jb = jointNoEncProjBatched {
            let K = jointNoEncProjBatchedK
            if let encProj = try? MLMultiArray(shape: [1, NSNumber(value: K), 640], dataType: .float32),
                let decStep = try? MLMultiArray(shape: [1, 640, 1], dataType: .float32)
            {
                encProj.reset(to: 0)
                decStep.reset(to: 0)
                let input = try? MLDictionaryFeatureProvider(dictionary: [
                    "encoder_proj": MLFeatureValue(multiArray: encProj),
                    "decoder": MLFeatureValue(multiArray: decStep),
                ])
                if let input = input {
                    if let opts = jointNoEncProjBatchedPredictionOptions {
                        _ = try? await jb.prediction(from: input, options: opts)
                    } else {
                        _ = try? await jb.prediction(from: input)
                    }
                }
            }
        }

        logger.info("Model warm-up complete")
    }

    /// Build an `MLPredictionOptions` with pre-allocated output backings.
    /// IMPORTANT: skip outputs that loop back as inputs to the next call
    /// (decoder/joint LSTM state, encoder cache state) — sharing those
    /// backings across calls causes the model to overwrite the value
    /// before the next call has finished reading it. Causes catastrophic
    /// WER regression. Only backing-allocate "one-shot" outputs.
    private static let loopbackOutputNames: Set<String> = [
        "h_out", "c_out",
        "cache_channel_out", "cache_time_out", "cache_len_out",
        "mel_cache_out",
    ]

    /// Per-model compute-unit override. Reads the named env var; if set
    /// to CPU / CPU_AND_NE / CPU_AND_GPU / ALL, builds a fresh
    /// MLModelConfiguration with that compute unit. Falls back to the
    /// base config otherwise. Logs once when overridden.
    internal static func computeUnitOverride(
        name: String,
        base: MLModelConfiguration,
        logger: AppLogger
    ) -> MLModelConfiguration {
        guard let raw = ProcessInfo.processInfo.environment[name]?.uppercased() else {
            return base
        }
        let override: MLComputeUnits
        switch raw {
        case "CPU", "CPU_ONLY": override = .cpuOnly
        case "CPU_AND_NE": override = .cpuAndNeuralEngine
        case "CPU_AND_GPU": override = .cpuAndGPU
        case "ALL": override = .all
        default:
            logger.warning("Unknown value for \(name): \(raw); ignoring")
            return base
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = override
        logger.info("\(name)=\(raw) → MLComputeUnits.\(override.rawValue)")
        return cfg
    }

    internal static func makePredictionOptions(for model: MLModel?) -> MLPredictionOptions? {
        guard let model = model else { return nil }
        var backings: [String: Any] = [:]
        for (name, desc) in model.modelDescription.outputDescriptionsByName {
            if loopbackOutputNames.contains(name) { continue }
            guard let cons = desc.multiArrayConstraint else { continue }
            let shape = cons.shape.map { $0 }
            if shape.contains(where: { $0.intValue <= 0 }) { continue }
            guard let arr = try? MLMultiArray(shape: shape, dataType: cons.dataType) else { continue }
            backings[name] = arr
        }
        guard !backings.isEmpty else { return nil }
        let options = MLPredictionOptions()
        options.outputBackings = backings
        return options
    }

    /// Like `locateModelBundle` but returns nil if NEITHER the compiled
    /// nor the uncompiled bundle exists. Use for optional models so the
    /// load site still gets the caching behavior (mlpackage → cached
    /// .mlmodelc next to source) instead of compiling to a temp dir per
    /// cold start.
    private func locateOptionalModelBundle(in directory: URL, compiled: String, uncompiled: String) async throws -> URL?
    {
        let compiledURL = directory.appendingPathComponent(compiled)
        let uncompiledURL = directory.appendingPathComponent(uncompiled)
        if !FileManager.default.fileExists(atPath: compiledURL.path)
            && !FileManager.default.fileExists(atPath: uncompiledURL.path)
        {
            return nil
        }
        return try await locateModelBundle(in: directory, compiled: compiled, uncompiled: uncompiled)
    }

    private func locateModelBundle(in directory: URL, compiled: String, uncompiled: String) async throws -> URL {
        let compiledURL = directory.appendingPathComponent(compiled)
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }
        let uncompiledURL = directory.appendingPathComponent(uncompiled)
        if FileManager.default.fileExists(atPath: uncompiledURL.path) {
            // `MLModel.load(contentsOf:)` requires a compiled `.mlmodelc`. Compile
            // the `.mlpackage` to a sibling `.mlmodelc` (cached for reuse).
            let baseName = (compiled as NSString).deletingPathExtension
            let cachedCompiledURL = directory.appendingPathComponent("\(baseName).mlmodelc")
            if FileManager.default.fileExists(atPath: cachedCompiledURL.path) {
                return cachedCompiledURL
            }
            logger.info("Compiling \(uncompiled) to .mlmodelc (first run only)...")
            let tempCompiledURL = try await MLModel.compileModel(at: uncompiledURL)
            // Try to cache next to the .mlpackage so subsequent loads skip
            // compilation. Falls back to the temp URL if the directory isn't
            // writable.
            do {
                if FileManager.default.fileExists(atPath: cachedCompiledURL.path) {
                    try FileManager.default.removeItem(at: cachedCompiledURL)
                }
                try FileManager.default.moveItem(at: tempCompiledURL, to: cachedCompiledURL)
                return cachedCompiledURL
            } catch {
                logger.warning(
                    "Could not cache compiled model next to .mlpackage (\(error.localizedDescription)); using temp path."
                )
                return tempCompiledURL
            }
        }
        throw ASRError.processingFailed(
            "Could not find \(compiled) or \(uncompiled) in \(directory.path)"
        )
    }

    /// Reset all states for a new transcription session.
    /// Preserves the currently selected prompt id and ml configuration.
    public func reset() async {
        StreamingAsrUtils.resetSharedState(
            audioBuffer: &audioBuffer,
            accumulatedTokenIds: &accumulatedTokenIds,
            processedChunks: &processedChunks
        )
        accumulatedTokenTimings.removeAll()
        absoluteFrameBase = 0
        lastFinishTokenTimings.removeAll()
        audioBufferOffset = 0
        firstDetectedLanguage = nil
        do {
            try resetStates()
        } catch {
            logger.error("Failed to reset states: \(error.localizedDescription)")
        }
        if useForcedPrefix {
            do {
                try await applyForcedPrefixIfNeeded()
            } catch {
                logger.error("Forced prefix seeding failed: \(error.localizedDescription)")
            }
        }
    }

    /// Run the decoder once with the lang-tag token for the currently selected
    /// language and write the resulting state back to `hState`/`cState`/`lastToken`.
    /// No-op if forced-prefix is disabled, no language is set, the tokenizer/
    /// decoder isn't loaded, or the language has no matching lang-tag token.
    private func applyForcedPrefixIfNeeded() async throws {
        guard useForcedPrefix,
            let language = currentLanguageCode,
            let tokenizer = tokenizer,
            let decoder = decoder,
            let h = hState,
            let c = cState
        else { return }

        guard let langTagId = tokenizer.langTagTokenId(forLanguage: language) else {
            logger.info("Forced prefix: no lang-tag token for \(language); skipping seed")
            return
        }

        let tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenInput[0] = NSNumber(value: langTagId)

        let tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
        tokenLen[0] = 1

        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenInput),
            "token_length": MLFeatureValue(multiArray: tokenLen),
            "h_in": MLFeatureValue(multiArray: h),
            "c_in": MLFeatureValue(multiArray: c),
        ])

        let decoderOutput = try await decoder.prediction(from: decoderInput)
        guard let hOut = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
        else {
            logger.warning("Forced prefix: decoder did not return h_out/c_out; skipping")
            return
        }

        self.hState = hOut
        self.cState = cOut
        self.lastToken = Int32(langTagId)
        // Mirror what the pipeline would do when it observes a lang-tag token.
        recordDetectedLanguage(language)
        logger.info("Forced prefix: seeded decoder with lang-tag token id \(langTagId) for \(language)")
    }

    public func cleanup() async {
        await reset()
        preprocessor = nil
        encoder = nil
        decoder = nil
        joint = nil
        tokenizer = nil
        cacheChannel = nil
        cacheTime = nil
        cacheLen = nil
        melCache = nil
        hState = nil
        cState = nil
        logger.info("StreamingNemotronMultilingualAsrManager resources cleaned up")
    }

    internal func resetStates() throws {
        let cacheConfig = EncoderCacheManager.CacheConfig(
            channelShape: config.cacheChannelShape,
            timeShape: config.cacheTimeShape,
            lenShape: [1]
        )
        let caches = try EncoderCacheManager.createInitialCaches(config: cacheConfig)
        cacheChannel = caches.channel
        cacheTime = caches.time
        cacheLen = caches.len
        // Stateful encoder: a fresh state has zero-initialized buffers, which
        // matches what createInitialCaches produces. Recreate it on reset.
        if #available(macOS 15, iOS 18, *) {
            if let enc = encoder, !enc.modelDescription.stateDescriptionsByName.isEmpty {
                encoderState = enc.makeState()
            }
        }
        // Seed cache_len with 1 instead of 0 so the encoder's
        // `ios17.slice_by_index` op never sees a zero-length slice, which would
        // fail CoreML shape inference and skip MPSGraph caching on every
        // session start. The cache buffers are zero, so this is equivalent to
        // 1 frame of silence preamble.
        cacheLen?[0] = 1

        // Mel cache (will be initialized on first chunk)
        melCache = nil
        // Drop any prefetched mel/encoder outputs from a previous session
        prefetchedMel = nil
        prefetchedEncoded = nil
        prefetchedEncoderProj = nil
        prefetchedCacheChannel = nil
        prefetchedCacheTime = nil
        prefetchedCacheLen = nil
        // Reset native LSTM state if present

        // Decoder LSTM states
        hState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        cState = try EncoderCacheManager.createZeroArray(
            shape: [config.decoderLayers, 1, config.decoderHidden]
        )

        lastToken = Int32(config.blankIdx)
    }

    /// Append audio buffer for processing
    public func appendAudio(_ buffer: AVAudioPCMBuffer) throws {
        try StreamingAsrUtils.appendAudio(buffer, using: audioConverter, to: &audioBuffer)
    }

    /// Process audio. Returns the empty string because the partial transcript
    /// is delivered via the partial callback or `getPartialTranscript()`.
    public func process(audioBuffer: sending AVAudioPCMBuffer) async throws -> String {
        let samples = try audioConverter.resampleBuffer(audioBuffer)
        return try await process(samples: samples)
    }

    /// Process pre-resampled 16 kHz Float samples directly. Skips
    /// `AVAudioPCMBuffer` + `AudioConverter.resampleBuffer` overhead — use
    /// this when the caller already has the audio as `[Float]` at 16 kHz
    /// (e.g. predecoded-PCM benchmarks or memory-cached pipelines).
    public func process(samples: [Float]) async throws -> String {
        // decoder/joint are optional (lean B1 ships omit them); require a
        // valid decode path instead — fused B1/B3/B2, or the bare pair.
        let hasDecodePath =
            decoderJoint != nil || decoderJointNoEncProj != nil || decoderJointArgmax != nil
            || (decoder != nil && joint != nil)
        guard preprocessor != nil, encoder != nil, hasDecodePath else {
            throw ASRError.notInitialized
        }

        self.audioBuffer.append(contentsOf: samples)

        let chunkSamples = config.chunkSamples
        // Drain the buffer using offset arithmetic instead of removeFirst —
        // removeFirst would memmove the entire remaining buffer on every
        // chunk, giving O(N²) cost for long files (3.6 GB shifted 51k times
        // for a 16h file).
        while (self.audioBuffer.count - self.audioBufferOffset) >= chunkSamples {
            let chunkStart = self.audioBufferOffset
            let chunkEnd = chunkStart + chunkSamples
            let chunk = Array(self.audioBuffer[chunkStart..<chunkEnd])
            // Pipelining: peek at the NEXT chunk for preprocessor[t+1] on CPU
            // concurrent with encoder[t] on ANE.
            let nextStart = chunkEnd
            let nextEnd = nextStart + chunkSamples
            let nextChunkSamples: [Float]? =
                (self.audioBuffer.count - nextStart) >= chunkSamples
                ? Array(self.audioBuffer[nextStart..<nextEnd])
                : nil
            try await processChunk(chunk, nextChunkSamples: nextChunkSamples)
            self.audioBufferOffset += chunkSamples

            // Periodic compaction: once we've consumed enough prefix, do a
            // single memmove to drop it. Amortized O(1) per chunk.
            if self.audioBufferOffset > 16 * chunkSamples {
                self.audioBuffer.removeFirst(self.audioBufferOffset)
                self.audioBufferOffset = 0
            }
        }

        return ""
    }

    /// Finish processing remaining audio (padded if needed) and return the
    /// final transcript text. The detected language is available via
    /// `detectedLanguage()` after this returns.
    public func finish() async throws -> String {
        let hasDecodePath =
            decoderJoint != nil || decoderJointNoEncProj != nil || decoderJointArgmax != nil
            || (decoder != nil && joint != nil)
        guard let tokenizer = tokenizer,
            preprocessor != nil,
            encoder != nil,
            hasDecodePath
        else {
            throw ASRError.notInitialized
        }

        let remaining = audioBuffer.count - audioBufferOffset
        if remaining > 0 {
            let paddingNeeded = config.chunkSamples - remaining
            if paddingNeeded > 0 {
                audioBuffer.append(contentsOf: Array(repeating: 0.0, count: paddingNeeded))
            }

            let chunkStart = audioBufferOffset
            let chunkEnd = chunkStart + config.chunkSamples
            let chunk = Array(audioBuffer[chunkStart..<chunkEnd])
            try await processChunk(chunk)
            audioBuffer.removeAll()
            audioBufferOffset = 0
        }

        let decoded = tokenizer.decode(ids: accumulatedTokenIds)
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = decoded.detectedLanguage
        }
        // Capture diagnostic stats before we clear state.
        lastFinishTokenCount = accumulatedTokenIds.count
        lastFinishDetectedLanguage = firstDetectedLanguage ?? decoded.detectedLanguage
        lastFinishProcessedChunks = processedChunks
        // Snapshot timings before clearing so finishWithTokenTimings() can return
        // them; clear the working buffers atomically with the ids.
        lastFinishTokenTimings = accumulatedTokenTimings
        accumulatedTokenIds.removeAll()
        accumulatedTokenTimings.removeAll()

        if appendTerminalPunctuation {
            return Self.tidyTerminalPunctuation(
                decoded.text, language: lastFinishDetectedLanguage)
        }
        return decoded.text
    }

    /// Display-only heuristic: capitalize the first character and append a
    /// terminal mark if the text doesn't already end in one. Cannot recover a
    /// missing mid-clause comma; always uses a period (`。` for zh/ja). Gated
    /// behind `appendTerminalPunctuation`.
    static func tidyTerminalPunctuation(_ text: String, language: String?) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.last else { return text }
        var result = trimmed.prefix(1).uppercased() + trimmed.dropFirst()
        let terminals: Set<Character> = [".", "!", "?", "。", "！", "？", "…"]
        if !terminals.contains(last) {
            let isCJK = (language?.hasPrefix("zh") ?? false) || (language?.hasPrefix("ja") ?? false)
            result.append(isCJK ? "。" : ".")
        }
        return result
    }

    /// Diagnostic stats captured at the most recent `finish()` call.
    /// `tokenCount` is the number of raw token ids accumulated (including any
    /// lang-tag tokens). `detectedLanguage` is the first lang-tag piece seen
    /// (e.g. "es-419") or nil if none. `processedChunks` is how many chunks
    /// were fed through the encoder for that session.
    public func lastDecodeStats() -> (tokenCount: Int, detectedLanguage: String?, processedChunks: Int) {
        return (lastFinishTokenCount, lastFinishDetectedLanguage, lastFinishProcessedChunks)
    }

    /// Get current partial transcript without finishing
    public func getPartialTranscript() -> String {
        guard let tokenizer = tokenizer else { return "" }
        let decoded = tokenizer.decode(ids: accumulatedTokenIds)
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = decoded.detectedLanguage
        }
        return decoded.text
    }

    /// Finish processing and return the final transcript together with per-token
    /// timings (absolute seconds from the start of the fed audio). The timings
    /// are aligned 1:1 with the user-visible (lang-tag-stripped) token stream;
    /// group them by the SentencePiece word-boundary marker to obtain word-level
    /// timestamps. Note: when `appendTerminalPunctuation` is enabled the returned
    /// text gains an untimed terminal mark, so callers that need strict
    /// text/timing parity should leave that flag off (the default).
    public func finishWithTokenTimings() async throws -> (text: String, timings: [TokenTiming]) {
        let text = try await finish()
        return (text, lastFinishTokenTimings)
    }

    /// Get per-token timings accumulated so far without finishing. Aligned 1:1
    /// with the tokens behind getPartialTranscript(). Use this when a caller must
    /// salvage a partially-processed session that cannot safely call finish()
    /// (e.g. after a mid-stream decode failure).
    public func getTokenTimings() -> [TokenTiming] {
        return accumulatedTokenTimings
    }

    /// Internal getter for the current prompt id, used by the pipeline.
    internal func currentPromptIdValue() -> Int32 { currentPromptId }

    /// Internal setter used by the pipeline when it encounters a lang-tag
    /// token in the decoder output.
    internal func recordDetectedLanguage(_ language: String) {
        if firstDetectedLanguage == nil {
            firstDetectedLanguage = language
        }
    }
}
