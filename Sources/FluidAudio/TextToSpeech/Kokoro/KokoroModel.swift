import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using unified CoreML model
/// Uses centralized model names from ModelNames.TTS
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModel {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModel")

    // Single model reference
    private static var kokoroModel: MLModel?
    private static var isModelLoaded = false

    // Legacy: Phoneme dictionary with frame counts (kept for backward compatibility)
    private static var phonemeDictionary: [String: (frameCount: Float, phonemes: [String])] = [:]
    private static var isDictionaryLoaded = false

    // Preferred: Simple word -> phonemes mapping from centralized file
    private static var wordToPhonemes: [String: [String]] = [:]
    private static var isSimpleDictLoaded = false

    /// Ensure required dictionary files exist (now handled by DownloadUtils during model loading)
    public static func ensureRequiredFiles() async throws {
        // Dictionary files are now downloaded automatically by DownloadUtils
        // when the kokoro model is loaded. This method is kept for backward compatibility.
    }

    /// Load the Kokoro model
    public static func loadModel() async throws {
        guard !isModelLoaded else { return }

        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let localPackage = cwd.appendingPathComponent("\(ModelNames.TTS.kokoroModel).mlpackage")

        if fm.fileExists(atPath: localPackage.path) {
            logger.info("Loading Kokoro model from local package: \(localPackage.path)")
            // Compile the .mlpackage to a .mlmodelc bundle before loading
            let compiledURL = try await MLModel.compileModel(at: localPackage)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            kokoroModel = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } else {
            // Delegate to TtsModels which uses DownloadUtils for all model downloads
            let models = try await TtsModels.download()
            kokoroModel = models.kokoro
        }
        logger.info("Loaded \(ModelNames.TTS.kokoroModel) model")

        isModelLoaded = true
        logger.info("Kokoro model successfully loaded")
    }

    /// Load simple word->phonemes dictionary (preferred)
    /// Supports two formats:
    /// 1) { "word_to_phonemes": { word: [token, ...] } }
    /// 2) { word: "ipa string" | [token, ...] } (flat map)
    public static func loadSimplePhonemeDictionary() throws {
        guard !isSimpleDictLoaded else { return }

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let dictURL = kokoroDir.appendingPathComponent(ModelNames.TTS.wordPhonemesFile)

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw TTSError.processingFailed("Invalid \(ModelNames.TTS.wordPhonemesFile) (not a JSON object)")
        }

        // Case 1: expected wrapped format
        if let mapping = json["word_to_phonemes"] as? [String: [String]] {
            wordToPhonemes = mapping
            isSimpleDictLoaded = true
            logger.info("Loaded \(wordToPhonemes.count) words from \(ModelNames.TTS.wordPhonemesFile) (wrapped)")
            return
        }

        // Case 2: flat map format (word -> String or [String])
        var tmp: [String: [String]] = [:]
        let vocabulary = KokoroVocabulary.getVocabulary()
        let allowed = Set(vocabulary.keys)

        var converted = 0
        for (k, v) in json {
            if let s = v as? String {
                let toks = tokenizeIPAString(s)
                let filtered = toks.filter { allowed.contains($0) }
                if !filtered.isEmpty {
                    tmp[k.lowercased()] = filtered
                    converted += 1
                }
            } else if let arr = v as? [String] {
                let filtered = arr.filter { allowed.contains($0) }
                if !filtered.isEmpty {
                    tmp[k.lowercased()] = filtered
                    converted += 1
                }
            }
        }

        guard !tmp.isEmpty else {
            throw TTSError.processingFailed("Unsupported \(ModelNames.TTS.wordPhonemesFile) format (no usable entries)")
        }

        wordToPhonemes = tmp
        isSimpleDictLoaded = true
        logger.info("Loaded \(wordToPhonemes.count) words from \(ModelNames.TTS.wordPhonemesFile) (flat)")
    }

    /// Tokenize an IPA string into model tokens.
    /// If the string contains whitespace, split on whitespace; otherwise split into Unicode scalars.
    private static func tokenizeIPAString(_ s: String) -> [String] {
        if s.contains(where: { $0.isWhitespace }) {
            return
                s
                .components(separatedBy: .whitespacesAndNewlines)
                .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: ",.;:()[]{}\"'")) }
                .filter { !$0.isEmpty }
        }
        // Split into Unicode scalars to preserve single-codepoint IPA tokens (e.g., ʧ, ʤ, ˈ)
        return s.unicodeScalars.map { String($0) }
    }

    /// Structure to hold a chunk of text that fits within 3.17 seconds
    // TextChunk is defined in KokoroChunker.swift

    /// Chunk text into segments under the model token budget, using punctuation-driven pauses.
    private static func chunkText(_ text: String) throws -> [TextChunk] {
        try loadSimplePhonemeDictionary()
        let target = targetTokenLength()
        let hasLang = false
        return KokoroChunker.chunk(
            text: text,
            wordToPhonemes: wordToPhonemes,
            targetTokens: target,
            hasLanguageToken: hasLang
        )
    }

    /// Convert phonemes to input IDs
    public static func phonemesToInputIds(_ phonemes: [String]) -> [Int32] {
        let vocabulary = KokoroVocabulary.getVocabulary()
        var ids: [Int32] = [0]  // BOS/EOS token per Python harness
        for phoneme in phonemes {
            if let id = vocabulary[phoneme] {
                ids.append(id)
            } else {
                logger.warning("Missing phoneme in vocab: '\(phoneme)'")
            }
        }
        ids.append(0)

        // Debug: validate id range
        #if DEBUG
        if !vocabulary.isEmpty {
            let maxId = vocabulary.values.max() ?? 0
            let minId = vocabulary.values.min() ?? 0
            let outOfRange = ids.filter { $0 != 0 && ($0 < minId || $0 > maxId) }
            if !outOfRange.isEmpty {
                print(
                    "Warning: Found \(outOfRange.count) token IDs out of range [\(minId), \(maxId)] (excluding BOS/EOS=0)"
                )
            }
            print("Tokenized \(ids.count) ids; first 32: \(ids.prefix(32))")
        }
        #endif

        return ids
    }

    /// Inspect model to determine the expected token length for input_ids
    private static func targetTokenLength() -> Int {
        if let model = kokoroModel {
            let inputs = model.modelDescription.inputDescriptionsByName
            if let inputDesc = inputs["input_ids"], let constraint = inputDesc.multiArrayConstraint {
                let shape = constraint.shape
                if shape.count >= 2 {
                    let n = shape.last!.intValue
                    if n > 0 { return n }
                }
            }
        }
        // Fallback to a common unified length if not discoverable
        return 124
    }

    /// Load voice embedding (simplified for 3-second model)
    public static func loadVoiceEmbedding(voice: String = "af_heart", phonemeCount: Int) throws -> MLMultiArray {
        let voice = "af_heart"
        // Try to load from Application Support directory
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        // Try multiple candidates (prefer local repo files first)
        let candidates: [URL] = [
            cwd.appendingPathComponent("voices/\(voice).json"),
            cwd.appendingPathComponent("\(voice).json"),
            voicesDir.appendingPathComponent("\(voice).json"),
        ]
        let voiceJSON = candidates.first { FileManager.default.fileExists(atPath: $0.path) } ?? candidates[0]

        var vector: [Float]?
        if FileManager.default.fileExists(atPath: voiceJSON.path) {
            do {
                let data = try Data(contentsOf: voiceJSON)
                let json = try JSONSerialization.jsonObject(with: data)
                func parseArray(_ any: Any) -> [Float]? {
                    if let ds = any as? [Double] { return ds.map { Float($0) } }
                    if let fs = any as? [Float] { return fs }
                    if let ns = any as? [NSNumber] { return ns.map { $0.floatValue } }
                    if let arr = any as? [Any] {
                        var out: [Float] = []
                        out.reserveCapacity(arr.count)
                        for v in arr {
                            if let n = v as? NSNumber {
                                out.append(n.floatValue)
                            } else if let d = v as? Double {
                                out.append(Float(d))
                            } else if let f = v as? Float {
                                out.append(f)
                            } else {
                                return nil
                            }
                        }
                        return out
                    }
                    return nil
                }

                if let arr = parseArray(json) {
                    vector = arr
                } else if let dict = json as? [String: Any] {
                    if let embed = dict["embedding"], let arr = parseArray(embed) {
                        vector = arr
                    } else if let byVoice = dict[voice], let arr = parseArray(byVoice) {
                        vector = arr
                    } else {
                        let keys = dict.keys.compactMap { Int($0) }.sorted()
                        var chosen: [Float]? = nil
                        if let exact = dict["\(phonemeCount)"] {
                            chosen = parseArray(exact)
                        } else if let k = keys.last(where: { $0 <= phonemeCount }), let cand = dict["\(k)"] {
                            chosen = parseArray(cand)
                        }
                        if let c = chosen {
                            vector = c
                        } else if let any = dict.values.first {
                            vector = parseArray(any)
                        }
                    }
                }
            } catch {
                // Ignore parse errors; will fall back
            }
        }

        // Require a valid voice embedding; fail if missing or invalid
        let dim = refDimFromModel()
        guard let vec = vector, vec.count == dim else {
            throw TTSError.modelNotFound("Voice embedding for \(voice) not found or invalid at \(voiceJSON.path)")
        }
        let embedding = try MLMultiArray(shape: [1, NSNumber(value: dim)] as [NSNumber], dataType: .float32)
        var varsum: Float = 0
        for i in 0..<dim {
            let v = vec[i]
            embedding[i] = NSNumber(value: v)
            varsum += v * v
        }
        logger.info(
            "Loaded voice embedding: \(voice), dim=\(dim), l2norm=\(String(format: "%.3f", sqrt(Double(varsum))))")
        return embedding
    }

    /// Helper to fetch ref_s expected dimension from model
    private static func refDimFromModel() -> Int {
        guard let model = kokoroModel else { return 256 }
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
            let shape = desc.multiArrayConstraint?.shape,
            shape.count >= 2
        {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }

    /// Synthesize a single chunk of text
    private static func synthesizeChunk(_ chunk: TextChunk, voice: String) async throws -> [Float] {
        // Convert phonemes to input IDs
        let phonemeSeq = chunk.phonemes
        // No language token prefix; avoid audible leading vowel
        let inputIds = phonemesToInputIds(phonemeSeq)

        guard !inputIds.isEmpty else {
            throw TTSError.processingFailed("No input IDs generated for chunk: \(chunk.words.joined(separator: " "))")
        }

        // Get voice embedding
        let refStyle = try loadVoiceEmbedding(voice: voice, phonemeCount: inputIds.count)

        // Determine target token length from model and pad/truncate accordingly
        let targetTokens = targetTokenLength()
        var trimmedIds = inputIds
        if trimmedIds.count > targetTokens {
            trimmedIds = Array(trimmedIds.prefix(targetTokens))
        } else if trimmedIds.count < targetTokens {
            trimmedIds.append(contentsOf: Array(repeating: Int32(0), count: targetTokens - trimmedIds.count))
        }

        // Create model inputs
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)] as [NSNumber], dataType: .int32)
        for (i, id) in trimmedIds.enumerated() {
            inputArray[i] = NSNumber(value: id)
        }

        // Create attention mask (1 for real tokens up to original count, 0 for padding)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: targetTokens)] as [NSNumber], dataType: .int32)
        let trueLen = min(inputIds.count, targetTokens)
        for i in 0..<targetTokens {
            attentionMask[i] = NSNumber(value: i < trueLen ? 1 : 0)
        }

        // Use zeros for phases for determinism (works well for 3s model)
        let phasesArray = try MLMultiArray(shape: [1, 9] as [NSNumber], dataType: .float32)
        for i in 0..<9 { phasesArray[i] = 0 }

        // Debug: print model IO

        // Run inference
        let modelInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputArray,
            "attention_mask": attentionMask,
            "ref_s": refStyle,
            "random_phases": phasesArray,
        ])

        guard let output = try kokoroModel?.prediction(from: modelInput) else {
            throw TTSError.processingFailed("Model prediction failed")
        }

        // Extract audio output explicitly by key used by model
        guard let audioArrayUnwrapped = output.featureValue(for: "audio")?.multiArrayValue,
            audioArrayUnwrapped.count > 0
        else {
            let names = Array(output.featureNames)
            throw TTSError.processingFailed("Failed to extract 'audio' output. Features: \(names)")
        }

        // Optional: trim to audio_length_samples if provided
        var effectiveCount = audioArrayUnwrapped.count
        if let lenFV = output.featureValue(for: "audio_length_samples") {
            var n: Int = 0
            if let lenArray = lenFV.multiArrayValue, lenArray.count > 0 {
                n = lenArray[0].intValue
            } else if lenFV.type == .int64 {
                n = Int(lenFV.int64Value)
            } else if lenFV.type == .double {
                n = Int(lenFV.doubleValue)
            }
            n = max(0, n)
            if n > 0 && n <= audioArrayUnwrapped.count {
                effectiveCount = n
            }
        }

        // Convert to float samples
        var samples: [Float] = []
        for i in 0..<effectiveCount {
            samples.append(audioArrayUnwrapped[i].floatValue)
        }

        // Basic sanity logging
        let minVal = samples.min() ?? 0
        let maxVal = samples.max() ?? 0
        if maxVal - minVal == 0 {
            logger.warning("Prediction produced constant signal (min=max=\(minVal)).")
        } else {
            logger.info("Audio range: [\(String(format: "%.4f", minVal)), \(String(format: "%.4f", maxVal))]")
        }

        return samples
    }

    /// Main synthesis function with chunking support
    public static func synthesize(text: String, voice: String = "af_heart") async throws -> Data {
        let synthesisStart = Date()

        logger.info("Starting synthesis: '\(text)'")
        logger.info("Input length: \(text.count) characters")

        // Ensure required files are downloaded
        try await ensureRequiredFiles()
        // Ensure voice embedding if available
        try? await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: voice)

        // Load model if needed
        if !isModelLoaded {
            try await loadModel()
        }

        // Chunk the text based on frame counts
        let chunks = try chunkText(text)

        if chunks.isEmpty {
            throw TTSError.processingFailed("No valid words found in text")
        }

        if chunks.count == 1 {
            // Single chunk - process normally
            logger.info("Text fits in single chunk")
            let chunk = chunks[0]
            logger.info("Processing chunk: \(chunk.words.count) words, \(chunk.totalFrames) frames")

            let samples = try await synthesizeChunk(chunk, voice: voice)

            // Normalize and convert to WAV
            let maxVal = samples.map { abs($0) }.max() ?? 1.0
            let normalizedSamples = maxVal > 0 ? samples.map { $0 / maxVal } : samples
            let audioData = try AudioWAV.data(from: normalizedSamples, sampleRate: 24000)

            let totalTime = Date().timeIntervalSince(synthesisStart)
            logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
            logger.info("Audio size: \(audioData.count) bytes")

            return audioData
        } else {
            // Multiple chunks - process and concatenate with boundary handling
            logger.info("Text split into \(chunks.count) chunks")
            var allSamples: [Float] = []

            // let sampleRate = 24000  // implicit by model; reserved for future resampling logic
            let crossfadeMs = 8
            let crossfadeN = max(0, Int(Double(crossfadeMs) * 24.0))

            for i in 0..<chunks.count {
                let chunk = chunks[i]
                logger.info(
                    "Processing chunk \(i+1)/\(chunks.count): \(chunk.words.count) words, \(chunk.totalFrames) frames")
                let chunkSamples = try await synthesizeChunk(chunk, voice: voice)
                if i == 0 {
                    allSamples.append(contentsOf: chunkSamples)
                } else {
                    let prevPause = chunks[i - 1].pauseAfterMs
                    if prevPause > 0 {
                        let silenceCount = Int(Double(prevPause) * 24.0)
                        if silenceCount > 0 {
                            allSamples.append(contentsOf: Array(repeating: 0.0, count: silenceCount))
                        }
                        allSamples.append(contentsOf: chunkSamples)
                    } else {
                        // Micro crossfade join (only when no punctuation-driven pause)
                        let n = min(crossfadeN, allSamples.count, chunkSamples.count)
                        if n > 0 {
                            // Fade out last n of allSamples, fade in first n of chunkSamples
                            for k in 0..<n {
                                let aIdx = allSamples.count - n + k
                                let bIdx = k
                                let t = Float(k) / Float(n)
                                let fadeOut = 1.0 - t
                                let fadeIn = t
                                allSamples[aIdx] = allSamples[aIdx] * fadeOut + chunkSamples[bIdx] * fadeIn
                            }
                            // Append remainder of chunk after crossfade overlap
                            if chunkSamples.count > n {
                                allSamples.append(contentsOf: chunkSamples[n...])
                            }
                        } else {
                            allSamples.append(contentsOf: chunkSamples)
                        }
                    }
                }
            }

            // Normalize all samples together
            let maxVal = allSamples.map { abs($0) }.max() ?? 1.0
            let normalizedSamples = maxVal > 0 ? allSamples.map { $0 / maxVal } : allSamples

            // Convert to WAV (24 kHz mono)
            let audioData = try AudioWAV.data(from: normalizedSamples, sampleRate: 24000)

            let totalTime = Date().timeIntervalSince(synthesisStart)
            logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
            logger.info("Total audio size: \(audioData.count) bytes")
            logger.info("Total samples: \(allSamples.count)")

            return audioData
        }
    }

    // convertSamplesToWAV moved to AudioWAV

    // convertToWAV removed (unused); use convertSamplesToWAV instead
}
