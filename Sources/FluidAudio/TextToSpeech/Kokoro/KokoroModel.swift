import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using unified CoreML model
/// Uses kokoro_completev20.mlmodelc with word_frames_phonemes.json dictionary
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModel {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroModel")

    // Single model reference
    private static var kokoroModel: MLModel?
    private static var isModelLoaded = false

    // Legacy: Phoneme dictionary with frame counts (kept for backward compatibility)
    private static var phonemeDictionary: [String: (frameCount: Float, phonemes: [String])] = [:]
    private static var isDictionaryLoaded = false

    // Preferred: Simple word -> phonemes mapping from word_phonemes.json
    private static var wordToPhonemes: [String: [String]] = [:]
    private static var isSimpleDictLoaded = false

    // Model and data URLs
    private static let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"

    /// Download file from URL if needed
    private static func downloadFileIfNeeded(filename: String, urlPath: String) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create directory if needed
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)

        guard !FileManager.default.fileExists(atPath: localURL.path) else {
            logger.info("File already exists: \(filename)")
            return
        }

        logger.info("Downloading \(filename)...")
        print("  Downloading \(filename)...")
        let downloadURL = URL(string: "\(baseURL)/\(urlPath)")!

        let (data, response) = try await URLSession.shared.data(from: downloadURL)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TTSError.modelNotFound("Failed to download \(filename)")
        }

        try data.write(to: localURL)
        logger.info("Downloaded \(filename) (\(data.count) bytes)")
    }

    /// Download model files if needed
    private static func downloadModelIfNeeded() async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create Models/kokoro directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        logger.info("Model directory: \(modelDir.path)")
        print("Model directory: \(modelDir.path)")

        let modelPath = modelDir.appendingPathComponent("kokoro_completev20.mlmodelc")

        if FileManager.default.fileExists(atPath: modelPath.path) {
            logger.info("Model already downloaded")
            print("Model already downloaded")
            return
        }

        logger.info("Downloading kokoro_completev20 model...")
        print("Downloading kokoro_completev20 model...")

        // Create model directory
        try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

        // Download the compiled mlmodelc files (some optional depending on packaging)
        let filesToDownload = [
            "coremldata.bin",
            "metadata.json",              // optional
            "model.mil",
            "weights/weight.bin",        // optional
            "analytics/coremldata.bin",  // optional
        ]

        // Create subdirectories
        try FileManager.default.createDirectory(
            at: modelPath.appendingPathComponent("weights"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: modelPath.appendingPathComponent("analytics"),
            withIntermediateDirectories: true
        )

        for file in filesToDownload {
            let fileURL = URL(string: "\(baseURL)/kokoro_completev20.mlmodelc/\(file)")!
            let destPath: URL

            if file == "weights/weight.bin" {
                destPath = modelPath.appendingPathComponent("weights").appendingPathComponent("weight.bin")
            } else if file == "analytics/coremldata.bin" {
                destPath = modelPath.appendingPathComponent("analytics").appendingPathComponent("coremldata.bin")
            } else {
                destPath = modelPath.appendingPathComponent(file)
            }

            do {
                logger.info("  Downloading \(file)...")
                let (data, response) = try await URLSession.shared.data(from: fileURL)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 200
                {
                    try data.write(to: destPath)
                    logger.info("  ✓ Downloaded \(file) (\(data.count) bytes)")
                } else {
                    logger.warning("  File \(file) returned status \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                }
            } catch {
                logger.warning("  Could not download \(file): \(error.localizedDescription)")
            }
        }

        logger.info("✓ Downloaded kokoro_completev20.mlmodelc (required files)")
    }

    /// Ensure all required files are downloaded
    public static func ensureRequiredFiles() async throws {
        // Download dictionary files
        print("Checking word_phonemes.json...")
        try await downloadFileIfNeeded(filename: "word_phonemes.json", urlPath: "word_phonemes.json")
        print("Checking word_frames_phonemes.json...")
        try await downloadFileIfNeeded(filename: "word_frames_phonemes.json", urlPath: "word_frames_phonemes.json")

        // Download model
        try await downloadModelIfNeeded()
    }

    /// Load the Kokoro model
    public static func loadModel() async throws {
        guard !isModelLoaded else { return }

        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let localPackage = cwd.appendingPathComponent("kokoro_completev20.mlpackage")

        if fm.fileExists(atPath: localPackage.path) {
            logger.info("Loading Kokoro model from local package: \(localPackage.path)")
            print("Loading Kokoro model from local package: \(localPackage.path)")
            // Compile the .mlpackage to a .mlmodelc bundle before loading
            let compiledURL = try await MLModel.compileModel(at: localPackage)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            kokoroModel = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } else {
            let cacheDir = try TtsModels.cacheDirectoryURL()
            let modelDir = cacheDir.appendingPathComponent("Models/kokoro")
            let modelPath = modelDir.appendingPathComponent("kokoro_completev20.mlmodelc")

            logger.info("Loading Kokoro model from \(modelDir.path)")
            print("Loading Kokoro model from \(modelDir.path)")

            if !fm.fileExists(atPath: modelPath.path) {
                logger.warning("Model not found in cache, downloading...")
                print("Model not found in cache, downloading...")
                try await downloadModelIfNeeded()
            }
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            kokoroModel = try MLModel(contentsOf: modelPath, configuration: configuration)
        }
        logger.info("Loaded kokoro_completev20 model")

        isModelLoaded = true
        logger.info("Kokoro model successfully loaded")
    }

    /// Load simple word->phonemes dictionary (preferred)
    public static func loadSimplePhonemeDictionary() throws {
        guard !isSimpleDictLoaded else { return }

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let dictURL = kokoroDir.appendingPathComponent("word_phonemes.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
           let mapping = json["word_to_phonemes"] as? [String: [String]] {
            wordToPhonemes = mapping
            isSimpleDictLoaded = true
            logger.info("Loaded \(wordToPhonemes.count) words from word_phonemes.json")
        } else {
            throw TTSError.processingFailed("Invalid word_phonemes.json format")
        }
    }

    /// Structure to hold word with its phonemes and frame count
    private struct WordInfo {
        let word: String
        let phonemes: [String]
        let frameCount: Float
    }

    /// Structure to hold a chunk of text that fits within 3.17 seconds
    private struct TextChunk {
        let words: [String]
        let phonemes: [String]
        let totalFrames: Float
    }

    /// Convert text to phonemes with frame timing
    public static func textToPhonemes(_ text: String) throws -> (phonemes: [String], totalFrames: Float) {
        try loadSimplePhonemeDictionary()

        let words = text.lowercased().split(separator: " ")
        var allPhonemes: [String] = []

        for word in words {
            let cleanWord = String(word).filter { $0.isLetter || $0.isNumber }

            if let phonemes = wordToPhonemes[cleanWord] {
                allPhonemes.append(contentsOf: phonemes)
                allPhonemes.append(" ")  // Space between words
            } else {
                logger.warning("Word '\(cleanWord)' not in dictionary")
            }
        }

        if allPhonemes.last == " " { allPhonemes.removeLast() }

        return (allPhonemes, 0)
    }

    /// Chunk text into segments that fit within 3.17 seconds
    private static func chunkText(_ text: String) throws -> [TextChunk] {
        // Prefer simple dictionary and produce a single chunk (matches main4.py)
        do {
            try loadSimplePhonemeDictionary()
            let words = text.lowercased().split(separator: " ").map { String($0) }
            let (phonemes, _) = try textToPhonemes(text)
            return [TextChunk(words: words, phonemes: phonemes, totalFrames: 0)]
        } catch {
            // Fallback to legacy frame-based chunking if needed
            try loadFramePhonemeDictionary()

            let maxFrames: Float = 76080.0  // 3.17 seconds at 24kHz
            let words = text.lowercased().split(separator: " ").map { String($0) }
            var chunks: [TextChunk] = []

            var currentChunkWords: [String] = []
            var currentChunkPhonemes: [String] = []
            var currentChunkFrames: Float = 0.0

            for word in words {
                let cleanWord = word.filter { $0.isLetter || $0.isNumber }

                if let (frameCount, phonemes) = phonemeDictionary[cleanWord] {
                    if currentChunkFrames + frameCount > maxFrames && !currentChunkWords.isEmpty {
                        if currentChunkPhonemes.last == " " { currentChunkPhonemes.removeLast() }
                        chunks.append(TextChunk(words: currentChunkWords, phonemes: currentChunkPhonemes, totalFrames: currentChunkFrames))
                        currentChunkWords = [word]
                        currentChunkPhonemes = phonemes + [" "]
                        currentChunkFrames = frameCount
                    } else {
                        currentChunkWords.append(word)
                        currentChunkPhonemes.append(contentsOf: phonemes)
                        currentChunkPhonemes.append(" ")
                        currentChunkFrames += frameCount
                    }
                }
            }

            if !currentChunkWords.isEmpty {
                if currentChunkPhonemes.last == " " { currentChunkPhonemes.removeLast() }
                chunks.append(TextChunk(words: currentChunkWords, phonemes: currentChunkPhonemes, totalFrames: currentChunkFrames))
            }

            return chunks
        }
    }

    /// Legacy frame-aware dictionary loader (word -> (frames, phonemes))
    private static func loadFramePhonemeDictionary() throws {
        guard !isDictionaryLoaded else { return }

        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let dictURL = kokoroDir.appendingPathComponent("word_frames_phonemes.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: [Any]]

        // Parse the dictionary format: {"word": [frame_count, [phonemes]]}
        for (word, value) in json {
            let array = value
            if array.count == 2,
                let frameCount = array[0] as? Double,
                let phonemes = array[1] as? [String]
            {
                phonemeDictionary[word] = (Float(frameCount), phonemes)
            }
        }

        isDictionaryLoaded = true
        logger.info("Loaded \(phonemeDictionary.count) words from frame-aware dictionary")
    }

    /// Convert phonemes to input IDs
    public static func phonemesToInputIds(_ phonemes: [String]) -> [Int32] {
        let vocabulary = KokoroVocabulary.getVocabulary()
        var ids: [Int32] = [0] // BOS/EOS token per Python harness
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
            let outOfRange = ids.filter { $0 < minId || $0 > maxId }
            if !outOfRange.isEmpty {
                print("Warning: Found \(outOfRange.count) token IDs out of range [\(minId), \(maxId)]")
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
        return 139
    }

    /// Load voice embedding (simplified for 3-second model)
    public static func loadVoiceEmbedding(voice: String = "af_heart", phonemeCount: Int) throws -> MLMultiArray {
        // Try to load from cache: ~/.cache/fluidaudio/Models/kokoro/voices/<voice>.json
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
                            if let n = v as? NSNumber { out.append(n.floatValue) }
                            else if let d = v as? Double { out.append(Float(d)) }
                            else if let f = v as? Float { out.append(f) }
                            else { return nil }
                        }
                        return out
                    }
                    return nil
                }

                if let arr = parseArray(json) {
                    vector = arr
                } else if let dict = json as? [String: Any] {
                    if let embed = dict["embedding"], let arr = parseArray(embed) { vector = arr }
                    else if let byVoice = dict[voice], let arr = parseArray(byVoice) { vector = arr }
                    else {
                        let keys = dict.keys.compactMap { Int($0) }.sorted()
                        var chosen: [Float]? = nil
                        if let exact = dict["\(phonemeCount)"] { chosen = parseArray(exact) }
                        else if let k = keys.last(where: { $0 <= phonemeCount }), let cand = dict["\(k)"] { chosen = parseArray(cand) }
                        if let c = chosen { vector = c }
                        else if let any = dict.values.first { vector = parseArray(any) }
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
        print("ref_s dim=\(dim), loaded=true, l2norm=\(String(format: "%.3f", sqrt(Double(varsum))))")
        return embedding
    }

    /// Helper to fetch ref_s expected dimension from model
    private static func refDimFromModel() -> Int {
        guard let model = kokoroModel else { return 256 }
        if let desc = model.modelDescription.inputDescriptionsByName["ref_s"],
           let shape = desc.multiArrayConstraint?.shape,
           shape.count >= 2 {
            let n = shape.last!.intValue
            if n > 0 { return n }
        }
        return 256
    }

    

    /// Synthesize a single chunk of text
    private static func synthesizeChunk(_ chunk: TextChunk, voice: String) async throws -> [Float] {
        // Convert phonemes to input IDs
        var phonemeSeq = chunk.phonemes
        // Prepend language token if present in vocab
        let vocab = KokoroVocabulary.getVocabulary()
        if vocab["a"] != nil { phonemeSeq.insert("a", at: 0) }
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

        print("targetTokens=\(targetTokens), trueLen=\(trueLen), maskOn=\((0..<targetTokens).reduce(0){ $0 + (attentionMask[$1].intValue) })")

        // Use zeros for phases for determinism (works well for 3s model)
        let phasesArray = try MLMultiArray(shape: [1, 9] as [NSNumber], dataType: .float32)
        for i in 0..<9 { phasesArray[i] = 0 }

        // Debug: print model IO
        if let model = kokoroModel {
            let inputs = model.modelDescription.inputDescriptionsByName
            let outputs = model.modelDescription.outputDescriptionsByName
            // Print to console for quick debugging
            print("Model inputs: \(Array(inputs.keys))")
            print("Model outputs: \(Array(outputs.keys))")
            if let idsDesc = inputs["input_ids"], let c = idsDesc.multiArrayConstraint {
                print("input_ids shape: \(c.shape.map{ $0.intValue })")
            }
            if let attnDesc = inputs["attention_mask"], let c = attnDesc.multiArrayConstraint {
                print("attention_mask shape: \(c.shape.map{ $0.intValue })")
            }
            if let refDesc = inputs["ref_s"], let c = refDesc.multiArrayConstraint {
                print("ref_s shape: \(c.shape.map{ $0.intValue })")
            }
        }

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
              audioArrayUnwrapped.count > 0 else {
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
            let audioData = try convertSamplesToWAV(normalizedSamples)

            let totalTime = Date().timeIntervalSince(synthesisStart)
            logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
            logger.info("Audio size: \(audioData.count) bytes")

            return audioData
        } else {
            // Multiple chunks - process and concatenate
            logger.info("Text split into \(chunks.count) chunks")
            var allSamples: [Float] = []

            for (i, chunk) in chunks.enumerated() {
                logger.info(
                    "Processing chunk \(i+1)/\(chunks.count): \(chunk.words.count) words, \(chunk.totalFrames) frames")
                let chunkSamples = try await synthesizeChunk(chunk, voice: voice)
                allSamples.append(contentsOf: chunkSamples)

                // Add small silence between chunks (0.1 seconds = 2400 samples at 24kHz)
                if i < chunks.count - 1 {
                    let silenceSamples = Array(repeating: Float(0.0), count: 2400)
                    allSamples.append(contentsOf: silenceSamples)
                }
            }

            // Normalize all samples together
            let maxVal = allSamples.map { abs($0) }.max() ?? 1.0
            let normalizedSamples = maxVal > 0 ? allSamples.map { $0 / maxVal } : allSamples

            // Convert to WAV
            let audioData = try convertSamplesToWAV(normalizedSamples)

            let totalTime = Date().timeIntervalSince(synthesisStart)
            logger.info("Synthesis complete in \(String(format: "%.3f", totalTime))s")
            logger.info("Total audio size: \(audioData.count) bytes")
            logger.info("Total samples: \(allSamples.count)")

            return audioData
        }
    }

    /// Convert float samples to WAV data
    private static func convertSamplesToWAV(_ samples: [Float]) throws -> Data {
        let sampleRate: Double = 24000  // 24 kHz sample rate

        // Convert to 16-bit PCM
        var pcmData = Data()
        for sample in samples {
            let clippedSample = max(-1.0, min(1.0, sample))
            let pcmSample = Int16(clippedSample * 32767)
            pcmData.append(contentsOf: withUnsafeBytes(of: pcmSample) { Array($0) })
        }

        // Create WAV header
        var wavData = Data()

        // RIFF header
        wavData.append(contentsOf: "RIFF".data(using: .ascii)!)
        let fileSize = UInt32(36 + pcmData.count)
        wavData.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        wavData.append(contentsOf: "WAVE".data(using: .ascii)!)

        // fmt chunk
        wavData.append(contentsOf: "fmt ".data(using: .ascii)!)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // Mono
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })  // Block align
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })  // Bits per sample

        // data chunk
        wavData.append(contentsOf: "data".data(using: .ascii)!)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(pcmData.count).littleEndian) { Array($0) })
        wavData.append(pcmData)

        return wavData
    }

    /// Convert MLMultiArray to WAV data
    private static func convertToWAV(_ audioArray: MLMultiArray) throws -> Data {
        let sampleRate: Double = 24000  // 24 kHz sample rate
        let count = audioArray.count

        // Extract float samples
        var samples: [Float] = []
        for i in 0..<count {
            samples.append(audioArray[i].floatValue)
        }

        // Normalize
        let maxVal = samples.map { abs($0) }.max() ?? 1.0
        if maxVal > 0 {
            samples = samples.map { $0 / maxVal }
        }

        // Convert to 16-bit PCM
        var pcmData = Data()
        for sample in samples {
            let pcmSample = Int16(max(-32768, min(32767, sample * 32767)))
            pcmData.append(contentsOf: withUnsafeBytes(of: pcmSample) { Array($0) })
        }

        // Create WAV header
        var wavData = Data()

        // RIFF header
        wavData.append(contentsOf: "RIFF".data(using: .ascii)!)
        let fileSize = UInt32(36 + pcmData.count)
        wavData.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        wavData.append(contentsOf: "WAVE".data(using: .ascii)!)

        // fmt chunk
        wavData.append(contentsOf: "fmt ".data(using: .ascii)!)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // Mono
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate * 2).littleEndian) { Array($0) })
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(2).littleEndian) { Array($0) })  // Block align
        wavData.append(contentsOf: withUnsafeBytes(of: UInt16(16).littleEndian) { Array($0) })  // Bits per sample

        // data chunk
        wavData.append(contentsOf: "data".data(using: .ascii)!)
        wavData.append(contentsOf: withUnsafeBytes(of: UInt32(pcmData.count).littleEndian) { Array($0) })
        wavData.append(pcmData)

        return wavData
    }
}
