import CoreML
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using multiple CoreML models
/// Uses pregenerator, decoder blocks, and generator architecture
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroTTS {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroTTS")

    // Model references
    private static var frontend: MLModel?  // BERT encoder
    private static var pregenerator: MLModel?
    private static var decoderBlocks: [MLModel] = []
    private static var generator: MLModel?
    private static var isModelsLoaded = false

    // Phoneme dictionary and vocabulary
    private static var phonemeDictionary: [String: [String]] = [:]
    private static var isDictionaryLoaded = false

    // Model and data URLs
    private static let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"
    private static let requiredFiles = [
        "word_phonemes.json",
        "vocab_index.json",
    ]
    private static let modelNames = [
        "kokoro_frontend",  // BERT encoder
        "pregenerator",
        "decoder_block_0",
        "decoder_block_1",
        "decoder_block_2",
        "decoder_block_3",
        "generator",
    ]

    /// Download file from URL if needed
    private static func downloadFileIfNeeded(filename: String, urlPath: String) async throws {
        let cacheDir = try TTSModels.cacheDirectoryURL()
        let localURL = cacheDir.appendingPathComponent(filename)

        guard !FileManager.default.fileExists(atPath: localURL.path) else {
            logger.info("File already exists: \(filename)")
            return
        }

        logger.info("Downloading \(filename)...")
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
    private static func downloadModelsIfNeeded() async throws {
        // Use the proper cache directory from TTSModels
        let modelDir = try TTSModels.cacheDirectoryURL()

        // Download each model's files
        for modelName in modelNames {
            let modelPath = modelDir.appendingPathComponent("\(modelName).mlmodelc")

            if !FileManager.default.fileExists(atPath: modelPath.path) {
                logger.info("Downloading \(modelName) model files...")

                // Create model directory
                try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

                // Download model.mil
                let milURL = URL(string: "\(baseURL)/\(modelName).mlmodelc/model.mil")!
                logger.info("Downloading \(modelName) model.mil from \(milURL)")
                let (milData, _) = try await URLSession.shared.data(from: milURL)
                try milData.write(to: modelPath.appendingPathComponent("model.mil"))
                logger.info("Saved model.mil (\(milData.count) bytes)")

                // Download weights
                let weightsDir = modelPath.appendingPathComponent("weights")
                try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)

                let weightURL = URL(string: "\(baseURL)/\(modelName).mlmodelc/weights/weight.bin")!
                logger.info("Downloading \(modelName) weight.bin from \(weightURL)")
                let (weightData, _) = try await URLSession.shared.data(from: weightURL)
                try weightData.write(to: weightsDir.appendingPathComponent("weight.bin"))
                logger.info("Saved weight.bin (\(weightData.count) bytes)")

                // Special handling for kokoro_frontend which has different files
                if modelName == "kokoro_frontend" {
                    // Create a proper Manifest.json for frontend without escaped slashes
                    let manifestString = """
                        {
                          "fileFormatVersion": "1.0.0",
                          "itemInfoEntries": {
                            "model.mil": {
                              "author": "com.apple.CoreML",
                              "description": "CoreML Model Specification"
                            },
                            "weights/weight.bin": {
                              "author": "com.apple.CoreML",
                              "description": "CoreML Model Weights"
                            }
                          },
                          "rootModelIdentifier": "model.mil"
                        }
                        """

                    let manifestData = manifestString.data(using: .utf8)!
                    try manifestData.write(to: modelPath.appendingPathComponent("Manifest.json"))
                    logger.info("Created proper Manifest.json for kokoro_frontend")
                } else {
                    // For other models, download Manifest.json directly
                    let manifestURL = URL(string: "\(baseURL)/\(modelName).mlmodelc/Manifest.json")!
                    logger.info("Downloading \(modelName) Manifest.json")
                    let (manifestData, _) = try await URLSession.shared.data(from: manifestURL)
                    try manifestData.write(to: modelPath.appendingPathComponent("Manifest.json"))
                    logger.info("Saved Manifest.json (\(manifestData.count) bytes)")
                }

                logger.info("Downloaded \(modelName) model to \(modelPath.path)")

                // Verify the model can be loaded
                do {
                    _ = try MLModel(contentsOf: modelPath)
                    logger.info("Verified \(modelName) model loads correctly")
                } catch {
                    logger.error("Model verification failed for \(modelName): \(error)")
                    // Try to compile it explicitly
                    logger.info("Attempting to compile \(modelName) model...")
                    let compiledURL = try await MLModel.compileModel(at: modelPath)
                    logger.info("Compiled \(modelName) to \(compiledURL.path)")
                    // Move compiled model to expected location
                    if compiledURL.path != modelPath.path {
                        try FileManager.default.removeItem(at: modelPath)
                        try FileManager.default.moveItem(at: compiledURL, to: modelPath)
                        logger.info("Moved compiled model to \(modelPath.path)")
                    }
                }
            }
        }
    }

    /// Ensure all required files are downloaded
    public static func ensureRequiredFiles() async throws {
        // Download data files
        for filename in requiredFiles {
            try await downloadFileIfNeeded(filename: filename, urlPath: filename)
        }

        // Download models
        try await downloadModelsIfNeeded()
    }

    /// Load all Kokoro models
    public static func loadModels() throws {
        guard !isModelsLoaded else { return }

        // Use the proper cache directory from TTSModels
        let modelDir = try TTSModels.cacheDirectoryURL()

        logger.info("Loading Kokoro models from \(modelDir.path)")

        // Check if models exist, if not they need to be downloaded
        let frontendURL = modelDir.appendingPathComponent("kokoro_frontend.mlmodelc")
        guard FileManager.default.fileExists(atPath: frontendURL.path) else {
            logger.error("Models not found at \(modelDir.path). Please ensure models are downloaded first.")
            throw TTSError.modelNotFound("Kokoro models not found. Run with auto-download or download manually.")
        }

        // Load frontend (BERT encoder)
        frontend = try MLModel(contentsOf: frontendURL)
        logger.info("Loaded frontend model")

        // Load pregenerator
        let pregeneratorURL = modelDir.appendingPathComponent("pregenerator.mlmodelc")
        pregenerator = try MLModel(contentsOf: pregeneratorURL)
        logger.info("Loaded pregenerator model")

        // Load decoder blocks
        decoderBlocks = []
        for i in 0..<4 {
            let decoderURL = modelDir.appendingPathComponent("decoder_block_\(i).mlmodelc")
            let decoder = try MLModel(contentsOf: decoderURL)
            decoderBlocks.append(decoder)
            logger.info("Loaded decoder block \(i)")
        }

        // Load generator
        let generatorURL = modelDir.appendingPathComponent("generator.mlmodelc")
        generator = try MLModel(contentsOf: generatorURL)
        logger.info("Loaded generator model")

        isModelsLoaded = true
    }

    /// Load phoneme dictionary
    public static func loadPhonemeDictionary() throws {
        guard !isDictionaryLoaded else { return }

        let currentDir = FileManager.default.currentDirectoryPath
        let dictURL = URL(fileURLWithPath: currentDir).appendingPathComponent("word_phonemes.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw TTSError.modelNotFound("Phoneme dictionary not found at \(dictURL.path)")
        }

        let data = try Data(contentsOf: dictURL)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        if let wordToPhonemes = json["word_to_phonemes"] as? [String: [String]] {
            phonemeDictionary = wordToPhonemes
            isDictionaryLoaded = true
            logger.info("Loaded \(phonemeDictionary.count) words from phoneme dictionary")
        }
    }

    /// Convert text to phonemes
    public static func textToPhonemes(_ text: String) throws -> [String] {
        try loadPhonemeDictionary()

        let words = text.lowercased().split(separator: " ")
        var allPhonemes: [String] = []

        for word in words {
            let cleanWord = String(word).filter { $0.isLetter || $0.isNumber }

            if let phonemes = phonemeDictionary[cleanWord] {
                allPhonemes.append(contentsOf: phonemes)
                allPhonemes.append(" ")  // Space between words
            } else {
                logger.warning("Word '\(cleanWord)' not in dictionary")
            }
        }

        // Remove trailing space
        if allPhonemes.last == " " {
            allPhonemes.removeLast()
        }

        return allPhonemes
    }

    /// Convert phonemes to input IDs
    public static func phonemesToInputIds(_ phonemes: [String]) -> [Int32] {
        let vocabulary = KokoroVocabulary.getVocabulary()

        // Debug: check vocab
        logger.info("Vocabulary has \(vocabulary.count) entries")

        var ids: [Int32] = [0]  // Start token

        for phoneme in phonemes {
            if let id = vocabulary[phoneme] {
                ids.append(id)
            } else {
                logger.warning("Missing phoneme in vocab: '\(phoneme)'")
                ids.append(0)  // Use 0 for unknown
            }
        }

        ids.append(0)  // End token

        // Pad to 272 tokens (frontend model requirement)
        while ids.count < 272 {
            ids.append(0)
        }
        if ids.count > 272 {
            ids = Array(ids.prefix(272))
        }

        return ids
    }

    /// Load voice embedding
    public static func loadVoiceEmbedding(voice: String = "af_heart") throws -> MLMultiArray {
        let currentDir = FileManager.default.currentDirectoryPath
        let voiceFile = "\(voice).json"
        let voiceURL = URL(fileURLWithPath: currentDir).appendingPathComponent(voiceFile)

        // For multi-model architecture, we use 128-dimensional embeddings
        let embedding = try MLMultiArray(shape: [1, 128] as [NSNumber], dataType: .float32)

        if FileManager.default.fileExists(atPath: voiceURL.path) {
            // Load from JSON if available
            let data = try Data(contentsOf: voiceURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Try different formats - Python might save as Double array
                if let values = json[voice] as? [Double] {
                    for (i, val) in values.enumerated() where i < 128 {
                        embedding[i] = NSNumber(value: Float(val))
                    }
                    logger.info("Loaded voice embedding from \(voiceFile)")
                    return embedding
                } else if let values = json[voice] as? [Float] {
                    for (i, val) in values.enumerated() where i < 128 {
                        embedding[i] = NSNumber(value: val)
                    }
                    logger.info("Loaded voice embedding from \(voiceFile)")
                    return embedding
                }
            }
            logger.warning("Could not parse voice embedding from \(voiceFile)")
        }

        // Use default embedding if file not found or parsing failed
        if !FileManager.default.fileExists(atPath: voiceURL.path) {
            logger.warning("Voice file not found, using default embedding")
        } else {
            logger.warning("Using default embedding (failed to parse JSON)")
        }

        // Fill with small random values like main2.py
        for i in 0..<128 {
            embedding[i] = NSNumber(value: Float.random(in: -0.1...0.1))
        }

        return embedding
    }

    /// Create ASR features using the frontend BERT encoder
    private static func createASRFeatures(
        inputIds: [Int32],
        refStyle: MLMultiArray
    ) throws -> (
        asr: MLMultiArray,
        f0: MLMultiArray,
        n: MLMultiArray,
        refStyleOut: MLMultiArray
    ) {
        // Convert input IDs to MLMultiArray
        let inputArray = try MLMultiArray(shape: [1, 272] as [NSNumber], dataType: .int32)
        for (i, id) in inputIds.enumerated() where i < 272 {
            inputArray[i] = NSNumber(value: id)
        }

        // Create full 256-dim ref_s by padding the 128-dim voice embedding
        let fullRefS = try MLMultiArray(shape: [1, 256] as [NSNumber], dataType: .float32)
        for i in 0..<128 {
            fullRefS[i] = refStyle[i]
        }
        // Pad with zeros for the remaining 128 dimensions
        for i in 128..<256 {
            fullRefS[i] = NSNumber(value: 0.0)
        }

        // Run frontend model
        logger.info("Running frontend (BERT encoder)...")
        let frontendInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputArray,
            "ref_s": fullRefS,
        ])

        guard let frontendOutput = try frontend?.prediction(from: frontendInput) else {
            throw TTSError.processingFailed("Frontend prediction failed")
        }

        // Extract outputs
        guard let asrRaw = frontendOutput.featureValue(for: "asr")?.multiArrayValue,
            let f0Raw = frontendOutput.featureValue(for: "F0_pred")?.multiArrayValue,
            let nRaw = frontendOutput.featureValue(for: "N_pred")?.multiArrayValue,
            let refStyleOut = frontendOutput.featureValue(for: "ref_style")?.multiArrayValue
        else {
            throw TTSError.processingFailed("Failed to extract frontend outputs")
        }

        // Fix shape mismatches:
        // - pregenerator expects ASR (1, 512, 621) but frontend outputs (1, 512, 678)
        // - pregenerator expects F0/N (1, 1242) but frontend outputs (1, 1356)
        let expectedAsrFrames = 621
        let expectedF0Frames = 1242

        // Fix ASR shape
        let asr = try MLMultiArray(shape: [1, 512, expectedAsrFrames] as [NSNumber], dataType: .float32)
        let sourceAsrFrames = min(asrRaw.shape[2].intValue, expectedAsrFrames)
        for c in 0..<512 {
            for f in 0..<sourceAsrFrames {
                let srcIdx = c * asrRaw.shape[2].intValue + f
                let dstIdx = c * expectedAsrFrames + f
                asr[dstIdx] = asrRaw[srcIdx]
            }
            // Pad with zeros if needed
            for f in sourceAsrFrames..<expectedAsrFrames {
                let dstIdx = c * expectedAsrFrames + f
                asr[dstIdx] = NSNumber(value: 0.0)
            }
        }

        // Fix F0/N shapes - TRUNCATE like main2.py (not pad)
        let f0: MLMultiArray
        let n: MLMultiArray

        if f0Raw.shape[1].intValue > expectedF0Frames {
            // Truncate from 1356 to 1242 frames
            f0 = try MLMultiArray(shape: [1, expectedF0Frames] as [NSNumber], dataType: .float32)
            n = try MLMultiArray(shape: [1, expectedF0Frames] as [NSNumber], dataType: .float32)

            for i in 0..<expectedF0Frames {
                f0[i] = f0Raw[i]
                n[i] = nRaw[i]
            }
            logger.info("Truncated F0/N from \(f0Raw.shape[1]) to \(expectedF0Frames) frames")
        } else {
            // Use as-is if already correct size or smaller
            f0 = f0Raw
            n = nRaw
        }

        return (asr, f0, n, refStyleOut)
    }

    /// Main synthesis function using multi-model pipeline
    public static func synthesize(text: String, voice: String = "af_heart") async throws -> Data {
        logger.info("Synthesizing: '\(text)'")

        // Ensure required files are downloaded
        do {
            logger.info("Checking and downloading required files...")
            try await ensureRequiredFiles()
            logger.info("Required files ready")
        } catch {
            logger.error("Failed to download required files: \(error)")
            throw error
        }

        // Load models if needed
        do {
            logger.info("Loading models...")
            try loadModels()
            logger.info("Models loaded successfully")
        } catch {
            logger.error("Failed to load models: \(error)")
            throw error
        }

        // Step 1: Text to phonemes
        let phonemes = try textToPhonemes(text)
        logger.info("Generated \(phonemes.count) phonemes: \(phonemes.prefix(20))")

        // Step 2: Phonemes to input IDs
        let inputIds = phonemesToInputIds(phonemes)
        logger.info("Generated \(inputIds.count) input IDs: \(inputIds.prefix(20))")

        // Step 3: Get voice embedding
        let refStyle = try loadVoiceEmbedding(voice: voice)

        // Step 4: Create ASR features using frontend BERT encoder
        let (asr, f0Pred, nPred, _) = try createASRFeatures(inputIds: inputIds, refStyle: refStyle)

        // Step 5: Run pregenerator
        logger.info("Running pregenerator...")
        let pregenInput = try MLDictionaryFeatureProvider(dictionary: [
            "asr": asr,
            "F0_pred": f0Pred,
            "N_pred": nPred,
            "ref_s": refStyle,
        ])

        guard let pregenOutput = try pregenerator?.prediction(from: pregenInput) else {
            throw TTSError.processingFailed("Pregenerator prediction failed")
        }

        // Extract pregenerator outputs - map correctly based on Python reference
        // From Python: F0_processed = pre_outputs['F0']
        //             N_processed = pre_outputs['N']
        //             asr_res = pre_outputs['var_157']  # 64 channels
        //             x_encoded = pre_outputs['var_141']  # 1024 channels
        let f0Processed = pregenOutput.featureValue(for: "F0")?.multiArrayValue ?? f0Pred
        let nProcessed = pregenOutput.featureValue(for: "N")?.multiArrayValue ?? nPred
        let asrRes = pregenOutput.featureValue(for: "var_157")?.multiArrayValue ?? asr
        var xCurrent = pregenOutput.featureValue(for: "var_141")?.multiArrayValue ?? asr

        // Step 6: Run decoder blocks with proper residual logic
        logger.info("Running decoder blocks...")
        var useResidual = true  // Start with residual connections

        for (i, decoder) in decoderBlocks.enumerated() {
            logger.info("Decoder block \(i): \(useResidual ? "with" : "without") residual")

            // Save current shape for upsampling detection
            let previousShape = xCurrent.shape

            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "x_input": xCurrent,
                "asr_res": asrRes,
                "F0": f0Processed,
                "N": nProcessed,
                "ref_s": refStyle,
            ])

            let decoderOutput = try decoder.prediction(from: decoderInput)

            // Get the output (usually the first/only output)
            if let outputNames = decoderOutput.featureNames.first,
                let outputArray = decoderOutput.featureValue(for: outputNames)?.multiArrayValue
            {
                // Check if this block does upsampling (shape change indicates upsampling)
                if outputArray.shape != previousShape {
                    logger.info("Decoder block \(i): Detected upsampling, disabling residual for next blocks")
                    useResidual = false
                }
                xCurrent = outputArray
            }
        }

        // Step 7: Generate random phases
        let randomPhases = try MLMultiArray(shape: [1, 9] as [NSNumber], dataType: .float32)
        for i in 0..<9 {
            randomPhases[i] = NSNumber(value: Float.random(in: -Float.pi...Float.pi))
        }

        // Step 8: Run generator
        logger.info("Running generator...")
        let generatorInput = try MLDictionaryFeatureProvider(dictionary: [
            "x_final": xCurrent,
            "ref_s": refStyle,
            "F0_pred": f0Pred,
            "random_phases": randomPhases,
        ])

        guard let generatorOutput = try generator?.prediction(from: generatorInput) else {
            throw TTSError.processingFailed("Generator prediction failed")
        }

        // Extract audio output
        guard
            let audioArray = generatorOutput.featureValue(for: generatorOutput.featureNames.first ?? "")?
                .multiArrayValue
        else {
            throw TTSError.processingFailed("Failed to extract audio from generator")
        }

        // Convert to audio data
        let audioData = try convertToWAV(audioArray)

        logger.info("Generated audio: \(audioData.count) bytes")

        return audioData
    }

    /// Convert MLMultiArray to WAV data
    private static func convertToWAV(_ audioArray: MLMultiArray) throws -> Data {
        let sampleRate: Double = 22050  // Match Python reference
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
