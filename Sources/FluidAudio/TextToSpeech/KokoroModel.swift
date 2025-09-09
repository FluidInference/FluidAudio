import CoreML
import Darwin
import Foundation
import OSLog

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Kokoro TTS implementation using multiple CoreML models
/// Uses pregenerator, decoder blocks, and generator architecture
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModel {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroModel")

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
    private static func downloadModelsIfNeeded() async throws {
        // Use Models/kokoro subdirectory in cache
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create Models/kokoro directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        logger.info("Model directory: \(modelDir.path)")
        print("Model directory: \(modelDir.path)")

        var modelsToDownload: [String] = []

        // Check which models need downloading
        for modelName in modelNames {
            let modelPath = modelDir.appendingPathComponent("\(modelName).mlmodelc")
            if !FileManager.default.fileExists(atPath: modelPath.path) {
                modelsToDownload.append(modelName)
            }
        }

        if modelsToDownload.isEmpty {
            logger.info("All models already downloaded")
            print("All models already downloaded")
            return
        }

        logger.info("Need to download \(modelsToDownload.count) models: \(modelsToDownload.joined(separator: ", "))")
        print("Need to download \(modelsToDownload.count) models: \(modelsToDownload.joined(separator: ", "))")

        // Download each model's files
        for (index, modelName) in modelsToDownload.enumerated() {
            let modelPath = modelDir.appendingPathComponent("\(modelName).mlmodelc")

            logger.info("[\(index + 1)/\(modelsToDownload.count)] Downloading \(modelName)...")
            print("[\(index + 1)/\(modelsToDownload.count)] Downloading \(modelName)...")

            // Create model directory
            try FileManager.default.createDirectory(at: modelPath, withIntermediateDirectories: true)

            // Download the compiled mlmodelc files
            let filesToDownload = [
                "coremldata.bin",
                "model.mil",
                "weights/weight.bin",
                "analytics/coremldata.bin",
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
                let fileURL = URL(string: "\(baseURL)/\(modelName).mlmodelc/\(file)")!
                let destPath: URL

                if file == "weights/weight.bin" {
                    destPath = modelPath.appendingPathComponent("weights").appendingPathComponent("weight.bin")
                } else if file == "analytics/coremldata.bin" {
                    destPath = modelPath.appendingPathComponent("analytics").appendingPathComponent(
                        "coremldata.bin")
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
                        logger.warning(
                            "  File \(file) returned status \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                    }
                } catch {
                    logger.warning("  Could not download \(file): \(error.localizedDescription)")
                }
            }

            logger.info("✓ Downloaded \(modelName).mlmodelc")
            // Skip verification - compiled mlmodelc files will be loaded later
        }

        logger.info("All models downloaded successfully")
    }

    /// Ensure all required files are downloaded
    public static func ensureRequiredFiles() async throws {
        // Skip download if we're using mlpackage models
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelDir = cacheDir.appendingPathComponent("Models/kokoro")
        let frontendPackageURL = modelDir.appendingPathComponent("kokoro_frontend.mlpackage")

        if FileManager.default.fileExists(atPath: frontendPackageURL.path) {
            logger.info("Using existing mlpackage models, skipping download")

            // Still download data files if needed
            for filename in requiredFiles {
                try await downloadFileIfNeeded(filename: filename, urlPath: filename)
            }

            // Download voice embedding if needed
            try await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: "af_heart")
            return
        }

        // Check and download data files
        print("Checking JSON data files...")
        var jsonToDownload: [String] = []
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        for filename in requiredFiles {
            let filePath = kokoroDir.appendingPathComponent(filename)
            if !FileManager.default.fileExists(atPath: filePath.path) {
                jsonToDownload.append(filename)
            }
        }

        if !jsonToDownload.isEmpty {
            print("Need to download \(jsonToDownload.count) JSON files: \(jsonToDownload.joined(separator: ", "))")
            for filename in jsonToDownload {
                try await downloadFileIfNeeded(filename: filename, urlPath: filename)
            }
        } else {
            print("JSON data files already present")
        }

        // Download models
        try await downloadModelsIfNeeded()

        // Download default voice embedding
        try await VoiceEmbeddingDownloader.ensureVoiceEmbedding(voice: "af_heart")
    }

    /// Load all Kokoro models
    public static func loadModels() async throws {
        guard !isModelsLoaded else { return }

        // Use Models/kokoro subdirectory in cache
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let modelDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        logger.info("Loading Kokoro models from \(modelDir.path)")
        print("Loading Kokoro models from \(modelDir.path)")

        // Check if any models are missing
        var missingModels: [String] = []
        for modelName in modelNames {
            let compiledURL = modelDir.appendingPathComponent("\(modelName).mlmodelc")
            let packageURL = modelDir.appendingPathComponent("\(modelName).mlpackage")

            if !FileManager.default.fileExists(atPath: compiledURL.path)
                && !FileManager.default.fileExists(atPath: packageURL.path)
            {
                missingModels.append(modelName)
            }
        }

        // If models are missing, download them
        if !missingModels.isEmpty {
            logger.warning("Missing models: \(missingModels.joined(separator: ", "))")
            print("Missing models: \(missingModels.joined(separator: ", "))")
            logger.info("Downloading missing models...")
            print("Downloading missing models...")
            try await downloadModelsIfNeeded()
            logger.info("Models downloaded successfully")
            print("Models downloaded successfully")
        }

        // Check if models exist after download attempt
        let frontendPackageURL = modelDir.appendingPathComponent("kokoro_frontend.mlpackage")
        let frontendCompiledURL = modelDir.appendingPathComponent("kokoro_frontend.mlmodelc")

        // Determine which format to use
        let useMlpackage = FileManager.default.fileExists(atPath: frontendPackageURL.path)

        if !useMlpackage && !FileManager.default.fileExists(atPath: frontendCompiledURL.path) {
            logger.error("❌ Models still not found after download attempt at \(modelDir.path)")
            throw TTSError.modelNotFound("Failed to download Kokoro models. Check network connection.")
        }

        // Load frontend (BERT encoder)
        if useMlpackage {
            logger.info("Loading frontend from mlpackage...")
            let compiledURL = try await MLModel.compileModel(at: frontendPackageURL)
            frontend = try MLModel(contentsOf: compiledURL)
            logger.info("Loaded frontend model from mlpackage")
        } else {
            frontend = try MLModel(contentsOf: frontendCompiledURL)
            logger.info("Loaded frontend model from mlmodelc")
        }

        // Helper to load a model from mlpackage or mlmodelc
        func loadModel(name: String) async throws -> MLModel {
            let packageURL = modelDir.appendingPathComponent("\(name).mlpackage")
            let compiledURL = modelDir.appendingPathComponent("\(name).mlmodelc")

            if FileManager.default.fileExists(atPath: packageURL.path) {
                logger.info("Loading \(name) from mlpackage...")
                let compiled = try await MLModel.compileModel(at: packageURL)
                return try MLModel(contentsOf: compiled)
            } else if FileManager.default.fileExists(atPath: compiledURL.path) {
                return try MLModel(contentsOf: compiledURL)
            } else {
                throw TTSError.modelNotFound("\(name) model not found")
            }
        }

        // Load pregenerator
        pregenerator = try await loadModel(name: "pregenerator")
        logger.info("Loaded pregenerator model")

        // Load decoder blocks
        decoderBlocks = []
        for i in 0..<4 {
            let decoder = try await loadModel(name: "decoder_block_\(i)")
            decoderBlocks.append(decoder)
            logger.info("Loaded decoder block \(i)")
        }

        // Load generator
        generator = try await loadModel(name: "generator")
        logger.info("Loaded generator model")

        // Verify all models are properly loaded
        try verifyModelsLoaded()

        isModelsLoaded = true
        logger.info("All Kokoro models successfully loaded and verified")

        // Perform warm-up inference to initialize models (especially important for M1)
        await performWarmupInference()
    }

    /// Perform warm-up inference to initialize models
    private static func performWarmupInference() async {
        logger.info("Performing warm-up inference...")
        print("Performing warm-up inference...")
        let warmupStart = Date()

        do {
            // Create minimal dummy inputs for warm-up
            let dummyInputIds = try MLMultiArray(shape: [1, 272] as [NSNumber], dataType: .int32)
            let dummyRefS = try MLMultiArray(shape: [1, 256] as [NSNumber], dataType: .float32)

            // Fill with minimal valid data
            for i in 0..<272 {
                dummyInputIds[i] = NSNumber(value: 0)
            }
            for i in 0..<256 {
                dummyRefS[i] = NSNumber(value: 0.0)
            }

            // Run frontend warm-up
            let frontendInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": dummyInputIds,
                "ref_s": dummyRefS,
            ])

            _ = try? frontend?.prediction(from: frontendInput)

            let warmupTime = Date().timeIntervalSince(warmupStart)
            logger.info("Warm-up completed in \(String(format: "%.3f", warmupTime))s")
            print("Warm-up completed in \(String(format: "%.3f", warmupTime))s")

            if warmupTime > 3.0 {
                logger.warning(
                    "⚠️ Slow warm-up detected (\(String(format: "%.1f", warmupTime))s) - first synthesis may be delayed")
                print(
                    "WARNING: Slow warm-up detected (\(String(format: "%.1f", warmupTime))s) - first synthesis may be delayed"
                )
            }
        } catch {
            logger.warning("⚠️ Warm-up inference failed (non-critical): \(error)")
        }
    }

    /// Log system information for debugging
    private static func logSystemInfo() {
        #if arch(arm64)
        logger.info("🖥️ Architecture: Apple Silicon (arm64)")

        // Detect chip type
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        let modelString = String(cString: model)

        if modelString.contains("Mac14") || modelString.contains("Mac15") {
            logger.info("Chip: Apple M2 or newer (\(modelString))")
            print("Chip: Apple M2 or newer (\(modelString))")
        } else if modelString.contains("Mac13") || modelString.contains("MacBookAir10")
            || modelString.contains("MacBookPro17") || modelString.contains("MacBookPro18")
        {
            logger.info("Chip: Apple M1 (\(modelString))")
            print("Chip: Apple M1 (\(modelString))")
            logger.warning("M1 chip detected - may experience compatibility issues")
            print("WARNING: M1 chip detected - may experience compatibility issues")
        } else {
            logger.info("Chip: \(modelString)")
            print("Chip: \(modelString)")
        }

        // Log memory
        var memInfo = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &memInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }

        if result == KERN_SUCCESS {
            let pageSize = vm_kernel_page_size
            let totalMemory =
                Double(memInfo.free_count + memInfo.active_count + memInfo.inactive_count + memInfo.wire_count)
                * Double(pageSize) / 1_073_741_824
            logger.info("🧠 Memory: \(String(format: "%.1f", totalMemory)) GB")
        }
        #else
        logger.info("🖥️ Architecture: Intel x86_64")
        #endif
    }

    /// Verify all models are properly loaded and can perform inference
    private static func verifyModelsLoaded() throws {
        logger.info("Verifying model initialization...")

        // Check each model is not nil
        guard let _ = frontend else {
            throw TTSError.modelNotFound("Frontend model failed to load")
        }
        guard let _ = pregenerator else {
            throw TTSError.modelNotFound("Pregenerator model failed to load")
        }
        guard let _ = generator else {
            throw TTSError.modelNotFound("Generator model failed to load")
        }
        guard decoderBlocks.count == 4 else {
            throw TTSError.modelNotFound("Not all decoder blocks loaded (expected 4, got \(decoderBlocks.count))")
        }

        // Verify model descriptions (ensures they're fully initialized)
        logger.info(
            "Frontend input: \(frontend?.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )
        logger.info(
            "Frontend output: \(frontend?.modelDescription.outputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )

        logger.info(
            "Pregenerator input: \(pregenerator?.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )
        logger.info(
            "Pregenerator output: \(pregenerator?.modelDescription.outputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )

        for (i, decoder) in decoderBlocks.enumerated() {
            logger.info(
                "Decoder \(i) input: \(decoder.modelDescription.inputDescriptionsByName.keys.joined(separator: ", "))")
            logger.info(
                "Decoder \(i) output: \(decoder.modelDescription.outputDescriptionsByName.keys.joined(separator: ", "))"
            )
        }

        logger.info(
            "Generator input: \(generator?.modelDescription.inputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )
        logger.info(
            "Generator output: \(generator?.modelDescription.outputDescriptionsByName.keys.joined(separator: ", ") ?? "none")"
        )

        logger.info("✓ All models verified and ready for inference")
    }

    /// Load phoneme dictionary
    public static func loadPhonemeDictionary() throws {
        guard !isDictionaryLoaded else { return }

        // Use Models/kokoro subdirectory
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        let dictURL = kokoroDir.appendingPathComponent("word_phonemes.json")

        // Download if missing
        if !FileManager.default.fileExists(atPath: dictURL.path) {
            logger.info("Phoneme dictionary not found in cache, downloading...")
            try downloadDictionaryFiles()
        }

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

    /// Download dictionary files if missing
    private static func downloadDictionaryFiles() throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")

        // Create directory if needed
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let baseURL = "https://huggingface.co/FluidInference/kokoro-82m-coreml/resolve/main"
        let files = ["word_phonemes.json", "vocab_index.json"]

        for file in files {
            let localPath = kokoroDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: localPath.path) {
                let remoteURL = URL(string: "\(baseURL)/\(file)")!
                logger.info("Downloading \(file)...")
                let data = try Data(contentsOf: remoteURL)
                try data.write(to: localPath)
                logger.info("Downloaded \(file) to cache")
            }
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
        // Use Models/kokoro/voices subdirectory in cache
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")

        // Create voices directory if needed
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        // For multi-model architecture, we use 128-dimensional embeddings
        let embedding = try MLMultiArray(shape: [1, 128] as [NSNumber], dataType: .float32)

        // Try JSON file first (we can't parse .pt files directly in Swift)
        let jsonFile = "\(voice).json"
        let jsonURL = voicesDir.appendingPathComponent(jsonFile)

        // Download voice embedding if missing
        if !FileManager.default.fileExists(atPath: jsonURL.path) {
            print("Downloading voice embedding: \(voice)")

            // First check if JSON exists locally in current directory
            let localURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(
                jsonFile)
            if FileManager.default.fileExists(atPath: localURL.path) {
                // Copy local file to cache
                try FileManager.default.copyItem(at: localURL, to: jsonURL)
                print("Voice embedding cached: \(voice)")
            } else {
                // Note: .pt files from HuggingFace need to be converted to JSON format
                // The conversion requires Python torch library
                print("Note: Voice embedding \(voice).pt needs to be converted to JSON format")
                print("Run: python3 extract_voice_embeddings.py to convert .pt files")
                // HuggingFace URL for reference: https://huggingface.co/FluidInference/kokoro-82m-coreml/blob/main/voices/\(voice).pt
            }
        }

        if FileManager.default.fileExists(atPath: jsonURL.path) {
            // Load from cached JSON
            let data = try Data(contentsOf: jsonURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // Try different formats - Python might save as Double array
                if let values = json[voice] as? [Double] {
                    for (i, val) in values.enumerated() where i < 128 {
                        embedding[i] = NSNumber(value: Float(val))
                    }
                    print("Loaded voice embedding: \(voice) from cache")
                    return embedding
                } else if let values = json[voice] as? [Float] {
                    for (i, val) in values.enumerated() where i < 128 {
                        embedding[i] = NSNumber(value: val)
                    }
                    print("Loaded voice embedding: \(voice) from cache")
                    return embedding
                }
            }
            logger.warning("Could not parse voice embedding from \(jsonFile)")
        }

        // Use default embedding if file not found or parsing failed
        if !FileManager.default.fileExists(atPath: jsonURL.path) {
            print("Voice embedding not found in cache, using default")
        } else {
            print("Using default embedding (failed to parse JSON)")
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
        let synthesisStart = Date()
        var stepTimings: [(String, TimeInterval)] = []

        func logStep(_ name: String, start: Date) {
            let elapsed = Date().timeIntervalSince(start)
            stepTimings.append((name, elapsed))
            logger.info("⏱️ \(name): \(String(format: "%.3f", elapsed))s")
        }

        // Log system info
        logger.info("\("=".repeated(50))")
        logSystemInfo()
        logger.info("🎤 Starting synthesis: '\(text)')")
        logger.info("📏 Input length: \(text.count) characters, \(text.split(separator: " ").count) words")

        // Warn about short inputs
        if text.count < 10 {
            logger.warning("⚠️ Very short input (\(text.count) chars) may produce poor quality audio")
        }

        logger.info("\("=".repeated(50))")

        // Ensure required files are downloaded
        let downloadStart = Date()
        do {
            logger.info("📥 Checking and downloading required files...")
            try await ensureRequiredFiles()
            logger.info("✅ Required files ready")
        } catch {
            logger.error("❌ Failed to download required files: \(error)")
            throw error
        }
        logStep("File verification", start: downloadStart)

        // Verify models are loaded (should be done externally now)
        if !isModelsLoaded {
            logger.warning("Models not loaded - calling loadModels() from synthesize is deprecated")
            let loadStart = Date()
            try await loadModels()
        } else {
            logger.info("Models already loaded")
        }

        // Step 1: Text to phonemes
        let phonemeStart = Date()
        let phonemes = try textToPhonemes(text)
        logger.info("🔤 Generated \(phonemes.count) phonemes: \(phonemes.prefix(20))")
        logStep("Text to phonemes", start: phonemeStart)

        // Step 2: Phonemes to input IDs
        let idStart = Date()
        let inputIds = phonemesToInputIds(phonemes)
        logger.info("🆔 Generated \(inputIds.count) input IDs: \(inputIds.prefix(20))")
        logStep("Phonemes to IDs", start: idStart)

        // Step 3: Get voice embedding
        let voiceStart = Date()
        let refStyle = try loadVoiceEmbedding(voice: voice)
        logger.info("🎤 Loaded voice embedding: \(voice)")
        logStep("Voice loading", start: voiceStart)

        // Step 4: Create ASR features using frontend BERT encoder
        let frontendStart = Date()
        logger.info("🤖 Running frontend BERT encoder...")
        let (asr, f0Pred, nPred, _) = try createASRFeatures(inputIds: inputIds, refStyle: refStyle)
        logStep("Frontend (BERT)", start: frontendStart)

        // Step 5: Run pregenerator
        let pregenStart = Date()
        logger.info("🎯 Running pregenerator...")
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
        logStep("Pregenerator", start: pregenStart)

        // Step 6: Run decoder blocks with proper residual logic
        let decoderStart = Date()
        logger.info("🔄 Running decoder blocks...")
        var useResidual = true  // Start with residual connections

        for (i, decoder) in decoderBlocks.enumerated() {
            let blockStart = Date()
            logger.info("📦 Decoder block \(i): \(useResidual ? "with" : "without") residual")

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
            logStep("  Decoder block \(i)", start: blockStart)
        }
        logStep("All decoder blocks", start: decoderStart)

        // Step 7: Generate random phases
        let randomPhases = try MLMultiArray(shape: [1, 9] as [NSNumber], dataType: .float32)
        for i in 0..<9 {
            randomPhases[i] = NSNumber(value: Float.random(in: -Float.pi...Float.pi))
        }

        // Step 8: Run generator
        let generatorStart = Date()
        logger.info("🎧 Running final generator...")
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
        logStep("Generator", start: generatorStart)

        // Convert to audio data
        let conversionStart = Date()
        let audioData = try convertToWAV(audioArray)
        logStep("WAV conversion", start: conversionStart)

        // Log summary
        let totalTime = Date().timeIntervalSince(synthesisStart)
        logger.info("\("=".repeated(50))")
        logger.info("✅ Synthesis complete!")
        logger.info("📊 Total time: \(String(format: "%.3f", totalTime))s")
        logger.info("📀 Audio size: \(audioData.count) bytes")

        // Log timing breakdown
        logger.info("📈 Timing breakdown:")
        for (step, time) in stepTimings {
            let percentage = (time / totalTime) * 100
            logger.info("  \(step): \(String(format: "%.3f", time))s (\(String(format: "%.1f", percentage))%)")
        }
        logger.info("\("=".repeated(50))")

        return audioData
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

// MARK: - Helper Extensions
extension String {
    func repeated(_ count: Int) -> String {
        return String(repeating: self, count: count)
    }
}
