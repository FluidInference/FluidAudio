#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// VAD benchmark implementation
struct VadBenchmark {

    static func runVadBenchmark(arguments: [String]) async {
        do {
            try await runVadBenchmarkWithErrorHandling(arguments: arguments)
        } catch {
            print("VAD Benchmark failed: \(error)")
            // Don't exit - return gracefully so comparison can continue
        }
    }

    static func runVadBenchmarkWithErrorHandling(arguments: [String]) async throws {
        print("🚀 Starting VAD Benchmark")
        var numFiles = -1  // Default to all files
        var useAllFiles = true  // Default to all files
        var vadThreshold: Float = 0.3
        var outputFile: String?
        var dataset = "mini50"  // Default to mini50 dataset
        print("   📝 Parsing arguments...")

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--num-files":
                if i + 1 < arguments.count {
                    numFiles = Int(arguments[i + 1]) ?? -1
                    useAllFiles = false  // Override default when specific count is given
                    i += 1
                }
            case "--all-files":
                useAllFiles = true
                numFiles = -1
            case "--threshold":
                if i + 1 < arguments.count {
                    vadThreshold = Float(arguments[i + 1]) ?? 0.3
                    i += 1
                }
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("🚀 Starting VAD Benchmark")
        print("   Test files: \(numFiles)")
        print("   VAD threshold: \(vadThreshold)")

        // Use VadManager with the trained model
        let vadManager = try await VadManager(
            config: VadConfig(
                threshold: vadThreshold,
                debugMode: false
            ))

        print("VAD system initialized")

        // Download test files
        let testFiles = try await downloadVadTestFiles(
            count: useAllFiles ? -1 : numFiles, dataset: dataset)

        // Run benchmark
        let result = try await runVadBenchmarkInternal(
            vadManager: vadManager, testFiles: testFiles, threshold: vadThreshold)

        // Print results
        // Calculate RTFx for display
        let rtfx = try await calculateRTFx(result: result, testFiles: testFiles)

        print("\n📊 VAD Benchmark Results:")
        print("   Accuracy: \(String(format: "%.1f", result.accuracy))%")
        print("   Precision: \(String(format: "%.1f", result.precision))%")
        print("   Recall: \(String(format: "%.1f", result.recall))%")
        print("   F1-Score: \(String(format: "%.1f", result.f1Score))%")
        print("   Total Time: \(String(format: "%.2f", result.processingTime))s")
        if rtfx < 1.0 && rtfx > 0 {
            print("   RTFx: \(String(format: "%.1f", 1.0/rtfx))x faster than real-time")
        } else if rtfx >= 1.0 {
            print("   RTFx: \(String(format: "%.1f", rtfx))x slower than real-time")
        } else {
            print("   RTFx: N/A")
        }
        print("   Files Processed: \(result.totalFiles)")
        print(
            "   Avg Time per File: \(String(format: "%.3f", result.processingTime / Double(result.totalFiles)))s"
        )

        // Save results with RTFx
        if let outputFile = outputFile {
            try await saveVadBenchmarkResultsWithRTFx(
                result, testFiles: testFiles, to: outputFile)
            print("💾 Results saved to: \(outputFile)")
        } else {
            try await saveVadBenchmarkResultsWithRTFx(
                result, testFiles: testFiles, to: "vad_benchmark_results.json")
            print("💾 Results saved to: vad_benchmark_results.json")
        }

        // Performance assessment
        if result.f1Score >= 70.0 {
            print("\nEXCELLENT: F1-Score above 70%")
        } else if result.f1Score >= 60.0 {
            print("\n⚠️ ACCEPTABLE: F1-Score above 60%")
        } else {
            print("\nNEEDS IMPROVEMENT: F1-Score below 60%")
            // Don't exit - just report the poor performance
        }
    }

    static func downloadVadTestFiles(
        count: Int, dataset: String = "mini50"
    ) async throws
        -> [VadTestFile]
    {
        if count == -1 {
            print("📥 Loading all available test audio files...")
        } else {
            print("📥 Loading \(count) test audio files...")
        }

        // First check if this is full MUSAN dataset
        if dataset == "musan-full" {
            if let musanFiles = try await loadFullMusanDataset(count: count) {
                return musanFiles
            }
        }

        // Check if this is VOiCES subset
        if dataset == "voices-subset" {
            if let voicesFiles = try await loadVoicesSubset(count: count) {
                return voicesFiles
            }
        }

        // First try to load from local dataset directory
        if let localFiles = try await loadLocalDataset(count: count) {
            return localFiles
        }

        // Second, try to load from Hugging Face cache
        if let cachedFiles = try await loadHuggingFaceVadDataset(count: count, dataset: dataset) {
            return cachedFiles
        }

        // Finally, download from Hugging Face
        print("🌐 Downloading VAD dataset from Hugging Face...")
        if let hfFiles = try await downloadHuggingFaceVadDataset(count: count, dataset: dataset) {
            return hfFiles
        }

        // No fallback to mock data - fail cleanly
        print("Failed to load VAD dataset from all sources:")
        print("   • Local dataset not found")
        print("   • Hugging Face cache empty")
        print("   • Hugging Face download failed")
        print("💡 Try: swift run fluidaudio download --dataset vad")
        throw NSError(
            domain: "VadError", code: 404,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "No VAD dataset available. Use 'download --dataset vad' to get real data."
            ])
    }

    static func loadLocalDataset(count: Int) async throws -> [VadTestFile]? {
        // Check for local VAD dataset directories
        let possiblePaths = [
            "VADDataset/",
            "vad_test_data/",
            "datasets/vad/",
            "../datasets/vad/",
        ]

        for basePath in possiblePaths {
            let datasetDir = URL(fileURLWithPath: basePath)

            guard FileManager.default.fileExists(atPath: datasetDir.path) else {
                continue
            }

            print("🗂️ Found local dataset at: \(basePath)")

            var testFiles: [VadTestFile] = []

            // Look for speech and non-speech subdirectories
            let speechDir = datasetDir.appendingPathComponent("speech")
            let nonSpeechDir = datasetDir.appendingPathComponent("non_speech")

            if FileManager.default.fileExists(atPath: speechDir.path) {
                let maxSpeechFiles = count == -1 ? Int.max : count / 2
                let speechFiles = try loadAudioFiles(
                    from: speechDir, expectedLabel: 1, maxCount: maxSpeechFiles)
                testFiles.append(contentsOf: speechFiles)
                print("   Loaded \(speechFiles.count) speech files")
            }

            if FileManager.default.fileExists(atPath: nonSpeechDir.path) {
                let maxNoiseFiles = count == -1 ? Int.max : count - testFiles.count
                let nonSpeechFiles = try loadAudioFiles(
                    from: nonSpeechDir, expectedLabel: 0, maxCount: maxNoiseFiles)
                testFiles.append(contentsOf: nonSpeechFiles)
                print("   Loaded \(nonSpeechFiles.count) non-speech files")
            }

            if !testFiles.isEmpty {
                print("📂 Using local dataset: \(testFiles.count) files total")
                return testFiles
            }
        }

        return nil
    }

    static func loadAudioFiles(
        from directory: URL, expectedLabel: Int, maxCount: Int
    ) throws
        -> [VadTestFile]
    {
        let fileManager = FileManager.default
        let audioExtensions = ["wav", "mp3", "m4a", "aac", "aiff"]

        guard
            let enumerator = fileManager.enumerator(
                at: directory, includingPropertiesForKeys: nil)
        else {
            return []
        }

        var files: [VadTestFile] = []

        for case let fileURL as URL in enumerator {
            guard files.count < maxCount else { break }

            let fileExtension = fileURL.pathExtension.lowercased()
            guard audioExtensions.contains(fileExtension) else { continue }

            let fileName = fileURL.lastPathComponent
            files.append(
                VadTestFile(name: fileName, expectedLabel: expectedLabel, url: fileURL))
        }

        return files
    }

    /// Load VAD dataset from Hugging Face cache
    static func loadHuggingFaceVadDataset(
        count: Int, dataset: String = "mini50"
    ) async throws
        -> [VadTestFile]?
    {
        let cacheDir = getVadDatasetCacheDirectory()

        // Check if cache exists and has the required structure
        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")

        guard
            FileManager.default.fileExists(atPath: speechDir.path)
                && FileManager.default.fileExists(atPath: noiseDir.path)
        else {
            return nil
        }

        // Load files from cache
        var testFiles: [VadTestFile] = []

        // Determine max files based on dataset
        let maxFilesForDataset = dataset == "mini100" ? 100 : 50

        // If count is -1, use all available files (but respect dataset limit)
        if count == -1 {
            print("📂 Loading all available files from Hugging Face cache...")

            // Load speech files (half of dataset)
            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: maxFilesForDataset / 2)
            testFiles.append(contentsOf: speechFiles)

            // Load noise files (half of dataset)
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: maxFilesForDataset / 2)
            testFiles.append(contentsOf: noiseFiles)
        } else {
            let speechCount = count / 2
            let noiseCount = count - speechCount

            // Load speech files
            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: speechCount)
            testFiles.append(contentsOf: speechFiles)

            // Load noise files
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: noiseCount)
            testFiles.append(contentsOf: noiseFiles)
        }

        if testFiles.isEmpty {
            return nil
        }

        print("🗂️ Found cached Hugging Face dataset: \(testFiles.count) files total")
        return testFiles
    }

    /// Download VAD dataset from Hugging Face musan_mini50 or musan_mini100 repository
    static func downloadHuggingFaceVadDataset(
        count: Int, dataset: String = "mini50"
    )
        async throws -> [VadTestFile]?
    {
        let cacheDir = getVadDatasetCacheDirectory()

        // Create cache directories
        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")
        try FileManager.default.createDirectory(
            at: speechDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: noiseDir, withIntermediateDirectories: true)

        // Select repository based on dataset parameter
        let repoName = dataset == "mini100" ? "musan_mini100" : "musan_mini50"
        let repoBase = "https://huggingface.co/datasets/alexwengg/\(repoName)/resolve/main"

        var testFiles: [VadTestFile] = []

        // If count is -1, download many files (large number)
        let maxFiles = dataset == "mini100" ? 100 : 50
        let speechCount = count == -1 ? maxFiles / 2 : count / 2
        let noiseCount = count == -1 ? maxFiles / 2 : count - speechCount

        do {
            // Download speech files
            print("   📢 Downloading speech samples...")
            let speechFiles = try await DatasetDownloader.downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/speech",
                targetDir: speechDir,
                expectedLabel: 1,
                count: speechCount,
                filePrefix: "speech",
                repoName: repoName
            )
            testFiles.append(contentsOf: speechFiles)

            // Download noise files
            print("   🔇 Downloading noise samples...")
            let noiseFiles = try await DatasetDownloader.downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/noise",
                targetDir: noiseDir,
                expectedLabel: 0,
                count: noiseCount,
                filePrefix: "noise",
                repoName: repoName
            )
            testFiles.append(contentsOf: noiseFiles)

            if !testFiles.isEmpty {
                print("Downloaded VAD dataset from Hugging Face: \(testFiles.count) files")
                return testFiles
            }

        } catch {
            print("Failed to download from Hugging Face: \(error)")
            // Clean up partial downloads
            try? FileManager.default.removeItem(at: cacheDir)
        }

        return nil
    }

    /// Get VAD dataset cache directory
    static func getVadDatasetCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let cacheDir = appSupport.appendingPathComponent(
            "FluidAudio/vadDataset", isDirectory: true)

        try? FileManager.default.createDirectory(
            at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    static func runVadBenchmarkInternal(
        vadManager: VadManager, testFiles: [VadTestFile], threshold: Float
    ) async throws -> VadBenchmarkResult {
        print("\n🔍 Running VAD benchmark on \(testFiles.count) files...")

        let startTime = Date()
        var predictions: [Int] = []
        var groundTruth: [Int] = []
        var fileDurations: [TimeInterval] = []
        var loadingTime: TimeInterval = 0
        var inferenceTime: TimeInterval = 0
        var totalAudioDuration: TimeInterval = 0

        for (index, testFile) in testFiles.enumerated() {
            let fileStartTime = Date()
            print("   Processing \(index + 1)/\(testFiles.count): \(testFile.name)")

            do {
                // Load audio file with optimized loading
                let loadStartTime = Date()
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioData = try await loadVadAudioData(audioFile)
                loadingTime += Date().timeIntervalSince(loadStartTime)

                // Calculate audio duration
                let audioDuration =
                    Double(audioFile.length) / audioFile.processingFormat.sampleRate
                totalAudioDuration += audioDuration

                // Process with VAD
                let inferenceStartTime = Date()
                let vadResults = try await vadManager.processAudioFile(audioData)
                inferenceTime += Date().timeIntervalSince(inferenceStartTime)

                // Free audio data immediately after processing
                // This helps with GitHub Actions memory constraints

                // Aggregate results (use max probability as file-level decision)
                let maxProbability = vadResults.map { $0.probability }.max() ?? 0.0
                let prediction = maxProbability >= threshold ? 1 : 0

                predictions.append(prediction)
                groundTruth.append(testFile.expectedLabel)

                let fileProcessingTime = Date().timeIntervalSince(fileStartTime)
                fileDurations.append(fileProcessingTime)

                // Calculate RTFx for this file
                let fileRTFx = audioDuration > 0 ? fileProcessingTime / audioDuration : 0
                let rtfxDisplay =
                    if fileRTFx < 1.0 && fileRTFx > 0 {
                        String(format: "%.1fx", 1.0 / fileRTFx)
                    } else {
                        String(format: "%.1fx", fileRTFx)
                    }

                print(
                    "      Result: max_prob=\(String(format: "%.3f", maxProbability)), prediction=\(prediction), expected=\(testFile.expectedLabel), time=\(String(format: "%.3f", fileProcessingTime))s, RTFx=\(rtfxDisplay)"
                )

            } catch {
                print("      Error: \(error)")
                // Use default prediction on error
                predictions.append(0)
                groundTruth.append(testFile.expectedLabel)
                fileDurations.append(Date().timeIntervalSince(fileStartTime))
            }
        }

        let processingTime = Date().timeIntervalSince(startTime)

        // Calculate metrics
        let metrics = calculateVadMetrics(predictions: predictions, groundTruth: groundTruth)

        // Calculate timing statistics
        let avgProcessingTime =
            fileDurations.isEmpty ? 0 : fileDurations.reduce(0, +) / Double(fileDurations.count)
        let minProcessingTime = fileDurations.min() ?? 0
        let maxProcessingTime = fileDurations.max() ?? 0

        // Calculate RTFx (Real-Time Factor)
        let rtfx = totalAudioDuration > 0 ? processingTime / totalAudioDuration : 0

        print("\n⏱️ Timing Statistics:")
        print("   Total processing time: \(String(format: "%.2f", processingTime))s")
        print("   Total audio duration: \(String(format: "%.2f", totalAudioDuration))s")
        if rtfx < 1.0 && rtfx > 0 {
            print("   RTFx: \(String(format: "%.1f", 1.0/rtfx))x faster than real-time")
        } else if rtfx >= 1.0 {
            print("   RTFx: \(String(format: "%.1f", rtfx))x slower than real-time")
        } else {
            print("   RTFx: N/A")
        }
        print(
            "   Audio loading time: \(String(format: "%.2f", loadingTime))s (\(String(format: "%.1f", loadingTime/processingTime*100))%)"
        )
        print(
            "   VAD inference time: \(String(format: "%.2f", inferenceTime))s (\(String(format: "%.1f", inferenceTime/processingTime*100))%)"
        )
        print("   Average per file: \(String(format: "%.3f", avgProcessingTime))s")
        print("   Min per file: \(String(format: "%.3f", minProcessingTime))s")
        print("   Max per file: \(String(format: "%.3f", maxProcessingTime))s")

        return VadBenchmarkResult(
            testName: "VAD_Benchmark_\(testFiles.count)_Files",
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1Score: metrics.f1Score,
            processingTime: processingTime,
            totalFiles: testFiles.count,
            correctPredictions: zip(predictions, groundTruth).filter { $0 == $1 }.count
        )
    }

    static func loadVadAudioData(_ audioFile: AVAudioFile) async throws -> [Float] {
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        // Early exit if already 16kHz - avoid resampling overhead
        let needsResampling = format.sampleRate != 16000

        // Use smaller buffer size for GitHub Actions memory constraints
        let bufferSize: AVAudioFrameCount = min(frameCount, 4096)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: bufferSize) else {
            throw NSError(domain: "AudioError", code: 1, userInfo: nil)
        }

        var allSamples: [Float] = []
        allSamples.reserveCapacity(Int(frameCount))

        // Read file in chunks to reduce memory pressure
        var remainingFrames = frameCount

        while remainingFrames > 0 {
            let framesToRead = min(remainingFrames, bufferSize)
            buffer.frameLength = 0  // Reset buffer

            try audioFile.read(into: buffer, frameCount: framesToRead)

            guard let floatData = buffer.floatChannelData?[0] else {
                throw NSError(domain: "AudioError", code: 2, userInfo: nil)
            }

            let actualFrameCount = Int(buffer.frameLength)
            if actualFrameCount == 0 { break }

            // Direct append without intermediate array creation
            let bufferPointer = UnsafeBufferPointer(start: floatData, count: actualFrameCount)
            allSamples.append(contentsOf: bufferPointer)

            remainingFrames -= AVAudioFrameCount(actualFrameCount)
        }

        // Resample to 16kHz if needed
        if needsResampling {
            allSamples = try await AudioProcessor.resampleAudio(
                allSamples, from: format.sampleRate, to: 16000)
        }

        return allSamples
    }

    static func calculateVadMetrics(
        predictions: [Int], groundTruth: [Int]
    ) -> (
        accuracy: Float, precision: Float, recall: Float, f1Score: Float
    ) {
        guard predictions.count == groundTruth.count && !predictions.isEmpty else {
            return (0, 0, 0, 0)
        }

        var truePositives = 0
        var falsePositives = 0
        var trueNegatives = 0
        var falseNegatives = 0

        for (pred, truth) in zip(predictions, groundTruth) {
            switch (pred, truth) {
            case (1, 1): truePositives += 1
            case (1, 0): falsePositives += 1
            case (0, 0): trueNegatives += 1
            case (0, 1): falseNegatives += 1
            default: break
            }
        }

        let accuracy = Float(truePositives + trueNegatives) / Float(predictions.count) * 100
        let precision =
            truePositives + falsePositives > 0
            ? Float(truePositives) / Float(truePositives + falsePositives) * 100 : 0
        let recall =
            truePositives + falseNegatives > 0
            ? Float(truePositives) / Float(truePositives + falseNegatives) * 100 : 0
        let f1Score =
            precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0

        return (accuracy, precision, recall, f1Score)
    }

    static func calculateRTFx(
        result: VadBenchmarkResult, testFiles: [VadTestFile]
    ) async throws
        -> Double
    {
        var totalAudioDuration: TimeInterval = 0

        for testFile in testFiles {
            do {
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioDuration =
                    Double(audioFile.length) / audioFile.processingFormat.sampleRate
                totalAudioDuration += audioDuration
            } catch {
                // Skip files that can't be read
                continue
            }
        }

        return totalAudioDuration > 0 ? result.processingTime / totalAudioDuration : 0
    }

    /// Load VOiCES subset dataset
    static func loadVoicesSubset(count: Int) async throws -> [VadTestFile]? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let voicesDir = appSupport.appendingPathComponent("FluidAudio/voicesSubset")

        guard FileManager.default.fileExists(atPath: voicesDir.path) else {
            print(
                "VOiCES subset not found. Run: swift run fluidaudio download --dataset voices-subset"
            )
            return nil
        }

        print("🗂️ Loading VOiCES subset with mixed speech/non-speech samples...")

        var testFiles: [VadTestFile] = []

        // Load clean and noisy speech files
        let cleanDir = voicesDir.appendingPathComponent("clean")
        let noisyDir = voicesDir.appendingPathComponent("noisy")

        // For a balanced VAD test, we need both speech and non-speech samples
        // VOiCES only has speech, so we'll also load non-speech from MUSAN
        let requestedSpeechCount = count == -1 ? 25 : count / 2

        if FileManager.default.fileExists(atPath: cleanDir.path) {
            let cleanFiles = try loadAudioFiles(
                from: cleanDir, expectedLabel: 1, maxCount: requestedSpeechCount / 2)
            testFiles.append(contentsOf: cleanFiles)
            print("   Loaded \(cleanFiles.count) clean speech files")
        }

        if FileManager.default.fileExists(atPath: noisyDir.path) {
            let noisyFiles = try loadAudioFiles(
                from: noisyDir, expectedLabel: 1, maxCount: requestedSpeechCount / 2)
            testFiles.append(contentsOf: noisyFiles)
            print("   Loaded \(noisyFiles.count) noisy speech files")
        }

        // Load non-speech samples from MUSAN mini dataset
        print("   📥 Loading non-speech samples from MUSAN...")
        let vadCacheDir = appSupport.appendingPathComponent("FluidAudio/vadDataset")
        let noiseDir = vadCacheDir.appendingPathComponent("noise")

        if FileManager.default.fileExists(atPath: noiseDir.path) {
            let requestedNoiseCount = count == -1 ? 25 : count - testFiles.count
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0, maxCount: requestedNoiseCount)
            testFiles.append(contentsOf: noiseFiles)
            print("   Loaded \(noiseFiles.count) non-speech files from MUSAN")
        } else {
            // If MUSAN noise samples aren't available, download them
            print("   🌐 Downloading non-speech samples from MUSAN...")
            if let musanFiles = try await downloadHuggingFaceVadDataset(
                count: testFiles.count, dataset: "mini50")
            {
                // Filter only non-speech samples
                let nonSpeechFiles = musanFiles.filter { $0.expectedLabel == 0 }
                testFiles.append(contentsOf: nonSpeechFiles)
                print("   Downloaded \(nonSpeechFiles.count) non-speech files")
            }
        }

        if testFiles.isEmpty {
            return nil
        }

        // Shuffle to mix speech and non-speech samples
        testFiles.shuffle()

        print("📂 Using VOiCES + MUSAN mixed dataset: \(testFiles.count) files total")
        print("   Speech samples: \(testFiles.filter { $0.expectedLabel == 1 }.count)")
        print("   Non-speech samples: \(testFiles.filter { $0.expectedLabel == 0 }.count)")
        print("💡 This tests VAD robustness in real-world acoustic conditions")
        return testFiles
    }

    /// Load full MUSAN dataset
    static func loadFullMusanDataset(count: Int) async throws -> [VadTestFile]? {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let musanDir = appSupport.appendingPathComponent("FluidAudio/musanFull/musan")

        guard FileManager.default.fileExists(atPath: musanDir.path) else {
            print(
                "Full MUSAN dataset not found. Run: swift run fluidaudio download --dataset musan-full"
            )
            return nil
        }

        print("🗂️ Loading full MUSAN dataset...")

        var testFiles: [VadTestFile] = []

        // Load speech files
        let speechDir = musanDir.appendingPathComponent("speech")
        if FileManager.default.fileExists(atPath: speechDir.path) {
            let speechFiles = try loadAudioFiles(
                from: speechDir, expectedLabel: 1, maxCount: count == -1 ? Int.max : count / 3)
            testFiles.append(contentsOf: speechFiles)
            print("   Loaded \(speechFiles.count) speech files")
        }

        // Load music files (treat as non-speech for VAD)
        let musicDir = musanDir.appendingPathComponent("music")
        if FileManager.default.fileExists(atPath: musicDir.path) {
            let musicFiles = try loadAudioFiles(
                from: musicDir, expectedLabel: 0, maxCount: count == -1 ? Int.max : count / 3)
            testFiles.append(contentsOf: musicFiles)
            print("   Loaded \(musicFiles.count) music files")
        }

        // Load noise files
        let noiseDir = musanDir.appendingPathComponent("noise")
        if FileManager.default.fileExists(atPath: noiseDir.path) {
            let noiseFiles = try loadAudioFiles(
                from: noiseDir, expectedLabel: 0,
                maxCount: count == -1 ? Int.max : count - testFiles.count)
            testFiles.append(contentsOf: noiseFiles)
            print("   Loaded \(noiseFiles.count) noise files")
        }

        if testFiles.isEmpty {
            return nil
        }

        print("📂 Using full MUSAN dataset: \(testFiles.count) files total")
        return testFiles.shuffled()  // Shuffle to mix different types
    }

    static func saveVadBenchmarkResultsWithRTFx(
        _ result: VadBenchmarkResult, testFiles: [VadTestFile], to file: String
    ) async throws {
        var totalAudioDuration: TimeInterval = 0

        for testFile in testFiles {
            do {
                let audioFile = try AVAudioFile(forReading: testFile.url)
                let audioDuration =
                    Double(audioFile.length) / audioFile.processingFormat.sampleRate
                totalAudioDuration += audioDuration
            } catch {
                // Skip files that can't be read
                continue
            }
        }

        let rtfx = totalAudioDuration > 0 ? result.processingTime / totalAudioDuration : 0

        let resultsDict: [String: Any] = [
            "test_name": result.testName,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1Score,
            "processing_time_seconds": result.processingTime,
            "total_audio_duration_seconds": totalAudioDuration,
            "rtfx": rtfx,
            "avg_time_per_file": result.processingTime / Double(result.totalFiles),
            "total_files": result.totalFiles,
            "correct_predictions": result.correctPredictions,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "environment": "CLI",
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: resultsDict, options: .prettyPrinted)
        try jsonData.write(to: URL(fileURLWithPath: file))
    }
}

#endif
