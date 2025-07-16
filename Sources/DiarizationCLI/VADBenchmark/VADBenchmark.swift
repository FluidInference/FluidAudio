#if os(macOS)
import AVFoundation
import Foundation
import FluidAudio

public struct VADBenchmark {

public static func runVadBenchmark(arguments: [String]) async {
    do {
        try await runVadBenchmarkWithErrorHandling(arguments: arguments)
    } catch {
        print("❌ VAD Benchmark failed: \(error)")
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

    let VadManager = VadManager(
        config: VadConfig(
            threshold: vadThreshold,
            chunkSize: 512,
            debugMode: true
        ))

    // VAD models will be automatically downloaded from Hugging Face if needed
    print("🔄 VAD models will be auto-downloaded from Hugging Face if needed")

    do {
        print("🔧 Initializing VAD manager...")
        try await VadManager.initialize()
        print("✅ VAD system initialized")
    } catch {
        print("❌ Failed to initialize VAD: \(error)")
        print("   Error type: \(type(of: error))")
        if let vadError = error as? VadError {
            print("   VAD Error: \(vadError.localizedDescription)")
        }
        print("   Make sure VAD models are available in vadCoreml/ or cached directory")
        throw error
    }

    // Download test files
    let testFiles = try await downloadVadTestFiles(
        count: useAllFiles ? -1 : numFiles, dataset: dataset)

    // Run benchmark
    let result = try await runVadBenchmarkInternal(
        VadManager: VadManager, testFiles: testFiles, threshold: vadThreshold)

    // Print results
    print("\n📊 VAD Benchmark Results:")
    print("   Accuracy: \(String(format: "%.1f", result.accuracy))%")
    print("   Precision: \(String(format: "%.1f", result.precision))%")
    print("   Recall: \(String(format: "%.1f", result.recall))%")
    print("   F1-Score: \(String(format: "%.1f", result.f1Score))%")
    print("   Processing Time: \(String(format: "%.2f", result.processingTime))s")
    print("   Files Processed: \(result.totalFiles)")

    // Save results
    if let outputFile = outputFile {
        try OutputFormatter.saveVadBenchmarkResults(result, to: outputFile)
        print("💾 Results saved to: \(outputFile)")
    } else {
        try OutputFormatter.saveVadBenchmarkResults(
            result, to: "vad_benchmark_results.json")
        print("💾 Results saved to: vad_benchmark_results.json")
    }

    // Performance assessment
    if result.f1Score >= 70.0 {
        print("\n✅ EXCELLENT: F1-Score above 70%")
    } else if result.f1Score >= 60.0 {
        print("\n⚠️ ACCEPTABLE: F1-Score above 60%")
    } else {
        print("\n❌ NEEDS IMPROVEMENT: F1-Score below 60%")
        // Don't exit - just report the poor performance
    }
}

static func downloadVadTestFiles(count: Int, dataset: String = "mini50") async throws
    -> [VadTestFile]
{
    if count == -1 {
        print("📥 Loading all available test audio files...")
    } else {
        print("📥 Loading \(count) test audio files...")
    }

    // First try to load from local dataset directory
    if let localFiles = try await loadLocalDataset(count: count) {
        return localFiles
    }

    // Second, try to load from Hugging Face cache
    if let cachedFiles = try await loadHuggingFaceVadDataset(count: count, dataset: dataset)
    {
        return cachedFiles
    }

    // Finally, download from Hugging Face
    print("🌐 Downloading VAD dataset from Hugging Face...")
    if let hfFiles = try await downloadHuggingFaceVadDataset(count: count, dataset: dataset)
    {
        return hfFiles
    }

    // No fallback to mock data - fail cleanly
    print("❌ Failed to load VAD dataset from all sources:")
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
            print("   ✅ Loaded \(speechFiles.count) speech files")
        }

        if FileManager.default.fileExists(atPath: nonSpeechDir.path) {
            let maxNoiseFiles = count == -1 ? Int.max : count - testFiles.count
            let nonSpeechFiles = try loadAudioFiles(
                from: nonSpeechDir, expectedLabel: 0, maxCount: maxNoiseFiles)
            testFiles.append(contentsOf: nonSpeechFiles)
            print("   ✅ Loaded \(nonSpeechFiles.count) non-speech files")
        }

        if !testFiles.isEmpty {
            print("📂 Using local dataset: \(testFiles.count) files total")
            return testFiles
        }
    }

    return nil
}

static func loadAudioFiles(from directory: URL, expectedLabel: Int, maxCount: Int) throws
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
static func loadHuggingFaceVadDataset(count: Int, dataset: String = "mini50") async throws
    -> [VadTestFile]?
{
    let cacheDir = DatasetDownloader.getVadDatasetCacheDirectory()

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
        print("🗂️ Found cached Hugging Face dataset: \(maxFilesForDataset) files total")

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
static func downloadHuggingFaceVadDataset(count: Int, dataset: String = "mini50")
    async throws -> [VadTestFile]?
{
    let cacheDir = DatasetDownloader.getVadDatasetCacheDirectory()

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
            print("✅ Downloaded VAD dataset from Hugging Face: \(testFiles.count) files")
            return testFiles
        }

    } catch {
        print("❌ Failed to download from Hugging Face: \(error)")
        // Clean up partial downloads
        try? FileManager.default.removeItem(at: cacheDir)
    }

    return nil
}

static func runVadBenchmarkInternal(
    VadManager: VadManager, testFiles: [VadTestFile], threshold: Float
) async throws -> VadBenchmarkResult {
    print("\n🔍 Running VAD benchmark on \(testFiles.count) files...")

    let startTime = Date()
    var predictions: [Int] = []
    var groundTruth: [Int] = []

    for (index, testFile) in testFiles.enumerated() {
        print("   Processing \(index + 1)/\(testFiles.count): \(testFile.name)")

        do {
            // Load audio file with optimized loading
            let audioFile = try AVAudioFile(forReading: testFile.url)
            let audioData = try await AudioProcessing.loadVadAudioData(audioFile)

            // Process with VAD
            let vadResults = try await VadManager.processAudioFile(audioData)

            // Free audio data immediately after processing
            // This helps with GitHub Actions memory constraints

            // Aggregate results (use max probability as file-level decision)
            let maxProbability = vadResults.map { $0.probability }.max() ?? 0.0
            let prediction = maxProbability >= threshold ? 1 : 0

            predictions.append(prediction)
            groundTruth.append(testFile.expectedLabel)

            print(
                "      Result: max_prob=\(String(format: "%.3f", maxProbability)), prediction=\(prediction), expected=\(testFile.expectedLabel)"
            )

        } catch {
            print("      ❌ Error: \(error)")
            // Use default prediction on error
            predictions.append(0)
            groundTruth.append(testFile.expectedLabel)
        }
    }

    let processingTime = Date().timeIntervalSince(startTime)

    // Calculate metrics
    let metrics = MetricsCalculator.calculateVadMetrics(
        predictions: predictions, groundTruth: groundTruth)

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
}
#endif
