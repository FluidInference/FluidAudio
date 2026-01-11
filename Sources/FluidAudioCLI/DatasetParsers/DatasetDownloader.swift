#if os(macOS)
import AVFoundation
import Foundation
import FluidAudio

/// Dataset downloading functionality for AMI and VAD datasets
struct DatasetDownloader {
    private static let logger = AppLogger(category: "Dataset")

    enum AMIVariant: String, CaseIterable {
        case sdm = "sdm"  // Single Distant Microphone (Mix-Headset.wav)
        case ihm = "ihm"  // Individual Headset Microphones (Headset-0.wav)

        var displayName: String {
            switch self {
            case .sdm: return "Single Distant Microphone"
            case .ihm: return "Individual Headset Microphones"
            }
        }

        var filePattern: String {
            switch self {
            case .sdm: return "Mix-Headset.wav"
            case .ihm: return "Headset-0.wav"
            }
        }
    }

    static func downloadAMIDataset(
        variant: AMIVariant, force: Bool, singleFile: String? = nil
    )
        async
    {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let baseDir = homeDir.appendingPathComponent("FluidAudioDatasets")
        let amiDir = baseDir.appendingPathComponent("ami_official")
        let variantDir = amiDir.appendingPathComponent(variant.rawValue)

        // Create directories if needed
        do {
            try FileManager.default.createDirectory(
                at: variantDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directory: \(error)")
            return
        }

        logger.info("📥 Downloading AMI \(variant.displayName) dataset...")
        logger.info("   Target directory: \(variantDir.path)")

        // Download AMI annotations first (required for proper benchmarking)
        await downloadAMIAnnotations(force: force)

        // Official AMI SDM test set (16 meetings) - matches NeMo evaluation
        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
            logger.info("📋 Downloading single file: \(singleFile)")
        } else {
            commonMeetings = [
                // Full 16-meeting AMI SDM test set
                "EN2002a", "EN2002b", "EN2002c", "EN2002d",
                "ES2004a", "ES2004b", "ES2004c", "ES2004d",
                "IS1009a", "IS1009b", "IS1009c", "IS1009d",
                "TS3003a", "TS3003b", "TS3003c", "TS3003d",
            ]
            logger.info("📋 Downloading official AMI SDM test set (16 meetings)")
        }

        var downloadedFiles = 0
        var skippedFiles = 0

        for meetingId in commonMeetings {
            let fileName = "\(meetingId).\(variant.filePattern)"
            let filePath = variantDir.appendingPathComponent(fileName)

            // Skip if file exists and not forcing download
            if !force && FileManager.default.fileExists(atPath: filePath.path) {
                logger.info("   ⏭️ Skipping \(fileName) (already exists)")
                skippedFiles += 1
                continue
            }

            // Try to download from AMI corpus mirror
            let success = await downloadAMIFile(
                meetingId: meetingId,
                variant: variant,
                outputPath: filePath
            )

            if success {
                downloadedFiles += 1
                logger.info("   Downloaded \(fileName)")
            } else {
                logger.warning("   Failed to download \(fileName)")
            }
        }

        logger.info("🎉 AMI \(variant.displayName) download completed")
        logger.info("   Downloaded: \(downloadedFiles) files")
        logger.info("   Skipped: \(skippedFiles) files")
        logger.info("   Total files: \(downloadedFiles + skippedFiles)/\(commonMeetings.count)")

        if downloadedFiles == 0 && skippedFiles == 0 {
            logger.warning("⚠️ No files were downloaded. You may need to download manually from:")
            logger.warning("   https://groups.inf.ed.ac.uk/ami/download/")
        }
    }

    static func downloadAMIFile(
        meetingId: String, variant: AMIVariant, outputPath: URL
    ) async
        -> Bool
    {
        // Try multiple URL patterns - the AMI corpus mirror structure has some variations
        let baseURLs = [
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Double slash pattern (from user's working example)
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus",  // Single slash pattern
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Alternative with extra slash
        ]

        for (_, baseURL) in baseURLs.enumerated() {
            let urlString = "\(baseURL)/\(meetingId)/audio/\(meetingId).\(variant.filePattern)"

            guard let url = URL(string: urlString) else {
                logger.error("     ⚠️ Invalid URL: \(urlString)")
                continue
            }

            do {
                logger.info("     📥 Downloading from: \(urlString)")
                let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: outputPath)

                        // Verify it's a valid audio file
                        if await isValidAudioFile(outputPath) {
                            let fileSizeMB = Double(data.count) / (1024 * 1024)
                            logger.info("     Downloaded \(String(format: "%.1f", fileSizeMB)) MB")
                            return true
                        } else {
                            logger.warning("     ⚠️ Downloaded file is not valid audio")
                            try? FileManager.default.removeItem(at: outputPath)
                            // Try next URL
                            continue
                        }
                    } else if httpResponse.statusCode == 404 {
                        logger.warning("     ⚠️ File not found (HTTP 404) - trying next URL...")
                        continue
                    } else {
                        logger.warning("     ⚠️ HTTP error: \(httpResponse.statusCode) - trying next URL...")
                        continue
                    }
                }
            } catch {
                logger.warning("     ⚠️ Download error: \(error.localizedDescription) - trying next URL...")
                continue
            }
        }

        logger.error("     Failed to download from all available URLs")
        return false
    }

    static func isValidAudioFile(_ url: URL) async -> Bool {
        do {
            let _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }

    /// Download AMI annotations to working directory for benchmarking
    static func downloadAMIAnnotations(force: Bool = false) async {
        let workingDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let annotationsDir = workingDir.appendingPathComponent("Datasets/ami_public_1.6.2")

        // Check if annotations already exist
        let segmentsDir = annotationsDir.appendingPathComponent("segments")
        let meetingsFile = annotationsDir.appendingPathComponent("corpusResources/meetings.xml")

        if !force && FileManager.default.fileExists(atPath: segmentsDir.path)
            && FileManager.default.fileExists(atPath: meetingsFile.path)
        {
            logger.info("📂 AMI annotations already exist in \(annotationsDir.path)")
            return
        }

        logger.info("📥 Downloading AMI annotations from Edinburgh University...")
        logger.info("   Target directory: \(annotationsDir.path)")

        // Create required directories
        do {
            try FileManager.default.createDirectory(
                at: annotationsDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create annotation directories: \(error)")
            return
        }

        // Download and extract AMI manual annotations v1.6.2
        let zipURL =
            "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        let zipFile = annotationsDir.appendingPathComponent("ami_public_manual_1.6.2.zip")

        logger.info("📥 Downloading AMI manual annotations archive (22MB)...")
        let zipSuccess = await downloadAnnotationFile(from: zipURL, to: zipFile)

        if !zipSuccess {
            logger.error("Failed to download AMI annotations archive")
            return
        }

        logger.info("📦 Extracting AMI annotations archive...")

        // Extract the ZIP file using the system unzip command
        let extractSuccess = await extractZipFile(zipFile, to: annotationsDir)

        if extractSuccess {
            // Clean up ZIP file
            try? FileManager.default.removeItem(at: zipFile)

            // Verify extraction was successful
            if FileManager.default.fileExists(atPath: segmentsDir.path)
                && FileManager.default.fileExists(atPath: meetingsFile.path)
            {
                logger.info("AMI annotations download and extraction completed")
                logger.info("💡 Benchmarks will now use real AMI ground truth data")
            } else {
                logger.warning("⚠️ Extraction completed but expected files not found")
                logger.warning("   Looking for: \(segmentsDir.path)")
                logger.warning("   Looking for: \(meetingsFile.path)")
            }
        } else {
            logger.error("Failed to extract AMI annotations archive")
        }
    }

    /// Download a single annotation file from AMI corpus
    static func downloadAnnotationFile(from urlString: String, to outputPath: URL) async -> Bool {
        guard let url = URL(string: urlString) else {
            logger.error("     ⚠️ Invalid URL: \(urlString)")
            return false
        }

        do {
            let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 200 {
                    try data.write(to: outputPath)

                    // Check if it's a ZIP file or XML file
                    if outputPath.pathExtension.lowercased() == "zip" {
                        // For ZIP files, just verify it's not empty
                        return data.count > 0
                    } else {
                        // Verify it's valid XML
                        if let xmlString = String(data: data, encoding: .utf8),
                            xmlString.contains("<?xml") || xmlString.contains("<nite:")
                        {
                            return true
                        } else {
                            logger.warning("     ⚠️ Downloaded file is not valid XML")
                            try? FileManager.default.removeItem(at: outputPath)
                            return false
                        }
                    }
                } else {
                    logger.warning("     ⚠️ HTTP error: \(httpResponse.statusCode)")
                    return false
                }
            }
        } catch {
            logger.warning("     ⚠️ Download error: \(error.localizedDescription)")
            return false
        }

        return false
    }

    /// Extract ZIP file using system unzip command
    static func extractZipFile(_ zipFile: URL, to targetDir: URL) async -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-q", "-o", zipFile.path, "-d", targetDir.path]

        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            logger.error("     ⚠️ Failed to extract ZIP file: \(error)")
            return false
        }
    }

    /// Download VAD dataset from Hugging Face
    static func downloadVadDataset(force: Bool, dataset: String = "mini50") async {
        let cacheDir = getVadDatasetCacheDirectory()

        logger.info("📥 Downloading VAD dataset from Hugging Face...")
        logger.info("   Target directory: \(cacheDir.path)")

        // Create cache directories
        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")

        do {
            try FileManager.default.createDirectory(
                at: speechDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: noiseDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create cache directories: \(error)")
            return
        }

        // Check if we should skip download
        if !force {
            let existingSpeechFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: speechDir, includingPropertiesForKeys: nil)) ?? []
            let existingNoiseFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: noiseDir, includingPropertiesForKeys: nil)) ?? []

            if !existingSpeechFiles.isEmpty && !existingNoiseFiles.isEmpty {
                logger.info("📂 VAD dataset already exists (use --force to re-download)")
                logger.info("   Speech files: \(existingSpeechFiles.count)")
                logger.info("   Noise files: \(existingNoiseFiles.count)")
                return
            }
        } else {
            // Force download - clean existing files
            try? FileManager.default.removeItem(at: speechDir)
            try? FileManager.default.removeItem(at: noiseDir)
            try? FileManager.default.createDirectory(
                at: speechDir, withIntermediateDirectories: true)
            try? FileManager.default.createDirectory(
                at: noiseDir, withIntermediateDirectories: true)
        }

        // Use specified dataset for download command
        let repoName = dataset == "mini100" ? "musan_mini100" : "musan_mini50"
        let repoBase = ModelRegistry.resolveDatasetBase("alexwengg/\(repoName)")

        var downloadedFiles = 0
        var failedFiles = 0

        // Download speech files
        logger.info("📢 Downloading speech samples...")
        let speechCount = dataset == "mini100" ? 50 : 25
        do {
            let speechFiles = try await downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/speech",
                targetDir: speechDir,
                expectedLabel: 1,
                count: speechCount,
                filePrefix: "speech",
                repoName: repoName
            )
            downloadedFiles += speechFiles.count
            logger.info("   Downloaded \(speechFiles.count) speech files")
        } catch {
            logger.error("   Failed to download speech files: \(error)")
            failedFiles += 1
        }

        // Download noise files
        logger.info("🔇 Downloading noise samples...")
        let noiseCount = dataset == "mini100" ? 50 : 25
        do {
            let noiseFiles = try await downloadVadFilesFromHF(
                baseUrl: "\(repoBase)/noise",
                targetDir: noiseDir,
                expectedLabel: 0,
                count: noiseCount,
                filePrefix: "noise",
                repoName: repoName
            )
            downloadedFiles += noiseFiles.count
            logger.info("   Downloaded \(noiseFiles.count) noise files")
        } catch {
            logger.error("   Failed to download noise files: \(error)")
            failedFiles += 1
        }

        logger.info("📊 VAD Dataset Download Summary:")
        logger.info("   Downloaded: \(downloadedFiles) files")
        logger.info("   Failed: \(failedFiles) categories")

        if downloadedFiles > 0 {
            logger.info("VAD dataset download completed")
            logger.info("💡 You can now run VAD benchmarks with the downloaded dataset")
        } else {
            logger.warning("No files were downloaded successfully")
            logger.warning("⚠️ VAD benchmarks will fall back to legacy URLs")
        }
    }

    /// Download VAD audio files from Hugging Face
    static func downloadVadFilesFromHF(
        baseUrl: String,
        targetDir: URL,
        expectedLabel: Int,
        count: Int,
        filePrefix: String,
        repoName: String
    ) async throws -> [VadTestFile] {
        var testFiles: [VadTestFile] = []

        // Get files directly from the directory (simplified structure in dataset)
        let repoApiUrl = try ModelRegistry.apiDatasets("alexwengg/\(repoName)", "tree/main/\(filePrefix)")
            .absoluteString
        var allFiles: [String] = []

        do {
            let directoryFiles = try await getHuggingFaceFileList(apiUrl: repoApiUrl)
            let audioFiles = directoryFiles.filter { fileName in
                let ext = URL(fileURLWithPath: fileName).pathExtension.lowercased()
                return ["wav", "mp3", "flac", "m4a"].contains(ext)
            }
            allFiles.append(contentsOf: audioFiles)
        } catch {
            logger.warning("      ⚠️ Could not access \(filePrefix): \(error)")
        }

        logger.info("      Found \(allFiles.count) audio files in \(filePrefix)/ directory")
        logger.debug("      Debug: requesting \(count) files from \(allFiles.count) available")

        if !allFiles.isEmpty {
            let filesToDownload = Array(allFiles.prefix(count))
            var downloadedCount = 0

            for fileName in filesToDownload {
                let fileUrl = "\(baseUrl)/\(fileName)"
                let destination = targetDir.appendingPathComponent(fileName)

                do {
                    let downloadedFile = try await downloadAudioFile(
                        from: fileUrl, to: destination)
                    testFiles.append(
                        VadTestFile(
                            name: fileName,
                            expectedLabel: expectedLabel,
                            url: downloadedFile
                        ))
                    downloadedCount += 1
                    logger.info("      Downloaded: \(fileName)")

                } catch {
                    logger.warning("      ⚠️ Failed to download \(fileName): \(error)")
                    continue
                }
            }
        } else {
            logger.warning("      No audio files found in subdirectories")
        }

        // If no files downloaded via API, try pattern-based download
        if testFiles.isEmpty {
            logger.warning(
                "      ⚠️ API method failed or no files found, trying pattern-based download...")

            // Fallback to pattern-based download
            let extensions = ["wav", "mp3", "flac"]
            let patterns = [
                // Common MUSAN file patterns
                "\(filePrefix)-music-",
                "\(filePrefix)-speech-",
                "\(filePrefix)-noise-",
                "musan-\(filePrefix)-",
                // Simple numbered patterns
                "\(filePrefix)-",
                "\(filePrefix)_",
            ]

            var downloadedCount = 0

            for pattern in patterns {
                if downloadedCount >= count { break }

                for i in 0..<(count * 2) {  // Try more files than needed
                    if downloadedCount >= count { break }

                    for ext in extensions {
                        if downloadedCount >= count { break }

                        let fileName = "\(pattern)\(String(format: "%04d", i)).\(ext)"
                        let fileUrl = "\(baseUrl)/\(fileName)"
                        let destination = targetDir.appendingPathComponent(fileName)

                        do {
                            let downloadedFile = try await downloadAudioFile(
                                from: fileUrl, to: destination)
                            testFiles.append(
                                VadTestFile(
                                    name: fileName,
                                    expectedLabel: expectedLabel,
                                    url: downloadedFile
                                ))
                            downloadedCount += 1
                            logger.info("      Downloaded: \(fileName)")

                        } catch {
                            // File doesn't exist, continue trying
                            continue
                        }
                    }
                }
            }
        }

        return testFiles
    }

    /// Get file list from HuggingFace using the API
    static func getHuggingFaceFileList(apiUrl: String) async throws -> [String] {
        guard let url = URL(string: apiUrl) else {
            throw NSError(
                domain: "APIError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid API URL"])
        }

        let (data, _) = try await DownloadUtils.sharedSession.data(from: url)

        // Parse the JSON response to extract file names
        if let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return json.compactMap { item in
                if let type = item["type"] as? String,
                    let path = item["path"] as? String,
                    type == "file"
                {
                    // Extract just the filename from the path
                    return URL(fileURLWithPath: path).lastPathComponent
                }
                return nil
            }
        }

        return []
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

    static func downloadAudioFile(
        from urlString: String, to destination: URL
    ) async throws
        -> URL
    {
        // Skip if already exists
        if FileManager.default.fileExists(atPath: destination.path) {
            return destination
        }

        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "DownloadError", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"])
        }

        // Create destination directory
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // Download using URLSession
        let (data, _) = try await DownloadUtils.sharedSession.data(from: url)
        try data.write(to: destination)

        // Verify it's valid audio
        do {
            let _ = try AVAudioFile(forReading: destination)
        } catch {
            throw NSError(
                domain: "DownloadError", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Downloaded file is not valid audio"])
        }

        return destination
    }

    /// Download full MUSAN dataset from OpenSLR
    static func downloadFullMusanDataset(force: Bool) async {
        let cacheDir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        .appendingPathComponent("FluidAudio/musanFull", isDirectory: true)

        logger.info("📥 Downloading full MUSAN dataset from OpenSLR...")
        logger.info("   Target directory: \(cacheDir.path)")
        logger.info("   Expected size: ~600MB compressed, ~4.5GB uncompressed")

        // Create cache directory
        do {
            try FileManager.default.createDirectory(
                at: cacheDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create cache directory: \(error)")
            return
        }

        // Check if already downloaded
        let musanDir = cacheDir.appendingPathComponent("musan")
        if !force && FileManager.default.fileExists(atPath: musanDir.path) {
            let subdirs = ["speech", "music", "noise"]
            var allExist = true
            for subdir in subdirs {
                if !FileManager.default.fileExists(
                    atPath: musanDir.appendingPathComponent(subdir).path)
                {
                    allExist = false
                    break
                }
            }

            if allExist {
                logger.info("📂 Full MUSAN dataset already exists (use --force to re-download)")
                logger.info("💡 Run: swift run fluidaudio vad-benchmark --dataset musan-full")
                return
            }
        }

        // Download from OpenSLR
        let musanURL = "https://www.openslr.org/resources/17/musan.tar.gz"
        let downloadPath = cacheDir.appendingPathComponent("musan.tar.gz")

        logger.info("🌐 Downloading from: \(musanURL)")

        do {
            // Download the tar.gz file
            let (downloadURL, response) = try await DownloadUtils.sharedSession.download(
                from: URL(string: musanURL)!)

            // Check response
            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode != 200 {
                logger.error("Download failed with status code: \(httpResponse.statusCode)")
                return
            }

            // Move downloaded file
            try FileManager.default.moveItem(at: downloadURL, to: downloadPath)
            logger.info("Download complete")

            // Extract tar.gz
            logger.info("📦 Extracting archive...")
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            task.arguments = ["-xzf", downloadPath.path, "-C", cacheDir.path]

            try task.run()
            task.waitUntilExit()

            if task.terminationStatus == 0 {
                logger.info("Extraction complete")

                // Clean up tar file
                try? FileManager.default.removeItem(at: downloadPath)

                // Count files
                let speechFiles = countFiles(in: musanDir.appendingPathComponent("speech"))
                let musicFiles = countFiles(in: musanDir.appendingPathComponent("music"))
                let noiseFiles = countFiles(in: musanDir.appendingPathComponent("noise"))

                logger.info("📊 Full MUSAN Dataset Summary:")
                logger.info("   Speech files: \(speechFiles)")
                logger.info("   Music files: \(musicFiles)")
                logger.info("   Noise files: \(noiseFiles)")
                logger.info("   Total files: \(speechFiles + musicFiles + noiseFiles)")
                logger.info("Full MUSAN dataset ready for benchmarking")
                logger.info("💡 Run: swift run fluidaudio vad-benchmark --dataset musan-full")
            } else {
                logger.error("Extraction failed")
                try? FileManager.default.removeItem(at: downloadPath)
            }

        } catch {
            logger.error("Download failed: \(error)")
            try? FileManager.default.removeItem(at: downloadPath)
        }
    }

    /// Count files recursively in a directory
    private static func countFiles(in directory: URL) -> Int {
        var count = 0
        if let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        {
            for case let fileURL as URL in enumerator {
                if let isFile = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    .isRegularFile, isFile
                {
                    count += 1
                }
            }
        }
        return count
    }

    /// Download VOiCES subset dataset from GitHub
    static func downloadVoicesSubset(force: Bool) async {
        let cacheDir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        .appendingPathComponent("FluidAudio/voicesSubset", isDirectory: true)

        logger.info("📥 Downloading VOiCES subset from GitHub...")
        logger.info("   Target directory: \(cacheDir.path)")

        // Create cache directory
        do {
            try FileManager.default.createDirectory(
                at: cacheDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create cache directory: \(error)")
            return
        }

        // Check if already downloaded
        let cleanDir = cacheDir.appendingPathComponent("clean")
        let noisyDir = cacheDir.appendingPathComponent("noisy")

        if !force && FileManager.default.fileExists(atPath: cleanDir.path)
            && FileManager.default.fileExists(atPath: noisyDir.path)
        {
            let cleanFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: cleanDir, includingPropertiesForKeys: nil)) ?? []
            let noisyFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: noisyDir, includingPropertiesForKeys: nil)) ?? []

            if !cleanFiles.isEmpty || !noisyFiles.isEmpty {
                logger.info("📂 VOiCES subset already exists (use --force to re-download)")
                logger.info("   Clean files: \(cleanFiles.count)")
                logger.info("   Noisy files: \(noisyFiles.count)")
                logger.info("💡 Run: swift run fluidaudio vad-benchmark --dataset voices-subset")
                return
            }
        }

        // Clone the repository
        logger.info("🌐 Cloning VOiCES-subset repository...")
        let cloneDir = cacheDir.appendingPathComponent("temp_clone")

        do {
            // Remove any existing temp directory
            try? FileManager.default.removeItem(at: cloneDir)

            // Clone the repository
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            task.arguments = [
                "clone", "--depth", "1", "https://github.com/Lab41/VOiCES-subset.git",
                cloneDir.path,
            ]

            try task.run()
            task.waitUntilExit()

            if task.terminationStatus == 0 {
                logger.info("Repository cloned successfully")

                // Move the audio files to our cache structure
                let sourceCleanDir = cloneDir.appendingPathComponent("clean")
                let sourceNoisyDir = cloneDir.appendingPathComponent("noisy")

                // Create destination directories
                try FileManager.default.createDirectory(
                    at: cleanDir, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(
                    at: noisyDir, withIntermediateDirectories: true)

                // Move clean files
                if FileManager.default.fileExists(atPath: sourceCleanDir.path) {
                    let cleanFiles = try FileManager.default.contentsOfDirectory(
                        at: sourceCleanDir, includingPropertiesForKeys: nil)
                    for file in cleanFiles where file.pathExtension == "wav" {
                        let destination = cleanDir.appendingPathComponent(
                            file.lastPathComponent)
                        try FileManager.default.moveItem(at: file, to: destination)
                    }
                    logger.info(
                        "   Moved \(cleanFiles.filter { $0.pathExtension == "wav" }.count) clean files"
                    )
                }

                // Move noisy files
                if FileManager.default.fileExists(atPath: sourceNoisyDir.path) {
                    let noisyFiles = try FileManager.default.contentsOfDirectory(
                        at: sourceNoisyDir, includingPropertiesForKeys: nil)
                    for file in noisyFiles where file.pathExtension == "wav" {
                        let destination = noisyDir.appendingPathComponent(
                            file.lastPathComponent)
                        try FileManager.default.moveItem(at: file, to: destination)
                    }
                    logger.info(
                        "   Moved \(noisyFiles.filter { $0.pathExtension == "wav" }.count) noisy files"
                    )
                }

                // Clean up clone directory
                try? FileManager.default.removeItem(at: cloneDir)

                logger.info("VOiCES subset ready for benchmarking")
                logger.info("💡 Run: swift run fluidaudio vad-benchmark --dataset voices-subset")

            } else {
                logger.error("Git clone failed")
                try? FileManager.default.removeItem(at: cloneDir)
            }

        } catch {
            logger.error("Download failed: \(error)")
            try? FileManager.default.removeItem(at: cloneDir)
        }
    }
}

#endif
