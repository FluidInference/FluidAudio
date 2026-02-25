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

        logger.info("📥 Downloading AMI \(variant.displayName) to \(variantDir.path)")

        // Download AMI annotations first (required for proper benchmarking)
        await downloadAMIAnnotations(force: force)

        // Official AMI SDM test set (16 meetings) - matches NeMo evaluation
        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
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
            } else {
                logger.warning("Failed to download \(fileName)")
            }
        }

        logger.info("AMI \(variant.displayName): \(downloadedFiles) downloaded, \(skippedFiles) skipped")

        if downloadedFiles == 0 && skippedFiles == 0 {
            logger.warning("⚠️ No files downloaded. Manual download: https://groups.inf.ed.ac.uk/ami/download/")
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
                continue
            }

            do {
                let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: outputPath)

                        // Verify it's a valid audio file
                        if await isValidAudioFile(outputPath) {
                            return true
                        } else {
                            try? FileManager.default.removeItem(at: outputPath)
                            // Try next URL
                            continue
                        }
                    } else if httpResponse.statusCode == 404 {
                        continue
                    } else {
                        continue
                    }
                }
            } catch {
                continue
            }
        }

        logger.error("Failed to download \(meetingId) from all URLs")
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
            logger.info("📂 AMI annotations exist at \(annotationsDir.path)")
            return
        }

        logger.info("📥 Downloading AMI annotations to \(annotationsDir.path)")

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
        let zipSuccess = await downloadAnnotationFile(from: zipURL, to: zipFile)

        if !zipSuccess {
            logger.error("Failed to download AMI annotations")
            return
        }

        logger.info("📦 Extracting AMI annotations...")

        // Extract the ZIP file using the system unzip command
        let extractSuccess = await extractZipFile(zipFile, to: annotationsDir)

        if extractSuccess {
            // Clean up ZIP file
            try? FileManager.default.removeItem(at: zipFile)

            // Verify extraction was successful
            if FileManager.default.fileExists(atPath: segmentsDir.path)
                && FileManager.default.fileExists(atPath: meetingsFile.path)
            {
                logger.info("AMI annotations ready")
            } else {
                logger.warning("⚠️ Extraction completed but expected files not found")
            }
        } else {
            logger.error("Failed to extract AMI annotations")
        }
    }

    /// Download a single annotation file from AMI corpus
    static func downloadAnnotationFile(from urlString: String, to outputPath: URL) async -> Bool {
        guard let url = URL(string: urlString) else {
            return false
        }

        do {
            let (data, response) = try await DownloadUtils.sharedSession.data(from: url)

            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 200 {
                    try data.write(to: outputPath)

                    // Check if it's a ZIP file or XML file
                    if outputPath.pathExtension.lowercased() == "zip" {
                        return data.count > 0
                    } else {
                        // Verify it's valid XML
                        if let xmlString = String(data: data, encoding: .utf8),
                            xmlString.contains("<?xml") || xmlString.contains("<nite:")
                        {
                            return true
                        } else {
                            try? FileManager.default.removeItem(at: outputPath)
                            return false
                        }
                    }
                } else {
                    return false
                }
            }
        } catch {
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
            return false
        }
    }

    /// Download VAD dataset from Hugging Face
    static func downloadVadDataset(force: Bool, dataset: String = "mini50") async {
        let cacheDir = getVadDatasetCacheDirectory()

        logger.info("📥 Downloading VAD dataset to \(cacheDir.path)")

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
                logger.info(
                    "📂 VAD dataset exists (\(existingSpeechFiles.count) speech, \(existingNoiseFiles.count) noise)")
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
        } catch {
            logger.error("Failed to download speech files: \(error)")
            failedFiles += 1
        }

        // Download noise files
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
        } catch {
            logger.error("Failed to download noise files: \(error)")
            failedFiles += 1
        }

        if downloadedFiles > 0 {
            logger.info("VAD dataset ready: \(downloadedFiles) files")
        } else {
            logger.warning("⚠️ VAD download failed, will use legacy URLs")
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
            // Will try pattern-based download as fallback
        }

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
                } catch {
                    continue
                }
            }
        }

        // If no files downloaded via API, try pattern-based download
        if testFiles.isEmpty {

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
                        } catch {
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

        logger.info("📥 Downloading full MUSAN (~600MB) to \(cacheDir.path)")

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
                logger.info("📂 Full MUSAN exists at \(musanDir.path)")
                return
            }
        }

        // Download from OpenSLR
        let musanURL = "https://www.openslr.org/resources/17/musan.tar.gz"
        let downloadPath = cacheDir.appendingPathComponent("musan.tar.gz")

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

            // Extract tar.gz
            logger.info("📦 Extracting MUSAN archive...")
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            task.arguments = ["-xzf", downloadPath.path, "-C", cacheDir.path]

            try task.run()
            task.waitUntilExit()

            if task.terminationStatus == 0 {
                // Clean up tar file
                try? FileManager.default.removeItem(at: downloadPath)

                // Count files
                let speechFiles = countFiles(in: musanDir.appendingPathComponent("speech"))
                let musicFiles = countFiles(in: musanDir.appendingPathComponent("music"))
                let noiseFiles = countFiles(in: musanDir.appendingPathComponent("noise"))

                logger.info(
                    "Full MUSAN ready: \(speechFiles + musicFiles + noiseFiles) files (speech: \(speechFiles), music: \(musicFiles), noise: \(noiseFiles))"
                )
            } else {
                logger.error("MUSAN extraction failed")
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

    // MARK: - Earnings22 KWS Dataset

    /// Get Earnings22 KWS dataset cache directory
    static func getEarnings22Directory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("FluidAudio/earnings22-kws", isDirectory: true)
    }

    // MARK: - HuggingFace Datasets Server API types

    private struct HFRowsResponse: Decodable {
        let rows: [HFRow]
        let numRowsTotal: Int

        enum CodingKeys: String, CodingKey {
            case rows
            case numRowsTotal = "num_rows_total"
        }
    }

    private struct HFRow: Decodable {
        let row: HFRowData
    }

    private struct HFRowData: Decodable {
        let fileId: String
        let text: String
        let dictionary: [String]?
        let audio: [HFAudioSource]

        enum CodingKeys: String, CodingKey {
            case fileId = "file_id"
            case text
            case dictionary
            case audio
        }
    }

    private struct HFAudioSource: Decodable {
        let src: String
        let type: String
    }

    /// Download Earnings22 KWS dataset from argmaxinc/earnings22-kws-golden
    /// using the HuggingFace Datasets Server REST API (pure Swift, no Python dependency).
    static func downloadEarnings22KWS(force: Bool) async {
        let cacheDir = getEarnings22Directory()
        let testDatasetDir = cacheDir.appendingPathComponent("test-dataset")

        logger.info("📥 Downloading Earnings22 KWS to \(cacheDir.path)")

        // Check if already downloaded
        if !force && FileManager.default.fileExists(atPath: testDatasetDir.path) {
            let files =
                (try? FileManager.default.contentsOfDirectory(
                    at: testDatasetDir, includingPropertiesForKeys: nil
                )) ?? []
            let wavFiles = files.filter { $0.pathExtension == "wav" }
            if wavFiles.count > 100 {
                logger.info("📂 Earnings22 KWS exists (\(wavFiles.count) files)")
                return
            }
        }

        // Create directories
        do {
            try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: testDatasetDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directories: \(error)")
            return
        }

        // Fetch rows via HuggingFace Datasets Server API (paginated, max 100 per request)
        let baseURL = "https://datasets-server.huggingface.co/rows"
        let dataset = "argmaxinc/earnings22-kws-golden"
        let pageSize = 100
        var offset = 0
        var totalExtracted = 0
        var totalRows = Int.max

        logger.info("📦 Fetching Earnings22 dataset via HuggingFace API...")

        while offset < totalRows {
            let apiURLString =
                "\(baseURL)?dataset=\(dataset)&config=default&split=test&offset=\(offset)&length=\(pageSize)"

            guard let apiURL = URL(string: apiURLString) else {
                logger.error("Invalid API URL: \(apiURLString)")
                return
            }

            do {
                let (data, response) = try await DownloadUtils.sharedSession.data(from: apiURL)

                guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                    let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                    logger.error("API request failed (status \(statusCode)) at offset \(offset)")
                    return
                }

                let decoded = try JSONDecoder().decode(HFRowsResponse.self, from: data)
                totalRows = decoded.numRowsTotal

                guard !decoded.rows.isEmpty else { break }

                for hfRow in decoded.rows {
                    let row = hfRow.row
                    let fileId = row.fileId
                    let wavPath = testDatasetDir.appendingPathComponent("\(fileId).wav")

                    // Skip if already extracted (resume support)
                    if FileManager.default.fileExists(atPath: wavPath.path) {
                        totalExtracted += 1
                        continue
                    }

                    // Download audio from the src URL
                    guard let audioSource = row.audio.first,
                        let audioURL = URL(string: audioSource.src)
                    else {
                        logger.warning("No audio source for \(fileId)")
                        continue
                    }

                    let (audioData, audioResponse) = try await DownloadUtils.sharedSession.data(from: audioURL)
                    guard let audioHTTP = audioResponse as? HTTPURLResponse, audioHTTP.statusCode == 200 else {
                        logger.warning("Failed to download audio for \(fileId)")
                        continue
                    }

                    try audioData.write(to: wavPath)

                    // Write text file
                    let textPath = testDatasetDir.appendingPathComponent("\(fileId).text.txt")
                    try row.text.write(to: textPath, atomically: true, encoding: .utf8)

                    // Write dictionary file
                    let dictPath = testDatasetDir.appendingPathComponent("\(fileId).dictionary.txt")
                    let dictText = row.dictionary?.joined(separator: "\n") ?? ""
                    try dictText.write(to: dictPath, atomically: true, encoding: .utf8)

                    totalExtracted += 1
                    logger.info("Extracted: \(fileId)")
                }

                logger.info("Progress: \(min(offset + pageSize, totalRows))/\(totalRows)")

            } catch {
                logger.error("Failed at offset \(offset): \(error)")
                return
            }

            offset += pageSize
        }

        // Count final files
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: testDatasetDir, includingPropertiesForKeys: nil
            )) ?? []
        let wavFiles = files.filter { $0.pathExtension == "wav" }
        logger.info("Earnings22 KWS ready: \(wavFiles.count) files (\(totalExtracted) extracted)")
    }

    // MARK: - Buckeye Forced Alignment Dataset

    /// Get Buckeye dataset cache directory
    static func getBuckeyeDatasetDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return appSupport.appendingPathComponent("FluidAudio/Buckeye", isDirectory: true)
    }

    /// Download Buckeye forced alignment benchmark dataset from HuggingFace
    static func downloadBuckeyeDataset(force: Bool) async {
        let cacheDir = getBuckeyeDatasetDirectory()
        let audioDir = cacheDir.appendingPathComponent("audio")
        let manifestPath = cacheDir.appendingPathComponent("manifest.json")

        logger.info("Downloading Buckeye dataset to \(cacheDir.path)")

        do {
            try FileManager.default.createDirectory(at: audioDir, withIntermediateDirectories: true)
        } catch {
            logger.error("Failed to create directory: \(error)")
            return
        }

        // Check if already downloaded
        if !force && FileManager.default.fileExists(atPath: manifestPath.path) {
            let existingFiles =
                (try? FileManager.default.contentsOfDirectory(
                    at: audioDir, includingPropertiesForKeys: nil)) ?? []
            let wavCount = existingFiles.filter { $0.pathExtension == "wav" }.count
            if wavCount >= 2400 {
                logger.info("Buckeye dataset exists (\(wavCount) audio files)")
                return
            }
        }

        let repoBase = ModelRegistry.resolveDatasetBase("alexwengg/buckeye")

        // Download manifest.json
        logger.info("Downloading manifest.json...")
        do {
            guard let url = URL(string: "\(repoBase)/manifest.json") else {
                logger.error("Invalid manifest URL")
                return
            }
            let (data, response) = try await DownloadUtils.sharedSession.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                logger.error("Failed to download manifest.json")
                return
            }
            try data.write(to: manifestPath)
        } catch {
            logger.error("Failed to download manifest: \(error)")
            return
        }

        // Parse manifest for audio file list
        guard let manifestData = try? Data(contentsOf: manifestPath),
            let json = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
            let samples = json["samples"] as? [[String: Any]]
        else {
            logger.error("Failed to parse manifest.json")
            return
        }

        logger.info("Downloading \(samples.count) audio files...")

        var downloaded = 0
        var skipped = 0
        var failed = 0

        for (i, sample) in samples.enumerated() {
            guard let audioPath = sample["audio"] as? String else { continue }
            let destination = cacheDir.appendingPathComponent(audioPath)
            let downloadURL = "\(repoBase)/\(audioPath)"

            if !force && FileManager.default.fileExists(atPath: destination.path) {
                skipped += 1
                continue
            }

            guard let url = URL(string: downloadURL) else {
                failed += 1
                continue
            }

            do {
                let (data, response) = try await DownloadUtils.sharedSession.data(from: url)
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    failed += 1
                    continue
                }
                try data.write(to: destination)
                downloaded += 1
            } catch {
                failed += 1
            }

            let total = i + 1
            if total % 100 == 0 || total == samples.count {
                logger.info(
                    "Progress: \(total)/\(samples.count) "
                        + "(\(downloaded) new, \(skipped) skipped, \(failed) failed)")
            }
        }

        logger.info(
            "Buckeye dataset ready: \(downloaded) downloaded, \(skipped) skipped, \(failed) failed")
    }

    /// Download VOiCES subset dataset from GitHub
    static func downloadVoicesSubset(force: Bool) async {
        let cacheDir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        .appendingPathComponent("FluidAudio/voicesSubset", isDirectory: true)

        logger.info("📥 Downloading VOiCES subset to \(cacheDir.path)")

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
                logger.info("📂 VOiCES subset exists (\(cleanFiles.count) clean, \(noisyFiles.count) noisy)")
                return
            }
        }

        // Clone the repository
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
                // Move the audio files to our cache structure
                let sourceCleanDir = cloneDir.appendingPathComponent("clean")
                let sourceNoisyDir = cloneDir.appendingPathComponent("noisy")

                // Create destination directories
                try FileManager.default.createDirectory(
                    at: cleanDir, withIntermediateDirectories: true)
                try FileManager.default.createDirectory(
                    at: noisyDir, withIntermediateDirectories: true)

                // Move clean files
                var cleanCount = 0
                var noisyCount = 0
                if FileManager.default.fileExists(atPath: sourceCleanDir.path) {
                    let cleanFiles = try FileManager.default.contentsOfDirectory(
                        at: sourceCleanDir, includingPropertiesForKeys: nil)
                    for file in cleanFiles where file.pathExtension == "wav" {
                        let destination = cleanDir.appendingPathComponent(
                            file.lastPathComponent)
                        try FileManager.default.moveItem(at: file, to: destination)
                        cleanCount += 1
                    }
                }

                // Move noisy files
                if FileManager.default.fileExists(atPath: sourceNoisyDir.path) {
                    let noisyFiles = try FileManager.default.contentsOfDirectory(
                        at: sourceNoisyDir, includingPropertiesForKeys: nil)
                    for file in noisyFiles where file.pathExtension == "wav" {
                        let destination = noisyDir.appendingPathComponent(
                            file.lastPathComponent)
                        try FileManager.default.moveItem(at: file, to: destination)
                        noisyCount += 1
                    }
                }

                // Clean up clone directory
                try? FileManager.default.removeItem(at: cloneDir)

                logger.info("VOiCES subset ready: \(cleanCount) clean, \(noisyCount) noisy")

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
