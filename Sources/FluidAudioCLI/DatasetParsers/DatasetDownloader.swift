#if os(macOS)
import AVFoundation
import Foundation

/// Dataset downloading functionality for AMI and VAD datasets
struct DatasetDownloader {
    
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

    static func downloadAMIDataset(variant: AMIVariant, force: Bool, singleFile: String? = nil) async {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let baseDir = homeDir.appendingPathComponent("FluidAudioDatasets")
        let amiDir = baseDir.appendingPathComponent("ami_official")
        let variantDir = amiDir.appendingPathComponent(variant.rawValue)

        // Create directories if needed
        do {
            try FileManager.default.createDirectory(
                at: variantDir, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create directory: \(error)")
            return
        }

        print("📥 Downloading AMI \(variant.displayName) dataset...")
        print("   Target directory: \(variantDir.path)")
        
        // Download AMI annotations first (required for proper benchmarking)
        await downloadAMIAnnotations(force: force)

        // Core AMI test set - smaller subset for initial benchmarking
        let commonMeetings: [String]
        if let singleFile = singleFile {
            commonMeetings = [singleFile]
            print("📋 Downloading single file: \(singleFile)")
        } else {
            commonMeetings = [
                "ES2002a",
                "ES2003a",
                "ES2004a",
                "ES2005a",
                "IS1000a",
                "IS1001a",
                "IS1002b",
                "TS3003a",
                "TS3004a",
            ]
        }

        var downloadedFiles = 0
        var skippedFiles = 0

        for meetingId in commonMeetings {
            let fileName = "\(meetingId).\(variant.filePattern)"
            let filePath = variantDir.appendingPathComponent(fileName)

            // Skip if file exists and not forcing download
            if !force && FileManager.default.fileExists(atPath: filePath.path) {
                print("   ⏭️ Skipping \(fileName) (already exists)")
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
                print("   ✅ Downloaded \(fileName)")
            } else {
                print("   ❌ Failed to download \(fileName)")
            }
        }

        print("🎉 AMI \(variant.displayName) download completed")
        print("   Downloaded: \(downloadedFiles) files")
        print("   Skipped: \(skippedFiles) files")
        print("   Total files: \(downloadedFiles + skippedFiles)/\(commonMeetings.count)")

        if downloadedFiles == 0 && skippedFiles == 0 {
            print("⚠️ No files were downloaded. You may need to download manually from:")
            print("   https://groups.inf.ed.ac.uk/ami/download/")
        }
    }

    static func downloadAMIFile(meetingId: String, variant: AMIVariant, outputPath: URL) async -> Bool {
        // Try multiple URL patterns - the AMI corpus mirror structure has some variations
        let baseURLs = [
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Double slash pattern (from user's working example)
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus",  // Single slash pattern
            "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus",  // Alternative with extra slash
        ]

        for (_, baseURL) in baseURLs.enumerated() {
            let urlString = "\(baseURL)/\(meetingId)/audio/\(meetingId).\(variant.filePattern)"

            guard let url = URL(string: urlString) else {
                print("     ⚠️ Invalid URL: \(urlString)")
                continue
            }

            do {
                print("     📥 Downloading from: \(urlString)")
                let (data, response) = try await URLSession.shared.data(from: url)

                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        try data.write(to: outputPath)

                        // Verify it's a valid audio file
                        if await isValidAudioFile(outputPath) {
                            let fileSizeMB = Double(data.count) / (1024 * 1024)
                            print(
                                "     ✅ Downloaded \(String(format: "%.1f", fileSizeMB)) MB")
                            return true
                        } else {
                            print("     ⚠️ Downloaded file is not valid audio")
                            try? FileManager.default.removeItem(at: outputPath)
                            // Try next URL
                            continue
                        }
                    } else if httpResponse.statusCode == 404 {
                        print("     ⚠️ File not found (HTTP 404) - trying next URL...")
                        continue
                    } else {
                        print(
                            "     ⚠️ HTTP error: \(httpResponse.statusCode) - trying next URL...")
                        continue
                    }
                }
            } catch {
                print(
                    "     ⚠️ Download error: \(error.localizedDescription) - trying next URL...")
                continue
            }
        }

        print("     ❌ Failed to download from all available URLs")
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
        
        if !force && FileManager.default.fileExists(atPath: segmentsDir.path) && 
           FileManager.default.fileExists(atPath: meetingsFile.path) {
            print("📂 AMI annotations already exist in \(annotationsDir.path)")
            return
        }
        
        print("📥 Downloading AMI annotations from Edinburgh University...")
        print("   Target directory: \(annotationsDir.path)")
        
        // Create required directories
        do {
            try FileManager.default.createDirectory(at: annotationsDir, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create annotation directories: \(error)")
            return
        }
        
        // Download and extract AMI manual annotations v1.6.2
        let zipURL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        let zipFile = annotationsDir.appendingPathComponent("ami_public_manual_1.6.2.zip")
        
        print("📥 Downloading AMI manual annotations archive (22MB)...")
        let zipSuccess = await downloadAnnotationFile(from: zipURL, to: zipFile)
        
        if !zipSuccess {
            print("❌ Failed to download AMI annotations archive")
            return
        }
        
        print("📦 Extracting AMI annotations archive...")
        
        // Extract the ZIP file using the system unzip command
        let extractSuccess = await extractZipFile(zipFile, to: annotationsDir)
        
        if extractSuccess {
            // Clean up ZIP file
            try? FileManager.default.removeItem(at: zipFile)
            
            // Verify extraction was successful
            if FileManager.default.fileExists(atPath: segmentsDir.path) && 
               FileManager.default.fileExists(atPath: meetingsFile.path) {
                print("✅ AMI annotations download and extraction completed")
                print("💡 Benchmarks will now use real AMI ground truth data")
            } else {
                print("⚠️ Extraction completed but expected files not found")
                print("   Looking for: \(segmentsDir.path)")
                print("   Looking for: \(meetingsFile.path)")
            }
        } else {
            print("❌ Failed to extract AMI annotations archive")
        }
    }
    
    /// Download a single annotation file from AMI corpus
    static func downloadAnnotationFile(from urlString: String, to outputPath: URL) async -> Bool {
        guard let url = URL(string: urlString) else {
            print("     ⚠️ Invalid URL: \(urlString)")
            return false
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
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
                           xmlString.contains("<?xml") || xmlString.contains("<nite:") {
                            return true
                        } else {
                            print("     ⚠️ Downloaded file is not valid XML")
                            try? FileManager.default.removeItem(at: outputPath)
                            return false
                        }
                    }
                } else {
                    print("     ⚠️ HTTP error: \(httpResponse.statusCode)")
                    return false
                }
            }
        } catch {
            print("     ⚠️ Download error: \(error.localizedDescription)")
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
            print("     ⚠️ Failed to extract ZIP file: \(error)")
            return false
        }
    }

    /// Download VAD dataset from Hugging Face
    static func downloadVadDataset(force: Bool, dataset: String = "mini50") async {
        let cacheDir = getVadDatasetCacheDirectory()

        print("📥 Downloading VAD dataset from Hugging Face...")
        print("   Target directory: \(cacheDir.path)")

        // Create cache directories
        let speechDir = cacheDir.appendingPathComponent("speech")
        let noiseDir = cacheDir.appendingPathComponent("noise")

        do {
            try FileManager.default.createDirectory(at: speechDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: noiseDir, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create cache directories: \(error)")
            return
        }

        // Check if we should skip download
        if !force {
            let existingSpeechFiles = (try? FileManager.default.contentsOfDirectory(at: speechDir, includingPropertiesForKeys: nil)) ?? []
            let existingNoiseFiles = (try? FileManager.default.contentsOfDirectory(at: noiseDir, includingPropertiesForKeys: nil)) ?? []

            if !existingSpeechFiles.isEmpty && !existingNoiseFiles.isEmpty {
                print("📂 VAD dataset already exists (use --force to re-download)")
                print("   Speech files: \(existingSpeechFiles.count)")
                print("   Noise files: \(existingNoiseFiles.count)")
                return
            }
        } else {
            // Force download - clean existing files
            try? FileManager.default.removeItem(at: speechDir)
            try? FileManager.default.removeItem(at: noiseDir)
            try? FileManager.default.createDirectory(at: speechDir, withIntermediateDirectories: true)
            try? FileManager.default.createDirectory(at: noiseDir, withIntermediateDirectories: true)
        }

        // Use specified dataset for download command
        let repoName = dataset == "mini100" ? "musan_mini100" : "musan_mini50"
        let repoBase = "https://huggingface.co/datasets/alexwengg/\(repoName)/resolve/main"

        var downloadedFiles = 0
        var failedFiles = 0

        // Download speech files
        print("📢 Downloading speech samples...")
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
            print("   ✅ Downloaded \(speechFiles.count) speech files")
        } catch {
            print("   ❌ Failed to download speech files: \(error)")
            failedFiles += 1
        }

        // Download noise files
        print("🔇 Downloading noise samples...")
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
            print("   ✅ Downloaded \(noiseFiles.count) noise files")
        } catch {
            print("   ❌ Failed to download noise files: \(error)")
            failedFiles += 1
        }

        print("\n📊 VAD Dataset Download Summary:")
        print("   Downloaded: \(downloadedFiles) files")
        print("   Failed: \(failedFiles) categories")

        if downloadedFiles > 0 {
            print("✅ VAD dataset download completed")
            print("💡 You can now run VAD benchmarks with the downloaded dataset")
        } else {
            print("❌ No files were downloaded successfully")
            print("⚠️ VAD benchmarks will fall back to legacy URLs")
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
        let repoApiUrl = "https://huggingface.co/api/datasets/alexwengg/\(repoName)/tree/main/\(filePrefix)"
        var allFiles: [String] = []

        do {
            let directoryFiles = try await getHuggingFaceFileList(apiUrl: repoApiUrl)
            let audioFiles = directoryFiles.filter { fileName in
                let ext = URL(fileURLWithPath: fileName).pathExtension.lowercased()
                return ["wav", "mp3", "flac", "m4a"].contains(ext)
            }
            allFiles.append(contentsOf: audioFiles)
        } catch {
            print("      ⚠️ Could not access \(filePrefix): \(error)")
        }

        print("      Found \(allFiles.count) audio files in \(filePrefix)/ directory")
        print("      Debug: requesting \(count) files from \(allFiles.count) available")

        if !allFiles.isEmpty {
            let filesToDownload = Array(allFiles.prefix(count))
            var downloadedCount = 0

            for fileName in filesToDownload {
                let fileUrl = "\(baseUrl)/\(fileName)"
                let destination = targetDir.appendingPathComponent(fileName)

                do {
                    let downloadedFile = try await downloadAudioFile(from: fileUrl, to: destination)
                    testFiles.append(VadTestFile(
                        name: fileName,
                        expectedLabel: expectedLabel,
                        url: downloadedFile
                    ))
                    downloadedCount += 1
                    print("      ✅ Downloaded: \(fileName)")

                } catch {
                    print("      ⚠️ Failed to download \(fileName): \(error)")
                    continue
                }
            }
        } else {
            print("      No audio files found in subdirectories")
        }

        // If no files downloaded via API, try pattern-based download
        if testFiles.isEmpty {
            print("      ⚠️ API method failed or no files found, trying pattern-based download...")

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

                for i in 0..<(count * 2) { // Try more files than needed
                    if downloadedCount >= count { break }

                    for ext in extensions {
                        if downloadedCount >= count { break }

                        let fileName = "\(pattern)\(String(format: "%04d", i)).\(ext)"
                        let fileUrl = "\(baseUrl)/\(fileName)"
                        let destination = targetDir.appendingPathComponent(fileName)

                        do {
                            let downloadedFile = try await downloadAudioFile(from: fileUrl, to: destination)
                            testFiles.append(VadTestFile(
                                name: fileName,
                                expectedLabel: expectedLabel,
                                url: downloadedFile
                            ))
                            downloadedCount += 1
                            print("      ✅ Downloaded: \(fileName)")

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

    /// Get file list from HuggingFace API
    static func getHuggingFaceFileList(apiUrl: String) async throws -> [String] {
        guard let url = URL(string: apiUrl) else {
            throw NSError(domain: "APIError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid API URL"])
        }

        let (data, _) = try await URLSession.shared.data(from: url)

        // Parse the JSON response to extract file names
        if let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return json.compactMap { item in
                if let type = item["type"] as? String,
                   let path = item["path"] as? String,
                   type == "file" {
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
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let cacheDir = appSupport.appendingPathComponent("FluidAudio/vadDataset", isDirectory: true)

        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir
    }

    static func downloadAudioFile(from urlString: String, to destination: URL) async throws -> URL {
        // Skip if already exists
        if FileManager.default.fileExists(atPath: destination.path) {
            return destination
        }

        guard let url = URL(string: urlString) else {
            throw NSError(domain: "DownloadError", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"])
        }

        let (data, _) = try await URLSession.shared.data(from: url)
        try data.write(to: destination)

        // Verify it's valid audio
        do {
            let _ = try AVAudioFile(forReading: destination)
        } catch {
            throw NSError(domain: "DownloadError", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Downloaded file is not valid audio"])
        }

        return destination
    }
}

#endif