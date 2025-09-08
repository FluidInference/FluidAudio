import Foundation
import OSLog

/// Downloads Kokoro TTS model and data files from HuggingFace
@available(macOS 13.0, iOS 16.0, *)
public struct KokoroModelDownloader {
    private static let logger = Logger(subsystem: "com.fluidaudio.tts", category: "KokoroModelDownloader")

    private static let modelFiles = [
        (
            "kokoro.mlmodelc/model.mil",
            "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/kokoro.mlmodelc/model.mil"
        ),
        (
            "kokoro.mlmodelc/coremldata.bin",
            "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/kokoro.mlmodelc/coremldata.bin"
        ),
        (
            "kokoro.mlmodelc/weights/weight.bin",
            "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/kokoro.mlmodelc/weights/weight.bin"
        ),
        (
            "kokoro.mlmodelc/analytics/coremldata.bin",
            "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/kokoro.mlmodelc/analytics/coremldata.bin"
        ),
    ]

    private static let dataFiles = [
        ("vocab_index.json", "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/vocab_index.json"),
        ("word_phonemes.json", "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/word_phonemes.json"),
    ]

    /// Check if all required files exist
    public static func checkRequiredFiles() -> Bool {
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath

        // Check model directory
        let modelPath = URL(fileURLWithPath: currentDir).appendingPathComponent("kokoro.mlmodelc").path
        if !fm.fileExists(atPath: modelPath) {
            return false
        }

        // Check data files
        for (filename, _) in dataFiles {
            let path = URL(fileURLWithPath: currentDir).appendingPathComponent(filename).path
            if !fm.fileExists(atPath: path) {
                return false
            }
        }

        return true
    }

    /// Download a voice embedding file
    public static func downloadVoiceIfNeeded(voice: String) async throws {
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath
        let voiceFile = "\(voice).pt"
        let voicePath = URL(fileURLWithPath: currentDir).appendingPathComponent(voiceFile).path

        if fm.fileExists(atPath: voicePath) {
            logger.info("Voice file \(voiceFile) already exists")
            return
        }

        let voiceURL = "https://huggingface.co/FluidInference/coreml-kokoro/resolve/main/voices/\(voice).pt"
        logger.info("Downloading voice: \(voice) from \(voiceURL)")

        try await downloadFile(from: voiceURL, to: voiceFile)
    }

    /// Download all required files if they don't exist
    public static func downloadRequiredFilesIfNeeded() async throws {
        let fm = FileManager.default
        let currentDir = fm.currentDirectoryPath

        // Create model directories
        let modelDir = URL(fileURLWithPath: currentDir).appendingPathComponent("kokoro.mlmodelc")
        let weightsDir = modelDir.appendingPathComponent("weights")
        let analyticsDir = modelDir.appendingPathComponent("analytics")

        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: weightsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: analyticsDir, withIntermediateDirectories: true)

        // Download model files
        for (filename, url) in modelFiles {
            let path = URL(fileURLWithPath: currentDir).appendingPathComponent(filename).path
            if !fm.fileExists(atPath: path) {
                logger.info("Downloading: \(filename)")
                try await downloadFile(from: url, to: filename)
            } else {
                logger.info("File exists: \(filename)")
            }
        }

        // Download data files
        for (filename, url) in dataFiles {
            let path = URL(fileURLWithPath: currentDir).appendingPathComponent(filename).path
            if !fm.fileExists(atPath: path) {
                logger.info("Downloading: \(filename)")
                try await downloadFile(from: url, to: filename)
            } else {
                logger.info("File exists: \(filename)")
            }
        }

        logger.info("✅ All required files downloaded successfully")
    }

    /// Download a file from URL to local path
    private static func downloadFile(from urlString: String, to filename: String) async throws {
        guard let url = URL(string: urlString) else {
            throw TTSError.processingFailed("Invalid URL: \(urlString)")
        }

        let currentDir = FileManager.default.currentDirectoryPath
        let destinationURL = URL(fileURLWithPath: currentDir).appendingPathComponent(filename)

        let (tempURL, response) = try await URLSession.shared.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw TTSError.processingFailed("Failed to download \(filename)")
        }

        // Create parent directory if needed
        let parentDir = destinationURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        // Move downloaded file to destination
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: destinationURL)

        let fileSize = try FileManager.default.attributesOfItem(atPath: destinationURL.path)[.size] as? Int ?? 0
        logger.info("Downloaded \(filename): \(fileSize / 1024 / 1024) MB")
    }
}
