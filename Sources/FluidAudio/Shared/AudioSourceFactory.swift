@preconcurrency import AVFoundation
import Foundation
import OSLog

public struct AudioSourceFactory {
    private let logger = AppLogger(category: "AudioSourceFactory")

    public init() {}

    public func makeDiskBackedSource(
        from url: URL,
        targetSampleRate: Int
    ) throws -> (source: DiskBackedAudioSampleSource, loadDuration: TimeInterval) {
        do {
            let startTime = Date()

            // A 16-bit mono wav already at the target rate is read in place —
            // no conversion pass, no temporary copy — so a day of audio opens
            // in the time it takes to map the file.
            if url.pathExtension.lowercased() == "wav" {
                let mapped = try Data(contentsOf: url, options: [.mappedIfSafe])
                if let layout = PcmWavLayout.parse(mapped, sampleRate: targetSampleRate) {
                    logger.debug("Mapped PCM wav in place: \(layout.sampleCount) samples")
                    let source = DiskBackedAudioSampleSource(
                        pcm16MappedWav: mapped, dataOffset: layout.dataOffset, sampleCount: layout.sampleCount)
                    return (source, Date().timeIntervalSince(startTime))
                }
            }

            let audioFile = try AVAudioFile(forReading: url)
            let inputFormat = audioFile.processingFormat
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetSampleRate),
                channels: 1,
                interleaved: false
            )!

            let tempURL = try makeTemporaryURL()
            guard FileManager.default.createFile(atPath: tempURL.path, contents: nil) else {
                throw AudioSourceError.processingFailed("Failed to create temporary audio buffer at \(tempURL.path)")
            }

            let handle = try FileHandle(forWritingTo: tempURL)
            defer {
                try? handle.close()
            }

            logger.debug(
                "Streaming conversion \(inputFormat.sampleRate)Hz×\(inputFormat.channelCount)ch → \(targetFormat.sampleRate)Hz×\(targetFormat.channelCount)ch"
            )

            // The same conversion resampleAudioFile runs, so transcribe(url)
            // and transcribe(samples) see the same samples (#1112).
            var totalSamples = 0
            do {
                try AudioConverter(targetFormat: targetFormat).convertFile(audioFile) { chunk in
                    try handle.write(contentsOf: Data(buffer: chunk))
                    totalSamples += chunk.count
                }
            } catch {
                logger.error("Streaming conversion failed before file mapping: \(error.localizedDescription)")
                throw error
            }

            try handle.synchronize()
            try handle.close()

            let attributes = try FileManager.default.attributesOfItem(atPath: tempURL.path)
            if let fileSize = attributes[.size] as? NSNumber {
                logger.debug("Streaming audio temp file size=\(fileSize.intValue) bytes")
            }
            logger.debug("Streaming audio total samples=\(totalSamples)")

            let mappedData = try Data(contentsOf: tempURL, options: [.mappedIfSafe])
            let source = DiskBackedAudioSampleSource(mappedData: mappedData, fileURL: tempURL)

            if source.sampleCount != totalSamples {
                logger.warning(
                    "Mapped sample count mismatch (reported=\(source.sampleCount), tracked=\(totalSamples)); continuing"
                )
            }

            let duration = Date().timeIntervalSince(startTime)
            return (source, duration)
        } catch let streamingError as AudioSourceError {
            throw streamingError
        } catch {
            logger.error("Streaming audio source creation failed: \(error.localizedDescription)")
            throw AudioSourceError.processingFailed(
                "Streaming audio source creation failed: \(error.localizedDescription)"
            )
        }
    }

    private func makeTemporaryURL() throws -> URL {
        let tempDirectory = FileManager.default.temporaryDirectory
        let identifier = UUID().uuidString
        return tempDirectory.appendingPathComponent("fluidaudio-streaming-\(identifier).raw")
    }
}

public enum AudioSourceError: Error, LocalizedError {
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}
