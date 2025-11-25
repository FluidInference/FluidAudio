import FluidAudio
import Foundation

public struct CompareAudioCommand {
    private static let logger = AppLogger(category: "CompareAudioCommand")

    public static func run(arguments: [String]) async {
        if arguments.count == 2 {
            // Explicit file pair mode
            let refPath = arguments[0]
            let faPath = arguments[1]
            await compareFiles(refPath: refPath, faPath: faPath)
            return
        }

        guard let directoryPath = arguments.first else {
            print("Usage: fluidaudio compare-audio <directory> OR <ref_file> <fa_file>")
            return
        }

        let directoryURL = URL(fileURLWithPath: directoryPath)
        let fileManager = FileManager.default

        do {
            let files = try fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil)
            let refFiles = files.filter { $0.lastPathComponent.hasSuffix("-ref.wav") }

            let diarizer = OfflineDiarizerManager()
            // Ensure models are loaded
            try await diarizer.prepareModels()

            print("Prompt | Voice | Distance | Ref Dur | FA Dur")
            print("---|---|---|---|---")

            for refFile in refFiles {
                let refFilename = refFile.lastPathComponent
                let faFilename = refFilename.replacingOccurrences(of: "-ref.wav", with: "-fa.wav")
                let faFile = directoryURL.appendingPathComponent(faFilename)

                if !fileManager.fileExists(atPath: faFile.path) {
                    logger.warning("Missing FA file for \(refFilename)")
                    continue
                }

                await comparePair(diarizer: diarizer, refFile: refFile, faFile: faFile)
            }

        } catch {
            logger.error("Error comparing audio: \(error)")
        }
    }

    private static func compareFiles(refPath: String, faPath: String) async {
        let diarizer = OfflineDiarizerManager()
        do {
            try await diarizer.prepareModels()
            let refURL = URL(fileURLWithPath: refPath)
            let faURL = URL(fileURLWithPath: faPath)
            
            print("Ref File | FA File | Distance | Ref Dur | FA Dur")
            print("---|---|---|---|---")
            
            await comparePair(diarizer: diarizer, refFile: refURL, faFile: faURL, printHeader: false)
        } catch {
            logger.error("Error preparing models: \(error)")
        }
    }

    private static func comparePair(diarizer: OfflineDiarizerManager, refFile: URL, faFile: URL, printHeader: Bool = true) async {
        do {
            // Process Ref
            let refResult = try await diarizer.process(refFile)
            guard let refSpeaker = getDominantSpeaker(from: refResult) else {
                logger.warning("No speaker found in \(refFile.lastPathComponent)")
                return
            }

            // Process FA
            let faResult = try await diarizer.process(faFile)
            guard let faSpeaker = getDominantSpeaker(from: faResult) else {
                logger.warning("No speaker found in \(faFile.lastPathComponent)")
                return
            }

            // Compute Distance
            let distance = SpeakerUtilities.cosineDistance(refSpeaker.embedding, faSpeaker.embedding)
            
            // Parse info from filename if possible, otherwise just use filenames
            let refFilename = refFile.lastPathComponent
            let parts = refFilename.split(separator: "-")
            let prompt = parts.first ?? "?"
            let voice = parts.count > 1 ? parts[1] : "?"

            if printHeader {
                 print("\(prompt) | \(voice) | \(String(format: "%.4f", distance)) | \(String(format: "%.2fs", refSpeaker.duration)) | \(String(format: "%.2fs", faSpeaker.duration))")
            } else {
                 print("\(refFile.lastPathComponent) | \(faFile.lastPathComponent) | \(String(format: "%.4f", distance)) | \(String(format: "%.2fs", refSpeaker.duration)) | \(String(format: "%.2fs", faSpeaker.duration))")
            }
           
        } catch {
            logger.error("Error processing pair \(refFile.lastPathComponent): \(error)")
        }
    }

    private static func getDominantSpeaker(from result: DiarizationResult) -> (id: String, embedding: [Float], duration: Float)? {
        var durations: [String: Float] = [:]
        for segment in result.segments {
            durations[segment.speakerId, default: 0] += segment.durationSeconds
        }
        
        guard let (speakerId, duration) = durations.max(by: { $0.value < $1.value }) else {
            return nil
        }
        
        let embedding: [Float]
        if let db = result.speakerDatabase, let dbEmbedding = db[speakerId] {
            embedding = dbEmbedding
        } else {
            // Fallback to first segment embedding if database is missing
            guard let segment = result.segments.first(where: { $0.speakerId == speakerId }) else {
                return nil
            }
            embedding = segment.embedding
        }
        
        return (speakerId, embedding, duration)
    }
}
