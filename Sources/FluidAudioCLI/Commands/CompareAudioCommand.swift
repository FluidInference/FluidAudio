import Foundation
import FluidAudio
import Accelerate

enum CompareAudioCommand {
    private static let logger = AppLogger(category: "CompareAudio")

    static func run(arguments: [String]) async {
        guard arguments.count >= 2 else {
            print("Usage: fluidaudio compare-audio <file1.wav> <file2.wav>")
            return
        }

        let file1 = arguments[0]
        let file2 = arguments[1]

        do {
            logger.info("Comparing audio files...")
            logger.info("File 1: \(file1)")
            logger.info("File 2: \(file2)")

            // Initialize models
            let models = try await DiarizerModels.downloadIfNeeded()
            let config = DiarizerConfig(debugMode: false)
            let manager = DiarizerManager(config: config)
            manager.initialize(models: models)

            // Process files
            let emb1 = try await extractMainEmbedding(manager: manager, path: file1)
            let emb2 = try await extractMainEmbedding(manager: manager, path: file2)

            // Compute similarity
            let dist = SpeakerUtilities.cosineDistance(emb1, emb2)
            let similarity = 1.0 - dist

            print("\n--- Comparison Result ---")
            print("Cosine Distance: \(String(format: "%.4f", dist))")
            print("Cosine Similarity: \(String(format: "%.4f", similarity))")
            
            if dist < 0.2 {
                print("✅ Voices match (distance < 0.2)")
            } else {
                print("❌ Voices differ (distance >= 0.2)")
            }

        } catch {
            logger.error("Comparison failed: \(error)")
        }
    }

    private static func extractMainEmbedding(manager: DiarizerManager, path: String) async throws -> [Float] {
        let converter = AudioConverter()
        let samples = try converter.resampleAudioFile(path: path)
        
        // Process
        let result = try manager.performCompleteDiarization(samples)
        
        // Find dominant speaker
        var speakerDurations: [String: Float] = [:]
        var speakerEmbeddings: [String: [[Float]]] = [:]
        
        for segment in result.segments {
            let dur = segment.endTimeSeconds - segment.startTimeSeconds
            speakerDurations[segment.speakerId, default: 0] += dur
            speakerEmbeddings[segment.speakerId, default: []].append(segment.embedding)
        }
        
        guard let mainSpeaker = speakerDurations.max(by: { $0.value < $1.value })?.key else {
            throw NSError(domain: "CompareAudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "No speech detected in \(path)"])
        }
        
        // Average embeddings
        let embeddings = speakerEmbeddings[mainSpeaker]!
        // Simple average
        var avg = [Float](repeating: 0, count: 256)
        for emb in embeddings {
            vDSP_vadd(avg, 1, emb, 1, &avg, 1, 256)
        }
        var count = Float(embeddings.count)
        vDSP_vsdiv(avg, 1, &count, &avg, 1, 256)
        
        // Normalize
        return l2Normalize(avg)
    }

    private static func l2Normalize(_ vector: [Float]) -> [Float] {
        var vector = vector
        var sumSq: Float = 0
        vDSP_svesq(vector, 1, &sumSq, vDSP_Length(vector.count))
        let norm = sqrt(sumSq)
        if norm > 1e-6 {
            var divisor = norm
            vDSP_vsdiv(vector, 1, &divisor, &vector, 1, vDSP_Length(vector.count))
        }
        return vector
    }
}
