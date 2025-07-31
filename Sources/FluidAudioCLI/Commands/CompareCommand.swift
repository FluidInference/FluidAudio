#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Command to compare streaming vs non-streaming ASR
@available(macOS 13.0, *)
enum CompareCommand {
    
    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            print("❌ No audio file specified")
            print("Usage: fluidaudio compare <audio_file>")
            return
        }
        
        let audioFile = arguments[0]
        
        do {
            print("Loading audio file...")
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("Failed to create buffer")
                return
            }
            
            try audioFileHandle.read(into: buffer)
            
            // Convert to mono 16kHz for ASR
            var audioSamples: [Float] = []
            let channelData = buffer.floatChannelData!
            let channelCount = Int(format.channelCount)
            let frameLength = Int(buffer.frameLength)
            
            print("Converting \(channelCount) channels to mono...")
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                audioSamples.append(sum / Float(channelCount))
            }
            
            // Resample to 16kHz if needed
            if format.sampleRate != 16000 {
                print("Resampling from \(format.sampleRate)Hz to 16000Hz...")
                let ratio = Float(16000) / Float(format.sampleRate)
                let targetLength = Int(Float(audioSamples.count) * ratio)
                var resampled = [Float](repeating: 0, count: targetLength)
                
                for i in 0..<targetLength {
                    let sourceIndex = Float(i) / ratio
                    let index = Int(sourceIndex)
                    let fraction = sourceIndex - Float(index)
                    
                    if index < audioSamples.count - 1 {
                        resampled[i] = audioSamples[index] * (1 - fraction) + audioSamples[index + 1] * fraction
                    } else if index < audioSamples.count {
                        resampled[i] = audioSamples[index]
                    }
                }
                audioSamples = resampled
            }
            
            print("Audio duration: \(Float(audioSamples.count) / 16000.0) seconds")
            print("Sample count: \(audioSamples.count)")
            
            // Load models once
            print("\nLoading ASR models...")
            let models = try await AsrModels.downloadAndLoad()
            
            // Test 1: Direct ASR (non-streaming) with full audio
            print("\n=== TEST 1: Direct ASR (Full Audio) ===")
            let asrManager = AsrManager()
            try await asrManager.initialize(models: models)
            
            let startDirect = Date()
            let directResult = try await asrManager.transcribe(audioSamples)
            let directTime = Date().timeIntervalSince(startDirect)
            
            print("Direct transcription word count: \(directResult.text.split(separator: " ").count)")
            print("Processing time: \(String(format: "%.2f", directTime))s")
            
            // Save direct result
            try directResult.text.write(toFile: "yc_direct.txt", atomically: true, encoding: .utf8)
            
            // Test 2: Direct ASR with 10-second chunks (WITHOUT resetting decoder state)
            print("\n=== TEST 2: Direct ASR (10s chunks - state preserved) ===")
            // DO NOT reset decoder state - it should be preserved across chunks
            
            let chunkSize = Int(10.0 * 16000) // 10 seconds
            var position = 0
            var chunkTexts: [String] = []
            
            // Create a fresh ASR manager for chunked test
            let chunkedAsrManager = AsrManager()
            try await chunkedAsrManager.initialize(models: models)
            
            while position < audioSamples.count {
                let endPos = min(position + chunkSize, audioSamples.count)
                let chunk = Array(audioSamples[position..<endPos])
                
                let chunkResult = try await chunkedAsrManager.transcribe(chunk)
                chunkTexts.append(chunkResult.text)
                print("Chunk \(chunkTexts.count): \(chunkResult.text.split(separator: " ").count) words")
                
                position = endPos
            }
            
            let chunkedText = chunkTexts.joined(separator: " ")
            print("Total chunked word count: \(chunkedText.split(separator: " ").count)")
            
            // Save chunked result
            try chunkedText.write(toFile: "yc_chunked.txt", atomically: true, encoding: .utf8)
            
            // Test 3: Streaming ASR
            print("\n=== TEST 3: Streaming ASR ===")
            let streamingAsr = StreamingAsrManager()
            try await streamingAsr.start()
            
            var confirmedTexts: [String] = []
            var volatileTexts: [String] = []
            
            // Listen for updates
            let updateTask = Task {
                for await update in await streamingAsr.transcriptionUpdates {
                    if update.isConfirmed {
                        confirmedTexts.append(update.text)
                    } else {
                        volatileTexts.append(update.text)
                    }
                }
            }
            
            // Stream the entire audio
            await streamingAsr.streamAudio(buffer)
            let streamingResult = try await streamingAsr.finish()
            updateTask.cancel()
            
            print("Streaming transcription word count: \(streamingResult.split(separator: " ").count)")
            print("Confirmed segments: \(confirmedTexts.count)")
            print("Volatile segments: \(volatileTexts.count)")
            
            // Save streaming result
            try streamingResult.write(toFile: "yc_streaming.txt", atomically: true, encoding: .utf8)
            
            // Test 4: Direct ASR with smaller chunks (2.5s like streaming)
            print("\n=== TEST 4: Direct ASR (2.5s chunks - state preserved) ===")
            let smallChunkSize = Int(2.5 * 16000) // 2.5 seconds
            position = 0
            var smallChunkTexts: [String] = []
            
            // Create another fresh ASR manager
            let smallChunkedAsrManager = AsrManager()
            try await smallChunkedAsrManager.initialize(models: models)
            
            while position < audioSamples.count {
                let endPos = min(position + smallChunkSize, audioSamples.count)
                let chunk = Array(audioSamples[position..<endPos])
                
                let chunkResult = try await smallChunkedAsrManager.transcribe(chunk)
                if !chunkResult.text.isEmpty {
                    smallChunkTexts.append(chunkResult.text)
                }
                
                position = endPos
            }
            
            let smallChunkedText = smallChunkTexts.joined(separator: " ")
            print("Total small chunked word count: \(smallChunkedText.split(separator: " ").count)")
            print("Non-empty chunks: \(smallChunkTexts.count)")
            
            // Save small chunked result
            try smallChunkedText.write(toFile: "yc_small_chunked.txt", atomically: true, encoding: .utf8)
            
            // Compare results
            print("\n=== COMPARISON ===")
            print("Direct (full):         \(directResult.text.split(separator: " ").count) words")
            print("Direct (10s chunks):   \(chunkedText.split(separator: " ").count) words")
            print("Direct (2.5s chunks):  \(smallChunkedText.split(separator: " ").count) words")
            print("Streaming:             \(streamingResult.split(separator: " ").count) words")
            print("Reference:             10042 words")
            
        } catch {
            print("Error: \(error)")
        }
    }
}
#endif