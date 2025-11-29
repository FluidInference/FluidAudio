import AVFoundation
import CoreML
import FluidAudio
import Foundation

struct CanaryStreamCommand {
    static func run(arguments: [String]) async {
        guard let inputPath = arguments.first else {
            print("Usage: fluidaudio canary-stream <input.wav>")
            return
        }
        
        let fileURL = URL(fileURLWithPath: inputPath)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("File not found: \(inputPath)")
            return
        }
        
        print("Initializing Canary Manager...")
        let manager = CanaryManager()
        
        do {
            // Load models
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            
            // Assuming models are in cache, CanaryModels.load will find them
            // We need to point to the cache directory
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let cacheDir = appSupport.appendingPathComponent("FluidAudio").appendingPathComponent("canary")
            
            print("Loading models from \(cacheDir.path)...")
            let models = try await CanaryModels.load(from: cacheDir, configuration: config)
            
            manager.initialize(models: models)
            print("Models loaded.")
            
            // Read audio file
            let audioFile = try AVAudioFile(forReading: fileURL)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("Failed to create buffer")
                return
            }
            
            try audioFile.read(into: buffer)
            
            // Convert to 16kHz Float array
            // Simple conversion if needed, or assume input is compatible for test
            // For robustness, let's just take the first channel float data
            // Ideally we should resample, but for quick test let's assume 16kHz or close enough, 
            // or just take raw samples if format matches.
            // jfk.wav is usually 16kHz mono.
            
            guard let floatChannelData = buffer.floatChannelData else {
                print("No float channel data")
                return
            }
            
            let channelData = floatChannelData[0]
            let sampleCount = Int(buffer.frameLength)
            var samples = Array(UnsafeBufferPointer(start: channelData, count: sampleCount))
            
            // Append silence to fill the window (need at least 14s)
            // 5 seconds of silence
            let silence = Array(repeating: Float(0.0), count: 16000 * 5)
            samples.append(contentsOf: silence)
            
            print("Audio loaded: \(samples.count) samples (including silence)")
            
            // Stream in 2s chunks (32000 samples)
            let chunkSize = 32000
            var offset = 0
            
            print("Starting streaming...")
            let startTime = Date()
            
            while offset < samples.count {
                let end = min(offset + chunkSize, samples.count)
                let chunk = Array(samples[offset..<end])
                
                // Pad last chunk if needed? Manager handles buffering.
                
                let text = try await manager.processStreamingChunk(chunk)
                if !text.isEmpty {
                    print("Partial: \(text)")
                }
                
                offset += chunkSize
                // Simulate real-time?
                // try await Task.sleep(nanoseconds: 2 * 1_000_000_000)
            }
            
            let elapsed = Date().timeIntervalSince(startTime)
            print("Finished in \(String(format: "%.2f", elapsed))s")
            
        } catch {
            print("Error: \(error)")
        }
    }
}
