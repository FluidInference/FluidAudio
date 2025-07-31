#if os(macOS)
    import AVFoundation
    import FluidAudio
    import Foundation

    func printUsage() {
        print(
            """
            FluidAudio CLI

            Usage: fluidaudio <command> [options]

            Commands:
                process                 Process a single audio file for diarization
                diarization-benchmark   Run diarization benchmark on evaluation datasets
                vad-benchmark           Run VAD-specific benchmark
                asr-benchmark           Run ASR benchmark on LibriSpeech
                realtime-transcribe     Realtime transcription simulation
                streaming-transcribe    Test new StreamingAsrManager API
                multi-stream            Test multi-stream ASR with shared models
                download                Download evaluation datasets
                help                    Show this help message

            Run 'fluidaudio <command> --help' for command-specific options.

            Examples:
                fluidaudio process audio.wav --output results.json

                fluidaudio diarization-benchmark --dataset ami-sdm

                fluidaudio asr-benchmark --subset test-clean --max-files 100

                fluidaudio realtime-transcribe audio.wav --low-latency

                fluidaudio download --dataset ami-sdm
            """
        )
    }

    // Main entry point
    let arguments = CommandLine.arguments

    guard arguments.count > 1 else {
        printUsage()
        exit(1)
    }

    let command = arguments[1]
    let semaphore = DispatchSemaphore(value: 0)

    // Use Task to handle async commands
    Task {
        switch command {
        case "diarization-benchmark":
            await DiarizationBenchmark.run(arguments: Array(arguments.dropFirst(2)))
        case "vad-benchmark":
            await VadBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
        case "asr-benchmark":
            print("DEBUG: asr-benchmark command received")
            if #available(macOS 13.0, *) {
                print("DEBUG: macOS version check passed")
                await ASRBenchmark.runASRBenchmark(arguments: Array(arguments.dropFirst(2)))
            } else {
                print("❌ ASR benchmark requires macOS 13.0 or later")
                exit(1)
            }
        case "realtime-transcribe":
            if #available(macOS 13.0, *) {
                await RealtimeTranscribeCommand.run(arguments: Array(arguments.dropFirst(2)))
            } else {
                print("❌ Realtime transcribe requires macOS 13.0 or later")
                exit(1)
            }
        case "test-transcribe":
            if #available(macOS 13.0, *) {
                await testTranscribe(arguments: Array(arguments.dropFirst(2)))
            } else {
                print("❌ Test transcribe requires macOS 13.0 or later")
                exit(1)
            }
        case "streaming-transcribe":
            if #available(macOS 13.0, *) {
                await StreamingTranscribeCommand.run(arguments: Array(arguments.dropFirst(2)))
            } else {
                print("❌ Streaming transcribe requires macOS 13.0 or later")
                exit(1)
            }
        case "multi-stream":
            if #available(macOS 13.0, *) {
                await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))
            } else {
                print("❌ Multi-stream requires macOS 13.0 or later")
                exit(1)
            }
        case "process":
            await ProcessCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "download":
            await DownloadCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "help", "--help", "-h":
            printUsage()
            exit(0)
        default:
            print("❌ Unknown command: \(command)")
            printUsage()
            exit(1)
        }
        
        semaphore.signal()
    }

    // Wait for async task to complete
    semaphore.wait()
    
    // Test transcribe function
    @available(macOS 13.0, *)
    func testTranscribe(arguments: [String]) async {
        guard !arguments.isEmpty else {
            print("Usage: fluidaudio test-transcribe <audio-file>")
            return
        }
        
        let audioFile = arguments[0]
        
        do {
            // Load audio
            print("Loading audio file: \(audioFile)")
            let audioFileURL = URL(fileURLWithPath: audioFile)
            let audioFileHandle = try AVAudioFile(forReading: audioFileURL)
            let format = audioFileHandle.processingFormat
            let frameCount = AVAudioFrameCount(audioFileHandle.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                print("Failed to create buffer")
                return
            }
            
            try audioFileHandle.read(into: buffer)
            
            // Convert to mono 16kHz
            var audioSamples: [Float] = []
            let channelData = buffer.floatChannelData!
            let channelCount = Int(format.channelCount)
            let frameLength = Int(buffer.frameLength)
            
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += channelData[channel][frame]
                }
                audioSamples.append(sum / Float(channelCount))
            }
            
            // Resample to 16kHz if needed
            if format.sampleRate != 16000 {
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
            
            // Load models
            print("Loading ASR models...")
            let models = try await AsrModels.downloadAndLoad()
            
            // Create ASR manager
            let asrManager = AsrManager()
            try await asrManager.initialize(models: models)
            
            // Test 1: Transcribe full audio at once
            print("\n=== TEST 1: Full file transcription ===")
            let fullResult = try await asrManager.transcribe(audioSamples)
            print("Full transcription: '\(fullResult.text)'")
            print("Processing time: \(String(format: "%.3f", fullResult.processingTime))s")
            
            // Test 2: Transcribe with 1.5s chunks (decoder state preserved)
            print("\n=== TEST 2: 1.5s chunks with decoder state ===")
            try await asrManager.resetDecoderState(for: .microphone)
            
            let chunkSize = Int(1.5 * 16000)
            var position = 0
            var chunkIndex = 0
            
            while position < audioSamples.count {
                let endPos = min(position + chunkSize, audioSamples.count)
                let chunk = Array(audioSamples[position..<endPos])
                
                let chunkResult = try await asrManager.transcribe(chunk)
                print("Chunk \(chunkIndex) [\(position/16000)s-\(endPos/16000)s]: '\(chunkResult.text)'")
                
                position = endPos
                chunkIndex += 1
            }
            
            // Test 2b: Transcribe with 2.5s chunks (decoder state preserved)
            print("\n=== TEST 2b: 2.5s chunks with decoder state ===")
            try await asrManager.resetDecoderState(for: .microphone)
            
            let largerChunkSize = Int(2.5 * 16000)
            position = 0
            chunkIndex = 0
            
            while position < audioSamples.count {
                let endPos = min(position + largerChunkSize, audioSamples.count)
                let chunk = Array(audioSamples[position..<endPos])
                
                let chunkResult = try await asrManager.transcribe(chunk)
                print("Chunk \(chunkIndex) [\(position/16000)s-\(endPos/16000)s]: '\(chunkResult.text)'")
                
                position = endPos
                chunkIndex += 1
            }
            
            // Test 3: Transcribe with chunks but reset decoder state each time
            print("\n=== TEST 3: 1.5s chunks with decoder reset ===")
            position = 0
            chunkIndex = 0
            
            while position < audioSamples.count {
                try await asrManager.resetDecoderState(for: .microphone)
                
                let endPos = min(position + chunkSize, audioSamples.count)
                let chunk = Array(audioSamples[position..<endPos])
                
                let chunkResult = try await asrManager.transcribe(chunk)
                print("Chunk \(chunkIndex) [\(position/16000)s-\(endPos/16000)s]: '\(chunkResult.text)'")
                
                position = endPos
                chunkIndex += 1
            }
            
        } catch {
            print("Error: \(error)")
        }
    }
#else
    #error("FluidAudioCLI is only supported on macOS")
#endif
