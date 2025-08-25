//
// VoiceProcessingTestApp - main.swift
// FluidAudio Voice Processing Compatibility Test
//
// This app reproduces the exact voice processing issue described by the user
// and tests FluidAudio's handling of the timestamp validation error.
//
// Usage: swift run voice-processing-test
//

import AVFoundation
import Cocoa
import FluidAudio
import Foundation

@available(macOS 13.0, *)
class VoiceProcessingTestApp: NSObject, ObservableObject {

    // Audio components
    private let audioEngine = AVAudioEngine()
    private var inputNode: AVAudioInputNode { audioEngine.inputNode }

    // FluidAudio components
    private var streamingManager: StreamingAsrManager?
    var isFluidAudioAvailable = false

    // Test state
    @Published var isRecording = false
    @Published var isVoiceProcessingEnabled = false
    @Published var currentFormat = ""
    @Published var errorLog: [String] = []
    @Published var transcriptionLog: [String] = []

    // Full transcription
    private var fullTranscript = ""

    // Diagnostic data
    private var tapInstalled = false
    private var bufferCount = 0
    private var timestampErrorCount = 0
    private var validTimestampCount = 0

    override init() {
        super.init()
        setupFluidAudio()
        requestMicrophonePermission()
    }

    // MARK: - Setup

    private func setupFluidAudio() {
        Task {
            do {
                log("Initializing FluidAudio...")
                log("Downloading models if needed (this may take a moment)...")
                streamingManager = StreamingAsrManager(config: .default)
                log("StreamingAsrManager created, starting...")
                try await streamingManager?.start()
                log("StreamingAsrManager started successfully")

                // Listen for transcription updates
                if let streamingManager = streamingManager {
                    Task {
                        for await update in await streamingManager.transcriptionUpdates {
                            await MainActor.run {
                                let logEntry = "[\(update.isConfirmed ? "CONFIRMED" : "HYPOTHESIS")] \(update.text)"
                                self.transcriptionLog.append(logEntry)

                                // Update full transcript
                                if update.isConfirmed {
                                    self.fullTranscript = update.text
                                }

                                // Print transcription in real-time
                                print("\n>>> TRANSCRIPTION: \(update.text)")
                                print(
                                    "    (Confidence: \(String(format: "%.2f", update.confidence)), Confirmed: \(update.isConfirmed))"
                                )
                            }
                        }
                    }
                }

                await MainActor.run {
                    self.isFluidAudioAvailable = true
                    self.log("FluidAudio initialized successfully")
                }
            } catch {
                await MainActor.run {
                    self.log("FluidAudio initialization failed: \(error)")
                    self.isFluidAudioAvailable = false
                }
            }
        }
    }

    private func requestMicrophonePermission() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            log("Microphone permission granted")
        case .notDetermined:
            log("Requesting microphone permission...")
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.log("Microphone permission granted")
                    } else {
                        self.log("Microphone permission denied")
                    }
                }
            }
        case .denied, .restricted:
            log("Microphone permission denied or restricted")
        @unknown default:
            log("Unknown microphone permission status")
        }
    }

    // MARK: - Voice Processing Setup (Exact User Configuration)

    func setupVoiceProcessing() throws {
        log("Setting up voice processing...")

        do {
            try inputNode.setVoiceProcessingEnabled(true)
            inputNode.volume = 1.0

            // Try to disable voice processing bypass mode which might help with timestamps
            inputNode.isVoiceProcessingBypassed = false

            log("Voice processing enabled")

            // Configure ducking for macOS 14+ to prevent microphone muting
            if #available(macOS 14.0, *) {
                // Create ducking configuration with minimal ducking
                let duckingConfig = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false,  // Disable dynamic ducking
                    duckingLevel: .min  // Use minimum ducking level
                )

                inputNode.voiceProcessingOtherAudioDuckingConfiguration = duckingConfig
                inputNode.isVoiceProcessingAGCEnabled = false

                log("Voice processing ducking configured for macOS 14+")
            }

            isVoiceProcessingEnabled = true

        } catch {
            log("Failed to enable voice processing: \(error)")
            throw error
        }
    }

    func disableVoiceProcessing() throws {
        log("Disabling voice processing...")

        do {
            try inputNode.setVoiceProcessingEnabled(false)
            isVoiceProcessingEnabled = false
            log("Voice processing disabled")
        } catch {
            log("Failed to disable voice processing: \(error)")
            throw error
        }
    }

    // MARK: - Audio Recording

    func startRecording() throws {
        guard !isRecording else { return }

        log("Starting recording...")

        // Clear previous transcript
        fullTranscript = ""
        transcriptionLog.removeAll()

        // Remove existing tap if any
        if tapInstalled {
            inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }

        // Reset diagnostic counters
        bufferCount = 0
        timestampErrorCount = 0
        validTimestampCount = 0

        // Get the format AFTER setting up voice processing
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        updateCurrentFormat(recordingFormat)

        log("Recording format: \(recordingFormat)")
        log("  - Sample rate: \(recordingFormat.sampleRate) Hz")
        log("  - Channels: \(recordingFormat.channelCount)")
        log("  - Format: \(recordingFormat.commonFormat.rawValue)")
        log("  - Interleaved: \(recordingFormat.isInterleaved)")

        // Install tap with diagnostic logging
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, time in
            self?.processAudioBuffer(buffer, at: time, source: .microphone)
        }

        tapInstalled = true

        // Start the audio engine
        try audioEngine.start()

        isRecording = true
        log("Recording started - SPEAK NOW!")
    }

    func stopRecording() async -> String {
        guard isRecording else { return "" }

        log("Stopping recording...")

        audioEngine.stop()

        if tapInstalled {
            inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }

        isRecording = false

        // Get final transcription
        var finalText = ""
        if let streamingManager = streamingManager {
            do {
                finalText = try await streamingManager.finish()
            } catch {
                log("Error getting final transcription: \(error)")
            }
        }

        // Log diagnostic summary
        log("Recording session summary:")
        log("  - Total buffers processed: \(bufferCount)")
        log("  - Valid timestamps: \(validTimestampCount)")
        log("  - Invalid timestamps: \(timestampErrorCount)")
        if bufferCount > 0 {
            let errorRate = Double(timestampErrorCount) / Double(bufferCount) * 100.0
            log("  - Timestamp error rate: \(String(format: "%.1f", errorRate))%")
        }

        log("Recording stopped")
        return finalText
    }

    // MARK: - Audio Processing (Exact User Pattern)

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, at time: AVAudioTime, source: AudioSource) {
        bufferCount += 1

        // Diagnostic logging every 100 buffers to avoid spam
        if bufferCount % 100 == 0 {
            print(".", terminator: "")
            fflush(stdout)
        }

        // Check timestamp validity (this is where the error occurs)
        let hasValidSampleTime = time.sampleTime != AVAudioFramePosition.max && time.sampleTime >= 0
        let hasValidHostTime = time.hostTime != 0

        if hasValidSampleTime && hasValidHostTime {
            validTimestampCount += 1
        } else {
            timestampErrorCount += 1

            // Log timestamp issues (but not every single one to avoid spam)
            if timestampErrorCount <= 5 || timestampErrorCount % 50 == 0 {
                DispatchQueue.main.async {
                    let sampleTimeStr = hasValidSampleTime ? "valid" : "INVALID"
                    let hostTimeStr = hasValidHostTime ? "valid" : "INVALID"
                    self.log(
                        "Timestamp issue #\(self.timestampErrorCount): sampleTime=\(sampleTimeStr), hostTime=\(hostTimeStr)"
                    )

                    if self.timestampErrorCount == 5 {
                        self.log("   (Further timestamp errors will be logged every 50 occurrences)")
                    }
                }
            }
        }

        // Copy buffer to avoid any threading issues
        guard let audioBuffer = buffer.copy() as? AVAudioPCMBuffer else {
            DispatchQueue.main.async {
                self.log("Failed to copy audio buffer")
            }
            return
        }

        // The key test: Can FluidAudio handle this regardless of timestamp validity?
        if let streamingManager = streamingManager {
            Task {
                // This should work regardless of timestamp validity
                // FluidAudio's streamAudio method only uses the buffer, not the timestamp
                await streamingManager.streamAudio(audioBuffer)
            }
        } else {
            // Log if streaming manager is not available
            if bufferCount == 1 {
                DispatchQueue.main.async {
                    self.log("Warning: StreamingManager not available for audio processing")
                }
            }
        }
    }

    // MARK: - Utility

    private func updateCurrentFormat(_ format: AVAudioFormat) {
        currentFormat = "\(Int(format.sampleRate))Hz \(format.channelCount)ch \(format.commonFormat.rawValue)"
    }

    private func log(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let logEntry = "[\(timestamp)] \(message)"

        errorLog.append(logEntry)
        print(logEntry)

        // Keep log size manageable
        if errorLog.count > 500 {
            errorLog.removeFirst(100)
        }
    }

    // MARK: - Test Scenarios

    func runTestScenarioA() async -> String {
        log("TEST SCENARIO A: Voice Processing Enabled")
        do {
            if isRecording { _ = await stopRecording() }

            try setupVoiceProcessing()
            do {
                try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds
            } catch {}
            try startRecording()

            print("\n========================================")
            print("RECORDING WITH VOICE PROCESSING ENABLED")
            print("Please speak clearly for 10 seconds...")
            print("========================================\n")

            do {
                try await Task.sleep(nanoseconds: 10_000_000_000)  // 10 seconds
            } catch {}

            let finalText = await stopRecording()
            log("Test Scenario A completed")
            return finalText

        } catch {
            log("Test Scenario A failed: \(error)")
            return ""
        }
    }

    func runTestScenarioB() async -> String {
        log("TEST SCENARIO B: Voice Processing Disabled")
        do {
            if isRecording { _ = await stopRecording() }

            try disableVoiceProcessing()
            do {
                try await Task.sleep(nanoseconds: 500_000_000)  // 0.5 seconds
            } catch {}
            try startRecording()

            print("\n========================================")
            print("RECORDING WITHOUT VOICE PROCESSING")
            print("Please speak clearly for 10 seconds...")
            print("========================================\n")

            do {
                try await Task.sleep(nanoseconds: 10_000_000_000)  // 10 seconds
            } catch {}

            let finalText = await stopRecording()
            log("Test Scenario B completed")
            return finalText

        } catch {
            log("Test Scenario B failed: \(error)")
            return ""
        }
    }

    func printStatus() {
        print("\nCurrent Status:")
        print("- FluidAudio Available: \(isFluidAudioAvailable ? "YES" : "NO")")
        print("- Recording: \(isRecording ? "ON" : "OFF")")
        print("- Voice Processing: \(isVoiceProcessingEnabled ? "ON" : "OFF")")
        print("- Current Format: \(currentFormat.isEmpty ? "N/A" : currentFormat)")
    }
}

// MARK: - AudioSource Extension
extension VoiceProcessingTestApp {
    enum AudioSource {
        case microphone
        case system
    }
}

// MARK: - Main Entry Point

@available(macOS 13.0, *)
func main() async {
    print("FluidAudio Voice Processing Test App")
    print("====================================")
    print("This test will record audio and transcribe it in real-time")
    print("Testing both with and without voice processing\n")

    let app = VoiceProcessingTestApp()

    // Wait for FluidAudio to actually initialize
    print("Waiting for FluidAudio to initialize (downloading models if needed)...")

    var waitTime = 0
    while !app.isFluidAudioAvailable && waitTime < 60 {
        do {
            try await Task.sleep(nanoseconds: 1_000_000_000)  // 1 second
            waitTime += 1
            if waitTime % 5 == 0 {
                print("Still waiting for FluidAudio... (\(waitTime)s elapsed)")
            }
        } catch {
            print("Sleep interrupted: \(error)")
        }
    }

    if !app.isFluidAudioAvailable {
        print("ERROR: FluidAudio failed to initialize after 60 seconds")
        exit(1)
    }

    print("FluidAudio ready!")
    app.printStatus()
    print()

    // Run test without voice processing first
    print("\n=== TEST 1: WITHOUT Voice Processing ===")
    let transcriptB = await app.runTestScenarioB()
    print("\nFINAL TRANSCRIPT (No Voice Processing):")
    print(transcriptB.isEmpty ? "(No transcription received)" : transcriptB)
    print()

    // Wait between tests
    do {
        try await Task.sleep(nanoseconds: 2_000_000_000)  // 2 seconds
    } catch {}

    // Run test with voice processing
    print("\n=== TEST 2: WITH Voice Processing ===")
    let transcriptA = await app.runTestScenarioA()
    print("\nFINAL TRANSCRIPT (With Voice Processing):")
    print(transcriptA.isEmpty ? "(No transcription received)" : transcriptA)
    print()

    app.printStatus()
    print()

    print("============== TEST RESULTS ==============")
    print("Without Voice Processing:")
    print("  Transcript: \(transcriptB.isEmpty ? "NONE" : "SUCCESS")")
    print("\nWith Voice Processing:")
    print("  Transcript: \(transcriptA.isEmpty ? "NONE" : "SUCCESS")")
    print("\nKey findings:")
    print("- Voice processing changed format to: \(app.currentFormat)")
    print("- Both scenarios should produce transcripts")
    print("- FluidAudio handles format changes automatically")
    print("==========================================")
}

// Check macOS version compatibility
if #available(macOS 13.0, *) {
    await main()
} else {
    print("This app requires macOS 13.0 or later")
    exit(1)
}
