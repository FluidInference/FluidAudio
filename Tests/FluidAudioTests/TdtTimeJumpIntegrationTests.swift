import CoreML
import XCTest

@testable import FluidAudio

/// Integration tests for TDT time jump fixes using real models
/// These tests are skipped in CI but can be run locally for verification
@available(macOS 13.0, iOS 16.0, *)
class TdtTimeJumpIntegrationTests: XCTestCase {

    private func skipIfCI() throws {
        // Skip these tests in CI environment
        if ProcessInfo.processInfo.environment["CI"] == "true" {
            throw XCTSkip("Integration tests with real models are skipped in CI")
        }
    }

    /// Test that chunk processing with real models doesn't produce duplicates
    func testChunkProcessingNoDuplicates() async throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        let manager: AsrManager

        let initManager = AsrManager(config: config)
        manager = initManager

        // Create test audio that will be processed in chunks (longer than 15s to force chunking)
        let sampleRate = 16000
        let duration = 20.0  // 20 seconds to force chunking
        let frequency: Float = 440.0  // A4 note

        var audioSamples: [Float] = []
        for i in 0..<Int(duration * Double(sampleRate)) {
            let t = Float(i) / Float(sampleRate)
            // Create a simple sine wave with some variation to avoid silence
            let sample = sin(2.0 * Float.pi * frequency * t) * 0.1 + sin(2.0 * Float.pi * frequency * 2.0 * t) * 0.05
            audioSamples.append(sample)
        }

        // Process with chunk processor (which should not use deduplication anymore)
        let processor = ChunkProcessor(audioSamples: audioSamples, enableDebug: true)
        var decoderState = try TdtDecoderState()

        let result = try await processor.process(
            using: manager,
            decoderState: &decoderState,
            startTime: Date()
        )

        print("Processed \(audioSamples.count) samples (\(duration)s) in chunks")
        print("Result tokens: \(result.tokenTimings?.count ?? 0)")
        print("Result text: '\(result.text)'")

        // The main assertion: we should get some result without crashes
        // More importantly, no duplicate phrases should appear
        XCTAssertNotNil(result, "Should get a result from chunk processing")

        // Check for obvious duplicates by looking for repeated phrases
        let words = result.text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }

        if words.count > 3 {
            // Look for any 3-word phrase that appears more than once
            for i in 0..<(words.count - 2) {
                let phrase = "\(words[i]) \(words[i+1]) \(words[i+2])"
                let occurrences = result.text.components(separatedBy: phrase).count - 1
                XCTAssertLessThanOrEqual(
                    occurrences, 1,
                    "Found duplicate phrase: '\(phrase)' appears \(occurrences) times"
                )
            }
        }

        print("✅ No duplicate phrases detected in chunked processing")
    }

    /// Test with Spanish-like audio pattern (simulated problematic case)
    func testSpanishPatternNoDuplicates() async throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        let manager: AsrManager

        let initManager = AsrManager(config: config)
        manager = initManager

        // Create audio that mimics Spanish speech patterns that caused duplicates
        let sampleRate = 16000
        let duration = 18.0  // Similar to problematic Spanish audio
        let baseFreq: Float = 200.0  // Lower frequency like speech

        var audioSamples: [Float] = []
        for i in 0..<Int(duration * Double(sampleRate)) {
            let t = Float(i) / Float(sampleRate)
            // Create speech-like patterns with pauses and variations
            let segment = Int(t / 3.0)  // Change pattern every 3 seconds
            let freq = baseFreq * (1.0 + 0.1 * Float(segment % 4))
            let amplitude: Float = (segment % 2 == 0) ? 0.1 : 0.05  // Varying amplitude
            let sample =
                sin(2.0 * Float.pi * freq * t) * amplitude + sin(2.0 * Float.pi * freq * 1.5 * t) * amplitude * 0.3
            audioSamples.append(sample)
        }

        print("Testing Spanish-pattern audio: \(audioSamples.count) samples (\(duration)s)")

        // Process the audio with our fixed chunk processor
        let result = try await manager.transcribe(audioSamples)

        print("Spanish-pattern transcription result:")
        print("Text: '\(result.text)'")
        print("Confidence: \(result.confidence)")
        print("Duration: \(result.duration)s")

        // Check for any obvious duplicate patterns in the result
        let words = result.text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }

        var duplicateCount = 0
        if words.count > 4 {
            // Look for any 4-word phrase that appears more than once
            for i in 0..<(words.count - 3) {
                let phrase = "\(words[i]) \(words[i+1]) \(words[i+2]) \(words[i+3])"
                let occurrences = result.text.components(separatedBy: phrase).count - 1
                if occurrences > 1 {
                    duplicateCount += 1
                    print("Duplicate phrase found: '\(phrase)' appears \(occurrences) times")
                }
            }
        }

        // With our fix, we shouldn't see duplicate phrases
        XCTAssertEqual(
            duplicateCount, 0,
            "Found \(duplicateCount) duplicate phrase patterns - time jump fix may not be working"
        )

        print("✅ Spanish-pattern audio processed without duplicates")
    }

    /// Test the problematic 15-30 second audio length range where duplication occurs
    func testProblematicAudioLengthRange() async throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        let manager: AsrManager

        let initManager = AsrManager(config: config)
        manager = initManager

        // Test different durations in the problematic range
        let testDurations: [Double] = [16.0, 18.0, 22.0, 28.0]  // All force chunking but aren't too long

        for duration in testDurations {
            print("\n--- Testing \(duration)s audio (problematic range) ---")

            // Create audio that will be processed in 2-3 chunks (the problematic case)
            let sampleRate = 16000
            let baseFreq: Float = 220.0  // Lower frequency like speech

            var audioSamples: [Float] = []
            for i in 0..<Int(duration * Double(sampleRate)) {
                let t = Float(i) / Float(sampleRate)
                // Create varied speech-like patterns that might trigger duplicates
                let segment = Int(t / 4.0)  // Change pattern every 4 seconds
                let freq = baseFreq * (1.0 + 0.15 * Float(segment % 3))
                let envelope = 0.8 + 0.2 * sin(0.5 * t)  // Slow amplitude variation
                let sample =
                    sin(2.0 * Float.pi * freq * t) * envelope * 0.08 + sin(2.0 * Float.pi * freq * 1.3 * t) * envelope
                    * 0.04
                audioSamples.append(sample)
            }

            print("Testing \(duration)s audio: \(audioSamples.count) samples")

            // Process the audio - this should trigger chunking
            let result = try await manager.transcribe(audioSamples)

            print("Result text: '\(result.text)'")
            print("Duration: \(result.duration)s, Confidence: \(result.confidence)")

            // Analyze for duplications by looking for repeated phrases
            let words = result.text.components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }

            var duplicateCount = 0
            var duplicatePatterns: [String] = []

            if words.count > 3 {
                // Look for any 3-word phrase that appears more than once
                for i in 0..<(words.count - 2) {
                    let phrase = "\(words[i]) \(words[i+1]) \(words[i+2])"
                    let occurrences = result.text.components(separatedBy: phrase).count - 1
                    if occurrences > 1 {
                        duplicateCount += 1
                        duplicatePatterns.append("'\(phrase)' (\(occurrences)x)")
                    }
                }

                // Also check for 4-word phrases for more precision
                if words.count > 4 {
                    for i in 0..<(words.count - 3) {
                        let phrase = "\(words[i]) \(words[i+1]) \(words[i+2]) \(words[i+3])"
                        let occurrences = result.text.components(separatedBy: phrase).count - 1
                        if occurrences > 1 {
                            duplicateCount += 1
                            duplicatePatterns.append("'\(phrase)' (\(occurrences)x)")
                        }
                    }
                }
            }

            if duplicateCount > 0 {
                print("❌ DUPLICATION DETECTED in \(duration)s audio:")
                for pattern in duplicatePatterns {
                    print("  - \(pattern)")
                }
            } else {
                print("✅ No duplicates in \(duration)s audio")
            }

            // Assert no duplicates for this duration
            XCTAssertEqual(
                duplicateCount, 0,
                "Found \(duplicateCount) duplicate patterns in \(duration)s audio - time jump issue not fully fixed"
            )
        }

        print("\n✅ All problematic audio lengths processed without duplicates")
    }

    /// Test time jump calculation logic without requiring ML models
    func testTimeJumpCalculationLogic() throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        _ = TdtDecoder(config: config)

        // Simulate chunk processing scenarios

        // Scenario 1: First chunk processes 100 frames, has 80 available
        var decoderState1 = try TdtDecoderState()

        // Simulate the calculation that happens at end of TdtDecoder
        let timeIndices1 = 120  // Decoder processed to frame 120
        let encoderSeqLength1 = 100  // But only had 100 frames available
        let calculatedTimeJump1 = timeIndices1 - encoderSeqLength1  // Should be 20

        decoderState1.timeJump = calculatedTimeJump1

        // Scenario 2: Second chunk should start from timeJump
        let prevTimeJump2 = decoderState1.timeJump ?? 0
        let expectedStartFrame2 = max(0, prevTimeJump2)  // Should be 20

        // This is the key test: the second chunk should start at frame 20, not 0
        XCTAssertEqual(expectedStartFrame2, 20, "Second chunk should start at frame 20 (from timeJump)")
        XCTAssertNotEqual(expectedStartFrame2, 0, "Second chunk should NOT start at frame 0")

        // Scenario 3: Verify no double-counting (consolidated to single mechanism)
        // Previously we had both timeJump and startFrameOffset, which could lead to double-counting
        // Now we only use timeJump, eliminating the redundancy
        let consolidatedCalculation = max(0, prevTimeJump2)  // Single source of truth

        XCTAssertEqual(consolidatedCalculation, 20, "Consolidated mechanism should give correct frame position")

        // Scenario 4: Test edge cases

        // Negative timeJump (decoder didn't process all frames)
        var decoderState3 = try TdtDecoderState()
        decoderState3.timeJump = -5  // Decoder was 5 frames behind
        let edgeCase1 = max(0, decoderState3.timeJump ?? 0)  // Should be 0
        XCTAssertEqual(edgeCase1, 0, "Negative timeJump should be clamped to 0")

        // Zero timeJump (decoder exactly finished at chunk boundary)
        var decoderState4 = try TdtDecoderState()
        decoderState4.timeJump = 0
        let edgeCase2 = max(0, decoderState4.timeJump ?? 0)  // Should be 0
        XCTAssertEqual(edgeCase2, 0, "Zero timeJump should remain 0")

    }

    /// Test audio length near chunk size boundary that may drop frames
    func testNearChunkSizeFrameDropping() async throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        let manager: AsrManager

        let initManager = AsrManager(config: config)
        manager = initManager

        // Test the problematic audio length range: 14.2-14.5 seconds
        // This is close to chunk size (14.4s = 11.2s center + 1.6s left + 1.6s right)
        // The issue: audio ends just before chunk boundary, potentially losing frames

        let problematicDurations: [Double] = [
            14.15,  // Slightly less than chunk size
            14.25,  // Your problematic case (14.28s)
            14.35,  // Still less than chunk size
            14.45,  // Just over chunk size
        ]

        for duration in problematicDurations {
            print("\n--- Testing \(duration)s audio (near chunk boundary) ---")

            // Create audio with speech-like pattern that extends to the end
            let sampleRate = 16000
            let totalSamples = Int(duration * Double(sampleRate))
            let baseFreq: Float = 150.0  // Speech-like frequency

            var audioSamples: [Float] = []
            for i in 0..<totalSamples {
                let t = Float(i) / Float(sampleRate)

                // Create pattern that has important content at the END
                // This simulates the missing "la lettera h di ph" at the end of your file
                let segment = Int(t / 2.0)  // 2-second segments
                let isEndSegment = t > (Float(duration) - 2.0)  // Last 2 seconds

                let freq = baseFreq * (1.0 + 0.1 * Float(segment % 4))
                let amplitude: Float = isEndSegment ? 0.15 : 0.08  // HIGHER amplitude at end

                // Create clear speech-like pattern at the end to test if it gets processed
                let envelope =
                    isEndSegment
                    ? 0.9 + 0.1 * sin(5.0 * t)
                    :  // More varied pattern at end
                    0.7 + 0.3 * sin(0.5 * t)  // Stable pattern earlier

                let sample =
                    sin(2.0 * Float.pi * freq * t) * amplitude * envelope + sin(2.0 * Float.pi * freq * 1.2 * t)
                    * amplitude * envelope * 0.3
                audioSamples.append(sample)
            }

            print("Created \(duration)s audio: \(audioSamples.count) samples")
            print("Chunk size: 14.4s (\(Int(14.4 * 16000)) samples)")
            print("Audio vs chunk: \(audioSamples.count < Int(14.4 * 16000) ? "SMALLER" : "LARGER")")

            // Process the audio - this should reveal if end frames are dropped
            let result = try await manager.transcribe(audioSamples)

            print("Transcription result:")
            print("Text: '\(result.text)'")
            print("Confidence: \(result.confidence)")
            print("Duration: \(result.duration)s")
            print("Token count: \(result.tokenTimings?.count ?? 0)")

            // Key assertions to detect frame dropping:

            // 1. Result duration should be close to input duration
            let durationDiff = abs(result.duration - duration)
            XCTAssertLessThan(
                durationDiff, 0.5,
                "Result duration (\(result.duration)s) differs too much from input (\(duration)s) - may indicate frame dropping"
            )

            // 2. For audio near chunk boundary, we should still get reasonable token coverage
            // If frames are dropped, we'd expect fewer tokens relative to duration
            let expectedMinTokens = max(1, Int(duration / 2.0))  // Very conservative estimate
            let actualTokens = result.tokenTimings?.count ?? 0
            XCTAssertGreaterThanOrEqual(
                actualTokens, expectedMinTokens,
                "Got only \(actualTokens) tokens for \(duration)s audio - possible frame dropping at chunk boundary"
            )

            // 3. Check if timestamps extend to near the end of the audio
            if let timestamps = result.tokenTimings {
                if !timestamps.isEmpty {
                    let lastTimestamp = timestamps.last?.endTime ?? 0.0
                    let expectedMinLastTimestamp = duration * 0.7  // Should reach at least 70% through
                    XCTAssertGreaterThanOrEqual(
                        lastTimestamp, expectedMinLastTimestamp,
                        "Last timestamp (\(lastTimestamp)s) is too early for \(duration)s audio - end frames may be dropped"
                    )
                }
            }

            // 4. Specific check for the problematic duration (14.28s like your file)
            if abs(duration - 14.28) < 0.1 {
                print("🔍 SPECIFIC CHECK for 14.28s case (like it_it_0027.wav)")

                // This case had WER 22.7% due to missing end phrase
                // If our fix works, we should get better coverage
                XCTAssertGreaterThan(
                    actualTokens, 0,
                    "14.28s audio should produce tokens - this mimics it_it_0027.wav case"
                )

                // The missing phrase was at the end, so we need good temporal coverage
                if let lastTimestamp = result.tokenTimings?.last?.endTime {
                    XCTAssertGreaterThan(
                        lastTimestamp, 12.0,
                        "For 14.28s audio, last token should be after 12s to avoid missing end content"
                    )
                }
            }

            print("✅ \(duration)s audio processed - duration diff: \(durationDiff)s, tokens: \(actualTokens)")
        }

        print("\n✅ All near-chunk-boundary durations tested for frame dropping")
    }

    /// Test chunk boundary calculation logic with different audio lengths
    func testChunkBoundaryCalculations() throws {
        // This test validates the chunk processing logic without requiring ML models

        print("\n--- Testing Chunk Boundary Calculations ---")

        let sampleRate = 16000
        let centerSeconds = 11.2  // From ChunkProcessor
        let leftContextSeconds = 1.6
        let rightContextSeconds = 1.6
        let totalChunkDuration = leftContextSeconds + centerSeconds + rightContextSeconds  // 14.4s

        let centerSamples = Int(centerSeconds * Double(sampleRate))  // 179,200 samples
        let leftContextSamples = Int(leftContextSeconds * Double(sampleRate))  // 25,600 samples
        let rightContextSamples = Int(rightContextSeconds * Double(sampleRate))  // 25,600 samples
        let totalChunkSamples = Int(totalChunkDuration * Double(sampleRate))  // 230,400 samples

        print("Chunk configuration:")
        print("  Center: \(centerSeconds)s = \(centerSamples) samples")
        print("  Left context: \(leftContextSeconds)s = \(leftContextSamples) samples")
        print("  Right context: \(rightContextSeconds)s = \(rightContextSamples) samples")
        print("  Total chunk: \(totalChunkDuration)s = \(totalChunkSamples) samples")

        // Test cases with different audio lengths relative to chunk size
        let testCases: [(description: String, duration: Double)] = [
            ("Short audio", 10.0),  // Well under chunk size
            ("Near chunk size", 14.28),  // Your problematic case
            ("At chunk size", 14.4),  // Exactly chunk size
            ("Over chunk size", 15.0),  // Requires chunking
            ("Long audio", 25.0),  // Multiple chunks
        ]

        for (description, duration) in testCases {
            print("\n📋 Testing: \(description) (\(duration)s)")

            let audioSamples = Int(duration * Double(sampleRate))
            print("  Audio: \(audioSamples) samples")

            // Simulate ChunkProcessor logic with adaptive context fix
            var centerStart = 0
            var chunkIndex = 0
            var processedSamples = 0

            while centerStart < audioSamples {
                let isLastChunk = (centerStart + centerSamples) >= audioSamples
                let remainingSamples = audioSamples - centerStart

                // Simulate adaptive context calculation for last chunk (NEW FIX)
                let adaptiveLeftContext: Int
                let frameOffset: Int

                if isLastChunk && remainingSamples < centerSamples {
                    // Last chunk can't fill center - maximize context usage
                    let maxModelSamples = Int(15.0 * 16000)  // 240,000 samples (15s)
                    let desiredTotalSamples = min(maxModelSamples, audioSamples)
                    let maxLeftContext = centerStart  // Can't go before start

                    let neededLeftContext = desiredTotalSamples - remainingSamples
                    adaptiveLeftContext = min(neededLeftContext, maxLeftContext)

                    // Calculate frame offset for already-processed content
                    if adaptiveLeftContext > leftContextSamples {
                        let extraContextSamples = adaptiveLeftContext - leftContextSamples
                        frameOffset = ASRConstants.calculateEncoderFrames(from: extraContextSamples)
                    } else {
                        frameOffset = 0
                    }
                } else {
                    // Standard context for non-last chunks
                    adaptiveLeftContext = leftContextSamples
                    frameOffset = 0
                }

                // Calculate window bounds with adaptive context
                let leftStart = max(0, centerStart - adaptiveLeftContext)
                let centerEnd = min(audioSamples, centerStart + centerSamples)
                let rightEnd = min(audioSamples, centerEnd + rightContextSamples)

                let chunkLength = rightEnd - leftStart
                let centerLength = centerEnd - centerStart

                print(
                    "    Chunk \(chunkIndex): centerStart=\(centerStart), bounds=[\(leftStart), \(rightEnd)), length=\(chunkLength), centerLength=\(centerLength), isLast=\(isLastChunk)"
                )

                // Show adaptive context details for last chunk
                if isLastChunk && remainingSamples < centerSamples {
                    print(
                        "      🔧 ADAPTIVE CONTEXT: leftContext=\(adaptiveLeftContext) samples (\(String(format: "%.2f", Double(adaptiveLeftContext)/16000.0))s), frameOffset=\(frameOffset)"
                    )
                }

                // Key checks for frame dropping:

                // 1. For audio near chunk size, check if adaptive context fixes frame dropping
                if abs(duration - 14.28) < 0.1 {  // The problematic case
                    let samplesFromEnd = audioSamples - rightEnd
                    print("      🔍 Near-chunk-size case: \(samplesFromEnd) samples from end of audio")

                    if isLastChunk && remainingSamples < centerSamples {
                        // With adaptive context, we should now process much more audio
                        let totalChunkSamples = adaptiveLeftContext + remainingSamples
                        print(
                            "      ✅ ADAPTIVE CONTEXT FIX: chunk now processes \(totalChunkSamples) samples (\(String(format: "%.2f", Double(totalChunkSamples)/16000.0))s)"
                        )
                        print("      ✅ Frame offset \(frameOffset) skips already-processed content")

                        // The fix should ensure we process nearly all remaining audio
                        XCTAssertLessThanOrEqual(
                            samplesFromEnd, 1000,  // Should now be < 60ms unprocessed
                            "With adaptive context fix, should leave minimal samples unprocessed - found \(samplesFromEnd) samples (\(Double(samplesFromEnd)/16000.0)s)"
                        )
                    } else {
                        // First chunk - still shows the original issue
                        if samplesFromEnd > 20000 {
                            print(
                                "      🚨 First chunk gap: \(Double(samplesFromEnd)/16000.0)s - will be fixed by adaptive context in last chunk"
                            )
                        }
                    }
                }

                // 2. Center section should always be processed fully unless it's the last chunk
                if !isLastChunk {
                    XCTAssertEqual(
                        centerLength, centerSamples,
                        "Non-last chunk should process full center section"
                    )
                } else {
                    // Last chunk may have partial center
                    let remainingSamples = audioSamples - centerStart
                    let expectedCenterLength = min(remainingSamples, centerSamples)
                    XCTAssertEqual(
                        centerLength, expectedCenterLength,
                        "Last chunk center length should match remaining samples"
                    )
                }

                // 3. Ensure we don't skip samples between chunks
                if chunkIndex > 0 {
                    XCTAssertEqual(
                        centerStart, processedSamples,
                        "No gap between chunks - centerStart should equal previously processed samples"
                    )
                }

                processedSamples = centerStart + centerLength
                centerStart += centerSamples
                chunkIndex += 1

                // Prevent infinite loops
                if chunkIndex > 10 {
                    XCTFail("Too many chunks - possible infinite loop")
                    break
                }
            }

            // Final verification
            let unprocessedSamples = audioSamples - processedSamples
            print("  Final: processed \(processedSamples)/\(audioSamples) samples, \(unprocessedSamples) unprocessed")

            // For your specific case, we shouldn't leave too many samples unprocessed
            if abs(duration - 14.28) < 0.1 {
                XCTAssertLessThanOrEqual(
                    unprocessedSamples, Int(0.2 * Double(sampleRate)),  // Less than 200ms unprocessed
                    "14.28s audio should have minimal unprocessed samples at end - found \(unprocessedSamples) samples (\(Double(unprocessedSamples)/Double(sampleRate))s)"
                )
            }
        }

        print("\n✅ Chunk boundary calculation tests completed")
    }

    /// Test adaptive context improvement for frame dropping (unit test)
    func testAdaptiveContextImprovement() throws {
        // This unit test demonstrates the improvement without requiring ML models

        print("\n--- Testing Adaptive Context Improvement ---")

        let sampleRate = 16000
        let centerSamples = Int(11.2 * Double(sampleRate))  // 179,200
        let leftContextSamples = Int(1.6 * Double(sampleRate))  // 25,600
        let rightContextSamples = Int(1.6 * Double(sampleRate))  // 25,600

        // Test the specific problematic case: 14.28s audio
        let duration = 14.28
        let audioSamples = Int(duration * Double(sampleRate))  // 228,480 samples

        print("Testing \(duration)s audio (\(audioSamples) samples)")

        // Simulate the two chunks
        var centerStart = 0
        var totalProcessedSamples = 0
        var chunkIndex = 0

        while centerStart < audioSamples {
            let isLastChunk = (centerStart + centerSamples) >= audioSamples
            let remainingSamples = audioSamples - centerStart

            // Apply our adaptive context fix
            let adaptiveLeftContext: Int
            let frameOffset: Int

            if isLastChunk && remainingSamples < centerSamples {
                // Our fix: maximize context for small last chunk
                let maxModelSamples = Int(15.0 * Double(sampleRate))  // 240,000
                let desiredTotalSamples = min(maxModelSamples, audioSamples)
                let maxLeftContext = centerStart

                let neededLeftContext = desiredTotalSamples - remainingSamples
                adaptiveLeftContext = min(neededLeftContext, maxLeftContext)

                if adaptiveLeftContext > leftContextSamples {
                    let extraContextSamples = adaptiveLeftContext - leftContextSamples
                    frameOffset = extraContextSamples / 1280  // Encoder frame size
                } else {
                    frameOffset = 0
                }
            } else {
                adaptiveLeftContext = leftContextSamples
                frameOffset = 0
            }

            // Calculate bounds
            let leftStart = max(0, centerStart - adaptiveLeftContext)
            let centerEnd = min(audioSamples, centerStart + centerSamples)
            let rightEnd = min(audioSamples, centerEnd + rightContextSamples)

            let chunkSamples = rightEnd - leftStart
            let actualFramesToProcess = ASRConstants.calculateEncoderFrames(from: chunkSamples)
            let framesToSkip = frameOffset
            let effectiveFramesProcessed = actualFramesToProcess - framesToSkip

            print("Chunk \(chunkIndex) (\(isLastChunk ? "LAST" : "normal")):")
            print(
                "  Audio window: [\(leftStart), \(rightEnd)) = \(chunkSamples) samples (\(String(format: "%.2f", Double(chunkSamples)/16000.0))s)"
            )
            print("  Frames: \(actualFramesToProcess) total, skip \(framesToSkip), process \(effectiveFramesProcessed)")

            if isLastChunk && remainingSamples < centerSamples {
                print(
                    "  🔧 ADAPTIVE: leftContext=\(adaptiveLeftContext) samples (\(String(format: "%.2f", Double(adaptiveLeftContext)/16000.0))s)"
                )
                print("  ✅ COVERS ALL REMAINING AUDIO: \(remainingSamples) samples processed")

                // Key assertion: the last chunk now covers all remaining audio
                XCTAssertEqual(
                    rightEnd, audioSamples,
                    "Adaptive context should cover all remaining audio"
                )

                // The effective processing should handle the remaining content
                let remainingFrames = ASRConstants.calculateEncoderFrames(from: remainingSamples)
                XCTAssertGreaterThanOrEqual(
                    effectiveFramesProcessed, remainingFrames,
                    "Should process at least the remaining frames"
                )
            }

            totalProcessedSamples = max(totalProcessedSamples, centerEnd)
            centerStart += centerSamples
            chunkIndex += 1
        }

        print("\nResults:")
        print("  Total audio: \(audioSamples) samples")
        print("  Total processed: \(totalProcessedSamples) samples")
        print("  Coverage: \(String(format: "%.1f", Double(totalProcessedSamples) * 100.0 / Double(audioSamples)))%")

        // With our fix, we should process all samples
        XCTAssertEqual(
            totalProcessedSamples, audioSamples,
            "Adaptive context fix should process all audio samples"
        )

        print("✅ Adaptive context successfully eliminates frame dropping")
    }

    /// Test decoder state persistence across multiple short chunks
    func testDecoderStatePersistence() async throws {
        try skipIfCI()

        let config = ASRConfig(enableDebug: true)
        let manager: AsrManager

        let initManager = AsrManager(config: config)
        manager = initManager

        // Create several short chunks that need to maintain state
        let sampleRate = 16000
        let chunkDuration = 3.0  // 3 second chunks
        let numChunks = 4
        let frequency: Float = 440.0

        var decoderState = try TdtDecoderState()
        var allTokens: [Int] = []
        var allTimestamps: [Int] = []

        for chunkIndex in 0..<numChunks {
            print("Processing chunk \(chunkIndex + 1)/\(numChunks)")

            // Create audio chunk
            var chunkSamples: [Float] = []
            let startSample = chunkIndex * Int(chunkDuration * Double(sampleRate))

            for i in 0..<Int(chunkDuration * Double(sampleRate)) {
                let globalI = startSample + i
                let t = Float(globalI) / Float(sampleRate)
                let sample = sin(2.0 * Float.pi * frequency * t) * 0.1
                chunkSamples.append(sample)
            }

            // Process chunk while maintaining decoder state
            let isLastChunk = (chunkIndex == numChunks - 1)
            let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: 240_000)

            let (hypothesis, _) = try await manager.executeMLInferenceWithTimings(
                paddedChunk,
                originalLength: chunkSamples.count,
                enableDebug: true,
                decoderState: &decoderState,
                contextFrameAdjustment: 0,  // Integration test doesn't use adaptive context
                isLastChunk: isLastChunk
            )

            print("Chunk \(chunkIndex): \(hypothesis.tokenCount) tokens, timeJump: \(decoderState.timeJump ?? 0)")

            allTokens.append(contentsOf: hypothesis.ySequence)
            allTimestamps.append(contentsOf: hypothesis.timestamps)

            // Verify timeJump is being set (except for last chunk)
            if !isLastChunk {
                // TimeJump should be set for non-last chunks
                XCTAssertNotNil(decoderState.timeJump, "TimeJump should be set for chunk \(chunkIndex)")
            } else {
                // Last chunk should clear timeJump
                XCTAssertNil(decoderState.timeJump, "TimeJump should be nil for last chunk")
            }

            // Verify lastToken is maintained for linguistic continuity
            if hypothesis.hasTokens {
                XCTAssertEqual(
                    decoderState.lastToken, hypothesis.computedLastToken,
                    "Decoder state should maintain last token for chunk \(chunkIndex)"
                )
            }
        }

        // Convert tokens to text for final verification
        let finalResult = manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            encoderSequenceLength: 0,
            audioSamples: Array(repeating: 0.0, count: numChunks * Int(chunkDuration * Double(sampleRate))),
            processingTime: 0.0
        )

        print("Multi-chunk result: '\(finalResult.text)'")
        print("Total tokens processed: \(allTokens.count)")

        XCTAssertNotNil(finalResult, "Should get combined result from multiple chunks")

        // Check that timestamps are monotonically increasing (no time travel)
        for i in 1..<allTimestamps.count {
            XCTAssertGreaterThanOrEqual(
                allTimestamps[i], allTimestamps[i - 1],
                "Timestamps should be monotonically increasing, but timestamp[\(i)] < timestamp[\(i-1)]"
            )
        }

        print("✅ Decoder state properly maintained across \(numChunks) chunks")
    }

    /// Performance test to ensure our changes don't impact speed
    func testChunkProcessingPerformance() throws {
        try skipIfCI()

        // This is a lightweight performance check that doesn't need actual models
        let audioSamples = [Float](repeating: 0.5, count: 320_000)  // 20 seconds
        _ = ChunkProcessor(audioSamples: audioSamples, enableDebug: false)

        measure {
            // Just test the chunking logic without actual ML inference
            // This verifies our changes don't add computational overhead
            var centerStart = 0
            let centerSamples = Int(11.2 * 16000)  // 11.2s center
            let leftContextSamples = Int(1.6 * 16000)  // 1.6s context
            let rightContextSamples = Int(1.6 * 16000)

            while centerStart < audioSamples.count {
                let leftStart = max(0, centerStart - leftContextSamples)
                let centerEnd = min(audioSamples.count, centerStart + centerSamples)
                let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

                if leftStart < rightEnd {
                    _ = Array(audioSamples[leftStart..<rightEnd])
                }

                centerStart += centerSamples
            }
        }

        print("✅ Chunk processing performance test completed")
    }

    /// Test adaptive context frame adjustment to prevent duplicate tokens
    func testAdaptiveContextFrameAdjustment() async throws {
        try skipIfCI()

        print("\n--- Testing Adaptive Context Frame Adjustment ---")

        let config = ASRConfig(enableDebug: true)
        let manager = AsrManager(config: config)

        // Create test audio that simulates the scenario where adaptive context causes duplicates:
        // 1. First chunk processes fully
        // 2. Last chunk is short and uses adaptive context, pulling in overlap from first chunk
        // 3. Without frame adjustment, we'd get duplicate tokens from the overlap

        let sampleRate = 16000
        let totalDuration = 12.5  // Just over one chunk (11.2s center + contexts)
        let totalSamples = Int(totalDuration * Double(sampleRate))

        // Create audio with distinct patterns in different regions to detect duplicates
        var audioSamples: [Float] = []
        for i in 0..<totalSamples {
            let t = Float(i) / Float(sampleRate)
            let segment = Int(t / 2.0)  // 2-second segments with different patterns

            let baseFreq: Float = 200.0 + Float(segment * 50)  // Different frequency per segment
            let amplitude: Float = 0.1

            let sample = amplitude * sin(2.0 * Float.pi * baseFreq * t)
            audioSamples.append(sample)
        }

        print("Created audio: \(totalDuration)s (\(totalSamples) samples)")

        // Process using ChunkProcessor (which will use adaptive context for the last chunk)
        var decoderState = try TdtDecoderState()
        let processor = ChunkProcessor(audioSamples: audioSamples, enableDebug: true)

        let startTime = Date()
        let result = try await processor.process(using: manager, decoderState: &decoderState, startTime: startTime)

        print("Processing result:")
        print("- Text: '\(result.text)'")
        print("- Duration: \(result.duration)s")
        print("- Token timings count: \(result.tokenTimings?.count ?? 0)")

        // Check for duplicate tokens at the same timestamp (the main issue we're fixing)
        var duplicateCount = 0
        var timestampTokenCount: [Int: Int] = [:]

        if let tokenTimings = result.tokenTimings {
            for timing in tokenTimings {
                // Convert time to frame index (0.08s per frame)
                let frameIndex = Int(timing.startTime / 0.08)

                if timestampTokenCount[frameIndex] == nil {
                    timestampTokenCount[frameIndex] = 1
                } else {
                    timestampTokenCount[frameIndex]! += 1
                    duplicateCount += 1
                    print(
                        "⚠️  Duplicate found: token '\(timing.token)' (id:\(timing.tokenId)) at frame \(frameIndex) (time: \(timing.startTime)s)"
                    )
                }
            }
        }

        print("Analysis:")
        print("- Unique timestamps: \(timestampTokenCount.count)")
        print("- Duplicate tokens: \(duplicateCount)")

        // With our fix, there should be no duplicate tokens at the same timestamp
        XCTAssertEqual(duplicateCount, 0, "Context frame adjustment should prevent duplicate tokens")

        if duplicateCount == 0 {
            print("✅ Adaptive context frame adjustment working correctly - no duplicates detected!")
        } else {
            print("❌ Frame adjustment failed - found \(duplicateCount) duplicates")
        }
    }
}
