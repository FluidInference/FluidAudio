import XCTest
@testable import SeamlessAudioSwift

final class BasicInitializationTests: XCTestCase {

    func testDiarizerFactoryCreatesCorrectBackends() {
        // Test SherpaOnnx backend
        let sherpaConfig = DiarizerConfig(backend: .sherpaOnnx)
        let sherpaManager = DiarizerFactory.createManager(config: sherpaConfig)
        XCTAssertEqual(sherpaManager.backend, .sherpaOnnx)
        XCTAssertTrue(sherpaManager is SherpaOnnxDiarizerManager)

        // Test CoreML backend
        let coremlConfig = DiarizerConfig(backend: .coreML)
        let coremlManager = DiarizerFactory.createManager(config: coremlConfig)
        XCTAssertEqual(coremlManager.backend, .coreML)
        XCTAssertTrue(coremlManager is CoreMLDiarizerManager)
    }

    func testConvenienceFactoryMethods() {
        // Test SherpaOnnx convenience factory
        let sherpaManager = DiarizerFactory.createSherpaOnnxManager()
        XCTAssertEqual(sherpaManager.backend, .sherpaOnnx)
        XCTAssertFalse(sherpaManager.isAvailable) // Not initialized yet

        // Test CoreML convenience factory
        let coremlManager = DiarizerFactory.createCoreMLManager()
        XCTAssertEqual(coremlManager.backend, .coreML)
        XCTAssertFalse(coremlManager.isAvailable) // Not initialized yet
    }

    func testBackwardCompatibilityAliases() {
        // Test that old type names still work
        let oldConfig = SpeakerDiarizationConfig()
        XCTAssertEqual(oldConfig.backend, .sherpaOnnx)

        let oldManager = SpeakerDiarizationManager(config: oldConfig)
        XCTAssertEqual(oldManager.backend, .sherpaOnnx)
        XCTAssertFalse(oldManager.isAvailable)
    }

    func testDiarizerBackendEnum() {
        // Test enum cases
        XCTAssertEqual(DiarizerBackend.sherpaOnnx.rawValue, "sherpa-onnx")
        XCTAssertEqual(DiarizerBackend.coreML.rawValue, "coreml")

        // Test all cases
        let allCases = DiarizerBackend.allCases
        XCTAssertEqual(allCases.count, 2)
        XCTAssertTrue(allCases.contains(.sherpaOnnx))
        XCTAssertTrue(allCases.contains(.coreML))
    }
}

// MARK: - Cross-Backend Tests

@available(macOS 13.0, iOS 16.0, *)
final class CrossBackendDiarizerTests: XCTestCase {

    /// Test both SherpaOnnx and CoreML backends with the same test logic
    func testBothBackendsInitialization() {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            XCTAssertEqual(manager.backend, backend, "Backend should match for \(backend)")
            XCTAssertFalse(manager.isAvailable, "Manager should not be available before initialization for \(backend)")
        }
    }

    func testBothBackendsNotInitializedErrors() async {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]
        let testSamples = Array(repeating: Float(0.5), count: 16000)

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test segmentation fails when not initialized
            do {
                _ = try await manager.performSegmentation(testSamples, sampleRate: 16000)
                XCTFail("Should have thrown notInitialized error for \(backend)")
            } catch DiarizerError.notInitialized {
                // Expected error
            } catch {
                XCTFail("Unexpected error for \(backend): \(error)")
            }

            // Test embedding extraction fails when not initialized
            do {
                _ = try await manager.extractEmbedding(from: testSamples)
                XCTFail("Should have thrown notInitialized error for \(backend)")
            } catch DiarizerError.notInitialized {
                // Expected error
            } catch {
                XCTFail("Unexpected error for \(backend): \(error)")
            }
        }
    }

    func testBothBackendsAudioValidation() {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        // Test valid audio
        let validSamples = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        // Test invalid audio (too short)
        let shortSamples = Array(repeating: Float(0.5), count: 8000) // 0.5 seconds

        // Test silent audio
        let silentSamples = Array(repeating: Float(0.0), count: 16000)

        // Test empty audio
        let emptySamples: [Float] = []

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test valid audio
            let validResult = manager.validateAudio(validSamples)
            XCTAssertTrue(validResult.isValid, "Valid audio should pass validation for \(backend)")
            XCTAssertEqual(validResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second for \(backend)")
            XCTAssertTrue(validResult.issues.isEmpty, "Valid audio should have no issues for \(backend)")

            // Test short audio
            let shortResult = manager.validateAudio(shortSamples)
            XCTAssertFalse(shortResult.isValid, "Short audio should fail validation for \(backend)")
            XCTAssertTrue(shortResult.issues.contains("Audio too short (minimum 1 second)"), "Short audio should have correct error for \(backend)")

            // Test silent audio
            let silentResult = manager.validateAudio(silentSamples)
            XCTAssertFalse(silentResult.isValid, "Silent audio should fail validation for \(backend)")
            XCTAssertTrue(silentResult.issues.contains("Audio too quiet or silent"), "Silent audio should have correct error for \(backend)")

            // Test empty audio
            let emptyResult = manager.validateAudio(emptySamples)
            XCTAssertFalse(emptyResult.isValid, "Empty audio should fail validation for \(backend)")
            XCTAssertTrue(emptyResult.issues.contains("No audio data"), "Empty audio should have correct error for \(backend)")
        }
    }

    func testBothBackendsCosineDistance() {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test identical embeddings
            let embedding1: [Float] = [1.0, 0.0, 0.0]
            let embedding2: [Float] = [1.0, 0.0, 0.0]
            let distance1 = manager.cosineDistance(embedding1, embedding2)
            XCTAssertEqual(distance1, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance for \(backend)")

            // Test orthogonal embeddings
            let embedding3: [Float] = [1.0, 0.0, 0.0]
            let embedding4: [Float] = [0.0, 1.0, 0.0]
            let distance2 = manager.cosineDistance(embedding3, embedding4)
            XCTAssertEqual(distance2, 1.0, accuracy: 0.001, "Orthogonal embeddings should have distance 1 for \(backend)")

            // Test opposite embeddings
            let embedding5: [Float] = [1.0, 0.0, 0.0]
            let embedding6: [Float] = [-1.0, 0.0, 0.0]
            let distance3 = manager.cosineDistance(embedding5, embedding6)
            XCTAssertEqual(distance3, 2.0, accuracy: 0.001, "Opposite embeddings should have distance 2 for \(backend)")
        }
    }

    func testBothBackendsEmbeddingValidation() {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test valid embedding
            let validEmbedding: [Float] = [0.5, 0.3, -0.2, 0.8]
            XCTAssertTrue(manager.validateEmbedding(validEmbedding), "Valid embedding should pass validation for \(backend)")

            // Test empty embedding
            let emptyEmbedding: [Float] = []
            XCTAssertFalse(manager.validateEmbedding(emptyEmbedding), "Empty embedding should fail validation for \(backend)")

            // Test embedding with NaN
            let nanEmbedding: [Float] = [0.5, Float.nan, 0.3]
            XCTAssertFalse(manager.validateEmbedding(nanEmbedding), "NaN embedding should fail validation for \(backend)")

            // Test embedding with infinity
            let infEmbedding: [Float] = [0.5, Float.infinity, 0.3]
            XCTAssertFalse(manager.validateEmbedding(infEmbedding), "Infinite embedding should fail validation for \(backend)")

            // Test very small magnitude embedding
            let smallEmbedding: [Float] = [0.01, 0.01, 0.01]
            XCTAssertFalse(manager.validateEmbedding(smallEmbedding), "Small magnitude embedding should fail validation for \(backend)")
        }
    }

    func testBothBackendsCleanup() async {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test cleanup doesn't crash
            await manager.cleanup()
            XCTAssertFalse(manager.isAvailable, "Manager should not be available after cleanup for \(backend)")
        }
    }

    func testBothBackendsSpeakerComparison() async {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        let audio1 = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }
        let audio2 = Array(0..<16000).map { i in
            sin(Float(i) * 0.02) * 0.5
        }

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            do {
                let similarity = try await manager.compareSpeakers(audio1: audio1, audio2: audio2)
                XCTAssertGreaterThanOrEqual(similarity, 0, "Similarity should be >= 0 for \(backend)")
                XCTAssertLessThanOrEqual(similarity, 100, "Similarity should be <= 100 for \(backend)")
            } catch DiarizerError.notInitialized {
                // Expected error in test environment
                print("Speaker comparison failed due to not being initialized (expected) for \(backend)")
            } catch {
                XCTFail("Unexpected error for \(backend): \(error)")
            }
        }
    }

    func testBothBackendsModelDownloadPaths() async {
        let backends: [DiarizerBackend] = [.sherpaOnnx, .coreML]

        for backend in backends {
            let config = DiarizerConfig(backend: backend)
            let manager = DiarizerFactory.createManager(config: config)

            // Test model download (this might fail in CI/test environment, but should return valid paths)
            do {
                let modelPaths: ModelPaths
                if let sherpaManager = manager as? SherpaOnnxDiarizerManager {
                    modelPaths = try await sherpaManager.downloadModels()
                } else if let coremlManager = manager as? CoreMLDiarizerManager {
                    modelPaths = try await coremlManager.downloadModels()
                } else {
                    XCTFail("Unknown manager type for \(backend)")
                    continue
                }

                XCTAssertFalse(modelPaths.segmentationPath.isEmpty, "Segmentation path should not be empty for \(backend)")
                XCTAssertFalse(modelPaths.embeddingPath.isEmpty, "Embedding path should not be empty for \(backend)")

                // Verify backend-specific model directories
                switch backend {
                case .sherpaOnnx:
                    XCTAssertTrue(modelPaths.segmentationPath.contains("sherpa-onnx"), "SherpaOnnx models should be in sherpa-onnx directory")
                case .coreML:
                    XCTAssertTrue(modelPaths.segmentationPath.contains("coreml"), "CoreML models should be in coreml directory")
                }

            } catch {
                // This may fail in test environment without network access - that's expected
                print("Model download failed (expected in test environment) for \(backend): \(error)")
            }
        }
    }
}

// MARK: - CoreML Backend Specific Test

@available(macOS 13.0, iOS 16.0, *)
final class CoreMLBackendTests: XCTestCase {

    func testCoreMLDiarizerCreationAndBasicFunctionality() async {
        // Test that CoreML diarizer can be created with the same config as WhisperState.swift
        let config = DiarizerConfig(
            backend: .coreML,
            clusteringThreshold: 0.7,
            minDurationOn: 1.0,
            minDurationOff: 0.5,
            numClusters: -1,
            debugMode: true
        )

        let diarizer = DiarizerFactory.createManager(config: config)

        // Verify it's the correct type
        XCTAssertTrue(diarizer is CoreMLDiarizerManager, "Should create CoreML diarizer")
        XCTAssertEqual(diarizer.backend, .coreML, "Should be CoreML backend")
        XCTAssertFalse(diarizer.isAvailable, "Should not be available before initialization")

        // Test basic validation functionality (doesn't require initialization)
        let validSamples = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        let validationResult = diarizer.validateAudio(validSamples)
        XCTAssertTrue(validationResult.isValid, "Valid audio should pass validation")
        XCTAssertEqual(validationResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second")

        // Test cosine distance calculation
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance = diarizer.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance")
    }

    func testCoreMLDiarizerInitializationAttempt() async {
        // Test that initialization attempt works (may fail due to model download but shouldn't crash)
        let config = DiarizerConfig(backend: .coreML, debugMode: true)
        let diarizer = DiarizerFactory.createManager(config: config)

        do {
            try await diarizer.initialize()
            XCTAssertTrue(diarizer.isAvailable, "Should be available after successful initialization")
            print("✅ CoreML diarizer initialized successfully!")

            // Test that we can perform basic operations
            let testSamples = Array(repeating: Float(0.5), count: 16000)

            do {
                let segments = try await diarizer.performSegmentation(testSamples, sampleRate: 16000)
                print("✅ CoreML segmentation completed, found \(segments.count) segments")
            } catch {
                print("ℹ️  Segmentation test completed (may need more realistic audio): \(error)")
            }

            await diarizer.cleanup()

        } catch {
            // This is expected in test environment - models might not download
            print("ℹ️  CoreML initialization test completed (expected in test environment): \(error)")
            XCTAssertFalse(diarizer.isAvailable, "Should not be available if initialization failed")
        }
    }
}
