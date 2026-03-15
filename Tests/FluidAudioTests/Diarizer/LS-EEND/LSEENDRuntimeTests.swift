import AVFoundation
import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class LSEENDRuntimeTests: XCTestCase {
    private static var rootURL: URL {
        if let path = ProcessInfo.processInfo.environment["LSEEND_WORKSPACE_ROOT"] {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("LS-EENDWorkspace")
    }

    private static let probeExecutableURL = rootURL.appendingPathComponent("artifacts/bin/lseend_runtime_probe")
    private static let probeSourceURLs: [URL] = [
        rootURL.appendingPathComponent("LS-EENDTest/Tools/LSEENDRuntimeProbe.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/Shared/IndexUtils.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/Shared/AppLogger.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/Shared/AudioConverter.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/Shared/NeMoMelSpectrogram.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/LS-EEND-Diarizer/LSEENDSupport.swift"),
        rootURL.appendingPathComponent(
            "LS-EENDTest/LS-EENDTest/FluidAudio/LS-EEND-Diarizer/LSEENDFeatureExtraction.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/LS-EEND-Diarizer/LSEENDInference.swift"),
        rootURL.appendingPathComponent("LS-EENDTest/LS-EENDTest/FluidAudio/LS-EEND-Diarizer/LSEENDEvaluation.swift"),
    ]
    nonisolated(unsafe) private static var cachedEngines: [LSEENDVariant: LSEENDInferenceEngine] = [:]
    nonisolated(unsafe) private static var didEnsureProbeExecutable = false
    private static let workspaceSetupHint =
        "Set LSEEND_WORKSPACE_ROOT to an LS-EEND workspace checkout that includes the runtime probe sources and parity artifacts."

    private struct ErrorStats {
        let maxAbs: Double
        let meanAbs: Double
    }

    private struct ProbeMatrix: Decodable {
        let rows: Int
        let columns: Int
        let values: [Float]

        func matrix() -> LSEENDMatrix {
            LSEENDMatrix(validatingRows: rows, columns: columns, values: values)
        }
    }

    private struct ProbeInferenceResult: Decodable {
        let logits: ProbeMatrix
        let probabilities: ProbeMatrix
        let fullLogits: ProbeMatrix
        let fullProbabilities: ProbeMatrix
        let frameHz: Double
        let durationSeconds: Double
    }

    private struct ProbeStreamingProgress: Decodable {
        let chunkIndex: Int
        let bufferSeconds: Double
        let numFramesEmitted: Int
        let totalFramesEmitted: Int
        let flush: Bool
    }

    private struct ProbeStreamingResult: Decodable {
        let result: ProbeInferenceResult
        let updates: [ProbeStreamingProgress]
    }

    private struct ProbeSessionCheckResult: Decodable {
        let firstUpdateRows: Int
        let firstUpdateTotalEmittedFrames: Int
        let committedProbabilities: ProbeMatrix
        let snapshotProbabilities: ProbeMatrix
        let repeatedFinalizeReturnedUpdate: Bool
        let repeatedSnapshotProbabilities: ProbeMatrix
    }

    private struct RepoEvalMetrics: Decodable {
        struct Artifacts: Decodable {
            let kaldiDir: String
            let rawLogitsNPY: String

            enum CodingKeys: String, CodingKey {
                case kaldiDir = "kaldi_dir"
                case rawLogitsNPY = "raw_logits_npy"
            }
        }

        struct Evaluation: Decodable {
            let collarSeconds: Double
            let frameRate: Double
            let medianWidth: Int
            let threshold: Float

            enum CodingKeys: String, CodingKey {
                case collarSeconds = "collar_seconds"
                case frameRate = "frame_rate"
                case medianWidth = "median_width"
                case threshold
            }
        }

        struct DERMetrics: Decodable {
            let der: Double
        }

        let artifacts: Artifacts
        let evaluation: Evaluation
        let correctedMappedDER: DERMetrics
        let referenceRTTM: String
        let recordingID: String
        let audio: String

        enum CodingKeys: String, CodingKey {
            case artifacts
            case evaluation
            case correctedMappedDER = "corrected_mapped_der"
            case referenceRTTM = "reference_rttm"
            case recordingID = "recording_id"
            case audio
        }
    }

    private struct CoreMLMetrics: Decodable {
        struct DERMetrics: Decodable {
            let der: Double
            let collarSeconds: Double
            let medianWidth: Int
            let threshold: Float

            enum CodingKeys: String, CodingKey {
                case der
                case collarSeconds = "collar_seconds"
                case medianWidth = "median_width"
                case threshold
            }
        }

        let der: DERMetrics
    }

    private struct StreamingMetrics: Decodable {
        struct Raw: Decodable {
            let der: Double
            let collarSeconds: Double
            let medianWidth: Int
            let threshold: Float

            enum CodingKeys: String, CodingKey {
                case der
                case collarSeconds = "collar_seconds"
                case medianWidth = "median_width"
                case threshold
            }
        }

        let rawLSEEND: Raw

        enum CodingKeys: String, CodingKey {
            case rawLSEEND = "raw_lseend"
        }
    }

    private struct StreamingUpdateFixtureFile: Decodable {
        let updates: [StreamingUpdateFixture]
    }

    private struct StreamingUpdateFixture: Decodable {
        let chunkIndex: Int
        let bufferSeconds: Double
        let numFramesEmitted: Int
        let totalFramesEmitted: Int

        enum CodingKeys: String, CodingKey {
            case chunkIndex = "chunk_index"
            case bufferSeconds = "buffer_seconds"
            case numFramesEmitted = "num_frames_emitted"
            case totalFramesEmitted = "total_frames_emitted"
        }
    }

    private struct RepoFixture {
        let variant: LSEENDVariant
        let metrics: RepoEvalMetrics
        let metricsURL: URL
        let audio8kURL: URL
        let rawLogitsURL: URL
        let referenceRTTMURL: URL
    }

    func testVariantRegistryResolvesAllExportedArtifacts() async throws {
        let expectedColumns: [LSEENDVariant: Int] = [
            .ami: 4,
            .callhome: 7,
            .dihard2: 10,
            .dihard3: 10,
        ]

        for variant in LSEENDVariant.allCases {
            let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: descriptor.modelURL.path),
                "Missing model package for \(variant.rawValue)")
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: descriptor.metadataURL.path),
                "Missing metadata for \(variant.rawValue)")

            let engine = try await makeEngine(variant: variant)
            let metadata = engine.metadata
            XCTAssertEqual(metadata.realOutputDim, expectedColumns[variant])
            XCTAssertEqual(metadata.fullOutputDim, (expectedColumns[variant] ?? 0) + 2)
        }
    }

    func testOfflineAMIParityAndDER() throws {
        try assertOfflineRepoFixtureMatchesSwiftRuntime(variant: .ami)
    }

    func testOfflineCALLHOMEParityAndDER() throws {
        try assertOfflineRepoFixtureMatchesSwiftRuntime(variant: .callhome)
    }

    func testOfflineDIHARD2ParityAndDER() throws {
        try assertOfflineRepoFixtureMatchesSwiftRuntime(variant: .dihard2)
    }

    func testOfflineDIHARD3ParityAndDER() throws {
        try assertOfflineRepoFixtureMatchesSwiftRuntime(variant: .dihard3)
    }

    func testEndToEndDIHARD3FLACMatchesCoreMLGolden() throws {
        let result = try runOfflineProbe(
            variant: .dihard3,
            audioURL: try requireWorkspaceArtifact("LDC2022S14.flac")
        )
        let actualProbabilities = result.probabilities.matrix()
        let actualFullProbabilities = result.fullProbabilities.matrix()

        let expectedProbabilities = try NPYReader.loadFloatArray(
            from: try requireWorkspaceArtifact("artifacts/coreml/LDC2022S14/LDC2022S14_coreml_probabilities.npy")
        ).matrix2D()
        let expectedFullProbabilities = try NPYReader.loadFloatArray(
            from: try requireWorkspaceArtifact("artifacts/coreml/LDC2022S14/LDC2022S14_coreml_full_probabilities.npy")
        ).matrix2D()

        let realStats = compare(actualProbabilities, expectedProbabilities)
        XCTAssertLessThanOrEqual(realStats.maxAbs, 0.1)
        XCTAssertLessThanOrEqual(realStats.meanAbs, 0.01)

        let fullStats = compare(actualFullProbabilities, expectedFullProbabilities)
        XCTAssertLessThanOrEqual(fullStats.maxAbs, 0.1)
        XCTAssertLessThanOrEqual(fullStats.meanAbs, 0.01)

        let metrics = try decode(
            CoreMLMetrics.self,
            from: try requireWorkspaceArtifact("artifacts/coreml/LDC2022S14/LDC2022S14_coreml_metrics.json")
        )
        let referenceBinary = try referenceBinaryMatrix(
            rttmURL: try requireWorkspaceArtifact("LDC2022S14.rttm"),
            numFrames: actualProbabilities.rows,
            frameRate: result.frameHz
        )
        let evaluation = LSEENDEvaluation.computeDER(
            probabilities: actualProbabilities,
            referenceBinary: referenceBinary,
            settings: LSEENDEvaluationSettings(
                threshold: metrics.der.threshold,
                medianWidth: metrics.der.medianWidth,
                collarSeconds: metrics.der.collarSeconds,
                frameRate: result.frameHz
            )
        )
        XCTAssertLessThanOrEqual(abs(evaluation.der - metrics.der.der), 0.01)
    }

    func testStreamingDIHARD3MatchesGoldenAndSessionBehavior() throws {
        let audioURL = try requireWorkspaceArtifact("LDC2022S14.flac")
        let simulation = try runStreamingProbe(
            variant: .dihard3,
            audioURL: audioURL,
            chunkSeconds: 0.5
        )
        let streamingFixture = try decode(
            StreamingUpdateFixtureFile.self,
            from: try requireWorkspaceArtifact(
                "artifacts/coreml/LDC2022S14_streaming/LDC2022S14_streaming_updates.json")
        )
        XCTAssertEqual(simulation.updates.count, streamingFixture.updates.count)
        for (actual, expected) in zip(simulation.updates, streamingFixture.updates) {
            XCTAssertEqual(actual.chunkIndex, expected.chunkIndex)
            XCTAssertEqual(actual.numFramesEmitted, expected.numFramesEmitted)
            XCTAssertEqual(actual.totalFramesEmitted, expected.totalFramesEmitted)
            XCTAssertEqual(actual.bufferSeconds, expected.bufferSeconds, accuracy: 1e-6)
        }

        let expectedStreamingFull = try NPYReader.loadFloatArray(
            from: try requireWorkspaceArtifact(
                "artifacts/coreml/LDC2022S14_streaming/LDC2022S14_full_probabilities.npy")
        ).matrix2D()
        let actualFullProbabilities = simulation.result.fullProbabilities.matrix()
        let actualProbabilities = simulation.result.probabilities.matrix()
        let streamingStats = compare(actualFullProbabilities, expectedStreamingFull)
        XCTAssertLessThanOrEqual(streamingStats.maxAbs, 0.1)
        XCTAssertLessThanOrEqual(streamingStats.meanAbs, 0.01)

        let streamingMetrics = try decode(
            StreamingMetrics.self,
            from: try requireWorkspaceArtifact("artifacts/coreml/LDC2022S14_streaming/LDC2022S14_metrics.json")
        )
        let referenceBinary = try referenceBinaryMatrix(
            rttmURL: try requireWorkspaceArtifact("LDC2022S14.rttm"),
            numFrames: actualProbabilities.rows,
            frameRate: simulation.result.frameHz
        )
        let evaluation = LSEENDEvaluation.computeDER(
            probabilities: actualProbabilities,
            referenceBinary: referenceBinary,
            settings: LSEENDEvaluationSettings(
                threshold: streamingMetrics.rawLSEEND.threshold,
                medianWidth: streamingMetrics.rawLSEEND.medianWidth,
                collarSeconds: streamingMetrics.rawLSEEND.collarSeconds,
                frameRate: simulation.result.frameHz
            )
        )
        XCTAssertLessThanOrEqual(abs(evaluation.der - streamingMetrics.rawLSEEND.der), 0.01)

        let sessionCheck = try runSessionCheckProbe(
            variant: .dihard3,
            audioURL: try findFirstWAV(
                in: try requireWorkspaceArtifact("artifacts/LDC2022S14_repo_eval/dihard3_nominal/kaldi")
            ),
            chunkSeconds: 0.5
        )
        XCTAssertEqual(sessionCheck.firstUpdateRows, 0)
        XCTAssertEqual(sessionCheck.firstUpdateTotalEmittedFrames, 0)

        let committed = sessionCheck.committedProbabilities.matrix()
        let snapshot = sessionCheck.snapshotProbabilities.matrix()
        let snapshotStats = compare(committed, snapshot)
        XCTAssertLessThanOrEqual(snapshotStats.maxAbs, 0)
        XCTAssertLessThanOrEqual(snapshotStats.meanAbs, 0)

        XCTAssertFalse(sessionCheck.repeatedFinalizeReturnedUpdate)

        let repeatedSnapshotStats = compare(
            snapshot,
            sessionCheck.repeatedSnapshotProbabilities.matrix()
        )
        XCTAssertLessThanOrEqual(repeatedSnapshotStats.maxAbs, 0)
        XCTAssertLessThanOrEqual(repeatedSnapshotStats.meanAbs, 0)
    }

    private func assertOfflineRepoFixtureMatchesSwiftRuntime(variant: LSEENDVariant) throws {
        let fixture = try repoFixture(for: variant)
        let result = try runOfflineProbe(variant: variant, audioURL: fixture.audio8kURL)
        let actualLogits = result.logits.matrix()
        let actualProbabilities = result.probabilities.matrix()

        let expectedLogits = try NPYReader.loadFloatArray(from: fixture.rawLogitsURL).matrix2D()
        let expectedProbabilities = expectedLogits.applyingSigmoid()
        let logitStats = compare(actualLogits, expectedLogits)
        let probabilityStats = compare(actualProbabilities, expectedProbabilities)

        let logitMaxTolerance: Double
        switch variant {
        case .dihard2:
            // The current DIHARD II float32 CoreML export stays tight in probability space
            // and DER, but shows a small, isolated logit-domain drift on CPU execution.
            logitMaxTolerance = 0.025
        default:
            logitMaxTolerance = 0.01
        }

        XCTAssertLessThanOrEqual(
            logitStats.maxAbs, logitMaxTolerance, "\(variant.rawValue) max abs error \(logitStats.maxAbs)")
        XCTAssertLessThanOrEqual(logitStats.meanAbs, 0.001, "\(variant.rawValue) mean abs error \(logitStats.meanAbs)")
        XCTAssertLessThanOrEqual(
            probabilityStats.maxAbs, 0.01, "\(variant.rawValue) probability max abs error \(probabilityStats.maxAbs)")
        XCTAssertLessThanOrEqual(
            probabilityStats.meanAbs, 0.001,
            "\(variant.rawValue) probability mean abs error \(probabilityStats.meanAbs)")

        let referenceBinary = try referenceBinaryMatrix(
            rttmURL: fixture.referenceRTTMURL,
            numFrames: actualProbabilities.rows,
            frameRate: result.frameHz
        )
        let evaluation = LSEENDEvaluation.computeDER(
            probabilities: actualProbabilities,
            referenceBinary: referenceBinary,
            settings: LSEENDEvaluationSettings(
                threshold: fixture.metrics.evaluation.threshold,
                medianWidth: fixture.metrics.evaluation.medianWidth,
                collarSeconds: fixture.metrics.evaluation.collarSeconds,
                frameRate: fixture.metrics.evaluation.frameRate
            )
        )
        XCTAssertLessThanOrEqual(
            abs(evaluation.der - fixture.metrics.correctedMappedDER.der),
            0.01,
            "\(variant.rawValue) DER drifted to \(evaluation.der)"
        )
    }

    private func repoFixture(for variant: LSEENDVariant) throws -> RepoFixture {
        let metricsRelativePath: String
        switch variant {
        case .ami:
            metricsRelativePath = "artifacts/ahnss_repo_eval/ami/ahnss_repo_eval_metrics.json"
        case .callhome:
            metricsRelativePath = "artifacts/ahnss_repo_eval/callhome/ahnss_repo_eval_metrics.json"
        case .dihard2:
            metricsRelativePath = "artifacts/czlvt_repo_eval/dihard2/czlvt_repo_eval_metrics.json"
        case .dihard3:
            metricsRelativePath = "artifacts/LDC2022S14_repo_eval/dihard3_nominal/LDC2022S14_repo_eval_metrics.json"
        }
        let metricsURL = try requireWorkspaceArtifact(metricsRelativePath)
        let metrics = try decode(RepoEvalMetrics.self, from: metricsURL)
        let kaldiDirURL = try requireExistingPath(
            URL(fileURLWithPath: metrics.artifacts.kaldiDir, isDirectory: true),
            description: metrics.artifacts.kaldiDir
        )
        let rawLogitsURL = try requireExistingPath(
            URL(fileURLWithPath: metrics.artifacts.rawLogitsNPY),
            description: metrics.artifacts.rawLogitsNPY
        )
        let referenceRTTMURL = try requireExistingPath(
            URL(fileURLWithPath: metrics.referenceRTTM),
            description: metrics.referenceRTTM
        )
        return RepoFixture(
            variant: variant,
            metrics: metrics,
            metricsURL: metricsURL,
            audio8kURL: try findFirstWAV(in: kaldiDirURL),
            rawLogitsURL: rawLogitsURL,
            referenceRTTMURL: referenceRTTMURL
        )
    }

    private func makeEngine(variant: LSEENDVariant) async throws -> LSEENDInferenceEngine {
        if let cached = Self.cachedEngines[variant] {
            return cached
        }
        let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
        let created = try LSEENDInferenceEngine(descriptor: descriptor, computeUnits: .cpuOnly)
        Self.cachedEngines[variant] = created
        return created
    }

    private func referenceBinaryMatrix(
        rttmURL: URL,
        numFrames: Int,
        frameRate: Double
    ) throws -> LSEENDMatrix {
        let parsed = try LSEENDEvaluation.parseRTTM(url: rttmURL)
        return LSEENDEvaluation.rttmToFrameMatrix(
            entries: parsed.entries,
            speakers: parsed.speakers,
            numFrames: numFrames,
            frameRate: frameRate
        )
    }

    private func compare(_ actual: LSEENDMatrix, _ expected: LSEENDMatrix) -> ErrorStats {
        XCTAssertEqual(actual.rows, expected.rows)
        XCTAssertEqual(actual.columns, expected.columns)
        XCTAssertEqual(actual.values.count, expected.values.count)

        guard
            actual.rows == expected.rows,
            actual.columns == expected.columns,
            actual.values.count == expected.values.count
        else {
            return ErrorStats(maxAbs: .infinity, meanAbs: .infinity)
        }

        var maxAbs = 0.0
        var sumAbs = 0.0
        for (lhs, rhs) in zip(actual.values, expected.values) {
            let diff = abs(Double(lhs - rhs))
            maxAbs = max(maxAbs, diff)
            sumAbs += diff
        }
        return ErrorStats(
            maxAbs: maxAbs,
            meanAbs: actual.values.isEmpty ? 0 : sumAbs / Double(actual.values.count)
        )
    }

    private func decode<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }

    private func requireWorkspaceArtifact(_ relativePath: String) throws -> URL {
        try requireExistingPath(
            Self.rootURL.appendingPathComponent(relativePath),
            description: relativePath
        )
    }

    private func requireExistingPath(_ url: URL, description: String) throws -> URL {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip(
                "Skipping LS-EEND runtime parity test: missing \(description) under \(Self.rootURL.path). "
                    + Self.workspaceSetupHint
            )
        }
        return url
    }

    private func findFirstWAV(in directory: URL) throws -> URL {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        guard
            let wavURL =
                contents
                .filter({ $0.pathExtension.lowercased() == "wav" })
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
                .first
        else {
            throw NSError(
                domain: "LSEENDRuntimeTests",
                code: 100,
                userInfo: [NSLocalizedDescriptionKey: "No WAV file found in \(directory.path)"]
            )
        }
        return wavURL
    }

    private func runProbe<T: Decodable>(_ type: T.Type, arguments: [String]) throws -> T {
        try Self.ensureProbeExecutable()
        let executablePath = Self.probeExecutableURL.path
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw NSError(
                domain: "LSEENDRuntimeTests",
                code: 101,
                userInfo: [NSLocalizedDescriptionKey: "Missing runtime probe executable at \(executablePath)."]
            )
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath)
        let outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        process.arguments = arguments + ["--output", outputURL.path]
        var environment = ProcessInfo.processInfo.environment
        environment["LSEEND_WORKSPACE_ROOT"] = Self.rootURL.path
        process.environment = environment
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
        let outputData = (try? Data(contentsOf: outputURL)) ?? Data()
        try? FileManager.default.removeItem(at: outputURL)
        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "LSEENDRuntimeTests",
                code: 102,
                userInfo: [
                    NSLocalizedDescriptionKey: """
                    Runtime probe failed.
                    executable: \(executablePath)
                    status: \(process.terminationStatus)
                    stdout:
                    \(String(decoding: outputData, as: UTF8.self))
                    stderr:
                    \(String(decoding: errorData, as: UTF8.self))
                    """
                ]
            )
        }

        return try JSONDecoder().decode(type, from: outputData)
    }

    private func runOfflineProbe(variant: LSEENDVariant, audioURL: URL) throws -> ProbeInferenceResult {
        try runProbe(
            ProbeInferenceResult.self,
            arguments: [
                "offline",
                "--variant", variantProbeToken(variant),
                "--audio", audioURL.path,
            ]
        )
    }

    private func runStreamingProbe(
        variant: LSEENDVariant,
        audioURL: URL,
        chunkSeconds: Double
    ) throws -> ProbeStreamingResult {
        try runProbe(
            ProbeStreamingResult.self,
            arguments: [
                "streaming",
                "--variant", variantProbeToken(variant),
                "--audio", audioURL.path,
                "--chunk-seconds", String(chunkSeconds),
            ]
        )
    }

    private func runSessionCheckProbe(
        variant: LSEENDVariant,
        audioURL: URL,
        chunkSeconds: Double
    ) throws -> ProbeSessionCheckResult {
        try runProbe(
            ProbeSessionCheckResult.self,
            arguments: [
                "session-check",
                "--variant", variantProbeToken(variant),
                "--audio", audioURL.path,
                "--chunk-seconds", String(chunkSeconds),
            ]
        )
    }

    private func variantProbeToken(_ variant: LSEENDVariant) -> String {
        switch variant {
        case .ami:
            return "ami"
        case .callhome:
            return "callhome"
        case .dihard2:
            return "dihard2"
        case .dihard3:
            return "dihard3"
        }
    }

    private static func ensureProbeExecutable() throws {
        guard !didEnsureProbeExecutable else {
            return
        }

        let fileManager = FileManager.default
        let missingSources = probeSourceURLs.filter { !fileManager.fileExists(atPath: $0.path) }
        guard missingSources.isEmpty else {
            throw XCTSkip(
                "Skipping LS-EEND runtime parity test: missing runtime probe sources under \(rootURL.path). "
                    + workspaceSetupHint
            )
        }
        let binaryExists = fileManager.fileExists(atPath: probeExecutableURL.path)
        let binaryDate = binaryExists ? try modificationDate(for: probeExecutableURL) : .distantPast
        let newestSourceDate =
            try probeSourceURLs
            .map(modificationDate(for:))
            .max() ?? .distantPast

        if !binaryExists || newestSourceDate > binaryDate {
            let outputDirectory = probeExecutableURL.deletingLastPathComponent()
            try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

            let process = Process()
            process.executableURL = try swiftCompilerURL()
            process.arguments =
                [
                    "-sdk", try macOSSDKURL().path,
                    "-target", currentTargetTriple(),
                    "-O",
                    "-o", probeExecutableURL.path,
                ] + probeSourceURLs.map(\.path)
            let stderr = Pipe()
            process.standardError = stderr
            try process.run()
            process.waitUntilExit()

            let errorOutput = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            guard process.terminationStatus == 0 else {
                throw NSError(
                    domain: "LSEENDRuntimeTests",
                    code: 103,
                    userInfo: [
                        NSLocalizedDescriptionKey: """
                        Failed to build runtime probe.
                        status: \(process.terminationStatus)
                        stderr:
                        \(errorOutput)
                        """
                    ]
                )
            }
        }

        didEnsureProbeExecutable = true
    }

    private static func modificationDate(for url: URL) throws -> Date {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return attributes[.modificationDate] as? Date ?? .distantPast
    }

    private static func swiftCompilerURL() throws -> URL {
        let candidates = [
            ProcessInfo.processInfo.environment["SWIFTC"],
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc",
            "/usr/bin/swiftc",
        ].compactMap { $0 }

        if let path = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
            return URL(fileURLWithPath: path)
        }

        throw NSError(
            domain: "LSEENDRuntimeTests",
            code: 104,
            userInfo: [NSLocalizedDescriptionKey: "Unable to locate a Swift compiler for the runtime probe build."]
        )
    }

    private static func macOSSDKURL() throws -> URL {
        let candidates = [
            ProcessInfo.processInfo.environment["SDKROOT"],
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
        ].compactMap { $0 }

        if let path = candidates.first(where: { FileManager.default.fileExists(atPath: $0) }) {
            return URL(fileURLWithPath: path)
        }

        throw NSError(
            domain: "LSEENDRuntimeTests",
            code: 105,
            userInfo: [NSLocalizedDescriptionKey: "Unable to locate a macOS SDK for the runtime probe build."]
        )
    }

    private static func currentTargetTriple() -> String {
        #if arch(arm64)
        let arch = "arm64"
        #elseif arch(x86_64)
        let arch = "x86_64"
        #else
        let arch = "arm64"
        #endif

        let version = ProcessInfo.processInfo.operatingSystemVersion
        return "\(arch)-apple-macos\(version.majorVersion).0"
    }
}
