import AVFoundation
import CoreML
import Foundation

enum LSEENDRuntimeProbeSupport {
    private static let probeFlag = "--lseend-probe"

    static func runIfRequested(arguments: [String] = CommandLine.arguments) async {
        guard let flagIndex = arguments.firstIndex(of: probeFlag) else {
            return
        }

        let probeArguments = Array(arguments[(flagIndex + 1)...])
        let outputURL = try? parseOptionalOutputURL(from: probeArguments)
        do {
            let payload = try await run(arguments: probeArguments)
            if let outputURL {
                try payload.write(to: outputURL, options: .atomic)
            } else {
                FileHandle.standardOutput.write(payload)
            }
            fflush(stdout)
            exit(0)
        } catch {
            let message = "\(error.localizedDescription)\n"
            if let data = message.data(using: .utf8) {
                FileHandle.standardError.write(data)
            }
            fflush(stderr)
            exit(1)
        }
    }

    private static func run(arguments: [String]) async throws -> Data {
        guard let command = arguments.first else {
            throw ProbeError.invalidArguments("Missing command.")
        }

        switch command {
        case "offline":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let engine = try LSEENDInferenceEngine(descriptor: await .loadFromHuggingFace(variant: variant), computeUnits: .cpuOnly)
            return try encodeJSON(ProbeInferenceResult(engine.infer(audioFileURL: audioURL)))
        case "streaming":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let chunkSeconds = try parseDouble(flag: "--chunk-seconds", from: arguments)
            let engine = try LSEENDInferenceEngine(descriptor: await .loadFromHuggingFace(variant: variant), computeUnits: .cpuOnly)
            return try encodeJSON(ProbeStreamingResult(engine.simulateStreaming(audioFileURL: audioURL, chunkSeconds: chunkSeconds)))
        case "session-check":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let chunkSeconds = try parseDouble(flag: "--chunk-seconds", from: arguments)
            return try encodeJSON(try runSessionCheck(variant: variant, audioURL: audioURL, chunkSeconds: chunkSeconds))
        default:
            throw ProbeError.invalidArguments("Unknown command: \(command)")
        }
    }

    private static func runSessionCheck(
        variant: LSEENDVariant,
        audioURL: URL,
        chunkSeconds: Double
    ) throws -> ProbeSessionCheckResult {
        
        let semaphore = DispatchSemaphore(value: 0)
        var descriptorResult: Result<LSEENDModelDescriptor, Error>!
        
        Task {
            do {
                let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
                descriptorResult = .success(descriptor)
            } catch {
                descriptorResult = .failure(error)
            }
            
            semaphore.signal()
        }
        
        semaphore.wait()
        
        let descriptor = try descriptorResult.get()
        let engine = try LSEENDInferenceEngine(descriptor: descriptor, computeUnits: .cpuOnly)
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(engine.targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        let audio = try converter.resampleAudioFile(audioURL)
        let chunkSize = max(1, Int(round(chunkSeconds * Double(engine.targetSampleRate))))
        let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

        let firstChunk = Array(audio.prefix(chunkSize))
        let firstUpdate = try session.pushAudio(firstChunk)

        var committed = LSEENDMatrix.empty(columns: engine.metadata.realOutputDim)
        if let firstUpdate, !firstUpdate.probabilities.isEmpty {
            committed = committed.appendingRows(firstUpdate.probabilities)
        }

        var startIndex = firstChunk.count
        while startIndex < audio.count {
            let stopIndex = min(audio.count, startIndex + chunkSize)
            if let update = try session.pushAudio(Array(audio[startIndex..<stopIndex])),
                !update.probabilities.isEmpty
            {
                committed = committed.appendingRows(update.probabilities)
            }
            startIndex = stopIndex
        }

        if let finalUpdate = try session.finalize(), !finalUpdate.probabilities.isEmpty {
            committed = committed.appendingRows(finalUpdate.probabilities)
        }

        let repeatedFinalize = try session.finalize()
        return ProbeSessionCheckResult(
            firstUpdateRows: firstUpdate?.probabilities.rows ?? 0,
            firstUpdateTotalEmittedFrames: firstUpdate?.totalEmittedFrames ?? 0,
            committedProbabilities: ProbeMatrix(committed),
            snapshotProbabilities: ProbeMatrix(session.snapshot().probabilities),
            repeatedFinalizeReturnedUpdate: repeatedFinalize != nil,
            repeatedSnapshotProbabilities: ProbeMatrix(session.snapshot().probabilities)
        )
    }

    private static func encodeJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    private static func parseVariant(from arguments: [String]) throws -> LSEENDVariant {
        let raw = try parseString(flag: "--variant", from: arguments)
        switch raw.lowercased() {
        case "ami":
            return .ami
        case "callhome":
            return .callhome
        case "dihard2":
            return .dihard2
        case "dihard3":
            return .dihard3
        default:
            throw ProbeError.invalidArguments("Unsupported variant: \(raw)")
        }
    }

    private static func parseAudioURL(from arguments: [String]) throws -> URL {
        URL(fileURLWithPath: try parseString(flag: "--audio", from: arguments))
    }

    private static func parseOptionalOutputURL(from arguments: [String]) throws -> URL? {
        guard let index = arguments.firstIndex(of: "--output") else {
            return nil
        }
        guard arguments.indices.contains(index + 1) else {
            throw ProbeError.invalidArguments("Missing --output path.")
        }
        return URL(fileURLWithPath: arguments[index + 1])
    }

    private static func parseDouble(flag: String, from arguments: [String]) throws -> Double {
        guard let value = Double(try parseString(flag: flag, from: arguments)) else {
            throw ProbeError.invalidArguments("Invalid numeric value for \(flag).")
        }
        return value
    }

    private static func parseString(flag: String, from arguments: [String]) throws -> String {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            throw ProbeError.invalidArguments("Missing \(flag).")
        }
        return arguments[index + 1]
    }
}

private struct ProbeMatrix: Codable {
    let rows: Int
    let columns: Int
    let values: [Float]

    init(_ matrix: LSEENDMatrix) {
        rows = matrix.rows
        columns = matrix.columns
        values = matrix.values
    }
}

private struct ProbeInferenceResult: Codable {
    let logits: ProbeMatrix
    let probabilities: ProbeMatrix
    let fullLogits: ProbeMatrix
    let fullProbabilities: ProbeMatrix
    let frameHz: Double
    let durationSeconds: Double

    init(_ result: LSEENDInferenceResult) {
        logits = ProbeMatrix(result.logits)
        probabilities = ProbeMatrix(result.probabilities)
        fullLogits = ProbeMatrix(result.fullLogits)
        fullProbabilities = ProbeMatrix(result.fullProbabilities)
        frameHz = result.frameHz
        durationSeconds = result.durationSeconds
    }
}

private struct ProbeStreamingProgress: Codable {
    let chunkIndex: Int
    let bufferSeconds: Double
    let numFramesEmitted: Int
    let totalFramesEmitted: Int
    let flush: Bool

    init(_ progress: LSEENDStreamingProgress) {
        chunkIndex = progress.chunkIndex
        bufferSeconds = progress.bufferSeconds
        numFramesEmitted = progress.numFramesEmitted
        totalFramesEmitted = progress.totalFramesEmitted
        flush = progress.flush
    }
}

private struct ProbeStreamingResult: Codable {
    let result: ProbeInferenceResult
    let updates: [ProbeStreamingProgress]

    init(_ simulation: LSEENDStreamingSimulationResult) {
        result = ProbeInferenceResult(simulation.result)
        updates = simulation.updates.map(ProbeStreamingProgress.init)
    }
}

private struct ProbeSessionCheckResult: Codable {
    let firstUpdateRows: Int
    let firstUpdateTotalEmittedFrames: Int
    let committedProbabilities: ProbeMatrix
    let snapshotProbabilities: ProbeMatrix
    let repeatedFinalizeReturnedUpdate: Bool
    let repeatedSnapshotProbabilities: ProbeMatrix
}

private enum ProbeError: LocalizedError {
    case invalidArguments(String)

    var errorDescription: String? {
        switch self {
        case .invalidArguments(let message):
            return message
        }
    }
}
