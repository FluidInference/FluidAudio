#if os(macOS)
import FluidAudio
import Foundation

/// Shared utilities for diarization benchmark commands (LS-EEND and Sortformer).
enum DiarizationBenchmarkUtils {

    /// Dataset corpora supported by diarization benchmarks.
    enum Dataset: String {
        case ami = "ami"
        case voxconverse = "voxconverse"
        case callhome = "callhome"
    }

    /// Per-meeting benchmark result shared across diarization benchmark commands.
    struct BenchmarkResult {
        let meetingName: String
        let der: Float
        let missRate: Float
        let falseAlarmRate: Float
        let speakerErrorRate: Float
        let rtfx: Float
        let processingTime: Double
        let totalFrames: Int
        let detectedSpeakers: Int
        let groundTruthSpeakers: Int
        let modelLoadTime: Double
        let audioLoadTime: Double?
    }

    // MARK: - File Paths

    static func getAMIFiles(maxFiles: Int?) -> [String] {
        let allMeetings = [
            "EN2002a", "EN2002b", "EN2002c", "EN2002d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d",
            "TS3003a", "TS3003b", "TS3003c", "TS3003d",
        ]

        var availableMeetings: [String] = []
        for meeting in allMeetings {
            let path = getAudioPath(for: meeting, dataset: .ami)
            if FileManager.default.fileExists(atPath: path) {
                availableMeetings.append(meeting)
            }
        }

        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    static func getAudioPath(for meeting: String, dataset: Dataset) -> String {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        switch dataset {
        case .ami:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/ami_official/sdm/\(meeting).Mix-Headset.wav"
            ).path
        case .voxconverse:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/voxconverse_test_wav/\(meeting).wav"
            ).path
        case .callhome:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/callhome_eng/\(meeting).wav"
            ).path
        }
    }

    static func getRTTMURL(for meeting: String, dataset: Dataset) -> URL? {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        switch dataset {
        case .ami:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/ami_official/rttm/\(meeting).rttm"
            )
        case .voxconverse:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/rttm_repo/test/\(meeting).rttm"
            )
        case .callhome:
            return homeDir.appendingPathComponent(
                "FluidAudioDatasets/callhome_eng/rttm/\(meeting).rttm"
            )
        }
    }

    static func getVoxConverseFiles(maxFiles: Int?) -> [String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let voxDir = homeDir.appendingPathComponent(
            "FluidAudioDatasets/voxconverse/voxconverse_test_wav"
        )

        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: voxDir,
                includingPropertiesForKeys: nil
            )
        else {
            return []
        }

        var availableMeetings: [String] = []
        for file in files where file.pathExtension == "wav" {
            let name = file.deletingPathExtension().lastPathComponent
            let rttmPath = homeDir.appendingPathComponent(
                "FluidAudioDatasets/voxconverse/rttm_repo/test/\(name).rttm"
            )
            if FileManager.default.fileExists(atPath: rttmPath.path) {
                availableMeetings.append(name)
            }
        }

        availableMeetings.sort()
        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    static func getCALLHOMEFiles(maxFiles: Int?) -> [String] {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let callhomeDir = homeDir.appendingPathComponent("FluidAudioDatasets/callhome_eng")

        guard
            let files = try? FileManager.default.contentsOfDirectory(
                at: callhomeDir,
                includingPropertiesForKeys: nil
            )
        else {
            return []
        }

        var availableMeetings: [String] = []
        for file in files where file.pathExtension == "wav" {
            let name = file.deletingPathExtension().lastPathComponent
            let rttmPath = callhomeDir.appendingPathComponent("rttm/\(name).rttm")
            if FileManager.default.fileExists(atPath: rttmPath.path) {
                availableMeetings.append(name)
            }
        }

        availableMeetings.sort()
        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }
        return availableMeetings
    }

    /// Returns files for the given dataset, filtering by availability.
    static func getFiles(for dataset: Dataset, maxFiles: Int?) -> [String] {
        switch dataset {
        case .ami:
            return getAMIFiles(maxFiles: maxFiles)
        case .voxconverse:
            return getVoxConverseFiles(maxFiles: maxFiles)
        case .callhome:
            return getCALLHOMEFiles(maxFiles: maxFiles)
        }
    }

    // MARK: - Summary & Output

    /// Prints a formatted benchmark summary table.
    ///
    /// - Parameters:
    ///   - results: Benchmark results to summarize.
    ///   - title: Header title (e.g. "LS-EEND BENCHMARK SUMMARY").
    ///   - derTargets: DER percentage thresholds to check, ordered from strictest to most lenient
    ///     (e.g. `[15, 25]` prints "DER < 15%" if met, else "DER < 25%", else "DER > 25%").
    static func printFinalSummary(
        results: [BenchmarkResult],
        title: String,
        derTargets: [Float]
    ) {
        guard !results.isEmpty else { return }

        print("\n" + String(repeating: "=", count: 80))
        print(title)
        print(String(repeating: "=", count: 80))

        print("Results Sorted by DER:")
        print(String(repeating: "-", count: 70))
        print("Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx")
        print(String(repeating: "-", count: 70))

        for result in results.sorted(by: { $0.der < $1.der }) {
            let speakerInfo = "\(result.detectedSpeakers)/\(result.groundTruthSpeakers)"
            let meetingCol = result.meetingName.padding(toLength: 12, withPad: " ", startingAt: 0)
            let speakerCol = speakerInfo.padding(toLength: 10, withPad: " ", startingAt: 0)
            print(
                String(
                    format: "%@ %8.1f %8.1f %8.1f %8.1f %@ %8.1f",
                    meetingCol,
                    result.der,
                    result.missRate,
                    result.falseAlarmRate,
                    result.speakerErrorRate,
                    speakerCol,
                    result.rtfx))
        }
        print(String(repeating: "-", count: 70))

        let count = Float(results.count)
        let avgDER = results.map { $0.der }.reduce(0, +) / count
        let avgMiss = results.map { $0.missRate }.reduce(0, +) / count
        let avgFA = results.map { $0.falseAlarmRate }.reduce(0, +) / count
        let avgSE = results.map { $0.speakerErrorRate }.reduce(0, +) / count
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / count

        print(
            String(
                format: "AVERAGE      %8.1f %8.1f %8.1f %8.1f         - %8.1f",
                avgDER, avgMiss, avgFA, avgSE, avgRTFx))
        print(String(repeating: "=", count: 70))

        print("\nTarget Check:")
        var matched = false
        for target in derTargets.sorted() {
            if avgDER < target {
                print("   DER < \(String(format: "%.0f", target))% (achieved: \(String(format: "%.1f", avgDER))%)")
                matched = true
                break
            }
        }
        if !matched, let highest = derTargets.max() {
            print(
                "   DER > \(String(format: "%.0f", highest))% (achieved: \(String(format: "%.1f", avgDER))%)")
        }

        if avgRTFx > 1 {
            print("   RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        } else {
            print("   RTFx < 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        }
    }

    static func saveJSONResults(results: [BenchmarkResult], to path: String) {
        let jsonData = results.map { resultToDict($0) }
        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
            print("JSON results saved to: \(path)")
        } catch {
            print("Failed to save JSON: \(error)")
        }
    }

    // MARK: - Progress Save/Load

    static func resultToDict(_ result: BenchmarkResult) -> [String: Any] {
        var dict: [String: Any] = [
            "meeting": result.meetingName,
            "der": result.der,
            "missRate": result.missRate,
            "falseAlarmRate": result.falseAlarmRate,
            "speakerErrorRate": result.speakerErrorRate,
            "rtfx": result.rtfx,
            "processingTime": result.processingTime,
            "totalFrames": result.totalFrames,
            "detectedSpeakers": result.detectedSpeakers,
            "groundTruthSpeakers": result.groundTruthSpeakers,
            "modelLoadTime": result.modelLoadTime,
        ]
        if let audioLoadTime = result.audioLoadTime {
            dict["audioLoadTime"] = audioLoadTime
        }
        return dict
    }

    static func saveProgress(results: [BenchmarkResult], to path: String) {
        let jsonData = results.map { resultToDict($0) }
        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            print("Failed to save progress: \(error)")
        }
    }

    static func loadProgress(from path: String) -> [BenchmarkResult]? {
        guard FileManager.default.fileExists(atPath: path) else { return nil }

        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            guard let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                return nil
            }

            return jsonArray.compactMap { dict -> BenchmarkResult? in
                guard let meeting = dict["meeting"] as? String,
                    let der = (dict["der"] as? NSNumber)?.floatValue,
                    let missRate = (dict["missRate"] as? NSNumber)?.floatValue,
                    let falseAlarmRate = (dict["falseAlarmRate"] as? NSNumber)?.floatValue,
                    let speakerErrorRate = (dict["speakerErrorRate"] as? NSNumber)?.floatValue,
                    let rtfx = (dict["rtfx"] as? NSNumber)?.floatValue,
                    let processingTime = (dict["processingTime"] as? NSNumber)?.doubleValue,
                    let totalFrames = (dict["totalFrames"] as? NSNumber)?.intValue,
                    let detectedSpeakers = (dict["detectedSpeakers"] as? NSNumber)?.intValue,
                    let groundTruthSpeakers = (dict["groundTruthSpeakers"] as? NSNumber)?.intValue,
                    let modelLoadTime = (dict["modelLoadTime"] as? NSNumber)?.doubleValue
                else {
                    return nil
                }

                let audioLoadTime = (dict["audioLoadTime"] as? NSNumber)?.doubleValue

                return BenchmarkResult(
                    meetingName: meeting,
                    der: der,
                    missRate: missRate,
                    falseAlarmRate: falseAlarmRate,
                    speakerErrorRate: speakerErrorRate,
                    rtfx: rtfx,
                    processingTime: processingTime,
                    totalFrames: totalFrames,
                    detectedSpeakers: detectedSpeakers,
                    groundTruthSpeakers: groundTruthSpeakers,
                    modelLoadTime: modelLoadTime,
                    audioLoadTime: audioLoadTime
                )
            }
        } catch {
            print("Failed to load progress: \(error)")
            return nil
        }
    }
}
#endif
