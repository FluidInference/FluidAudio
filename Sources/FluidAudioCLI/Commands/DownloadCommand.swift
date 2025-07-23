import Foundation
import FluidAudio

/// Handler for the 'download' command - downloads benchmark datasets
enum DownloadCommand {
    static func run(arguments: [String]) async {
        var dataset = "all"
        var forceDownload = false

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
            case "--force":
                forceDownload = true
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("📥 Starting dataset download")
        print("   Dataset: \(dataset)")
        print("   Force download: \(forceDownload ? "enabled" : "disabled")")

        switch dataset.lowercased() {
        case "ami-sdm":
            await DatasetDownloader.downloadAMIDataset(variant: .sdm, force: forceDownload)
        case "ami-ihm":
            await DatasetDownloader.downloadAMIDataset(variant: .ihm, force: forceDownload)
        case "ami-annotations":
            await DatasetDownloader.downloadAMIAnnotations(force: forceDownload)
        case "vad":
            await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")  // Default to mini100 for more test data
        case "vad-mini50":
            await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini50")
        case "vad-mini100":
            await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")
        case "librispeech-test-clean":
            let benchmark = ASRBenchmark()
            do {
                try await benchmark.downloadLibriSpeech(subset: "test-clean", forceDownload: forceDownload)
            } catch {
                print("❌ Failed to download LibriSpeech test-clean: \(error)")
                exit(1)
            }
        case "librispeech-test-other":
            let benchmark = ASRBenchmark()
            do {
                try await benchmark.downloadLibriSpeech(subset: "test-other", forceDownload: forceDownload)
            } catch {
                print("❌ Failed to download LibriSpeech test-other: \(error)")
                exit(1)
            }
        case "all":
            await DatasetDownloader.downloadAMIDataset(variant: .sdm, force: forceDownload)
            await DatasetDownloader.downloadAMIDataset(variant: .ihm, force: forceDownload)
            await DatasetDownloader.downloadVadDataset(force: forceDownload, dataset: "mini100")
        default:
            print("❌ Unsupported dataset: \(dataset)")
            printUsage()
            exit(1)
        }
    }
    
    private static func printUsage() {
        print(
            """
            
            Download Command Usage:
                fluidaudio download [options]
            
            Options:
                --dataset <name>    Dataset to download (default: all)
                --force            Force re-download even if exists
            
            Available datasets:
                ami-sdm                 AMI SDM subset
                ami-ihm                 AMI IHM subset
                ami-annotations         AMI annotation files
                vad, vad-mini50,       VAD evaluation datasets
                vad-mini100
                librispeech-test-clean  LibriSpeech test-clean subset
                librispeech-test-other  LibriSpeech test-other subset
                all                     All diarization datasets
            
            Examples:
                fluidaudio download --dataset ami-sdm
                fluidaudio download --dataset librispeech-test-clean --force
            """
        )
    }
}