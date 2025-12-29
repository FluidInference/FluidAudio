#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation
import MachTaskSelfWrapper

// Using @main instead of main.swift for Swift 6 compatibility.
// This provides an explicit async context and clear isolation semantics.
@main
struct FluidAudioCLI {
    static let cliLogger = AppLogger(category: "Main")

    static func main() async {
        let arguments = CommandLine.arguments

        guard arguments.count > 1 else {
            printUsage()
            exitWithPeakMemory(1)
        }

        // Log system information once at application startup
        await SystemInfo.logOnce(using: cliLogger)

        let command = arguments[1]

        defer {
            logPeakMemoryUsage()
        }

        switch command {
        case "vad-benchmark":
            await VadBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
        case "vad-analyze":
            await VadAnalyzeCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "asr-benchmark":
            await ASRBenchmark.runASRBenchmark(arguments: Array(arguments.dropFirst(2)))
        case "fleurs-benchmark":
            await FLEURSBenchmark.runCLI(arguments: Array(arguments.dropFirst(2)))
        case "transcribe":
            await TranscribeCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "multi-stream":
            await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "tts":
            await TTS.run(arguments: Array(arguments.dropFirst(2)))
        case "diarization-benchmark":
            await StreamDiarizationBenchmark.run(arguments: Array(arguments.dropFirst(2)))
        case "process":
            await ProcessCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "download":
            await DownloadCommand.run(arguments: Array(arguments.dropFirst(2)))
        case "parakeet-eou":
            await ParakeetEouCommand.main(Array(arguments.dropFirst(2)))
        case "help", "--help", "-h":
            printUsage()
        default:
            cliLogger.error("Unknown command: \(command)")
            printUsage()
            exit(1)
        }
    }

    static func printUsage() {
        cliLogger.info(
            """
            FluidAudio CLI

            Usage: fluidaudio <command> [options]

            Commands:
                process                 Process a single audio file for diarization
                diarization-benchmark   Run diarization benchmark
                vad-benchmark           Run VAD-specific benchmark
                vad-analyze             Inspect VAD segmentation and streaming events
                asr-benchmark           Run ASR benchmark on LibriSpeech
                fleurs-benchmark        Run multilingual ASR benchmark on FLEURS dataset
                transcribe              Transcribe audio file using streaming ASR
                multi-stream            Transcribe multiple audio files in parallel
                tts                     Synthesize speech from text using Kokoro TTS
                parakeet-eou            Run Parakeet EOU Streaming ASR on a single file
                download                Download evaluation datasets
                help                    Show this help message

            Run 'fluidaudio <command> --help' for command-specific options.

            Examples:
                fluidaudio process audio.wav --output results.json

                fluidaudio diarization-benchmark --single-file ES2004a

                fluidaudio asr-benchmark --subset test-clean --max-files 100

                fluidaudio fleurs-benchmark --languages en_us,fr_fr --samples 10

                fluidaudio transcribe audio.wav --low-latency

                fluidaudio multi-stream audio1.wav audio2.wav

                fluidaudio tts "Hello world" --output hello.wav

                fluidaudio vad-analyze audio.wav --streaming

                fluidaudio download --dataset ami-sdm
            """
        )
    }

    static func fetchPeakMemoryUsageBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count =
            mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size)
            / mach_msg_type_number_t(MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    get_current_task_port(),
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count)
            }
        }

        guard result == KERN_SUCCESS else {
            return nil
        }

        return info.resident_size_peak
    }

    static func logPeakMemoryUsage() {
        guard let peakBytes = fetchPeakMemoryUsageBytes() else {
            cliLogger.error("Unable to determine peak memory usage")
            return
        }

        let peakGigabytes = Double(peakBytes) / 1024.0 / 1024.0 / 1024.0
        let formatted = String(format: "%.3f", peakGigabytes)
        cliLogger.info(
            "Peak memory usage (process-wide): \(formatted) GB"
        )
    }

    static func exitWithPeakMemory(_ code: Int32) -> Never {
        logPeakMemoryUsage()
        exit(code)
    }
}
#else
#error("FluidAudioCLI is only supported on macOS")
#endif
