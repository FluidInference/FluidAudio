#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

let cliLogger = AppLogger(category: "Main")

func printUsage() {
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

// Returns the Mach high-water resident memory footprint for the current process.
// This captures the peak physical memory, including shared framework pages, rather than
// the CLI's current or private usage.
func fetchPeakMemoryUsageBytes() -> UInt64? {
    var info = task_vm_info_data_t()
    var count =
        mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size)
        / mach_msg_type_number_t(MemoryLayout<natural_t>.size)

    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(
                mach_task_self_,
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

func logPeakMemoryUsage() {
    guard let peakBytes = fetchPeakMemoryUsageBytes() else {
        cliLogger.error("Unable to determine peak memory usage")
        return
    }

    let peakGigabytes = Double(peakBytes) / 1024.0 / 1024.0 / 1024.0
    let formatted = String(format: "%.3f", peakGigabytes)
    print(
        "Peak memory usage (process-wide): \(formatted) GB"
    )
}

func exitWithPeakMemory(_ code: Int32) -> Never {
    logPeakMemoryUsage()
    exit(code)
}

// Main entry point
let arguments = CommandLine.arguments

guard arguments.count > 1 else {
    printUsage()
    exitWithPeakMemory(1)
}

// Log system information once at application startup
Task {
    await SystemInfo.logOnce(using: cliLogger)
}

let command = arguments[1]
let semaphore = DispatchSemaphore(value: 0)

// Use Task to handle async commands
Task {
    defer {
        logPeakMemoryUsage()
        semaphore.signal()
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
        if #available(macOS 13.0, *) {
            await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("Multi-stream requires macOS 13.0 or later")
            exitWithPeakMemory(1)
        }
        await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))

    case "tts":
        if #available(macOS 13.0, *) {
            await TTS.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("TTS requires macOS 13.0 or later")
            exitWithPeakMemory(1)
        }

    case "diarization-benchmark":
        await StreamDiarizationBenchmark.run(arguments: Array(arguments.dropFirst(2)))
    case "process":
        await ProcessCommand.run(arguments: Array(arguments.dropFirst(2)))
    case "download":
        await DownloadCommand.run(arguments: Array(arguments.dropFirst(2)))
    case "help", "--help", "-h":
        printUsage()
        exitWithPeakMemory(0)
    default:
        cliLogger.error("Unknown command: \(command)")
        printUsage()
        exitWithPeakMemory(1)
    }
}

// Wait for async task to complete
semaphore.wait()
#else
#error("FluidAudioCLI is only supported on macOS")
#endif
