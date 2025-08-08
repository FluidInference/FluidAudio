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
            diarization-benchmark  Run streaming diarization benchmark
            vad-benchmark           Run VAD-specific benchmark
            asr-benchmark           Run ASR benchmark on LibriSpeech
            transcribe              Transcribe audio file using streaming ASR
            multi-stream            Transcribe multiple audio files in parallel
            download                Download evaluation datasets
            help                    Show this help message

        Run 'fluidaudio <command> --help' for command-specific options.

        Examples:
            fluidaudio process audio.wav --output results.json

            fluidaudio diarization-benchmark --single-file ES2004a

            fluidaudio asr-benchmark --subset test-clean --max-files 100

            fluidaudio transcribe audio.wav --low-latency

            fluidaudio multi-stream audio1.wav audio2.wav

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
    case "vad-benchmark":
        await VadBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
    case "asr-benchmark":
        print("DEBUG: asr-benchmark command received")
        if #available(macOS 13.0, *) {
            print("DEBUG: macOS version check passed")
            await ASRBenchmark.runASRBenchmark(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("ASR benchmark requires macOS 13.0 or later")
            exit(1)
        }
    case "transcribe":
        if #available(macOS 13.0, *) {
            await StreamingTranscribeCommand.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("Transcribe requires macOS 13.0 or later")
            exit(1)
        }
    case "multi-stream":
        if #available(macOS 13.0, *) {
            await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("Multi-stream requires macOS 13.0 or later")
            exit(1)
        }
    case "diarization-benchmark":
        if #available(macOS 13.0, *) {
            await StreamDiarizationBenchmark.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("Stream diarization benchmark requires macOS 13.0 or later")
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
        print("Unknown command: \(command)")
        printUsage()
        exit(1)
    }

    semaphore.signal()
}

// Wait for async task to complete
semaphore.wait()
#else
#error("FluidAudioCLI is only supported on macOS")
#endif
