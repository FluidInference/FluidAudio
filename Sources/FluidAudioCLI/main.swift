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

// Mirror OSLog messages to console when running CLI
AppLogger.enableConsoleOutput(true)

let command = arguments[1]
let semaphore = DispatchSemaphore(value: 0)

// Use Task to handle async commands
Task {
    switch command {
    case "vad-benchmark":
        await VadBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
    case "asr-benchmark":
        cliLogger.debug("asr-benchmark command received")
        if #available(macOS 13.0, *) {
            cliLogger.debug("macOS version check passed")
            await ASRBenchmark.runASRBenchmark(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("ASR benchmark requires macOS 13.0 or later")
            exit(1)
        }
    case "fleurs-benchmark":
        if #available(macOS 13.0, *) {
            await FLEURSBenchmark.runCLI(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("FLEURS benchmark requires macOS 13.0 or later")
            exit(1)
        }
    case "transcribe":
        if #available(macOS 13.0, *) {
            await TranscribeCommand.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("Transcribe requires macOS 13.0 or later")
            exit(1)
        }
    case "multi-stream":
        if #available(macOS 13.0, *) {
            await MultiStreamCommand.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("Multi-stream requires macOS 13.0 or later")
            exit(1)
        }
    case "tts":
        if #available(macOS 13.0, *) {
            await TTS.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            print("TTS requires macOS 13.0 or later")
            exit(1)
        }
    
    case "diarization-benchmark":
        if #available(macOS 13.0, *) {
            await StreamDiarizationBenchmark.run(arguments: Array(arguments.dropFirst(2)))
        } else {
            cliLogger.error("Diarization benchmark requires macOS 13.0 or later")
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
        cliLogger.error("Unknown command: \(command)")
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
