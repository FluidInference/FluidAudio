#if os(macOS)
    import Foundation

    @main
    struct DiarizationCLI {

        static func main() async {
            let arguments = CommandLine.arguments

            guard arguments.count > 1 else {
                Commands.printUsage()
                exit(1)
            }

            let command = arguments[1]

            switch command {
            case "benchmark":
                await Commands.runBenchmark(arguments: Array(arguments.dropFirst(2)))
            case "vad-benchmark":
                await VADBenchmark.runVadBenchmark(arguments: Array(arguments.dropFirst(2)))
            case "process":
                await Commands.processFile(arguments: Array(arguments.dropFirst(2)))
            case "download":
                await Commands.downloadDataset(arguments: Array(arguments.dropFirst(2)))
            case "help", "--help", "-h":
                Commands.printUsage()
            default:
                print("❌ Unknown command: \(command)")
                Commands.printUsage()
                exit(1)
            }
        }
    }
#endif
