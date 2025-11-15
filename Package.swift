// swift-tools-version: 5.10
import PackageDescription
import Foundation

let enableTTS = ProcessInfo.processInfo.environment["FLUIDAUDIO_ENABLE_TTS"] == "1"

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "FluidAudio",
            targets: ["FluidAudio"]
        ),
        .executable(
            name: "fluidaudio",
            targets: ["FluidAudioCLI"]
        ),
    ],
    dependencies: [],
    targets: {
        var targets: [Target] = [
            .target(
                name: "FluidAudio",
                dependencies: [
                    "FastClusterWrapper",
                ],
                path: "Sources/FluidAudio",
                exclude: ["Frameworks"]
            ),
            .target(
                name: "FastClusterWrapper",
                path: "Sources/FastClusterWrapper",
                publicHeadersPath: "include"
            ),
        ]

        // CLI target: depend on TTS only when enabled so builds can exclude GPL bits
        var cliDependencies: [Target.Dependency] = ["FluidAudio"]

        if enableTTS {
            targets.append(
                .binaryTarget(
                    name: "ESpeakNG",
                    path: "Sources/FluidAudio/Frameworks/ESpeakNG.xcframework"
                )
            )
            targets.append(
                .target(
                    name: "FluidAudioTTS",
                    dependencies: [
                        "FluidAudio",
                        "ESpeakNG",
                    ],
                    path: "Sources/FluidAudioTTS"
                )
            )
            cliDependencies.append("FluidAudioTTS")
        }

        targets.append(
            .executableTarget(
                name: "FluidAudioCLI",
                dependencies: cliDependencies,
                path: "Sources/FluidAudioCLI",
                exclude: ["README.md"],
                resources: [
                    .process("Utils/english.json")
                ],
                swiftSettings: enableTTS ? [.define("ENABLE_TTS")] : []
            )
        )

        var testDeps: [Target.Dependency] = ["FluidAudio"]
        if enableTTS {
            testDeps.append("FluidAudioTTS")
        }
        targets.append(
            .testTarget(
                name: "FluidAudioTests",
                dependencies: testDeps
            )
        )

        return targets
    }(),
    cxxLanguageStandard: .cxx17
)
