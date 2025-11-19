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
        .library(
            name: "FluidAudioWithTTS",
            targets: ["FluidAudio", "FluidAudioTTS"]
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
                exclude: [
                    "Frameworks",
                    "ASR/ContextBiasing",
                    "ASR/CtcModels.swift",
                ]
            ),
            .target(
                name: "FastClusterWrapper",
                path: "Sources/FastClusterWrapper",
                publicHeadersPath: "include"
            ),
            // TTS targets are always available for FluidAudioWithTTS product
            .binaryTarget(
                name: "ESpeakNG",
                path: "Sources/FluidAudio/Frameworks/ESpeakNG.xcframework"
            ),
            .target(
                name: "FluidAudioTTS",
                dependencies: [
                    "FluidAudio",
                    "ESpeakNG",
                ],
                path: "Sources/FluidAudioTTS"
            ),
        ]

        // CLI target: depend on TTS only when enabled via env var (for local development)
        var cliDependencies: [Target.Dependency] = ["FluidAudio"]
        if enableTTS {
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
