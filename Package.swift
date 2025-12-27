// swift-tools-version: 6.0
import PackageDescription

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
            name: "FluidAudioTTS",
            targets: ["FluidAudioTTS"]
        ),
        .executable(
            name: "fluidaudio",
            targets: ["FluidAudioCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
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
        .target(
            name: "MachTaskSelfWrapper",
            path: "Sources/MachTaskSelfWrapper",
            publicHeadersPath: "include"
        ),
        // TTS targets are always available for FluidAudioWithTTS product
        .binaryTarget(
            name: "ESpeakNG",
            path: "Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "FluidAudioTTS",
            dependencies: [
                "FluidAudio",
                "ESpeakNG",
            ],
            path: "Sources/FluidAudioTTS"
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: [
                "FluidAudio",
                "FluidAudioTTS",
            ],
            path: "Sources/FluidAudioCLI",
            exclude: ["README.md"],
            resources: [
                .process("Utils/english.json")
            ]
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: [
                "FluidAudio",
                "FluidAudioTTS",
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
