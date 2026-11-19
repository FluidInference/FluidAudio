// swift-tools-version: 6.0
import PackageDescription
import Foundation

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v2),
        .tvOS(.v17),
    ],
    products: [
        .library(
            name: "FluidAudio",
            targets: ["FluidAudio"]
        ),
        .executable(
            name: "fluidaudiocli",
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
                // The prebuilt xcframework has no visionOS or tvOS slice.
                .target(name: "NemoTextProcessing", condition: .when(platforms: [.macOS, .iOS, .macCatalyst])),
                "LuxTtsG2pResources",
            ],
            path: "Sources/FluidAudio",
            exclude: ["ASR/Parakeet/Unified/benchmark.md"],
            swiftSettings: [
                // Package@swift-6.2.swift exposes these as opt-out traits (#990).
                .define("TTS"),
                .define("Diarizer"),
            ]
        ),
        .target(
            name: "LuxTtsG2pResources",
            path: "Sources/LuxTtsG2pResources",
            resources: [
                // Keep .process: .copy of a Resources-named directory breaks Apple code signing on iOS.
                .process("Resources")
            ]
        ),
        // Byte-exact NeMo text normalization (FST engine, all 7 languages).
        // Prebuilt xcframework from FluidInference/text-processing-rs v0.3.4
        // (macOS, iOS, iOS Simulator and Mac Catalyst slices).
        // Always linked on tools < 6.2; Package@swift-6.2.swift exposes it as
        // the opt-out `NemoTextProcessing` trait (#880, #888).
        .binaryTarget(
            name: "NemoTextProcessing",
            url:
                "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.4/NemoTextProcessing.xcframework.zip",
            checksum: "fdfebc5ce0938624435aa217d5425394f0b946cc0dd47bd01b66917e671be851"
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
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: ["FluidAudio"],
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
                "FluidAudioCLI",
            ],
            resources: [
                .process("TTS/LuxTts/Resources"),
                .process("TTS/PocketTTS/Fixtures"),
                // Real recordings (cleared for public release by the speaker) for the
                // streaming final-window regression, issue #855.
                .copy("ASR/Parakeet/SlidingWindow/Fixtures"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
