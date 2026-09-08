// swift-tools-version: 6.1
import PackageDescription
import Foundation

// Tools 6.1+ manifest: identical to Package.swift plus the `NemoTextProcessing`
// trait. Keep the two in sync; Package.swift serves toolchains < 6.1, which
// have no traits and always link the engine.

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
            name: "fluidaudiocli",
            targets: ["FluidAudioCLI"]
        ),
    ],
    traits: [
        // Opt out of the ~18 MB NeMo text-normalization engine (a prebuilt
        // Rust staticlib) for ASR/VAD/diarization-only apps, or when the app
        // links its own Rust runtime (#880, #888):
        //   .package(url: ..., traits: [])
        // TTS frontends and `TextNormalizer` then pass text through unchanged
        // and report `isNativeAvailable == false`.
        .trait(
            name: "NemoTextProcessing",
            description: "Link the bundled NeMo text-normalization engine (TTS frontends, ITN)."
        ),
        .default(enabledTraits: ["NemoTextProcessing"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
                .target(name: "NemoTextProcessing", condition: .when(traits: ["NemoTextProcessing"])),
            ],
            path: "Sources/FluidAudio",
            exclude: ["ASR/Parakeet/Unified/benchmark.md"],
            resources: [
                // Keep .process: .copy of a Resources-named directory breaks Apple code signing on iOS.
                .process("TTS/LuxTts/G2p/Resources")
            ]
        ),
        // Byte-exact NeMo text normalization (FST engine, all 7 languages).
        // Prebuilt xcframework from FluidInference/text-processing-rs v0.3.0.
        .binaryTarget(
            name: "NemoTextProcessing",
            url:
                "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.0/NemoTextProcessing.xcframework.zip",
            checksum: "76d0ee9a32b1ee2193231299180ca9bc4fc7e98794e771b3d55d66498352d85f"
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
                .process("TTS/LuxTts/Resources")
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
