// swift-tools-version: 6.2
import PackageDescription
import Foundation

// Tools 6.2+ manifest: identical to Package.swift plus the `NemoTextProcessing`
// trait. Keep the two in sync; Package.swift serves toolchains < 6.2, which
// always link the engine. (SwiftPM 6.1 accepts the trait syntax but still
// links a trait-conditioned binary target — verified on Xcode 16.4 — so the
// opt-out is gated at 6.2.)

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
        // Opt out of the NeMo text-normalization engine (~8 MB per slice, a prebuilt
        // Rust staticlib) for ASR/VAD/diarization-only apps, or when the app
        // links its own Rust runtime (#880, #888):
        //   .package(url: ..., traits: [])
        // TTS frontends and `TextNormalizer` then pass text through unchanged
        // and report `isNativeAvailable == false`.
        .trait(
            name: "NemoTextProcessing",
            description: "Link the bundled NeMo text-normalization engine (TTS frontends, ITN)."
        ),
        .trait(
            name: "JapaneseTextProcessing",
            description: "Link OpenJTalk for the Kokoro ANE Japanese text frontend."
        ),
        .default(enabledTraits: ["NemoTextProcessing", "JapaneseTextProcessing"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "CZlib",
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
                .target(name: "NemoTextProcessing", condition: .when(traits: ["NemoTextProcessing"])),
                .target(name: "voicevox_core", condition: .when(traits: ["JapaneseTextProcessing"])),
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
        // OpenJTalk runtime for the Kokoro ANE Japanese text frontend: VOICEVOX
        // CORE 0.17.0's xcframework (macOS + iOS + simulator slices), ad-hoc
        // re-signed — the upstream zip's macOS slice ships with an invalid code
        // signature and the kernel kills any process that loads it.
        // HOSTING_PLACEHOLDER: replace `path:` with the published `url:` + `checksum:`.
        .binaryTarget(
            name: "voicevox_core",
            path: ".mobius/voicevox_core.xcframework"
        ),
        .systemLibrary(
            name: "CZlib",
            path: "Sources/CZlib"
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
                // Real recordings (cleared for public release by the speaker) for the
                // streaming final-window regression, issue #855.
                .copy("ASR/Parakeet/SlidingWindow/Fixtures"),
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
