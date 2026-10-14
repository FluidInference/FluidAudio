// swift-tools-version: 6.2
import PackageDescription
import Foundation

// Tools 6.2+ manifest: identical to Package.swift plus the `NemoTextProcessing`,
// `TTS` and `Diarizer` traits. Keep the two in sync; Package.swift serves
// toolchains < 6.2, which always link the engine and compile every subsystem. (SwiftPM 6.1 accepts the trait syntax but still
// links a trait-conditioned binary target — verified on Xcode 16.4 — so the
// opt-out is gated at 6.2.)

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v2),
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
        // Opt out of whole subsystems for apps that never reach them (#990):
        //   .package(url: ..., traits: [])            // ASR/VAD only
        //   .package(url: ..., traits: ["Diarizer"])  // ASR/VAD + diarization
        .trait(
            name: "TTS",
            description: "Text-to-speech backends (Kokoro, PocketTTS, LuxTTS, ...) and their bundled G2P resources."
        ),
        .trait(
            name: "Diarizer",
            description: "Speaker diarization (pyannote, Sortformer, LS-EEND, Nemotron 3)."
        ),
        .default(enabledTraits: ["NemoTextProcessing", "TTS", "Diarizer"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
                // The prebuilt xcframework has no visionOS slice.
                .target(
                    name: "NemoTextProcessing",
                    condition: .when(platforms: [.macOS, .iOS, .macCatalyst], traits: ["NemoTextProcessing"])
                ),
                .target(name: "LuxTtsG2pResources", condition: .when(traits: ["TTS"])),
            ],
            path: "Sources/FluidAudio",
            exclude: ["ASR/Parakeet/Unified/benchmark.md"]
        ),
        // Separate target so the ~1 MB G2P bundle drops out with the `TTS` trait.
        .target(
            name: "LuxTtsG2pResources",
            path: "Sources/LuxTtsG2pResources",
            resources: [
                // Keep .process: .copy of a Resources-named directory breaks Apple code signing on iOS.
                .process("Resources")
            ]
        ),
        // Byte-exact NeMo text normalization (FST engine, all 7 languages).
        // Prebuilt xcframework from FluidInference/text-processing-rs v0.3.1
        // (macOS, iOS, iOS Simulator and Mac Catalyst slices).
        .binaryTarget(
            name: "NemoTextProcessing",
            url:
                "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.1/NemoTextProcessing.xcframework.zip",
            checksum: "5fa8c10d4ec26c1bb2413125f351a7222a4c68a23b74476680fbada7e26fc6aa"
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
