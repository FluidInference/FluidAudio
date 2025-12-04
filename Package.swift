// swift-tools-version: 5.10
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
        .binaryTarget(
            name: "SentencePiece",
            path: "Sources/FluidAudio/Frameworks/SentencePiece.xcframework"
        ),
        .target(
            name: "SentencePieceWrapper",
            dependencies: [
                "SentencePiece"
            ],
            path: "Sources/SentencePieceWrapper",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include")
            ]
        ),
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "SentencePieceWrapper",
            ],
            path: "Sources/FluidAudio",
            exclude: [
                "Frameworks",
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
