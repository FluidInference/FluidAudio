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
        .executable(
            name: "fluidaudio",
            targets: ["FluidAudioCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .binaryTarget(
            name: "SentencePiece",
            path: "smdesaiSentencePiece/SentencePiece.xcframework"
        ),
        .binaryTarget(
            name: "ESpeakNG",
            path: "Sources/FluidAudio/Frameworks/ESpeakNG.xcframework"
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
                "ESpeakNG",
                "FastClusterWrapper",
                "SentencePieceWrapper",
            ],
            path: "Sources/FluidAudio",
            exclude: ["Frameworks"]
        ),
        .target(
            name: "FastClusterWrapper",
            path: "Sources/FastClusterWrapper",
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
            dependencies: ["FluidAudio"]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
