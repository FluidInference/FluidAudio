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
            name: "FluidAudioEspeak",
            targets: ["FluidAudioEspeak"]
        ),
        .executable(
            name: "fluidaudiocli",
            targets: ["FluidAudioCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6")
    ],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/FluidAudio",
            exclude: [
                "Frameworks"
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
        // TTS targets are always available for FluidAudioEspeak product
        .binaryTarget(
            name: "ESpeakNG",
            path: "Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "FluidAudioEspeak",
            dependencies: [
                "FluidAudio",
                "ESpeakNG",
            ],
            path: "Sources/FluidAudioEspeak"
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: [
                "FluidAudio",
                "FluidAudioEspeak",
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
                "FluidAudioEspeak",
            ]
        ),
    ],
    cxxLanguageStandard: .cxx17
)
