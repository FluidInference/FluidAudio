// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
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
            name: "ESpeakNG",
            url: "https://github.com/FluidInference/FluidAudio/releases/download/v0.5.1/ESpeakNG.xcframework.zip",
            checksum: "054c5d1409b864c07b3612c6fd4b05eff6d696cbbef03901ac9ac575ede543f3"
        ),
        .target(
            name: "FluidAudio",
            dependencies: ["ESpeakNG"],
            path: "Sources/FluidAudio",
            exclude: []
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
    ]
)
