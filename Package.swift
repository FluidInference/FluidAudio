// swift-tools-version: 6.1
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
            targets: ["DiarizationCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [],
            path: "Sources/FluidAudio"
        ),
        .executableTarget(
            name: "DiarizationCLI",
            dependencies: ["FluidAudio"],
            path: "Sources/DiarizationCLI"
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: ["FluidAudio"],
            resources: [
                .copy("README_BENCHMARKS.md")
            ]
        ),
    ]
)
