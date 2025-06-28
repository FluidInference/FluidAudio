// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "FluidAudioSwift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(
            name: "FluidAudioSwift",
            targets: ["FluidAudioSwift"]
        ),
        .executable(
            name: "fluidaudio",
            targets: ["DiarizationCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudioSwift",
            dependencies: [],
            path: "Sources/FluidAudioSwift"
        ),
        .executableTarget(
            name: "DiarizationCLI",
            dependencies: ["FluidAudioSwift"],
            path: "Sources/DiarizationCLI"
        ),
        .testTarget(
            name: "FluidAudioSwiftTests",
            dependencies: ["FluidAudioSwift"],
            resources: [
                .copy("README_BENCHMARKS.md")
            ]
        ),
    ]
)
