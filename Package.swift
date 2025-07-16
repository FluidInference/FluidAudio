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
            targets: ["DiarizationCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [],
            path: "Sources/FluidAudio",
            exclude: []
        ),
        .executableTarget(
            name: "DiarizationCLI",
            dependencies: ["FluidAudio"],
            path: "Sources/DiarizationCLI",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: ["FluidAudio"]
        ),
    ]
)
