// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "FluidAudio",
    defaultLocalization: "en",
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
        .target(
            name: "FluidAudio",
            dependencies: [],
            path: "Sources/FluidAudio",
            exclude: [],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: ["FluidAudio"],
            path: "Sources/FluidAudioCLI",
            exclude: ["README.md"],
            resources: [
                .process("Utils/english.json")
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: ["FluidAudio"],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6, .v5]
)
