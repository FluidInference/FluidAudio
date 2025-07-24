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
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.17")
    ],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Sources/FluidAudio",
            exclude: []
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: ["FluidAudio"],
            path: "Sources/FluidAudioCLI"
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: ["FluidAudio"]
        ),
    ]
)
