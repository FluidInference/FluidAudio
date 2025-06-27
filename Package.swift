// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "FluidAudioSwift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "FluidAudioSwift",
            targets: ["FluidAudioSwift"]
        ),
    ],
    dependencies: [
        // Add any external dependencies here if needed
    ],
    targets: [
        .target(
            name: "FluidAudioSwift",
            dependencies: [],
            path: "Sources/FluidAudioSwift"
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
