// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PocketTtsSmallDemo",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "PocketTtsSmallDemo",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            resources: [
                // Three upstream PyTorch (fp32, CPU) takes for A/B against Core ML.
                .copy("Reference")
            ]
        )
    ]
)
