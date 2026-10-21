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
            ]
        )
    ]
)
