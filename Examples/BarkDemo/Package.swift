// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BarkDemo",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "BarkDemo",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)
