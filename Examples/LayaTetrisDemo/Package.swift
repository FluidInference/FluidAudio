// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaTetrisDemo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "LayaTetrisDemo",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)
