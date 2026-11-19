// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PhononDemo",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(name: "FluidAudio", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "PhononDemo",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        )
    ]
)
