// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "FluidAudioExample",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .executable(
            name: "FluidAudioExample",
            targets: ["FluidAudioExampleApp"]
        )
    ],
    dependencies: [
        .package(name: "FluidAudio", path: "../../..")
    ],
    targets: [
        .executableTarget(
            name: "FluidAudioExampleApp",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ],
            path: "Sources",
            resources: [
                .process("FluidAudioExample/Resources")
            ]
        )
    ]
)
