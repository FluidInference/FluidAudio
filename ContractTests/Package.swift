// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "FluidAudioTokenContractTests",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "FluidAudio", path: "..", traits: ["Diarizer"])],
    targets: [
        .testTarget(
            name: "TokenContractTests",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ], path: "Tests")
    ])
