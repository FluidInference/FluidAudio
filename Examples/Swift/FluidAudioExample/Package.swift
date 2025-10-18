// swift-tools-version: 5.10
import Foundation
import PackageDescription

let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let exampleResourcesDirectory = packageDirectory
    .appendingPathComponent("Sources/FluidAudioExample/Resources")
let infoPlistPath = exampleResourcesDirectory.appendingPathComponent("AppInfo.plist").path

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
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlistPath
                ])
            ]
        )
    ]
)
