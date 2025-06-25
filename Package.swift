// swift-tools-version: 6.1
import PackageDescription
import Foundation

// Use SwiftPM's built-in package directory resolution
let packageDir = Context.packageDirectory

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
            dependencies: ["FluidAudioSwift"]
        ),
    ]
)
