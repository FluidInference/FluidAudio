// swift-tools-version: 6.1
import PackageDescription
import Foundation

// Use SwiftPM's built-in package directory resolution
let packageDir = Context.packageDirectory

let package = Package(
    name: "SeamlessAudioSwift",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "SeamlessAudioSwift",
            targets: ["SeamlessAudioSwift"]
        ),
    ],
    dependencies: [
        // Add any external dependencies here if needed
    ],
    targets: [
        .target(
            name: "SeamlessAudioSwift",
            dependencies: [],
            path: "Sources/SeamlessAudioSwift"
        ),
        .testTarget(
            name: "SeamlessAudioSwiftTests",
            dependencies: ["SeamlessAudioSwift"]
        ),
    ]
)
