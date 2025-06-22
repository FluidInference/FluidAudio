// swift-tools-version: 5.9
import PackageDescription

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
            dependencies: ["SherpaOnnxWrapper"],
            path: "Sources/SeamlessAudioSwift"
        ),
        .systemLibrary(
            name: "SherpaOnnxWrapper",
            path: "Sources/SherpaOnnxWrapper",
            pkgConfig: "sherpa-onnx"
        ),
        .testTarget(
            name: "SeamlessAudioSwiftTests",
            dependencies: ["SeamlessAudioSwift"],
            path: "Tests/SeamlessAudioSwiftTests"
        ),
    ]
)
