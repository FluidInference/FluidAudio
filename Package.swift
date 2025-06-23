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
            dependencies: ["SherpaOnnxWrapper"],
            path: "Sources/SeamlessAudioSwift",
            linkerSettings: [
                .unsafeFlags(["-L\(packageDir)/Sources/SherpaOnnxWrapperC/lib"]),
                .linkedLibrary("onnxruntime"),
                .linkedLibrary("piper_phonemize"),
                .linkedLibrary("sherpa-onnx"),
                .linkedLibrary("sherpa-onnx-c-api"),
                .linkedLibrary("sherpa-onnx-core"),
                .linkedLibrary("sherpa-onnx-cxx-api"),
                .linkedLibrary("sherpa-onnx-fst"),
                .linkedLibrary("sherpa-onnx-fstfar"),
                .linkedLibrary("sherpa-onnx-kaldifst-core"),
                .linkedLibrary("sherpa-onnx-portaudio_static"),
                .linkedLibrary("ssentencepiece_core"),
                .linkedLibrary("ucd"),
                .linkedLibrary("c++")
            ]
        ),
        .target(
            name: "SherpaOnnxWrapper",
            dependencies: ["SherpaOnnxWrapperC"],
            path: "Sources/SherpaOnnxWrapper",
            exclude: ["lib/"]
        ),
        .systemLibrary(
            name: "SherpaOnnxWrapperC",
            path: "Sources/SherpaOnnxWrapperC"
        ),
        .testTarget(
            name: "SeamlessAudioSwiftTests",
            dependencies: ["SeamlessAudioSwift"]
        ),
    ]
)
