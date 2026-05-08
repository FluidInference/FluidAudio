// swift-tools-version: 6.0
import PackageDescription
import Foundation

// MARK: - Optional Metaphone3 integration
//
// Metaphone3 is a paid product. To enable phonetic-fallback matching in
// the vocabulary rescorer, drop `Metaphone3.xcframework` into
// `Frameworks/` at the package root before resolving. When the
// xcframework is present:
//   - the `Metaphone3Binary` binary target is added,
//   - `FluidAudioCLI` depends on it,
//   - the `METAPHONE3_AVAILABLE` Swift flag is defined for FluidAudioCLI.
// When absent, none of the above happens and FluidAudio builds and runs
// without any Metaphone3 references — appropriate for the open-source
// distribution.
let metaphone3FrameworkPath = "Frameworks/Metaphone3.xcframework"
let hasMetaphone3 = FileManager.default.fileExists(
    atPath: "\(FileManager.default.currentDirectoryPath)/\(metaphone3FrameworkPath)"
)

var cliDependencies: [Target.Dependency] = ["FluidAudio"]
var extraTargets: [Target] = []
var cliSwiftSettings: [SwiftSetting] = []

if hasMetaphone3 {
    extraTargets.append(
        .binaryTarget(
            name: "Metaphone3Binary",
            path: metaphone3FrameworkPath
        )
    )
    cliDependencies.append("Metaphone3Binary")
    cliSwiftSettings.append(.define("METAPHONE3_AVAILABLE"))
}

let package = Package(
    name: "FluidAudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "FluidAudio",
            targets: ["FluidAudio"]
        ),
        .executable(
            name: "fluidaudiocli",
            targets: ["FluidAudioCLI"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "FluidAudio",
            dependencies: [
                "FastClusterWrapper",
                "MachTaskSelfWrapper",
            ],
            path: "Sources/FluidAudio",
            exclude: [
                "Frameworks"
            ]
        ),
        .target(
            name: "FastClusterWrapper",
            path: "Sources/FastClusterWrapper",
            publicHeadersPath: "include"
        ),
        .target(
            name: "MachTaskSelfWrapper",
            path: "Sources/MachTaskSelfWrapper",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "FluidAudioCLI",
            dependencies: cliDependencies,
            path: "Sources/FluidAudioCLI",
            exclude: ["README.md"],
            resources: [
                .process("Utils/english.json")
            ],
            swiftSettings: cliSwiftSettings
        ),
        .testTarget(
            name: "FluidAudioTests",
            dependencies: [
                "FluidAudio",
                "FluidAudioCLI",
            ]
        ),
    ] + extraTargets,
    cxxLanguageStandard: .cxx17
)
