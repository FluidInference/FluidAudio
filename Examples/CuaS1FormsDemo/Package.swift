// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CuaS1FormsDemo",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "CuaS1FormsDemo", targets: ["CuaS1FormsDemo"])],
    dependencies: [.package(path: "../..")],
    targets: [
        .target(
            name: "CuaDemoCore",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")],
            resources: [.copy("Resources/demo.jsonl")]),
        .executableTarget(name: "CuaS1FormsDemo", dependencies: ["CuaDemoCore"]),
        .testTarget(name: "CuaDemoCoreTests", dependencies: ["CuaDemoCore"]),
    ])
