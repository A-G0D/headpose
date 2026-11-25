// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "headpose",
    platforms: [.macOS(.v15)],
    targets: [
        .target(name: "HeadPoseCore"),
        .executableTarget(name: "headpose", dependencies: ["HeadPoseCore"]),
        .executableTarget(name: "headpose-bench", dependencies: ["HeadPoseCore"]),
        .testTarget(name: "HeadPoseCoreTests", dependencies: ["HeadPoseCore"]),
    ],
    swiftLanguageModes: [.v5]
)
