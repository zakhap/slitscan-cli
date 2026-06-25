// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SlitscanStudio",
    platforms: [.macOS(.v14)],
    targets: [
        // Pure math: profiles + parameter model. No Metal, no AppKit — testable.
        .target(
            name: "SlitscanCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // The app: Metal gather, ingest, SwiftUI UI, export.
        .executableTarget(
            name: "SlitscanStudio",
            dependencies: ["SlitscanCore"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Parity gate: Swift profile math must reproduce the vendored NumPy reference.
        .testTarget(
            name: "SlitscanCoreTests",
            dependencies: ["SlitscanCore"],
            resources: [.copy("Resources/test_vectors.json")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
