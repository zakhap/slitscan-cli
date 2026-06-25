// SlitscanStudioApp.swift — entry point. Activates as a regular foreground app
// (needed when launched via `swift run` from an SPM executable).

import SwiftUI
import AppKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

@main
struct SlitscanStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var app: AppState

    init() {
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.run()
            exit(0)
        }
        guard let state = AppState() else {
            fatalError("Metal is required and unavailable on this device.")
        }
        _app = StateObject(wrappedValue: state)
    }

    var body: some Scene {
        WindowGroup("Slitscan Studio") {
            ContentView()
                .environmentObject(app)
                .preferredColorScheme(.dark)
                .tint(Theme.safelight)
        }
        .windowStyle(.titleBar)
    }
}
