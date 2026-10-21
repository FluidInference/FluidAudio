import AppKit
import SwiftUI

@main
struct BarkDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = DemoModel.shared

    init() {
        // demo.sh tees stdout into the log pane; flush every line.
        setvbuf(stdout, nil, _IOLBF, 0)
        // A killed run leaves saved window state that can relaunch the app with no window; always open fresh.
        UserDefaults.standard.set(true, forKey: "ApplePersistenceIgnoreState")
        // `swift run` starts a plain process; make it a regular foreground app with a window.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Bark Typewriter") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 900, minHeight: 640)
        }
        .windowStyle(.hiddenTitleBar)
    }
}

/// Applies `--file` / `--model` / `--autorun` once the app has launched, whether or not a window has appeared yet.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Task { @MainActor in DemoModel.shared.applyLaunchArguments() }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
