import AppKit
import SwiftUI

/// AppKit reads launch arguments as `-key value` pairs and treats a leftover word
/// as a document to open (`--autorun --text hello` leaves `hello`), which makes
/// SwiftUI skip the default window. Swallow those open requests so the window
/// always appears.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func application(_ application: NSApplication, open urls: [URL]) {}
    func applicationShouldOpenUntitledFile(_ sender: NSApplication) -> Bool { true }
}

@main
struct PhononDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = DemoModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Phonon on the Neural Engine") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 560, idealWidth: 620, minHeight: 640)
                .task { model.applyLaunchArguments(CommandLine.arguments) }
        }
        .defaultSize(width: 620, height: 720)
        .windowStyle(.hiddenTitleBar)
    }
}
