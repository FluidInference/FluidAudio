import AppKit
import SwiftUI

@main
struct PocketTtsSmallDemoApp: App {
    @StateObject private var model = DemoModel()

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)
        // Bare SwiftPM executables start as background processes; make this one a
        // regular windowed app with a menu bar and Dock presence.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("Pocket TTS small models on the Neural Engine") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 620, idealWidth: 700, minHeight: 360)
                .task { model.applyLaunchArguments(CommandLine.arguments) }
        }
        .defaultSize(width: 700, height: 400)
        .windowStyle(.hiddenTitleBar)
    }
}
