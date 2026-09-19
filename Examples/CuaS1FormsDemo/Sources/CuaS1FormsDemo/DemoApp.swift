import AppKit
import CuaDemoCore
import SwiftUI

@main
enum DemoLauncher {
    @MainActor
    static func main() async {
        let arguments = CommandLine.arguments
        if arguments.contains("--verify") {
            do {
                try await DemoVerification.run(modelURL: argumentURL("--model"))
            } catch {
                print("Verification failed: \(error.localizedDescription)")
                exit(1)
            }
            return
        }
        DemoApp.main()
    }

    static func argumentURL(_ name: String) -> URL? {
        argument(name).map { URL(fileURLWithPath: $0) }
    }

    static func argument(_ name: String) -> String? {
        let arguments = CommandLine.arguments
        guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else { return nil }
        return arguments[index + 1]
    }
}

@MainActor
final class DemoAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct DemoApp: App {
    @NSApplicationDelegateAdaptor(DemoAppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("CUA-S1-FORMS · FluidInference") {
            DemoView()
                .frame(minWidth: 1120, minHeight: 740)
                .preferredColorScheme(.light)
        }
        .defaultSize(width: 1380, height: 890)
        .windowStyle(.hiddenTitleBar)
    }
}

/// Captures only this application's content view, without screen-recording access.
@MainActor
func saveWindowSnapshot(to url: URL) throws {
    guard let view = NSApplication.shared.windows.first(where: { $0.isVisible })?.contentView,
        let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
    else { throw DemoError("No demo window is available to capture.") }
    view.cacheDisplay(in: view.bounds, to: bitmap)
    guard let data = bitmap.representation(using: .png, properties: [:]) else {
        throw DemoError("Could not encode the demo snapshot.")
    }
    try data.write(to: url)
}
