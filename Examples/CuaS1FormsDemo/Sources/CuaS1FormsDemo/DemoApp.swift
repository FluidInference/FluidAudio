import AppKit
import CuaDemoCore
import SwiftUI

@main
enum DemoLauncher {
    @MainActor
    static func main() {
        let arguments = CommandLine.arguments
        if arguments.contains("--benchmark") || arguments.contains("--verify") {
            Task {
                do {
                    try await runCommand()
                    exit(0)
                } catch {
                    print("Command failed: \(error.localizedDescription)")
                    exit(1)
                }
            }
            dispatchMain()
        }
        let app = NSApplication.shared
        let delegate = DemoAppDelegate()
        app.delegate = delegate
        app.run()
        withExtendedLifetime(delegate) {}
    }

    @MainActor
    private static func runCommand() async throws {
        if CommandLine.arguments.contains("--verify") {
            try await DemoVerification.run(modelURL: argumentURL("--model"))
            return
        }
        let baseline: URL
        let candidate: URL
        if let url = argumentURL("--model") { baseline = url } else { baseline = try await DemoAssets.modelURL() }
        if let url = argumentURL("--ane-model") {
            candidate = url
        } else {
            candidate = try await DemoAssets.modelURL(aneGather: true)
        }
        guard let output = argumentURL("--report") else { throw DemoError("Supply --report /path/to/report.json.") }
        _ = try await VariantBenchmark.run(
            baseline: baseline, candidate: candidate, output: output,
            hardware: argument("--hardware") ?? "Not reported")
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
    private var window: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        let root = Group {
            if CommandLine.arguments.contains("--browser") { BrowserAgentView() } else { DemoView() }
        }
        .frame(minWidth: 1280, minHeight: 800)
        .preferredColorScheme(.light)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1380, height: 890),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.title = "CUA-S1-FORMS · FluidInference"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: root)
        window.center()
        self.window = window
        window.makeKeyAndOrderFront(nil)
        let menu = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit CUA Forms", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        menu.addItem(appItem)
        let editItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        let redo = editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(.separator())
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = editMenu
        menu.addItem(editItem)
        NSApplication.shared.mainMenu = menu
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        window?.makeKeyAndOrderFront(nil)
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
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
