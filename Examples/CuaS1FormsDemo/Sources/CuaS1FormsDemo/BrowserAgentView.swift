import AppKit
import CuaBrowserCore
import CuaDemoCore
import SwiftUI
import WebKit

struct LiveBrowserView: NSViewRepresentable {
    let driver: BrowserDriver
    func makeNSView(context: Context) -> WKWebView { driver.webView }
    func updateNSView(_ nsView: WKWebView, context: Context) {}
}

struct BrowserAgentView: View {
    @State private var session = BrowserSession()
    @State private var isShowcase = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Label("FluidInference / CUA-S1-FORMS", systemImage: "square.stack.3d.up.fill")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Label("Two models. Two live browsers. All local.", systemImage: "desktopcomputer")
                    .font(.system(size: 11)).foregroundStyle(Palette.green)
            }.padding(.top, 24)
            HStack(alignment: .bottom) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Watch CUA use the form.").font(.system(size: 31, weight: .semibold, design: .rounded))
                    Text("Read a live control. Choose an action. Fill it. Check the result.")
                        .font(.system(size: 13)).foregroundStyle(Palette.muted)
                }
                Spacer()
                Picker(
                    "Form",
                    selection: Binding(
                        get: { session.source.scenarioIndex },
                        set: { index in
                            session.source.selectScenario(index)
                            perform { try await session.reset() }
                        })
                ) {
                    ForEach(session.source.scenarios.indices, id: \.self) { index in
                        Text(session.source.scenarios[index].shortTitle).tag(index)
                    }
                }.frame(width: 225).disabled(session.isBusy || isShowcase)
            }
            HStack(alignment: .top, spacing: 16) {
                ProfileEditorView(session: session.source).frame(width: 245)
                    .disabled(session.isBusy || isShowcase)
                ForEach(session.lanes) { lane in lanePanel(lane) }
            }.frame(maxHeight: .infinity)
            HStack(spacing: 12) {
                Button(session.isBusy ? "Stop" : "Run both agents") {
                    if session.isBusy {
                        session.stop()
                        return
                    }
                    perform { try await session.run() }
                }.buttonStyle(.borderedProminent).tint(Palette.ink)
                    .disabled(!session.isReady || session.isComplete || isShowcase || session.source.inputIssue != nil)
                Button("Step") { perform { try await session.run(singleStep: true) } }
                    .disabled(!session.isReady || session.isBusy || session.isComplete || isShowcase)
                Button("Reset forms") { perform { try await session.reset() } }
                    .disabled(!session.isReady || session.isBusy || isShowcase)
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Text(session.error ?? session.source.inputIssue ?? session.status)
                        .font(.system(size: 11)).foregroundStyle(session.error == nil ? Palette.ink : .red)
                    Text("Paced for visibility · Live call times exclude the pause · Submit buttons are yours to click")
                        .font(.system(size: 10)).foregroundStyle(Palette.muted)
                }
            }.controlSize(.large)
        }
        .padding(.horizontal, 24).padding(.bottom, 22)
        .foregroundStyle(Palette.ink).background(Palette.canvas)
        .onChange(of: session.source.details.map { "\($0.id):\($0.name):\($0.value)" }) { _, _ in
            guard session.isReady, !session.isBusy, !isShowcase else { return }
            perform { try await session.reset() }
        }
        .task {
            do {
                try await session.load(
                    baseline: DemoLauncher.argumentURL("--model"),
                    candidate: DemoLauncher.argumentURL("--ane-model"))
                guard let directory = DemoLauncher.argumentURL("--showcase") else { return }
                isShowcase = true
                try await BrowserShowcase.run(session: session, directory: directory)
                isShowcase = false
                if CommandLine.arguments.contains("--exit-after-showcase") { NSApplication.shared.terminate(nil) }
            } catch {
                isShowcase = false
                session.showError(error)
                print("Browser demo failed: \(error)")
                if let directory = DemoLauncher.argumentURL("--showcase") {
                    try? Data(error.localizedDescription.utf8).write(
                        to: directory.appendingPathComponent("failure.txt"))
                }
            }
        }
    }

    private func lanePanel(_ lane: BrowserLane) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text(lane.title).font(.system(size: 14, weight: .semibold))
                    Text(
                        lane.id == "baseline" ? "Baseline · same trained weights" : "ANE gather · same trained weights"
                    )
                    .font(.system(size: 10)).foregroundStyle(Palette.muted)
                }
                Spacer()
                Text("\(lane.decisions.count)/\(session.source.controls.count)")
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Palette.green)
            }.padding(15)
            Divider()
            LiveBrowserView(driver: lane.driver).frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(!session.isBusy && !isShowcase)
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Label(lane.status, systemImage: "checkmark.shield")
                        .lineLimit(1).font(.system(size: 10, weight: .medium)).foregroundStyle(Palette.green)
                    Spacer()
                }
                if let decision = lane.decisions.last {
                    Text(decision.selectedOption).font(.system(size: 12, weight: .medium))
                        .lineLimit(2).frame(height: 32, alignment: .top)
                    HStack {
                        Text(String(format: "%.2f ms Swift scoring", decision.scoreMilliseconds))
                        Spacer()
                        Text(String(format: "%.1f%% option score", decision.probability * 100))
                    }.font(.system(size: 10, design: .monospaced)).foregroundStyle(Palette.muted)
                } else {
                    Text("Waiting for your details.").font(.system(size: 12)).frame(height: 32)
                    Text("Actual model choices appear here.").font(.system(size: 10)).foregroundStyle(Palette.muted)
                }
            }.padding(14).background(Palette.mint.opacity(0.45))
        }.panel()
    }

    private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
        Task { do { try await operation() } catch { session.showError(error) } }
    }
}
