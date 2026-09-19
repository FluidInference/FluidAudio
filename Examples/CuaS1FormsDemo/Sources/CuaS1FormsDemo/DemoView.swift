import AppKit
import CuaDemoCore
import SwiftUI

private enum Palette {
    static let ink = Color(red: 0.10, green: 0.17, blue: 0.19)
    static let muted = Color(red: 0.42, green: 0.48, blue: 0.48)
    static let green = Color(red: 0.06, green: 0.45, blue: 0.34)
    static let mint = Color(red: 0.91, green: 0.96, blue: 0.92)
    static let canvas = Color(red: 0.96, green: 0.97, blue: 0.95)
    static let line = Color(red: 0.87, green: 0.90, blue: 0.87)
}

struct DemoView: View {
    @State private var session = DemoSession()

    var body: some View {
        VStack(spacing: 0) {
            header
            VStack(alignment: .leading, spacing: 20) {
                introduction
                HStack(alignment: .top, spacing: 18) {
                    sourcePanel.frame(width: 270)
                    formPanel.frame(maxWidth: .infinity)
                    inspectorPanel.frame(width: 310)
                }
                .frame(maxHeight: .infinity)
                footer
            }
            .padding(24)
        }
        .foregroundStyle(Palette.ink)
        .background(Palette.canvas)
        .font(.system(size: 13))
        .task {
            await session.load(modelURL: DemoLauncher.argumentURL("--model"))
            if CommandLine.arguments.contains("--snapshot-filled"), session.isReady {
                do { while !session.isComplete { try await session.scoreNext() } } catch {
                    print("Snapshot run failed: \(error)")
                }
                session.selectedControlID = session.controls.first?.id
            }
            guard let output = DemoLauncher.argumentURL("--snapshot") else { return }
            do {
                try await Task.sleep(for: .milliseconds(650))
                try saveWindowSnapshot(to: output)
                print("Saved demo snapshot: \(output.path)")
                if CommandLine.arguments.contains("--exit-after-snapshot") {
                    NSApplication.shared.terminate(nil)
                }
            } catch { print("Snapshot failed: \(error)") }
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "square.stack.3d.up.fill")
                .font(.system(size: 21)).foregroundStyle(Palette.green)
            Text("FluidInference").font(.system(size: 16, weight: .semibold))
            Text("/").foregroundStyle(Palette.line)
            Text("CUA-S1-FORMS").font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(Palette.muted)
            Spacer()
            HStack(spacing: 7) {
                Circle().fill(session.isReady ? Palette.green : .orange).frame(width: 6, height: 6)
                Text(session.isReady ? "Running locally on your Mac" : "Preparing Core ML model")
            }
            .font(.system(size: 11, weight: .medium))
            .padding(.horizontal, 12).padding(.vertical, 7)
            .background(.white, in: Capsule())
        }
        .padding(.leading, 88).padding(.trailing, 24).frame(height: 58)
        .overlay(alignment: .bottom) { Rectangle().fill(Palette.line).frame(height: 1) }
    }

    private var introduction: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 7) {
                Text("A form, filled on your Mac.").font(.system(size: 32, weight: .semibold, design: .rounded))
                Text("One tiny model chooses the next action. Swift carries it out.")
                    .foregroundStyle(Palette.muted)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 7) {
                Text("706K parameters  ·  1.51 MB").font(.system(size: 13, weight: .semibold))
                Text("SwiftUI + FluidAudio + Core ML").font(.system(size: 11)).foregroundStyle(Palette.muted)
            }
        }
    }

    private var sourcePanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            panelTitle("01", "Source document", symbol: "doc.text")
            VStack(alignment: .leading, spacing: 10) {
                Text(session.scenario?.documentTitle ?? "Sample document")
                    .font(.system(size: 20, weight: .medium, design: .serif))
                Text("Pre-extracted entities from Cua’s sample document.")
                    .font(.system(size: 11)).foregroundStyle(Palette.muted).fixedSize(horizontal: false, vertical: true)
            }.padding(18)
            Divider().overlay(Palette.line).padding(.horizontal, 18)
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ForEach(session.scenario?.entities ?? []) { entity in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(entity.name).font(.system(size: 10, weight: .medium)).foregroundStyle(Palette.muted)
                            Text(entity.value).font(.system(size: 12)).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }.padding(18)
            }
            Text("All entities are candidates, including irrelevant ones.")
                .font(.system(size: 10)).foregroundStyle(Palette.muted)
                .padding(16).frame(maxWidth: .infinity, alignment: .leading)
                .background(Palette.canvas.opacity(0.6))
        }.panel()
    }

    private var formPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            panelTitle("02", "Live form", symbol: "rectangle.and.pencil.and.ellipsis")
            HStack {
                Picker(
                    "Sample form",
                    selection: Binding(
                        get: { session.scenarioIndex }, set: { session.selectScenario($0) })
                ) {
                    ForEach(session.scenarios.indices, id: \.self) { index in
                        Text(session.scenarios[index].shortTitle).tag(index)
                    }
                }
                .labelsHidden().pickerStyle(.menu).disabled(session.isRunning)
                Spacer()
                Text("\(session.decisions.count)/\(session.controls.count)")
                    .font(.system(size: 11, design: .monospaced)).foregroundStyle(Palette.muted)
            }.padding(.horizontal, 17).padding(.vertical, 12)
            Divider().overlay(Palette.line)
            if session.isSubmitted {
                Label("Demo submitted. Everything stayed on this Mac.", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 12, weight: .medium)).foregroundStyle(Palette.green)
                    .padding(14).frame(maxWidth: .infinity, alignment: .leading).background(Palette.mint)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 17) {
                        Text(session.scenario?.title ?? "Loading sample forms…")
                            .font(.system(size: 17, weight: .semibold)).fixedSize(horizontal: false, vertical: true)
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 13) {
                            ForEach(session.controls.filter { $0.role == "Edit" }) { control in
                                textControl(control).id(control.id)
                            }
                        }
                        Divider().overlay(Palette.line)
                        ForEach(session.controls.filter { $0.role == "CheckBox" }) { control in
                            checkboxControl(control).id(control.id)
                        }
                        ForEach(session.controls.filter { $0.role == "Button" }) { control in
                            HStack {
                                Text(control.title).font(.system(size: 11)).foregroundStyle(Palette.muted)
                                Spacer()
                                if let decision = decision(for: control.id) { effectBadge(decision.effect) }
                            }.id(control.id)
                        }
                    }.padding(18)
                }
                .onChange(of: session.selectedControlID) { _, id in
                    guard let id, session.isRunning else { return }
                    withAnimation(.easeInOut(duration: 0.16)) { proxy.scrollTo(id, anchor: .center) }
                }
            }
            HStack {
                Text(
                    session.isComplete
                        ? "Review the filled form before submitting." : "The model scores each control independently."
                )
                .font(.system(size: 10)).foregroundStyle(Palette.muted)
                Spacer(minLength: 8)
                Button(session.isSubmitted ? "Submitted" : "Submit demo") { session.submitLocally() }
                    .buttonStyle(.bordered)
                    .disabled(!session.isComplete || session.isRunning || session.isSubmitted)
            }
            .padding(14).background(Palette.canvas.opacity(0.6))
        }.panel()
    }

    private func textControl(_ control: DemoControl) -> some View {
        let selected = session.selectedControlID == control.id
        return VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 4) {
                Text(control.title).font(.system(size: 10, weight: .medium)).lineLimit(1)
                Spacer(minLength: 0)
                if let decision = decision(for: control.id) {
                    Image(systemName: decision.effect == .filled ? "checkmark.circle.fill" : "minus.circle")
                        .font(.system(size: 10)).foregroundStyle(
                            decision.effect == .filled ? Palette.green : Palette.muted)
                }
            }
            TextField(
                "Empty",
                text: Binding(
                    get: { session.controls.first(where: { $0.id == control.id })?.value ?? "" },
                    set: { session.setValue($0, for: control.id) })
            )
            .textFieldStyle(.plain).font(.system(size: 12)).padding(.horizontal, 10).frame(height: 34)
            .background(selected ? Palette.mint.opacity(0.45) : .white, in: RoundedRectangle(cornerRadius: 7))
            .overlay { RoundedRectangle(cornerRadius: 7).stroke(selected ? Palette.green : Palette.line, lineWidth: 1) }
            .disabled(session.isRunning)
        }
        .contentShape(Rectangle())
        .onTapGesture { if !session.isRunning { session.selectedControlID = control.id } }
    }

    private func checkboxControl(_ control: DemoControl) -> some View {
        Toggle(
            isOn: Binding(
                get: { session.controls.first(where: { $0.id == control.id })?.isChecked ?? false },
                set: { session.setChecked($0, for: control.id) })
        ) {
            Text(control.title).font(.system(size: 11)).fixedSize(horizontal: false, vertical: true)
        }
        .toggleStyle(.checkbox).tint(Palette.green).disabled(session.isRunning)
    }

    private var inspectorPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            panelTitle("03", "Model decision", symbol: "point.3.connected.trianglepath.dotted")
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if let decision = session.inspectedDecision {
                        decisionDetails(decision)
                    } else {
                        VStack(alignment: .leading, spacing: 12) {
                            Image(systemName: "cursorarrow.rays").font(.system(size: 28)).foregroundStyle(Palette.green)
                            Text("See the choice happen.").font(.system(size: 18, weight: .medium))
                            Text("Fill the form or take one step. The live scores for every option will appear here.")
                                .font(.system(size: 12)).foregroundStyle(Palette.muted)
                                .fixedSize(horizontal: false, vertical: true)
                        }.padding(.vertical, 15)
                    }
                    if !session.decisions.isEmpty {
                        Divider().overlay(Palette.line)
                        Text("DECISIONS THIS PASS").font(.system(size: 9, weight: .semibold)).foregroundStyle(
                            Palette.muted)
                        ForEach(session.decisions.reversed()) { decision in
                            Button {
                                session.selectedControlID = decision.id
                            } label: {
                                HStack(spacing: 8) {
                                    Circle().fill(decision.effect == .skipped ? Palette.line : Palette.green)
                                        .frame(width: 5, height: 5)
                                    Text(decision.title).font(.system(size: 11)).lineLimit(1)
                                    Spacer(minLength: 4)
                                    Text(decision.effect.rawValue).font(.system(size: 9)).foregroundStyle(Palette.muted)
                                }
                            }.buttonStyle(.plain)
                        }
                    }
                }.padding(18)
            }
        }.panel()
    }

    private func decisionDetails(_ decision: DemoDecision) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                effectBadge(decision.effect)
                Spacer()
                Text(String(format: "%.2f ms", decision.milliseconds))
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
            }
            Text(decision.title).font(.system(size: 19, weight: .semibold))
            Text(decision.result.selectedOption).font(.system(size: 13, weight: .medium))
                .fixedSize(horizontal: false, vertical: true)
                .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                .background(Palette.mint, in: RoundedRectangle(cornerRadius: 8))
            Text("OPTION SCORES").font(.system(size: 9, weight: .semibold)).foregroundStyle(Palette.muted)
            let order = decision.result.probabilities.indices.sorted {
                decision.result.probabilities[$0] > decision.result.probabilities[$1]
            }
            ForEach(Array(order.prefix(4)), id: \.self) { index in scoreRow(index, decision: decision) }
            DisclosureGroup("All \(order.count) options") {
                VStack(spacing: 14) {
                    ForEach(Array(order.dropFirst(4)), id: \.self) { index in scoreRow(index, decision: decision) }
                }.padding(.top, 12)
            }.font(.system(size: 11)).tint(Palette.muted)
            DisclosureGroup("Input sent to the model") {
                Text(decision.context).font(.system(size: 10, design: .monospaced))
                    .textSelection(.enabled).padding(.top, 8).frame(maxWidth: .infinity, alignment: .leading)
            }.font(.system(size: 11)).tint(Palette.muted)
            Text("Model scores are not calibrated confidence.")
                .font(.system(size: 9)).foregroundStyle(Palette.muted)
        }
    }

    private func scoreRow(_ index: Int, decision: DemoDecision) -> some View {
        let probability = Double(decision.result.probabilities[index])
        return VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .top, spacing: 8) {
                Text(session.scenario?.options[index] ?? "Option \(index)")
                    .font(.system(size: 10)).lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                Text(String(format: "%.1f%%", probability * 100))
                    .font(.system(size: 10, design: .monospaced)).foregroundStyle(Palette.muted)
            }
            GeometryReader { geometry in
                Capsule().fill(Palette.canvas)
                Capsule().fill(index == decision.result.selectedIndex ? Palette.green : Palette.line)
                    .frame(width: max(0, geometry.size.width * probability))
            }.frame(height: 4)
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            if session.isRunning {
                Button {
                    session.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill").frame(width: 110)
                }
                .buttonStyle(.borderedProminent).tint(Palette.ink)
            } else {
                Button {
                    session.isComplete ? session.recheck() : session.start()
                } label: {
                    Label(session.isComplete ? "Recheck form" : "Fill form", systemImage: "play.fill").frame(width: 110)
                }
                .buttonStyle(.borderedProminent).tint(Palette.ink).disabled(!session.isReady)
                .keyboardShortcut(.return, modifiers: .command)
            }
            Button {
                session.start(singleStep: true)
            } label: {
                Label("Step", systemImage: "forward.end")
            }
            .buttonStyle(.bordered).disabled(!session.isReady || session.isRunning || session.isComplete)
            Button {
                session.reset()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
            }
            .buttonStyle(.borderless).foregroundStyle(Palette.muted)
            Spacer()
            if let error = session.errorMessage {
                Text(error).font(.system(size: 11)).foregroundStyle(.red).lineLimit(2)
                if !session.isReady {
                    Button("Retry") { Task { await session.load(modelURL: DemoLauncher.argumentURL("--model")) } }
                }
            } else if session.isLoading {
                ProgressView().controlSize(.small)
                Text("First launch downloads the model; later runs use the local cache.")
                    .font(.system(size: 11)).foregroundStyle(Palette.muted)
            } else {
                VStack(alignment: .trailing, spacing: 4) {
                    Text(
                        String(
                            format: "%d decisions · %.2f ms in Swift scoring calls", session.decisions.count,
                            session.totalMilliseconds)
                    )
                    .font(.system(size: 11, weight: .medium))
                    Text("Animation is paced for readability. Sample data stays on this Mac.")
                        .font(.system(size: 10)).foregroundStyle(Palette.muted)
                }
            }
        }.controlSize(.large).frame(minHeight: 42)
    }

    private func decision(for id: Int) -> DemoDecision? { session.decisions.last { $0.id == id } }

    private func effectBadge(_ effect: DemoEffect) -> some View {
        Text(effect.rawValue.uppercased()).font(.system(size: 9, weight: .semibold))
            .foregroundStyle(effect == .skipped ? Palette.muted : Palette.green)
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(effect == .skipped ? Palette.canvas : Palette.mint, in: Capsule())
    }

    private func panelTitle(_ number: String, _ title: String, symbol: String) -> some View {
        HStack(spacing: 8) {
            Text(number).font(.system(size: 10, design: .monospaced)).foregroundStyle(Palette.muted)
            Text(title).font(.system(size: 12, weight: .semibold))
            Spacer()
            Image(systemName: symbol).foregroundStyle(Palette.muted)
        }.padding(16).overlay(alignment: .bottom) { Rectangle().fill(Palette.line).frame(height: 1) }
    }
}

extension View {
    fileprivate func panel() -> some View {
        frame(maxHeight: .infinity, alignment: .top)
            .background(.white, in: RoundedRectangle(cornerRadius: 13))
            .clipShape(RoundedRectangle(cornerRadius: 13))
            .overlay { RoundedRectangle(cornerRadius: 13).stroke(Palette.line, lineWidth: 1) }
    }
}
