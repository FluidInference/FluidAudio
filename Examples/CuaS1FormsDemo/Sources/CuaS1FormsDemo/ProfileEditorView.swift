import CuaDemoCore
import SwiftUI

struct ProfileEditorView: View {
    @Bindable var session: DemoSession
    @FocusState private var focusedValue: UUID?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Text("01").font(.system(size: 10, design: .monospaced)).foregroundStyle(Palette.muted)
                Text("Your details").font(.system(size: 12, weight: .semibold))
                Spacer()
                Image(systemName: "person.text.rectangle").foregroundStyle(Palette.muted)
            }
            .padding(16)
            .overlay(alignment: .bottom) { Rectangle().fill(Palette.line).frame(height: 1) }
            VStack(alignment: .leading, spacing: 10) {
                Text("Start with you.").font(.system(size: 20, weight: .medium, design: .serif))
                Text("Add the details you want to use. They carry across all three forms.")
                    .font(.system(size: 11)).foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Button("Use example") { session.useExampleDetails() }.buttonStyle(.bordered)
                    Spacer()
                    Button("Clear") { session.clearDetails() }.buttonStyle(.borderless)
                        .foregroundStyle(Palette.muted)
                }.controlSize(.small).disabled(session.isRunning)
            }.padding(16)
            Divider().overlay(Palette.line).padding(.horizontal, 16)
            ScrollView {
                VStack(alignment: .leading, spacing: 13) {
                    ForEach(session.details) { detail in editor(for: detail) }
                    if let name = ProfileChoices.combinedName(in: session.details) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("FULL NAME").font(.system(size: 9, weight: .medium))
                            Text(name).font(.system(size: 12))
                            Text("Combined from your first and last name.").font(.system(size: 10))
                        }
                        .foregroundStyle(Palette.green).padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Palette.mint, in: RoundedRectangle(cornerRadius: 7))
                    }
                    Button {
                        session.addDetail()
                    } label: {
                        Label("Add another detail", systemImage: "plus.circle")
                    }
                    .buttonStyle(.borderless).foregroundStyle(Palette.green)
                    .disabled(session.isRunning || session.details.count >= 29)
                }.padding(16)
            }
            VStack(alignment: .leading, spacing: 5) {
                Text("\(session.detailCount) details ready · Empty values are omitted")
                    .font(.system(size: 10, weight: .medium))
                Text("Kept in memory on this Mac. Editing details resets the form preview.")
                    .font(.system(size: 10)).foregroundStyle(Palette.muted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14).frame(maxWidth: .infinity, alignment: .leading)
            .background(Palette.canvas.opacity(0.6))
        }
        .panel()
        .onAppear { focusedValue = session.details.first?.id }
        .onChange(of: session.details.first?.id) { _, id in focusedValue = id }
    }

    private func editor(for detail: ProfileDetail) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField(
                    "Label, e.g. Insurance provider",
                    text: Binding(
                        get: { session.details.first(where: { $0.id == detail.id })?.name ?? "" },
                        set: { session.updateDetail(detail.id, name: $0) })
                )
                .textFieldStyle(.plain).font(.system(size: 10, weight: .medium))
                .accessibilityLabel("Detail label")
                Button {
                    session.removeDetail(detail.id)
                } label: {
                    Image(systemName: "minus.circle").font(.system(size: 11)).foregroundStyle(Palette.muted)
                }
                .buttonStyle(.plain).help("Remove this detail")
                .accessibilityLabel("Remove \(detail.name)")
            }
            TextField(
                "Enter a value",
                text: Binding(
                    get: { session.details.first(where: { $0.id == detail.id })?.value ?? "" },
                    set: { session.updateDetail(detail.id, value: $0) })
            )
            .textFieldStyle(.plain).font(.system(size: 12))
            .padding(.horizontal, 10).frame(height: 33)
            .background(Palette.canvas.opacity(0.4), in: RoundedRectangle(cornerRadius: 7))
            .overlay { RoundedRectangle(cornerRadius: 7).stroke(Palette.line, lineWidth: 1) }
            .accessibilityLabel("\(detail.name) value")
            .focused($focusedValue, equals: detail.id)
        }.disabled(session.isRunning)
    }
}
