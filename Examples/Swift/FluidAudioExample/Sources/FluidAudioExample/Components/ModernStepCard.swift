import SwiftUI

struct ModernStepCard<Content: View>: View {
    let number: Int
    let title: String
    let caption: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.lg) {
            HStack(alignment: .top, spacing: DesignSpacing.lg) {
                ZStack {
                    Circle()
                        .fill(DesignColors.accentLight)
                        .frame(width: 44, height: 44)

                    Text("\(number)")
                        .font(DesignTypography.labelLarge)
                        .fontWeight(.semibold)
                        .foregroundColor(DesignColors.accent)
                }

                VStack(alignment: .leading, spacing: DesignSpacing.sm) {
                    Text(title)
                        .headingText()

                    Text(caption)
                        .secondaryText()
                        .lineLimit(2)
                }

                Spacer()
            }

            Divider()
                .opacity(DesignOpacity.divider)

            content
        }
        .padding(DesignSpacing.lg)
        .background(DesignColors.card)
        .cornerRadius(DesignRadius.medium)
        .shadow(
            color: DesignElevation.small.color,
            radius: DesignElevation.small.radius,
            x: DesignElevation.small.x,
            y: DesignElevation.small.y
        )
        .transition(.opacity.combined(with: .scale(scale: 0.95)))
    }
}

#Preview {
    VStack(spacing: DesignSpacing.lg) {
        ModernStepCard(
            number: 1,
            title: "Choose Audio File",
            caption: "Select a local clip to stream"
        ) {
            HStack {
                VStack(alignment: .leading, spacing: DesignSpacing.sm) {
                    Text("sample.wav")
                        .bodyText()
                    Text("Supported: WAV, MP3, AIFF, M4A")
                        .secondaryText()
                }
                Spacer()
                Image(systemName: "folder")
                    .foregroundColor(DesignColors.accent)
            }
        }

        ModernStepCard(
            number: 2,
            title: "Stream & Transcribe",
            caption: "Choose a source and run streaming ASR"
        ) {
            VStack(spacing: DesignSpacing.md) {
                Picker("Source", selection: .constant(0)) {
                    Text("File").tag(0)
                    Text("Microphone").tag(1)
                }
                .pickerStyle(.segmented)

                ModernButton("Start Streaming", icon: "waveform") {}
            }
        }

        Spacer()
    }
    .padding(DesignSpacing.lg)
    .background(DesignColors.background)
}
