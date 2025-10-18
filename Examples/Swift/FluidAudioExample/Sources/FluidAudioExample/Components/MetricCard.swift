import SwiftUI

struct ModernMetricCard: View {
    let title: String
    let value: String
    let icon: String?

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            HStack(spacing: DesignSpacing.sm) {
                if let icon = icon {
                    Image(systemName: icon)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(DesignColors.accent)
                        .frame(width: 20, height: 20)
                        .background(DesignColors.accentLight)
                        .cornerRadius(DesignRadius.small)
                }

                Text(title.uppercased())
                    .tertiaryText()
                    .lineLimit(1)

                Spacer()
            }

            Text(value)
                .font(DesignTypography.headingMedium)
                .foregroundColor(DesignColors.text)
                .lineLimit(2)
                .minimumScaleFactor(0.8)
        }
        .padding(DesignSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: DesignRadius.medium)
                .fill(DesignColors.secondaryBackground)
        )
        .transition(.opacity.combined(with: .scale(scale: 0.95)))
    }
}

struct ModernMetricsGrid<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        let columns = [
            GridItem(.flexible(), spacing: DesignSpacing.lg),
            GridItem(.flexible(), spacing: DesignSpacing.lg),
        ]

        LazyVGrid(columns: columns, alignment: .leading, spacing: DesignSpacing.lg) {
            content
        }
    }
}

#Preview {
    VStack(spacing: DesignSpacing.xl) {
        Text("Metrics Display")
            .headingText()

        ModernMetricsGrid {
            ModernMetricCard(title: "Processing Time", value: "2.45s", icon: "timer")
            ModernMetricCard(title: "Real-Time Factor", value: "0.95x", icon: "bolt.fill")
            ModernMetricCard(title: "First Token", value: "145ms", icon: "hare")
            ModernMetricCard(title: "Word Count", value: "42 words", icon: "textformat")
        }
        .padding(DesignSpacing.lg)

        Spacer()
    }
    .background(DesignColors.background)
}
