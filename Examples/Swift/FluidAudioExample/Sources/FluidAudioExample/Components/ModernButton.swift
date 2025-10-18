import SwiftUI

struct ModernButton: View {
    enum Style {
        case primary
        case secondary
        case tertiary
        case destructive
    }

    let title: String
    let icon: String?
    let style: Style
    let isLoading: Bool
    let isDisabled: Bool
    let action: () -> Void

    init(
        _ title: String,
        icon: String? = nil,
        style: Style = .primary,
        isLoading: Bool = false,
        isDisabled: Bool = false,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.icon = icon
        self.style = style
        self.isLoading = isLoading
        self.isDisabled = isDisabled
        self.action = action
    }

    var body: some View {
        Button(action: {
            withAnimation(DesignAnimation.quick) {
                action()
            }
        }) {
            HStack(spacing: DesignSpacing.sm) {
                if isLoading {
                    ProgressView()
                        .scaleEffect(0.8)
                        .frame(width: 16, height: 16)
                } else if let icon = icon {
                    Image(systemName: icon)
                        .font(.system(size: 14, weight: .semibold))
                }

                Text(title)
                    .font(DesignTypography.labelLarge)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, DesignSpacing.md)
            .padding(.horizontal, DesignSpacing.lg)
            .background(backgroundColor)
            .foregroundColor(foregroundColor)
            .cornerRadius(DesignRadius.medium)
            .overlay(borderOverlay)
            .opacity(isDisabled || isLoading ? DesignOpacity.disabled : 1.0)
        }
        .disabled(isDisabled || isLoading)
    }

    private var backgroundColor: Color {
        switch style {
        case .primary:
            return DesignColors.accent
        case .secondary:
            return DesignColors.accentLight
        case .tertiary:
            return Color.clear
        case .destructive:
            return DesignColors.error
        }
    }

    private var foregroundColor: Color {
        switch style {
        case .primary, .destructive:
            return .white
        case .secondary:
            return DesignColors.accent
        case .tertiary:
            return DesignColors.accent
        }
    }

    private var borderOverlay: some View {
        Group {
            if style == .tertiary {
                RoundedRectangle(cornerRadius: DesignRadius.medium)
                    .stroke(DesignColors.border, lineWidth: 1)
            }
        }
    }
}

#Preview {
    VStack(spacing: DesignSpacing.lg) {
        Text("Button Styles")
            .headingText()

        VStack(spacing: DesignSpacing.md) {
            ModernButton("Primary Button", icon: "play") {}
            ModernButton("Secondary Button", icon: "pause", style: .secondary) {}
            ModernButton("Tertiary Button", icon: "gear", style: .tertiary) {}
            ModernButton("Destructive Button", icon: "trash", style: .destructive) {}
        }

        Divider()
            .opacity(DesignOpacity.divider)

        VStack(spacing: DesignSpacing.md) {
            ModernButton("Disabled Button", icon: "lock", isDisabled: true) {}
            ModernButton("Loading Button", icon: "hourglass", isLoading: true) {}
        }

        Spacer()
    }
    .padding(DesignSpacing.lg)
    .background(DesignColors.background)
}
