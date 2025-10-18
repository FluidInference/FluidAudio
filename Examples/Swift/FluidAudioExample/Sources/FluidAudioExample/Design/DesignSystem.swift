import SwiftUI

// MARK: - Color Palette (Linear-inspired Muted)
struct DesignColors {
    static let background = Color(red: 0.99, green: 0.99, blue: 0.99)
    static let secondaryBackground = Color(red: 0.95, green: 0.95, blue: 0.95)
    static let card = Color.white
    static let border = Color(red: 0.9, green: 0.9, blue: 0.9)

    static let text = Color(red: 0.1, green: 0.1, blue: 0.1)
    static let textSecondary = Color(red: 0.6, green: 0.6, blue: 0.6)
    static let textTertiary = Color(red: 0.75, green: 0.75, blue: 0.75)

    static let accent = Color(red: 0.0, green: 0.5, blue: 1.0) // Modern blue
    static let accentLight = Color(red: 0.9, green: 0.95, blue: 1.0)
    static let success = Color(red: 0.2, green: 0.8, blue: 0.4)
    static let warning = Color(red: 1.0, green: 0.8, blue: 0.0)
    static let error = Color(red: 1.0, green: 0.3, blue: 0.3)

    static let recording = Color(red: 0.95, green: 0.2, blue: 0.2) // Red pulse for recording
}

// MARK: - Spacing
struct DesignSpacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let xl: CGFloat = 24
    static let xxl: CGFloat = 32
    static let xxxl: CGFloat = 48
}

// MARK: - Shadow/Elevation
struct DesignElevation {
    static let subtle = Shadow(
        color: .black.opacity(0.04),
        radius: 2,
        x: 0,
        y: 1
    )

    static let small = Shadow(
        color: .black.opacity(0.08),
        radius: 4,
        x: 0,
        y: 2
    )

    static let medium = Shadow(
        color: .black.opacity(0.12),
        radius: 8,
        x: 0,
        y: 4
    )

    static let large = Shadow(
        color: .black.opacity(0.16),
        radius: 12,
        x: 0,
        y: 6
    )
}

// MARK: - Border Radius
struct DesignRadius {
    static let small: CGFloat = 8
    static let medium: CGFloat = 12
    static let large: CGFloat = 16
    static let full: CGFloat = 999
}

// MARK: - Typography
struct DesignTypography {
    // Display
    static let displayLarge = Font.system(size: 32, weight: .semibold, design: .default)
    static let displayMedium = Font.system(size: 28, weight: .semibold, design: .default)

    // Heading
    static let headingLarge = Font.system(size: 24, weight: .semibold, design: .default)
    static let headingMedium = Font.system(size: 20, weight: .semibold, design: .default)
    static let headingSmall = Font.system(size: 16, weight: .semibold, design: .default)

    // Body
    static let bodyLarge = Font.system(size: 16, weight: .regular, design: .default)
    static let bodyMedium = Font.system(size: 14, weight: .regular, design: .default)
    static let bodySmall = Font.system(size: 13, weight: .regular, design: .default)

    // Label
    static let labelLarge = Font.system(size: 14, weight: .medium, design: .default)
    static let labelMedium = Font.system(size: 12, weight: .medium, design: .default)
    static let labelSmall = Font.system(size: 11, weight: .medium, design: .default)

    // Monospace
    static let monospaceBody = Font.system(size: 13, weight: .regular, design: .monospaced)
}

// MARK: - Animation
struct DesignAnimation {
    static let quick = Animation.easeInOut(duration: 0.15)
    static let standard = Animation.easeInOut(duration: 0.25)
    static let slow = Animation.easeInOut(duration: 0.35)
    static let slowest = Animation.easeInOut(duration: 0.5)

    static let spring = Animation.spring(response: 0.3, dampingFraction: 0.7)
    static let gentle = Animation.easeInOut(duration: 0.2)
}

// MARK: - Opacity
struct DesignOpacity {
    static let disabled: Double = 0.5
    static let hover: Double = 0.08
    static let pressed: Double = 0.12
    static let divider: Double = 0.08
    static let skeleton: Double = 0.12
}

struct Shadow {
    let color: Color
    let radius: CGFloat
    let x: CGFloat
    let y: CGFloat
}

// MARK: - View Modifiers
extension View {
    func cardStyle() -> some View {
        self
            .background(DesignColors.card)
            .cornerRadius(DesignRadius.medium)
            .shadow(color: DesignElevation.small.color, radius: DesignElevation.small.radius, x: DesignElevation.small.x, y: DesignElevation.small.y)
    }

    func headingText() -> some View {
        self
            .font(DesignTypography.headingMedium)
            .foregroundColor(DesignColors.text)
    }

    func bodyText() -> some View {
        self
            .font(DesignTypography.bodyMedium)
            .foregroundColor(DesignColors.text)
    }

    func secondaryText() -> some View {
        self
            .font(DesignTypography.bodySmall)
            .foregroundColor(DesignColors.textSecondary)
    }

    func tertiaryText() -> some View {
        self
            .font(DesignTypography.labelSmall)
            .foregroundColor(DesignColors.textTertiary)
    }
}
