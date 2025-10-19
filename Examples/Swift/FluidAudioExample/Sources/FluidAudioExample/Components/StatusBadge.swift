import SwiftUI

struct StatusBadge: View {
    let status: StatusType
    @State private var isAnimating = false

    enum StatusType: Equatable {
        case idle
        case loading
        case success
        case error(String)
        case recording

        var color: Color {
            switch self {
            case .idle:
                return DesignColors.textSecondary
            case .loading:
                return DesignColors.accent
            case .success:
                return DesignColors.success
            case .error:
                return DesignColors.error
            case .recording:
                return DesignColors.recording
            }
        }

        var icon: String {
            switch self {
            case .idle:
                return "checkmark.circle"
            case .loading:
                return "hourglass"
            case .success:
                return "checkmark.circle.fill"
            case .error:
                return "exclamationmark.circle.fill"
            case .recording:
                return "record.circle.fill"
            }
        }

        var label: String {
            switch self {
            case .idle:
                return "Ready"
            case .loading:
                return "Processing"
            case .success:
                return "Complete"
            case .error(let message):
                return message
            case .recording:
                return "Recording"
            }
        }
    }

    var body: some View {
        HStack(spacing: DesignSpacing.sm) {
            Image(systemName: status.icon)
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(status.color)
                .scaleEffect(isAnimating && (status == .loading || status == .recording) ? 1.1 : 1.0)

            Text(status.label)
                .font(DesignTypography.labelSmall)
                .foregroundColor(status.color)
        }
        .padding(.horizontal, DesignSpacing.md)
        .padding(.vertical, DesignSpacing.sm)
        .background(status.color.opacity(0.1))
        .cornerRadius(DesignRadius.small)
        .onAppear {
            if status == .loading || status == .recording {
                startAnimation()
            }
        }
        .onChange(of: status) { _, newStatus in
            if newStatus == .loading || newStatus == .recording {
                startAnimation()
            } else {
                isAnimating = false
            }
        }
    }

    private func startAnimation() {
        isAnimating = true
        withAnimation(Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
            isAnimating = true
        }
    }
}

#Preview {
    VStack(spacing: DesignSpacing.lg) {
        StatusBadge(status: .idle)
        StatusBadge(status: .loading)
        StatusBadge(status: .success)
        StatusBadge(status: .error("Microphone not available"))
        StatusBadge(status: .recording)
    }
    .padding(DesignSpacing.lg)
    .background(DesignColors.background)
}
