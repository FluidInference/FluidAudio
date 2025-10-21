import SwiftUI

struct SkeletonLoader: View {
    @State private var isShimmering = false

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            // Simulating a title line
            RoundedRectangle(cornerRadius: 6)
                .fill(DesignColors.secondaryBackground)
                .frame(width: 120, height: 16)
                .shimmer(isActive: isShimmering)

            VStack(alignment: .leading, spacing: DesignSpacing.sm) {
                RoundedRectangle(cornerRadius: 6)
                    .fill(DesignColors.secondaryBackground)
                    .frame(height: 12)
                    .shimmer(isActive: isShimmering)

                RoundedRectangle(cornerRadius: 6)
                    .fill(DesignColors.secondaryBackground)
                    .frame(width: 200, height: 12)
                    .shimmer(isActive: isShimmering)
            }
        }
        .onAppear {
            isShimmering = true
        }
    }
}

struct SkeletonCardLoader: View {
    @State private var isShimmering = false

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            HStack(spacing: DesignSpacing.md) {
                RoundedRectangle(cornerRadius: 24)
                    .fill(DesignColors.secondaryBackground)
                    .frame(width: 40, height: 40)
                    .shimmer(isActive: isShimmering)

                VStack(alignment: .leading, spacing: DesignSpacing.sm) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(DesignColors.secondaryBackground)
                        .frame(width: 100, height: 14)
                        .shimmer(isActive: isShimmering)

                    RoundedRectangle(cornerRadius: 6)
                        .fill(DesignColors.secondaryBackground)
                        .frame(width: 150, height: 12)
                        .shimmer(isActive: isShimmering)
                }

                Spacer()
            }

            Divider()
                .opacity(DesignOpacity.divider)

            VStack(alignment: .leading, spacing: DesignSpacing.md) {
                RoundedRectangle(cornerRadius: 6)
                    .fill(DesignColors.secondaryBackground)
                    .frame(height: 40)
                    .shimmer(isActive: isShimmering)

                HStack(spacing: DesignSpacing.md) {
                    RoundedRectangle(cornerRadius: 6)
                        .fill(DesignColors.secondaryBackground)
                        .shimmer(isActive: isShimmering)

                    RoundedRectangle(cornerRadius: 6)
                        .fill(DesignColors.secondaryBackground)
                        .shimmer(isActive: isShimmering)
                }
                .frame(height: 36)
            }
        }
        .padding(DesignSpacing.lg)
        .cardStyle()
        .onAppear {
            isShimmering = true
        }
    }
}

// MARK: - Shimmer Modifier
struct ShimmerModifier: ViewModifier {
    @State private var shimmerPosition: CGFloat = -1
    let isActive: Bool

    func body(content: Content) -> some View {
        content
            .background(
                LinearGradient(
                    gradient: Gradient(stops: [
                        .init(color: DesignColors.secondaryBackground, location: 0),
                        .init(color: DesignColors.card, location: 0.5),
                        .init(color: DesignColors.secondaryBackground, location: 1),
                    ]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: shimmerPosition * 400)
            )
            .clipped()
            .onAppear {
                if isActive {
                    startShimmer()
                }
            }
            .onChange(of: isActive) { _, newValue in
                if newValue {
                    startShimmer()
                } else {
                    shimmerPosition = -1
                }
            }
    }

    private func startShimmer() {
        withAnimation(Animation.linear(duration: 1.5).repeatForever(autoreverses: false)) {
            shimmerPosition = 1
        }
    }
}

extension View {
    func shimmer(isActive: Bool = true) -> some View {
        modifier(ShimmerModifier(isActive: isActive))
    }
}

#Preview {
    VStack(spacing: DesignSpacing.xl) {
        Text("Skeleton Loader")
            .headingText()

        SkeletonLoader()

        Divider()
            .opacity(DesignOpacity.divider)

        Text("Card Skeleton")
            .headingText()

        SkeletonCardLoader()

        Spacer()
    }
    .padding(DesignSpacing.lg)
    .background(DesignColors.background)
}
