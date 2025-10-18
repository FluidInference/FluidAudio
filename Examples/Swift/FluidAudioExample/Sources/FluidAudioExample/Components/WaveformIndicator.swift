import SwiftUI

struct WaveformIndicator: View {
    let isRecording: Bool
    @State private var pulseScale: CGFloat = 1.0
    @State private var pulseOpacity: Double = 1.0

    var body: some View {
        ZStack {
            // Outer pulse rings
            if isRecording {
                Circle()
                    .stroke(DesignColors.recording.opacity(0.3), lineWidth: 1.5)
                    .scaleEffect(pulseScale)
                    .opacity(pulseOpacity)

                Circle()
                    .stroke(DesignColors.recording.opacity(0.2), lineWidth: 1.5)
                    .scaleEffect(pulseScale * 1.3)
                    .opacity(pulseOpacity * 0.7)
            }

            // Center circle
            Circle()
                .fill(DesignColors.recording)
                .shadow(color: DesignColors.recording.opacity(0.4), radius: 6, x: 0, y: 2)
        }
        .frame(width: 16, height: 16)
        .onAppear {
            if isRecording {
                startPulse()
            }
        }
        .onChange(of: isRecording) { newValue in
            if newValue {
                startPulse()
            } else {
                pulseScale = 1.0
                pulseOpacity = 1.0
            }
        }
    }

    private func startPulse() {
        withAnimation(Animation.easeOut(duration: 1.5).repeatForever(autoreverses: false)) {
            pulseScale = 2.5
            pulseOpacity = 0
        }
    }
}

struct WaveformVisualization: View {
    @State private var displayBars: [CGFloat] = Array(repeating: 0.5, count: 12)
    let isRecording: Bool

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(0..<displayBars.count, id: \.self) { index in
                RoundedRectangle(cornerRadius: 2)
                    .fill(DesignColors.accent.opacity(0.6))
                    .frame(height: displayBars[index] * 32)
                    .opacity(isRecording ? 0.9 : 0.5)
            }
        }
        .frame(height: 32)
        .onAppear {
            if isRecording {
                startAnimation()
            }
        }
        .onChange(of: isRecording) { newValue in
            if newValue {
                startAnimation()
            }
        }
    }

    private func startAnimation() {
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            if isRecording {
                withAnimation(.easeInOut(duration: 0.1)) {
                    displayBars = displayBars.map { _ in
                        CGFloat.random(in: 0.2...1.0)
                    }
                }
            }
        }
    }
}

#Preview {
    VStack(spacing: DesignSpacing.xl) {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            Text("Recording Indicator")
                .headingText()

            HStack(spacing: DesignSpacing.md) {
                WaveformIndicator(isRecording: true)
                Text("Recording active")
                    .bodyText()
            }

            HStack(spacing: DesignSpacing.md) {
                WaveformIndicator(isRecording: false)
                Text("Idle state")
                    .bodyText()
            }
        }
        .cardStyle()
        .padding(DesignSpacing.lg)

        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            Text("Waveform Visualization")
                .headingText()

            WaveformVisualization(isRecording: true)
                .cardStyle()
                .padding(DesignSpacing.md)

            WaveformVisualization(isRecording: false)
                .cardStyle()
                .padding(DesignSpacing.md)
        }
        .padding(DesignSpacing.lg)

        Spacer()
    }
    .background(DesignColors.background)
}
