import SwiftUI

/// The speaking ring: three closed waves around a soft glow. At rest it breathes;
/// while audio plays, each wave's radius follows the live spectrum of what the
/// speaker is playing (`SpectrumMeter`, fed by the player's audio tap), so vowels
/// swell the low arcs and sibilants spike the high ones. The spectrum is mirrored
/// around the circle so the outline has no seam, and each wave rotates at its own
/// speed so the shape keeps moving even on a held vowel.
struct RingWaveView: View {
    let meter: SpectrumMeter
    /// Three wave colours, outer to inner.
    let palette: [Color]

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let snap = meter.snapshot(at: timeline.date)
                draw(
                    context: context, size: size, time: timeline.date.timeIntervalSinceReferenceDate,
                    level: snap.level, bands: snap.bands)
            }
        }
        .accessibilityLabel("Speech as a ring of waves")
    }

    private struct Wave {
        let amplitude: CGFloat  // fraction of the base radius at full energy
        let spin: Double  // radians per second
        let ripples: Double  // lobes of the idle ripple
        let rippleSpeed: Double
        let width: CGFloat
    }

    private let waves: [Wave] = [
        Wave(amplitude: 0.50, spin: 0.17, ripples: 3, rippleSpeed: 1.1, width: 2.6),
        Wave(amplitude: 0.38, spin: -0.23, ripples: 4, rippleSpeed: -1.6, width: 2.0),
        Wave(amplitude: 0.27, spin: 0.31, ripples: 5, rippleSpeed: 2.2, width: 1.5),
    ]

    private func draw(context: GraphicsContext, size: CGSize, time t: Double, level: Float, bands: [Float]) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let base = min(size.width, size.height) * 0.30
        let energy = CGFloat(tanh(Double(level) * 14))  // 0 at rest, ~1 on speech
        // Contrast: how far each band sits above the frame's mean, so the outline
        // forms lobes instead of a uniformly larger circle.
        let mean = bands.isEmpty ? 0 : bands.reduce(0, +) / Float(bands.count)
        let raw = bands.map { CGFloat(max(0, min(1, ($0 - mean) * 2.6 + 0.35))) }
        // 5-tap smoothing across neighbouring bands: round lobes, no jitter.
        let kernel: [CGFloat] = [0.1, 0.2, 0.4, 0.2, 0.1]
        let peaks = raw.indices.map { i in
            kernel.enumerated().reduce(CGFloat(0)) { acc, e in
                acc + e.element * raw[min(raw.count - 1, max(0, i + e.offset - 2))]
            }
        }
        let breathe = 1 + 0.018 * CGFloat(sin(t * 1.25))

        // Glow behind the waves, brighter with energy.
        let glowR = base * (1.15 + 0.35 * energy)
        context.fill(
            Path(ellipseIn: CGRect(x: center.x - glowR, y: center.y - glowR, width: 2 * glowR, height: 2 * glowR)),
            with: .radialGradient(
                Gradient(colors: [
                    palette[1].opacity(0.10 + 0.30 * Double(energy)), palette[2].opacity(0.04), .clear,
                ]),
                center: center, startRadius: 0, endRadius: glowR))

        // Mirrored spectrum around the circle: angle 0 (top) is the lowest band,
        // the bottom the highest, both sides symmetric. Cosine-interpolated.
        func band(_ angle: Double) -> CGFloat {
            guard peaks.count > 1 else { return 0 }
            var u = (angle / Double.pi).truncatingRemainder(dividingBy: 2)
            if u < 0 { u += 2 }
            let p = (u > 1 ? 2 - u : u) * Double(peaks.count - 1)
            let i = min(peaks.count - 2, Int(p))
            let f = CGFloat(0.5 * (1 - cos(Double.pi * (p - Double(i)))))
            return peaks[i] * (1 - f) + peaks[i + 1] * f
        }

        let steps = 240
        for (index, wave) in waves.enumerated().reversed() {
            var path = Path()
            let rotation = t * wave.spin
            for s in 0...steps {
                let theta = Double(s) / Double(steps) * 2 * Double.pi
                let idle = 0.014 * sin(wave.ripples * theta + t * wave.rippleSpeed)
                let shape =
                    0.88 * band(theta + rotation) + 0.12 * CGFloat(0.5 + 0.5 * sin(wave.ripples * theta - t * 2))
                let r = base * (breathe + CGFloat(idle) + wave.amplitude * energy * shape)
                let point = CGPoint(x: center.x + r * CGFloat(sin(theta)), y: center.y - r * CGFloat(cos(theta)))
                if s == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            path.closeSubpath()
            let color = palette[index]
            if index == waves.count - 1 {
                // tint the inside with the outer colour (the inner one can be yellow)
                context.fill(path, with: .color(palette[0].opacity(0.08 + 0.10 * Double(energy))))
            }
            // soft halo under each line
            var halo = context
            halo.addFilter(.blur(radius: 6 + 10 * energy))
            halo.stroke(path, with: .color(color.opacity(0.35 + 0.4 * Double(energy))), lineWidth: wave.width * 3)
            context.stroke(path, with: .color(color.opacity(0.85)), lineWidth: wave.width)
        }
    }
}
