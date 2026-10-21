import SwiftUI

/// Speech as sea: three translucent swells whose height follows the envelope
/// of the audio actually coming out of the speaker (not the generator, which
/// runs 20–50x ahead). Redrawn at display rate from `LevelMeter`, which the
/// player's audio tap feeds; nothing here is `@Published`, so the rest of the
/// UI never re-renders because of it.
struct OceanWaveView: View {
    let meter: LevelMeter

    var body: some View {
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                let t = timeline.date.timeIntervalSinceReferenceDate
                let snap = meter.snapshot(at: timeline.date)
                draw(context: context, size: size, time: t, level: snap.level, history: snap.history)
            }
        }
        .frame(height: 96)
        .background(
            LinearGradient(colors: [Nord.bg, Nord.panel], startPoint: .top, endPoint: .bottom)
        )
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Nord.border))
        .accessibilityLabel("Speech level as ocean waves")
    }

    private func draw(context: GraphicsContext, size: CGSize, time t: Double, level: Float, history: [Float]) {
        let w = size.width
        let h = size.height
        let baseline = h * 0.58
        let n = history.count
        // Envelope along x: oldest level at the left, what plays now at the right.
        func envelope(_ x: CGFloat) -> CGFloat {
            guard n > 1 else { return 0 }
            let pos = Double(x / w) * Double(n - 1)
            let i = min(n - 2, max(0, Int(pos)))
            let f = CGFloat(pos - Double(i))
            let v = CGFloat(history[i]) * (1 - f) + CGFloat(history[i + 1]) * f
            return min(1, v * 7)  // speech RMS 0.03–0.15 -> 0.2–1
        }
        let layers:
            [(wavelength: CGFloat, speed: Double, phase: Double, height: CGFloat, color: Color, alpha: Double)] = [
                (w / 2.2, 0.9, 0.0, 0.42, Nord.frost3, 0.6),
                (w / 3.6, -1.4, 1.7, 0.32, Nord.frost2, 0.55),
                (w / 5.5, 2.3, 3.1, 0.22, Nord.frost1, 0.55),
            ]
        for layer in layers {
            var path = Path()
            path.move(to: CGPoint(x: 0, y: h))
            var x: CGFloat = 0
            while x <= w {
                let swell = 0.035 + envelope(x) * layer.height  // idle swell keeps the sea alive
                let angle = Double(2 * .pi * x / layer.wavelength) + t * layer.speed + layer.phase
                let ripple = sin(angle) + 0.35 * sin(angle * 2.1 + t * 0.7)
                let y = baseline - CGFloat(ripple) * swell * h * 0.55
                path.addLine(to: CGPoint(x: x, y: y))
                x += 3
            }
            path.addLine(to: CGPoint(x: w, y: h))
            path.closeSubpath()
            context.fill(path, with: .color(layer.color.opacity(layer.alpha)))
            context.stroke(path, with: .color(layer.color.opacity(min(1, layer.alpha + 0.3))), lineWidth: 1)
        }
        // Foam at the current level, right edge.
        let glow = CGFloat(min(1, level * 7))
        if glow > 0.05 {
            let r = 6 + glow * 14
            let rect = CGRect(x: w - 24 - r / 2, y: baseline - glow * h * 0.3 - r / 2, width: r, height: r)
            context.fill(Path(ellipseIn: rect), with: .color(Nord.textBright.opacity(0.25 + 0.5 * glow)))
        }
    }
}
