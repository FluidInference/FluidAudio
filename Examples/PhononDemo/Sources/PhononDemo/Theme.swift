import SwiftUI

/// Nord palette (the user's terminal theme): polar-night backgrounds, snow-storm
/// text, frost blues for the sea and controls, aurora accents for status.
enum Nord {
    static let bg = Color(red: 0x2E / 255, green: 0x34 / 255, blue: 0x40 / 255)  // nord0
    static let panel = Color(red: 0x3B / 255, green: 0x42 / 255, blue: 0x52 / 255)  // nord1
    static let panelRaised = Color(red: 0x43 / 255, green: 0x4C / 255, blue: 0x5E / 255)  // nord2
    static let border = Color(red: 0x4C / 255, green: 0x56 / 255, blue: 0x6A / 255)  // nord3
    static let text = Color(red: 0xD8 / 255, green: 0xDE / 255, blue: 0xE9 / 255)  // nord4
    static let textBright = Color(red: 0xEC / 255, green: 0xEF / 255, blue: 0xF4 / 255)  // nord6
    static let textDim = Color(red: 0xA3 / 255, green: 0xAC / 255, blue: 0xBD / 255)
    static let frost0 = Color(red: 0x8F / 255, green: 0xBC / 255, blue: 0xBB / 255)  // nord7
    static let frost1 = Color(red: 0x88 / 255, green: 0xC0 / 255, blue: 0xD0 / 255)  // nord8
    static let frost2 = Color(red: 0x81 / 255, green: 0xA1 / 255, blue: 0xC1 / 255)  // nord9
    static let frost3 = Color(red: 0x5E / 255, green: 0x81 / 255, blue: 0xAC / 255)  // nord10
    static let red = Color(red: 0xBF / 255, green: 0x61 / 255, blue: 0x6A / 255)  // nord11
    static let orange = Color(red: 0xD0 / 255, green: 0x87 / 255, blue: 0x70 / 255)  // nord12
    static let yellow = Color(red: 0xEB / 255, green: 0xCB / 255, blue: 0x8B / 255)  // nord13
    static let green = Color(red: 0xA3 / 255, green: 0xBE / 255, blue: 0x8C / 255)  // nord14
    /// Three shades of the same frost blue (hue of nord9/nord10) for the sea
    /// layers: dark depth, mid swell, pale surface.
    static let seaDeep = Color(red: 0x3F / 255, green: 0x5A / 255, blue: 0x7E / 255)
    static let seaMid = Color(red: 0x6F / 255, green: 0x91 / 255, blue: 0xB5 / 255)
    static let seaLight = Color(red: 0xA8 / 255, green: 0xC3 / 255, blue: 0xDC / 255)
    static let purple = Color(red: 0xB4 / 255, green: 0x8E / 255, blue: 0xAD / 255)  // nord15
}

/// Rounded panel in the terminal's raised background, used for the text box,
/// the table and the log instead of the system gray/black.
struct NordPanel: ViewModifier {
    func body(content: Content) -> some View {
        content
            .scrollContentBackground(.hidden)
            .background(Nord.panel)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Nord.border))
    }
}

extension View {
    func nordPanel() -> some View { modifier(NordPanel()) }
}
