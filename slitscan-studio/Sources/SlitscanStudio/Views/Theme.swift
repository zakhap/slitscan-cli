// Theme.swift — the darkroom. One source of truth for the whole instrument:
// near-black room, mono type, hairline rules, a single safelight accent.
// No view should reach for a raw Color or .system font; pull from here.

import SwiftUI

enum Theme {
    // The room
    static let bg        = Color(hex: 0x0A0A0A)   // near-black floor
    static let panel     = Color(hex: 0x141414)   // inspector / filmstrip surface
    static let panelHi   = Color(hex: 0x1C1C1C)   // raised chips, hover
    static let hairline  = Color(hex: 0x2A2A2A)   // 1px rules between zones
    static let hairlineHi = Color(hex: 0x3A3A3A)  // active/hover rule

    // Ink
    static let ink   = Color(hex: 0xE8E8E8)        // primary text
    static let dim   = Color(hex: 0x6E6E6E)        // labels, secondary
    static let faint = Color(hex: 0x444444)        // disabled, ghost

    // The one accent
    static let safelight = Color(hex: 0xFF3B1D)    // markers, fills, focus, "live"

    // Type — mono everywhere, this is an instrument
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
    static let label   = mono(10, .semibold)       // UPPERCASE section + control labels
    static let body    = mono(12)                  // control text
    static let value   = mono(11, .medium)         // numeric readouts
    static let display = mono(13, .bold)           // wordmark
}

// MARK: - Reusable darkroom surfaces

extension View {
    /// Boxed monospace readout — the little numeric chip next to a control.
    func darkroomChip() -> some View {
        self
            .font(Theme.value)
            .foregroundStyle(Theme.ink)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Theme.panelHi)
            .overlay(RoundedRectangle(cornerRadius: 2).strokeBorder(Theme.hairline, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 2))
    }

    /// A full-width hairline rule.
    func hairlineUnderline() -> some View {
        overlay(alignment: .bottom) {
            Rectangle().fill(Theme.hairline).frame(height: 1)
        }
    }
}

// MARK: - Borderless mono action button (toolbar verbs)

struct DarkroomButtonStyle: ButtonStyle {
    var emphasized = false
    @State private var hovering = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Theme.body)
            .foregroundStyle(foreground(pressed: configuration.isPressed))
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(background(pressed: configuration.isPressed))
            .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(border, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: 3))
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .animation(.easeOut(duration: 0.12), value: hovering)
            .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
    }

    private func foreground(pressed: Bool) -> Color {
        if emphasized { return pressed ? Theme.ink : Theme.safelight }
        return hovering || pressed ? Theme.ink : Theme.dim
    }
    private func background(pressed: Bool) -> Color {
        if pressed { return Theme.panelHi }
        return hovering ? Theme.panel : .clear
    }
    private var border: Color {
        if emphasized { return Theme.safelight.opacity(hovering ? 0.9 : 0.5) }
        return hovering ? Theme.hairlineHi : Theme.hairline
    }
}

extension ButtonStyle where Self == DarkroomButtonStyle {
    static var darkroom: DarkroomButtonStyle { DarkroomButtonStyle() }
    static var darkroomEmphasized: DarkroomButtonStyle { DarkroomButtonStyle(emphasized: true) }
}

// MARK: - hex Color

extension Color {
    init(hex: UInt32, alpha: Double = 1) {
        self.init(
            .sRGB,
            red:   Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue:  Double(hex & 0xFF) / 255,
            opacity: alpha
        )
    }
}
