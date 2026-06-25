// FilmstripView.swift — makes the two time controls *visible* (spec §12).
// Thumbnails of the clip, a marker at buffer_origin, and a band showing how far
// back `spread` rakes from it. Drag to scrub the origin.
// Darkroom dress: safelight marker + rake, hairline frame, mono ticks.

import SwiftUI

struct FilmstripView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let originX = CGFloat(app.params.bufferOrigin) * w
            let spreadX = CGFloat(app.params.spread) * w

            ZStack(alignment: .leading) {
                if app.thumbnails.isEmpty {
                    Theme.bg
                    HStack(spacing: 8) {
                        Image(systemName: "film")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.faint)
                        Text("no clip — load a demo or open a video")
                            .font(Theme.value)
                            .foregroundStyle(Theme.faint)
                    }
                    .frame(maxWidth: .infinity)
                } else {
                    HStack(spacing: 0) {
                        ForEach(Array(app.thumbnails.enumerated()), id: \.offset) { _, img in
                            Image(nsImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .clipped()
                        }
                    }
                    // Darken the strip slightly so the safelight marks read clearly.
                    Theme.bg.opacity(0.15)
                }

                // Rake band: from (origin - spread) to origin — where delay reads.
                Rectangle()
                    .fill(Theme.safelight.opacity(0.18))
                    .overlay(alignment: .leading) {
                        Rectangle().fill(Theme.safelight.opacity(0.5)).frame(width: 1)
                    }
                    .frame(width: max(spreadX, 1))
                    .offset(x: max(originX - spreadX, 0))

                // Origin marker (the "now" edge of the window).
                Rectangle()
                    .fill(Theme.safelight)
                    .frame(width: 2)
                    .shadow(color: Theme.safelight.opacity(0.6), radius: 3)
                    .offset(x: min(max(originX - 1, 0), w - 2))
            }
            .overlay(alignment: .top) { ticks(width: w) }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onChanged { v in
                    app.params.bufferOrigin = Double(min(max(v.location.x / w, 0), 1))
                }
            )
        }
        .frame(height: 84)
        .background(Theme.bg)
        .overlay(Rectangle().strokeBorder(Theme.hairline, lineWidth: 1))
    }

    /// Hairline tick marks along the top edge — a darkroom ruler.
    private func ticks(width: CGFloat) -> some View {
        let count = 16
        return HStack(spacing: 0) {
            ForEach(0..<count, id: \.self) { i in
                Rectangle()
                    .fill(Theme.hairlineHi)
                    .frame(width: 1, height: i % 4 == 0 ? 7 : 4)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(width: width)
        .allowsHitTesting(false)
    }
}
