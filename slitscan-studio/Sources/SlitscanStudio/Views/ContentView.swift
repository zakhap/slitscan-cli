// ContentView.swift — the single window, three zones: canvas (dominant),
// inspector (right), filmstrip (bottom) — plus a top action bar (spec §12).
// Dressed as a darkroom: near-black room, mono verbs, hairline rules.

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            rule
            HStack(spacing: 0) {
                ZStack {
                    Theme.bg
                    MetalView(appState: app)
                    if app.isLoading {
                        ProgressView()
                            .controlSize(.large)
                            .tint(Theme.safelight)
                            .padding(24)
                            .background(Theme.panel)
                            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Theme.hairline, lineWidth: 1))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                }
                .frame(minWidth: 360, minHeight: 320)
                rule
                InspectorView()
            }
            rule
            VStack(alignment: .leading, spacing: 8) {
                FilmstripView()
                statusBar
            }
            .padding(12)
            .background(Theme.panel)
        }
        .frame(minWidth: 900, minHeight: 600)
        .background(Theme.bg)
    }

    private var rule: some View { Rectangle().fill(Theme.hairline).frame(height: 1) }

    // MARK: toolbar

    private var toolbar: some View {
        HStack(spacing: 14) {
            HStack(spacing: 8) {
                Circle().fill(Theme.safelight).frame(width: 7, height: 7)
                Text("THE CUT")
                    .font(Theme.display)
                    .tracking(2)
                    .foregroundStyle(Theme.ink)
            }

            Rectangle().fill(Theme.hairline).frame(width: 1, height: 16)

            Button("demo", action: app.loadDemo).buttonStyle(.darkroom)
            Button("open…", action: openVideo).buttonStyle(.darkroom)

            Spacer()

            Button("save recipe", action: saveRecipe).buttonStyle(.darkroom)
            Button("load recipe", action: loadRecipe).buttonStyle(.darkroom)
            Button("export still ⌘E", action: exportStill)
                .buttonStyle(.darkroomEmphasized)
                .keyboardShortcut("e", modifiers: .command)
                .disabled(app.cube == nil)
                .opacity(app.cube == nil ? 0.4 : 1)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(Theme.bg)
    }

    // MARK: status

    private var statusBar: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(app.isLoading ? Theme.safelight : Theme.faint)
                .frame(width: 6, height: 6)
            Text(app.status.isEmpty ? "ready" : app.status)
                .font(Theme.value)
                .foregroundStyle(Theme.dim)
                .lineLimit(1)
        }
    }

    // MARK: panels

    private func openVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .mpeg4Movie, .quickTimeMovie]
        panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url { app.loadVideo(url: url) }
    }

    private func exportStill() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png, .tiff, .jpeg]
        panel.nameFieldStringValue = "slitscan.png"
        if panel.runModal() == .OK, let url = panel.url { app.export(to: url) }
    }

    private func saveRecipe() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "cut.recipe.json"
        if panel.runModal() == .OK, let url = panel.url { app.saveRecipe(to: url) }
    }

    private func loadRecipe() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url { app.loadRecipe(from: url) }
    }
}
