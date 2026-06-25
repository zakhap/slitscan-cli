// InspectorView.swift — the precision complement to canvas gesture (spec §12).
// Every parameter as a labeled control, grouped by family (spec §9).
// Darkroom dress: uppercase mono section rules, boxed numeric readouts.

import SwiftUI
import SlitscanCore

struct InspectorView: View {
    @EnvironmentObject var app: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                group("Surface") {
                    labeled("Profile") {
                        Picker("", selection: $app.params.profile) {
                            ForEach(Profile.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                        }
                        .labelsHidden()
                        .font(Theme.body)
                    }
                    labeled("Axis") {
                        Picker("", selection: $app.params.axis) {
                            ForEach(Axis.allCases, id: \.self) { Text($0.rawValue.uppercased()).tag($0) }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                    }
                    slider("Vanguard", $app.params.vanguard, 0...1)
                }

                group("Time") {
                    slider("Origin (scrub)", $app.params.bufferOrigin, 0...1)
                    slider("Spread (rake)", $app.params.spread, 0...1)
                }

                group("Texture") {
                    labeled("Grain / slice") {
                        Stepper(value: $app.params.sliceWidth, in: 1...128) {
                            Text("\(app.params.sliceWidth) px").darkroomChip()
                        }
                    }
                }

                group("Boundary & quality") {
                    labeled("Fill") {
                        Picker("", selection: $app.params.fill) {
                            ForEach(FillMode.allCases, id: \.self) { Text($0.rawValue.capitalized).tag($0) }
                        }
                        .labelsHidden()
                        .font(Theme.body)
                    }
                    Toggle(isOn: $app.params.interpolate) {
                        Text("Interpolate (sub-frame)").font(Theme.body).foregroundStyle(Theme.ink)
                    }
                    .toggleStyle(.switch)
                }

                Button("reset all", action: app.resetParams)
                    .buttonStyle(.darkroom)
                    .padding(.top, 4)
            }
            .padding(18)
        }
        .frame(width: 300)
        .background(Theme.panel)
    }

    // MARK: building blocks

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(Theme.label)
                .tracking(1.5)
                .foregroundStyle(Theme.dim)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.bottom, 6)
                .hairlineUnderline()
            content()
        }
    }

    /// A label sitting above its control — the inspector's default row.
    private func labeled<Content: View>(_ label: String, @ViewBuilder _ control: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(label).font(Theme.body).foregroundStyle(Theme.dim)
            control()
        }
    }

    private func slider(_ label: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Text(label).font(Theme.body).foregroundStyle(Theme.dim)
                Spacer()
                Text(String(format: "%.3f", value.wrappedValue)).darkroomChip()
            }
            Slider(value: value, in: range)
                .controlSize(.small)
        }
    }
}
