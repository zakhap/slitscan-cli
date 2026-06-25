// Session.swift — persist the recipe (params + optional clip reference). The
// recipe is tiny and shareable independent of the clip (spec §14).

import Foundation
import SlitscanCore

struct SessionFile: Codable {
    var params: SurfaceParams
    var clipPath: String?
}

enum Session {
    static func save(params: SurfaceParams, clipURL: URL?, to url: URL) throws {
        let file = SessionFile(params: params, clipPath: clipURL?.path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(file).write(to: url)
    }

    static func load(from url: URL) throws -> SessionFile {
        try JSONDecoder().decode(SessionFile.self, from: Data(contentsOf: url))
    }
}
