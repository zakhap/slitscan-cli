// ProfileParityTests.swift — the §3 CI gate.
//
// Loads test vectors emitted by the vendored NumPy reference and asserts the
// Swift profile math reproduces them within tolerance. If this passes, the
// Swift definition agrees with the canonical spec by construction; the Metal
// shader transliterates the same Swift arithmetic.

import XCTest
@testable import SlitscanCore

struct VectorCase: Decodable {
    let profile: String
    let n: Int
    let vanguard: Double
    let delays: [Double]
}

struct VectorFile: Decodable {
    let cases: [VectorCase]
}

final class ProfileParityTests: XCTestCase {
    func testSwiftMatchesNumPyReference() throws {
        guard let url = Bundle.module.url(forResource: "test_vectors", withExtension: "json") else {
            return XCTFail("test_vectors.json missing — run reference/generate_vectors.py")
        }
        let file = try JSONDecoder().decode(VectorFile.self, from: Data(contentsOf: url))
        XCTAssertFalse(file.cases.isEmpty, "no test vectors loaded")

        let tolerance = 1e-9
        for c in file.cases {
            guard let profile = Profile(rawValue: c.profile) else {
                return XCTFail("unknown profile in fixture: \(c.profile)")
            }
            let got = delays(profile, n: c.n, vanguard: c.vanguard)
            XCTAssertEqual(got.count, c.delays.count,
                           "\(c.profile) n=\(c.n) v=\(c.vanguard): length mismatch")
            for (i, (a, b)) in zip(got, c.delays).enumerated() {
                XCTAssertEqual(a, b, accuracy: tolerance,
                               "\(c.profile) n=\(c.n) v=\(c.vanguard) band \(i): \(a) != \(b)")
            }
        }
    }
}
