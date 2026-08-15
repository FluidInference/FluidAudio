import XCTest

@testable import FluidAudio

/// Exact-parity gates for the pure host-side LuxTTS math against the Python
/// fixtures: solver timesteps, ratio duration estimate, avg-duration
/// expansion indices, and the anchor-Euler mini trajectory.
final class LuxTtsSolverTests: XCTestCase {

    func testTimeStepsMatchFixtureExactly() throws {
        let fixtures = try LuxTtsFixtures.load()
        let steps = LuxTtsSolver.timeSteps(
            numSteps: fixtures.solver.numSteps, tShift: fixtures.solver.tShift)
        XCTAssertEqual(steps.count, fixtures.solver.timesteps.count)
        for (swift, python) in zip(steps, fixtures.solver.timesteps) {
            // The fixture values are torch float32; the Swift Double result
            // must round to the identical float32 (the decoder consumes fp32).
            XCTAssertEqual(Float(swift), Float(python), "timestep mismatch")
        }
        XCTAssertEqual(steps.first, 0.0)
        XCTAssertEqual(steps.last, 1.0)
    }

    func testFeaturesLengthMatchesFixture() throws {
        let fixtures = try LuxTtsFixtures.load()
        for text in fixtures.texts {
            let featuresLength = LuxTtsSolver.featuresLength(
                promptFrames: fixtures.prompt.melFrames,
                promptTokenCount: fixtures.prompt.tokenIds.count,
                textTokenCount: text.tokenIds.count,
                speed: 1.0)
            XCTAssertEqual(featuresLength, text.featuresLenSpeed1, "for: \(text.text)")
        }
    }

    func testTokensIndexMatchesFixtureExactly() throws {
        let fixtures = try LuxTtsFixtures.load()
        for text in fixtures.texts {
            let expansion = text.expansion
            let index = try LuxTtsSolver.tokensIndex(
                tokensCount: expansion.tokensLen,
                featuresLength: expansion.featuresLen)
            XCTAssertEqual(index.count, expansion.tokensIndexLen)
            XCTAssertEqual(index, expansion.tokensIndex, "for: \(text.text)")
            // Remainder frames must point at the appended pad-slot row.
            XCTAssertEqual(index.last, expansion.tokensLen)
        }
    }

    func testTokensIndexThrowsOnDegenerateDuration() {
        // featuresLength < tokensCount ⇒ avg < 1 ⇒ every real token gets zero
        // frames and the whole sequence collapses to the pad slot. Must throw
        // rather than silently return an all-pad (silent) index.
        XCTAssertThrowsError(
            try LuxTtsSolver.tokensIndex(tokensCount: 10, featuresLength: 5)
        ) { error in
            guard case LuxTtsError.degenerateDuration(let fl, let tc) = error else {
                XCTFail("expected degenerateDuration, got \(error)")
                return
            }
            XCTAssertEqual(fl, 5)
            XCTAssertEqual(tc, 10)
        }
        // Boundary: avg == 1 (featuresLength == tokensCount) must NOT throw.
        XCTAssertNoThrow(
            try LuxTtsSolver.tokensIndex(tokensCount: 4, featuresLength: 4))
    }

    func testAnchorEulerMiniTrajectoryMatchesFixture() throws {
        let fixtures = try LuxTtsFixtures.load()
        let solver = fixtures.solver
        let trajectory = solver.miniTrajectory
        let steps = LuxTtsSolver.timeSteps(numSteps: solver.numSteps, tShift: solver.tShift)
        // The fixture trajectory was computed with the float32 timesteps.
        let f32Steps = solver.timesteps

        var x = trajectory.x0
        for step in 0..<solver.numSteps {
            // Guard that our Double timesteps round to the fixture's floats.
            XCTAssertEqual(Float(steps[step]), Float(f32Steps[step]))
            x = LuxTtsSolver.anchorEulerUpdate(
                x: x,
                v: trajectory.vSteps[step],
                tCur: f32Steps[step],
                tNext: f32Steps[step + 1],
                isLast: step == solver.numSteps - 1)
        }
        XCTAssertEqual(x.count, trajectory.xFinal.count)
        for (got, expected) in zip(x, trajectory.xFinal) {
            XCTAssertEqual(got, expected, accuracy: 1e-12)
        }
    }
}
