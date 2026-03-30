import Testing

@testable import FluidAudio

// Helper actor for capturing updates in callbacks
actor UpdateCapture {
    var lastUpdate: CommitLayerUpdate?
    var capturedUpdates: [CommitLayerUpdate] = []
    var updateCount: Int = 0
    var commitCount: Int = 0
    var timeoutCommitCount: Int = 0

    func capture(_ update: CommitLayerUpdate) {
        lastUpdate = update
        capturedUpdates.append(update)
        updateCount += 1
        if update.lastCommitReason == .debounceTimeout {
            timeoutCommitCount += 1
        }
    }

    func reset() {
        lastUpdate = nil
        capturedUpdates.removeAll()
        updateCount = 0
        commitCount = 0
        timeoutCommitCount = 0
    }
}

@Suite("PunctuationCommitLayer Tests")
struct PunctuationCommitLayerTests {

    // MARK: - Punctuation Detection Tests

    @Test("Detects punctuation and splits text correctly")
    func testPunctuationDetection() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello world. How are you")

        #expect(update.committedText == "Hello world. ")
        #expect(update.ghostText == "How are you")
        #expect(update.lastCommitReason == .punctuation("."))
    }

    @Test("No punctuation keeps all text as ghost")
    func testNoPunctuation() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello world")

        #expect(update.committedText == "")
        #expect(update.ghostText == "Hello world")
        #expect(update.lastCommitReason == nil)
    }

    @Test("Multiple punctuation marks commit all")
    func testMultiplePunctuation() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("First. Second! Third?")

        #expect(update.committedText == "First. Second! Third? ")
        #expect(update.ghostText == "")
        #expect(update.lastCommitReason == .punctuation("?"))
    }

    @Test("Exclamation mark commits text")
    func testExclamationMark() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Wow! Amazing")

        #expect(update.committedText == "Wow! ")
        #expect(update.ghostText == "Amazing")
        #expect(update.lastCommitReason == .punctuation("!"))
    }

    @Test("Question mark commits text")
    func testQuestionMark() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("How are you? I am fine")

        #expect(update.committedText == "How are you? ")
        #expect(update.ghostText == "I am fine")
        #expect(update.lastCommitReason == .punctuation("?"))
    }

    @Test("Incremental updates accumulate committed text")
    func testIncrementalUpdates() async throws {
        let layer = PunctuationCommitLayer()

        let update1 = await layer.processPartialText("Hello. ")
        #expect(update1.committedText == "Hello. ")
        #expect(update1.ghostText == "")

        let update2 = await layer.processPartialText("How are you")
        #expect(update2.committedText == "Hello. ")
        #expect(update2.ghostText == "How are you")

        let update3 = await layer.processPartialText("How are you? Great")
        #expect(update3.committedText == "Hello. How are you? ")
        #expect(update3.ghostText == "Great")
    }

    // MARK: - Debounce Timeout Tests

    @Test("Debounce timeout commits ghost text when commitOnTimeout is true")
    func testDebounceTimeoutCommits() async throws {
        let layer = PunctuationCommitLayer(
            debounceTimeout: 0.1,  // 100ms for fast testing
            commitOnTimeout: true
        )

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        let update1 = await layer.processPartialText("Hello world")
        #expect(update1.committedText == "")
        #expect(update1.ghostText == "Hello world")

        // Wait for debounce timeout + extra time for callback task to complete
        try await Task.sleep(nanoseconds: 150_000_000)  // 150ms
        try await Task.sleep(nanoseconds: 50_000_000)  // Extra 50ms for callback

        // Check callback was invoked with timeout commit
        let lastUpdate = await updateActor.lastUpdate
        #expect(lastUpdate?.committedText == "Hello world")
        #expect(lastUpdate?.ghostText == "")
        #expect(lastUpdate?.lastCommitReason == .debounceTimeout)
    }

    @Test("Debounce timeout keeps ghost text when commitOnTimeout is false")
    func testDebounceTimeoutNoCommit() async throws {
        let layer = PunctuationCommitLayer(
            debounceTimeout: 0.1,  // 100ms for fast testing
            commitOnTimeout: false
        )

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        let update1 = await layer.processPartialText("Hello world")
        #expect(update1.committedText == "")
        #expect(update1.ghostText == "Hello world")

        // Wait for debounce timeout
        try await Task.sleep(nanoseconds: 150_000_000)  // 150ms

        // Callback should not have been invoked for timeout (only initial processPartialText)
        let timeoutCommits = await updateActor.timeoutCommitCount
        #expect(timeoutCommits == 0)
    }

    @Test("New text cancels debounce timer")
    func testDebounceTimerCancellation() async throws {
        let layer = PunctuationCommitLayer(
            debounceTimeout: 0.1,  // 100ms for fast testing
            commitOnTimeout: true
        )

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        _ = await layer.processPartialText("Hello")
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms (before timeout)

        // New text should cancel previous timer
        _ = await layer.processPartialText("Hello world")
        try await Task.sleep(nanoseconds: 120_000_000)  // 120ms
        try await Task.sleep(nanoseconds: 50_000_000)  // Extra 50ms for callback

        // Should only commit once (from second update)
        let timeoutCommits = await updateActor.timeoutCommitCount
        #expect(timeoutCommits == 1)
    }

    // MARK: - EOU Integration Tests

    @Test("EOU commits all ghost text")
    func testEOUCommitsGhostText() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello world")
        let eouUpdate = await layer.processEOU()

        #expect(eouUpdate.committedText == "Hello world")
        #expect(eouUpdate.ghostText == "")
        #expect(eouUpdate.lastCommitReason == .endOfUtterance)
    }

    @Test("EOU with existing committed text")
    func testEOUWithCommittedText() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello. World")
        let eouUpdate = await layer.processEOU()

        #expect(eouUpdate.committedText == "Hello. World")
        #expect(eouUpdate.ghostText == "")
        #expect(eouUpdate.lastCommitReason == .endOfUtterance)
    }

    @Test("EOU with no ghost text")
    func testEOUWithNoGhostText() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello.")
        let eouUpdate = await layer.processEOU()

        #expect(eouUpdate.committedText == "Hello. ")
        #expect(eouUpdate.ghostText == "")
        #expect(eouUpdate.lastCommitReason == .endOfUtterance)
    }

    @Test("EOU cancels debounce timer")
    func testEOUCancelsDebounce() async throws {
        let layer = PunctuationCommitLayer(
            debounceTimeout: 0.1,
            commitOnTimeout: true
        )

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        _ = await layer.processPartialText("Hello")
        try await Task.sleep(nanoseconds: 50_000_000)  // 50ms

        // EOU should cancel debounce timer
        _ = await layer.processEOU()
        try await Task.sleep(nanoseconds: 100_000_000)  // Wait past original timeout

        // Should not have committed via timeout
        let timeoutCommits = await updateActor.timeoutCommitCount
        #expect(timeoutCommits == 0)
    }

    // MARK: - Manual Commit Tests

    @Test("Manual commit promotes ghost text")
    func testManualCommit() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello world")
        let commitUpdate = await layer.manualCommit()

        #expect(commitUpdate.committedText == "Hello world")
        #expect(commitUpdate.ghostText == "")
        #expect(commitUpdate.lastCommitReason == .manualCommit)
    }

    @Test("Manual commit with no ghost text")
    func testManualCommitNoGhost() async throws {
        let layer = PunctuationCommitLayer()

        let commitUpdate = await layer.manualCommit()

        #expect(commitUpdate.committedText == "")
        #expect(commitUpdate.ghostText == "")
        #expect(commitUpdate.lastCommitReason == .manualCommit)
    }

    @Test("Manual commit with existing committed text")
    func testManualCommitWithExistingCommitted() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello. World")
        let commitUpdate = await layer.manualCommit()

        #expect(commitUpdate.committedText == "Hello. World")
        #expect(commitUpdate.ghostText == "")
        #expect(commitUpdate.lastCommitReason == .manualCommit)
    }

    // MARK: - Reset Tests

    @Test("Reset clears all text")
    func testReset() async throws {
        let layer = PunctuationCommitLayer()

        _ = await layer.processPartialText("Hello. World")
        await layer.reset()

        let update = await layer.processPartialText("New text")
        #expect(update.committedText == "")
        #expect(update.ghostText == "New text")
    }

    @Test("Reset cancels debounce timer")
    func testResetCancelsDebounce() async throws {
        let layer = PunctuationCommitLayer(
            debounceTimeout: 0.1,
            commitOnTimeout: true
        )

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        _ = await layer.processPartialText("Hello")
        await layer.reset()
        try await Task.sleep(nanoseconds: 150_000_000)

        let timeoutCommits = await updateActor.timeoutCommitCount
        #expect(timeoutCommits == 0)
    }

    // MARK: - Callback Tests

    @Test("Callback is invoked on updates")
    func testCallbackInvoked() async throws {
        let layer = PunctuationCommitLayer()

        let updateActor = UpdateCapture()
        await layer.setUpdateCallback { update in
            Task {
                await updateActor.capture(update)
            }
        }

        _ = await layer.processPartialText("Hello")
        _ = await layer.processPartialText("Hello.")
        _ = await layer.manualCommit()

        // Give callbacks time to complete
        try await Task.sleep(nanoseconds: 10_000_000)  // 10ms

        let count = await updateActor.updateCount
        #expect(count == 3)
    }

    // MARK: - Edge Cases

    @Test("Empty string input")
    func testEmptyString() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("")

        #expect(update.committedText == "")
        #expect(update.ghostText == "")
    }

    @Test("Only punctuation")
    func testOnlyPunctuation() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText(".")

        #expect(update.committedText == ". ")
        #expect(update.ghostText == "")
        #expect(update.lastCommitReason == .punctuation("."))
    }

    @Test("Punctuation at start")
    func testPunctuationAtStart() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText(". Hello")

        #expect(update.committedText == ". ")
        #expect(update.ghostText == "Hello")
        #expect(update.lastCommitReason == .punctuation("."))
    }

    @Test("Multiple consecutive punctuation")
    func testConsecutivePunctuation() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("What...? Really")

        #expect(update.committedText == "What...? ")
        #expect(update.ghostText == "Really")
        #expect(update.lastCommitReason == .punctuation("?"))
    }

    @Test("Whitespace handling")
    func testWhitespaceHandling() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello.   World")

        #expect(update.committedText == "Hello.   ")
        #expect(update.ghostText == "World")
    }

    // MARK: - Total Text Tests

    @Test("Total text combines committed and ghost")
    func testTotalText() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello. World")

        #expect(update.totalText == "Hello. World")
    }

    @Test("Total text with no ghost")
    func testTotalTextNoGhost() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello.")

        #expect(update.totalText == "Hello. ")
    }

    @Test("Total text with no committed")
    func testTotalTextNoCommitted() async throws {
        let layer = PunctuationCommitLayer()

        let update = await layer.processPartialText("Hello")

        #expect(update.totalText == "Hello")
    }

    // MARK: - Concurrency Tests

    @Test("Concurrent updates are safe")
    func testConcurrentUpdates() async throws {
        let layer = PunctuationCommitLayer()

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    _ = await layer.processPartialText("Update \(i). ")
                }
            }
        }

        let finalUpdate = await layer.processPartialText("Final")
        // Should have accumulated some committed text from concurrent updates
        #expect(!finalUpdate.committedText.isEmpty)
    }

    @Test("Actor isolation prevents data races")
    func testActorIsolation() async throws {
        let layer = PunctuationCommitLayer()

        // This test verifies that the actor can be safely accessed from multiple tasks
        var resultCount = 0
        await withTaskGroup(of: CommitLayerUpdate.self) { group in
            group.addTask {
                await layer.processPartialText("First. ")
            }
            group.addTask {
                await layer.processEOU()
            }
            group.addTask {
                await layer.manualCommit()
            }

            for await _ in group {
                resultCount += 1
            }
        }

        // If we get here without crashes, actor isolation is working
        #expect(resultCount == 3)
    }
}
