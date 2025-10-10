# Code Review Methodology

This document outlines the systematic approach used for code reviews in FluidAudio, including detection strategies for common issues and architecture patterns.

## Review Framework

### 1. Initial Context Gathering

**Goal:** Understand the scope and purpose before diving into code.

**Steps:**

1. Read commit messages and PR description
2. Check file statistics (`git diff --stat`)
3. Identify new vs modified files
4. Review documentation changes first (they often explain intent)
5. Map out new component relationships

**Example from realtime-asr-2:**

```bash
git diff --stat origin/main...HEAD
# Revealed: 2,435 lines added, major refactor of streaming pipeline
# New components: StabilizedStreamingEmitter, StreamingVadPipeline, etc.
```

---

### 2. Architecture & Design Analysis

**Goal:** Evaluate high-level structure before implementation details.

#### 2.1 Component Boundaries

**What to look for:**

- Clear separation of concerns
- Single responsibility principle adherence
- Proper dependency injection
- Actor/class boundary decisions

**Detection Strategy:**

- Draw mental component diagram
- Look for tight coupling (too many dependencies)
- Check if components could be reused independently

**Red Flags:**

- Classes with >10 dependencies
- Circular dependencies
- God classes with multiple responsibilities
- Poor abstraction boundaries

**Example:**

```swift
// GOOD: Clear component separation
final class StreamingVadPipeline {
    private let config: StreamingAsrConfig
    private let injectedManager: VadManager?
    // Single responsibility: VAD processing
}

// BAD: Too many responsibilities
actor StreamingAsrManager {
    // Audio conversion + VAD + windowing + stabilization + ASR + error handling
    // 13+ mutable state variables
}
```

#### 2.2 Concurrency & Thread Safety

**What to look for:**

- Proper actor isolation
- Sendable conformance (NO `@unchecked Sendable`)
- Race condition potential
- Deadlock risks

**Detection Strategy:**

1. Search for `@unchecked` - should find zero occurrences
2. Look for shared mutable state
3. Check async boundary crossings
4. Verify Sendable conformance on types crossing actor boundaries

**Red Flags:**

- `@unchecked Sendable` usage
- Mutable state without synchronization
- Complex locking patterns
- Weak/unowned references in async contexts

---

### 3. Resource Management & Memory Issues

**Goal:** Identify leaks, unbounded growth, and lifecycle problems.

#### 3.1 Unbounded Growth Detection

**Strategy:**

1. Search for arrays/buffers that grow during processing
2. Trace the lifecycle - when is data removed?
3. Calculate worst-case growth (e.g., 1-hour session)
4. Check for cleanup in reset/cancel paths

**Detection Pattern:**

```swift
// Step 1: Find accumulation
private var accumulatedTokens: [Int] = []

// Step 2: Find append operations
accumulatedTokens.append(contentsOf: tokens)

// Step 3: Search for removal - NOT FOUND = LEAK
// Missing: accumulatedTokens.removeAll() or sliding window logic

// Step 4: Calculate impact
// 10 tokens/sec * 3600 sec = 36,000 tokens in 1 hour
```

**Common Patterns:**

- Accumulation without bounds: `array.append()` with no corresponding `.removeFirst()` or `.removeAll()`
- Cache without eviction policy
- String concatenation in loops
- Retained closures capturing large objects

#### 3.2 File Handle & Resource Leaks

**Strategy:**

1. Search for `FileHandle`, `URLSession`, `AVAudioEngine`, etc.
2. Verify every open has a corresponding close
3. Check error paths - do they still close resources?
4. Look for `defer` blocks or try-finally patterns

**Detection Pattern:**

```swift
// Opening a resource
let handle = try FileHandle(forWritingTo: url)
debugFileHandle = handle

// Check 1: Is there a close() call?
try? handle.close()  // Found, but...

// Check 2: Is close called on ALL paths?
// - Normal completion: ✅ finalizeAfterStreamEnd()
// - Error paths: ❌ May skip finalization
// - Cancel paths: ❌ May skip finalization

// Check 3: Is failure handled?
try? handle.close()  // Silent failure - RED FLAG
```

#### 3.3 Reference Cycle Detection

**Strategy:**

1. Look for closures capturing self
2. Check for weak/unowned references
3. Trace delegate/callback patterns
4. Verify lifecycle alignment

**Detection Pattern:**

```swift
// RED FLAG: Weak reference in critical path
let decoder: TokenDecoder = { [weak manager] tokenId in
    manager?.vocabulary[tokenId]  // Could return nil mid-stream
}

// Analysis:
// - Is manager expected to outlive the closure? YES
// - Could manager be deallocated early? Possibly
// - Is nil return handled gracefully? NO (silent failure)
// - Solution: [unowned manager] or explicit error handling
```

---

### 4. Code Quality & Maintainability

#### 4.1 Method Length Analysis

**Strategy:**

1. Sort methods by line count
2. Flag methods >50 lines for inspection
3. Check if they can be decomposed
4. Verify single responsibility

**Detection Pattern:**

```bash
# Generate method length report
rg "^\s*(func|init)\s+" --only-matching --line-number | \
  awk '{print $1}' | \
  # Count lines between function declarations
```

**Manual Review:**

```swift
// Example: 105-line method
func process(...) async {  // Line 75
    // Complex logic spanning 105 lines
}  // Line 183

// Decomposition opportunities:
// - Extract VAD processing
// - Extract buffer management
// - Extract event handling
```

**Thresholds:**

- <30 lines: Good
- 30-50 lines: Acceptable, consider refactoring
- 50-80 lines: Should refactor
- >80 lines: Must refactor

#### 4.2 Magic Number Detection

**Strategy:**

1. Search for numeric literals in logic (exclude 0, 1, -1)
2. Check if they have semantic meaning
3. Verify they're documented

**Detection Pattern:**

```bash
# Find numeric literals
rg '\b[0-9]+\.[0-9]+\b|\b[2-9][0-9]+\b' --type swift

# Manual review each occurrence
```

**Examples:**

```swift
// RED FLAG: Unexplained constants
let bufferSize: AVAudioFrameCount = 4096  // Why 4096?
let minimumSeconds: Double = 3.0  // Why 3.0?
if preSpeechBuffer.count > preSpeechSampleLimit  // What's the limit?

// GOOD: Named constants with documentation
/// VAD chunk size aligned with model input requirements (512ms at 16kHz)
static let vadChunkSize: Int = 8192
```

#### 4.3 State Complexity Analysis

**Strategy:**

1. Count mutable properties in actors/classes
2. Look for related state that should be grouped
3. Check for state machine patterns
4. Verify state transitions are documented

**Detection Pattern:**

```swift
// Count mutable properties
// If count > 7-8, consider complexity issue

public actor StreamingAsrManager {
    private var asrManager: AsrManager?              // 1
    private var recognizerTask: Task<Void, Error>?  // 2
    private var audioSource: AudioSource             // 3
    private var vadPipeline: StreamingVadPipeline    // 4
    private var windowProcessor: StreamingWindowProcessor // 5
    private var segmentIndex: Int                    // 6
    private var lastProcessedFrame: Int              // 7
    private var accumulatedTokens: [Int]             // 8
    private var stabilizerSink: StreamingStabilizerSink // 9
    private var cumulativeVadDroppedSamples: Int     // 10
    private var startTime: Date?                     // 11
    private var processedChunks: Int                 // 12
    private var updateContinuation: ...              // 13
}

// Solution: Group related state
struct StreamingState {
    var segmentIndex: Int = 0
    var lastProcessedFrame: Int = 0
    var accumulatedTokens: [Int] = []
    var processedChunks: Int = 0
    var startTime: Date?
}
```

---

### 5 Error Path Analysis

**Strategy:**

1. Find all throw/try statements
2. Trace error propagation
3. Verify cleanup happens on error paths
4. Check for error masking (`try?` without handling)

**Detection Pattern:**

```swift
// Find silent failures
try? handle.close()  // RED FLAG: Error ignored

// Trace error paths
do {
    let result = try await processStreamingChunk(...)
} catch {
    // Question: Are resources cleaned up here?
    // Question: Is state left consistent?
    await attemptErrorRecovery(error: error)  // Does this always succeed?
}
```

**Verification Checklist:**

- [ ] Resources closed on error paths
- [ ] State left in valid state after error
- [ ] Errors logged appropriately
- [ ] User-facing errors have helpful messages
- [ ] No silent failures with `try?`

---

### 6. Performance Analysis

#### 6.1 Algorithmic Complexity

**Strategy:**

1. Identify operations in hot paths
2. Calculate time complexity
3. Check for unnecessary copies
4. Look for N² operations

**Detection Pattern:**

```swift
// RED FLAG: O(n) operation in loop
while residualBuffer.count >= chunkSize {
    let chunk = Array(residualBuffer.prefix(chunkSize))  // O(n) copy
    residualBuffer.removeFirst(chunkSize)  // O(n) shift
}

// Analysis:
// - Called per audio chunk (many times per second)
// - Both operations are O(n) where n = buffer size
// - Combined: O(n²) over full buffer
// - Solution: Circular buffer or index-based approach
```

**Hot Path Identification:**

1. Find audio processing loops
2. Find per-frame operations
3. Check operations called per-token
4. Verify no blocking operations in async paths

#### 6.2 Memory Allocation Patterns

**Strategy:**

1. Look for allocations in loops
2. Check for unnecessary copies
3. Verify buffer reuse
4. Look for string concatenation patterns

**Detection Pattern:**

```swift
// RED FLAG: Repeated allocations
for update in updates {
    updateContinuation?.yield(update)  // Creates new update each time
}

// Better: Reuse or pass references
confirmedTranscript.append(trimmed)  // Multiple allocations

// Better: Reserve capacity upfront
confirmedTranscript.reserveCapacity(estimatedSize)
```

---

### 7. Testing Coverage Analysis

**Strategy:**

1. Map code paths to test cases
2. Identify untested branches
3. Check for edge case coverage
4. Verify error path testing

**Detection Pattern:**

```bash
# 1. List all test files
find Tests -name "*Tests.swift"

# 2. For each source file, check corresponding test
# Example: StreamingVadPipeline.swift
#   -> StreamingVadPipelineTests.swift (NOT FOUND)

# 3. Check test coverage of critical paths
# - Normal flow
# - Error paths
# - Edge cases (empty input, max values, etc.)
# - Concurrent access
```

**Coverage Checklist per Component:**

- [ ] Happy path tested
- [ ] Error paths tested
- [ ] Boundary conditions tested (empty, full, exact size)
- [ ] Concurrent access tested (if applicable)
- [ ] Memory behavior tested (for accumulators)
- [ ] Cleanup tested (reset, cancel)
- [ ] Integration tests for component interactions

**Specific Gaps in realtime-asr-2:**

1. StreamingVadPipeline: No dedicated tests
2. ❌ StreamingWindowProcessor: No tests for complex index logic
    - Needs: Edge cases, boundary conditions, index arithmetic
3. Error recovery paths: No tests
4. Long session memory: No tests
5. Cancel/reset: Limited tests

---

### 8. Security & Safety Review

#### 8.1 Resource Exhaustion

**Strategy:**

1. Find all unbounded operations
2. Calculate worst-case resource usage
3. Verify limits and quotas
4. Check for DOS potential

**Detection Pattern:**

```swift
// RED FLAG: Unbounded file growth
let filename = "fluid_audio_stabilizer_\(UUID().uuidString).jsonl"
// Questions:
// - Max file size?
// - Cleanup policy?
// - Disk space monitoring?

// Calculation:
// 10 updates/sec * 500 bytes/update * 3600 sec = 18 MB/hour
// 8-hour session = 144 MB (acceptable? needs rotation?)
```

#### 8.2 Input Validation

**Strategy:**

1. Find all public API boundaries
2. Check parameter validation
3. Verify range checks
4. Look for assertion vs error handling

**Detection Pattern:**

```swift
// Public API - check validation
public func streamAudio(_ buffer: AVAudioPCMBuffer) {
    inputBuilder.yield(buffer)
    // Questions:
    // - What if buffer is empty?
    // - What if format is wrong?
    // - What if called before start()?
    // - What if called after finish()?
}
```

---

## Review Checklist

Use this checklist for systematic code reviews:

### Architecture

- [ ] Components have clear, single responsibilities
- [ ] Dependencies are injected, not hard-coded
- [ ] Actor boundaries are appropriate
- [ ] No `@unchecked Sendable` usage

### Resource Management

- [ ] No unbounded growth (arrays, strings, caches)
- [ ] All file handles/resources are closed
- [ ] Cleanup works on all paths (normal, error, cancel)
- [ ] No reference cycles

### Code Quality

- [ ] Methods are <50 lines
- [ ] No magic numbers (all explained)
- [ ] State complexity is manageable (<8 properties)
- [ ] Code is self-documenting or well-commented

### Correctness

- [ ] Index arithmetic is correct (bounds checked)
- [ ] Error paths are handled properly
- [ ] No silent failures (`try?` without recovery)
- [ ] Edge cases are handled (empty, null, max values)

### Performance

- [ ] No O(n²) operations in hot paths
- [ ] Unnecessary copies are avoided
- [ ] Buffers are reused where possible
- [ ] String concatenation is optimized

### Testing

- [ ] Happy paths are tested
- [ ] Error paths are tested
- [ ] Edge cases are tested
- [ ] Memory behavior is tested
- [ ] Integration tests exist

### Security

- [ ] No unbounded resource consumption
- [ ] Input validation at API boundaries
- [ ] Error messages don't leak sensitive info
- [ ] File operations are safe

---

## Tools & Commands

### Automated Checks

```bash
# Code formatting
swift format lint --recursive --configuration .swift-format Sources/ Tests/

# Find @unchecked usage (should be 0)
rg "@unchecked" --type swift

# Find magic numbers
rg '\b[2-9][0-9]+\b' --type swift Sources/

# Find long methods (>50 lines)
# (Manual inspection of git diff)

# Find TODOs/FIXMEs
rg "TODO|FIXME|XXX|HACK" --type swift

# Find force unwraps (use sparingly)
rg "!\s*(//|$)" --type swift

# Find try? (check if errors should be handled)
rg "try\?" --type swift
```

### Manual Analysis

```bash
# Review commit
git show <commit-hash>

# Review branch vs main
git diff origin/main...HEAD

# Check file history
git log -p -- path/to/file.swift

# Review specific changes
git diff origin/main...HEAD -- Sources/FluidAudio/ASR/Streaming/
```

---

## Example: Applying the Framework

### Step-by-Step Review of `StreamingAsrManager`

1. **Initial scan** (1 min)
   - 507 lines - substantial refactor
   - Actor-based design ✓
   - 13+ mutable properties - complexity concern

2. **Architecture review** (5 min)
   - Component separation: Good boundaries
   - Concern: Too much responsibility in one actor
   - Actor isolation: Proper ✓

3. **Resource scan** (10 min)
   - Found: `accumulatedTokens` grows unbounded - CRITICAL
   - Found: File handles in stabilizer sink
   - Verified: No circular references ✓

4. **Code quality** (10 min)
   - `start()` method: 76 lines - refactor suggested
   - Magic number: 4096 buffer size - needs constant
   - State complexity: 13 properties - consider grouping

5. **Correctness** (10 min)
   - Index arithmetic in window processor - needs verification
   - Error paths: Recovery implemented but not tested
   - Weak reference in stabilizer - potential nil issue

6. **Performance** (5 min)
   - Array operations: `removeFirst()` in hot path - optimization suggested
   - String concat: Multiple appends - use reserveCapacity

7. **Testing** (5 min)
   - Integration tests exist ✓
   - Missing: VAD pipeline tests, long-session tests
   - Missing: Error recovery tests

8. **Security** (5 min)
   - Debug file: No size limit - add rotation
   - Input validation: Minimal - add checks

**Total time:** ~50 minutes for comprehensive review

**Output:** Prioritized list of issues with severity and recommendations

---

## Severity Ratings

### Critical (Must Fix Before Merge)

- Memory leaks
- Resource leaks (file handles, connections)
- Data corruption risks
- Security vulnerabilities
- `@unchecked Sendable` usage

### High (Should Fix Before Merge)

- Complex, untested logic (e.g., index arithmetic)
- Error handling gaps
- Performance issues in hot paths
- Missing critical tests

### Medium (Consider Fixing)

- Method length issues
- State complexity
- Magic numbers
- Code duplication
- Test coverage gaps

### Low (Future Improvement)

- Style inconsistencies
- Documentation gaps
- Minor optimizations
- Refactoring suggestions

---

## Continuous Improvement

After each review:

1. **Track patterns** - Note recurring issues
2. **Update checklists** - Add new categories as discovered
3. **Share findings** - Educate team on patterns
4. **Automate** - Add linters/analyzers for common issues
5. **Measure** - Track issue density over time

---

## References

- [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)
- [Swift Concurrency](https://docs.swift.org/swift-book/LanguageGuide/Concurrency.html)
- [Memory Management in Swift](https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html)
- FluidAudio [CLAUDE.md](../CLAUDE.md) - Project-specific guidelines
