# FluidAudio Planning Playbook

This guide defines how Codex should create comprehensive plans before tackling any complex or high-risk change. Use it to think like a principal engineer: deeply understand the problem, map unknowns, evaluate trade-offs, and chart a confident path to execution.

## When to Produce a Plan

- The request touches multiple subsystems, alters concurrency behavior, or impacts the audio pipeline.
- Unclear requirements, ambiguous success criteria, or new product surface areas.
- Any task where you would want peer alignment before coding (≈ 2+ hours of work).
- If in doubt, write the plan. Plans create clarity and expose risks early.

## Plan Creation Workflow

Before drafting the plan, conduct any necessary research using Context7 or DeepWiki (or both). Summarize the key findings and links from that research inside the plan so reviewers can trace assumptions. During execution, keep leveraging Context7/DeepWiki whenever you need additional context or verification, and note their use in updates.

1. **Restate the Mission**
   - Summarize the problem in your own words.
   - List explicit goals, non-goals, stakeholders (CLI users, diarizer pipeline, model loaders), and success criteria (latency/accuracy targets, UX expectations, guardrails).
2. **Gather Critical Context**
   - Identify relevant modules (e.g., `Sources/FluidAudio/ASR`, `Shared`, CLI commands).
   - Note existing patterns (managers, configs, async actors) and style constraints from `AGENTS.md`.
   - Capture external constraints: CoreML availability, threading rules, no mock models/tests without request.
3. **Baseline the Current State**
   - Highlight key entry points, data flows, and known bottlenecks.
   - Link to reference files with line numbers when known.
   - Record metrics or logs worth reproducing before changes.
4. **Clarify Unknowns & Research Needs**
   - Enumerate hard questions you must answer (APIs, model formats, concurrency requirements).
   - Plan validation steps: docs to read, experiments to run, SMEs to consult.
   - Flag “unknown unknowns” by stating the assumptions that could break and how to detect them early.
5. **Enumerate Solution Options**
   - Describe at least 2 viable approaches when possible.
   - For each option, assess complexity, blast radius, performance impact, and alignment with existing architecture.
   - Call out irreversible steps or migrations that need extra caution.
6. **Select a Direction with Rationale**
   - Explain why the chosen path wins (safety, maintainability, scalability).
   - Note mitigations for rejected options if they might resurface.
7. **Define Success Conditions & Test Strategy**
   - Translate success criteria into measurable checkpoints (CLI command output, diarization accuracy deltas, error budget).
   - Decide which test layers are necessary: targeted unit coverage, integration via `FluidAudioCLI`, or end-to-end audio runs.
   - Outline data and tooling needed (reference audio files, golden transcripts, log comparison scripts) and ensure availability without violating rules.
   - Plan how to validate concurrency or performance requirements (profiling, benchmarking harnesses, timing instrumentation).
8. **Implementation Strategy**
   - Break down into ordered work chunks that can be executed and validated independently.
   - Structure chunks into isolated deliverables: each should have a clear owner, inputs/outputs, review expectations, and demoable end state.
   - Include source files per step and expected code-level adjustments (new actor, config struct, CLI flag, etc.).
   - Consider concurrency, error handling, streaming behavior, and API surface consistency.
   - Capture outstanding tasks as TODO items with status (e.g., `[ ] perf benchmark`, `[ ] model warmup profiling`) so progress is transparent.
9. **Validation Plan**
   - Define how you will build confidence without adding new tests unless the user requests them; when tests are required, map them to the chosen strategy.
   - Outline manual workflows (audio samples to run, CLI commands, logging to inspect).
   - Specify regression risks and monitoring hooks (e.g., `AppLogger`, timing metrics).
10. **Risk & Rollback Analysis**
    - List failure modes (performance regressions, model load failures, memory spikes).
    - Provide rollback/feature flag strategies if applicable.
    - Call out any required coordination (model downloads, doc updates).
11. **Execution Checklist**
    - Pre-flight checks: models available, environment constraints understood.
    - Dependencies or blocking tasks resolved.
    - Communication artifacts (status updates, follow-ups) noted.

## Strategic Dimensions to Address

- **Performance Budgets**: Define latency, memory, and energy budgets up front; ensure plans track how each step impacts real-time ASR/VAD throughput on target hardware.
- **Resource Logistics**: Confirm model availability, download sizes, and CoreML compilation costs; decide how to stage large artifacts and prewarm caches or embeddings.
- **Observability Enhancements**: Identify logging, metrics, and tracing gaps; plan instrumentation so regressions surface quickly without noisy logs.
- **Concurrency & Isolation**: Validate actor boundaries, task cancellation behavior, and thread-hopping patterns; document how you prevent deadlocks, priority inversions, or data races.
- **Security & Privacy**: Check for surface areas handling user audio/text; ensure encryption, temporary file hygiene, and GDPR/CCPA considerations align with company policies.
- **Compatibility Matrix**: List supported OS versions, device classes, and architecture constraints; plan fallbacks or degradation strategies for unsupported environments.
- **Deployment & Rollout**: Decide on feature flags, CLI toggles, migration steps, and documentation updates; outline canary or staged rollout procedures when risk is high.
- **Coordination Points**: Identify stakeholders (research, infra, product) who must review decisions; schedule syncs and note dependencies on external deliverables.
- **Knowledge Transfer**: Plan for updating README, AGENTS.md, or architecture diagrams so future engineers inherit the context without guesswork.
- **Post-Delivery Follow-Up**: Define success metrics, monitoring windows, and retrospective checkpoints to capture learnings for the next iteration.

## Plan Quality Bar

- Concise yet thorough—capture reasoning, not code-level details.
- Reference facts with sources (file paths, docs) to ground assumptions.
- Spotlight ambiguities instead of guessing; propose how to eliminate them.
- Explicitly document trade-offs so future readers can revisit choices.
- Plans should be actionable for any senior engineer to execute without further clarification.

## Working Notes Etiquette

- Store the active plan in `.mobius/` with a descriptive filename (e.g., `.mobius/plan_offline_diarizer_refactor.md`); do not commit plan files.
- Update the plan after completing significant milestones or when assumptions change.
- Keep a changelog section if the plan evolves materially.

Treat the plan as the contract between intent and execution. Comprehensive planning shortens feedback loops, reduces risk, and makes implementation almost mechanical.
