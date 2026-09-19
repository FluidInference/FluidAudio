# CUA-S1-FORMS native Mac demo

A standalone SwiftUI app that calls FluidAudio's `CuaS1FormsManager` directly.
Enter your own details once, then watch the real Core ML model match them to
three different forms and explain each decision with live option scores. No Python or browser server
is involved in the running app.

[![Two real Core ML variants driving separate WebKit forms](browser-demo.gif)](browser-demo.mp4)

[Watch the recorded browser demo](browser-demo.mp4) · [Browser action trace](Reports/browser-validation.json) · [Matched Swift benchmark](Reports/variant-comparison.json)

## Run

Requires an Apple silicon Mac with macOS 14+ and Swift 6.0+. Command Line Tools
are enough to build and run the app; XCTest needs a full Xcode installation.

From the FluidAudio repository root:

```bash
Examples/CuaS1FormsDemo/run.sh
```

The script builds in release mode and opens `Examples/CuaS1FormsDemo/.build/CUA Forms.app`.
The first launch downloads the 1.51 MB portable model package from the exact
[reviewed HF commit](https://huggingface.co/FluidInference/cua-s1-forms-coreml/tree/c87b915d302bdbff644709408d8fbd8c8effe894),
checks every file's SHA-256, and compiles it locally. Subsequent launches reuse
`~/Library/Application Support/FluidAudio/Demos/CuaS1Forms/`. This works while
the model PR is still open; it does not depend on a model being released on `main`.

To use an existing local package or compiled bundle without downloading:

```bash
Examples/CuaS1FormsDemo/run.sh --model /absolute/path/to/cua_s1_forms_fp16_options32.mlpackage
```

The optional `ane-gather/` package in the model PR uses the same interface and
works with `--model` too. It places 98.2% of operations on ANE on the measured
M5 Pro, but was about 6% slower than the default in a matched model-call check.
See the [ANE profile](../../Documentation/Decision/CuaS1Forms.md#ane-profile).

You can also open this example's `Package.swift` in Xcode and run the
`CuaS1FormsDemo` executable scheme, or use:

```bash
swift run --package-path Examples/CuaS1FormsDemo CuaS1FormsDemo
```

## Live browser agent: both variants

```bash
Examples/CuaS1FormsDemo/run.sh --browser
```

The native app hosts **two separate WKWebViews**. Enter source details or click
**Use example**, choose one of the three forms, and press **Run both agents**.
The driver observes each live DOM label, role, form title, value, and checkbox
state; combines it with the supplied task; calls the actual Core ML model;
then executes the selected fill/check action and independently reads the DOM
back. Text changes dispatch input/change events; checkboxes receive clicks.
The outlined field and action card show the actual model decision. **Step**
processes one control in each browser. **Stop** prevents the next action.

Each browser's HTML is generated from the original control catalog, without
source values or answer keys. The agent does not read a field-to-value mapping.
A stale DOM observation is rejected before writing. Model-selected button clicks
are shown for review; only a user's explicit click creates a local receipt.
There is no external navigation or submission. The source editor stays in memory,
and untouched examples follow the selected form just as in the SwiftUI mode.
The displayed option score is not calibrated confidence.

Both portable packages are downloaded once and checked against pinned hashes.
The optional export is pinned to [f66dd2a](https://huggingface.co/FluidInference/cua-s1-forms-coreml/tree/f66dd2af1ee94f359b1e65305d35540263d4a2fe/ane-gather).
To supply both locally, add `--model /path/to/baseline.mlpackage`
and `--ane-model /path/to/ane-gather.mlpackage`.

The recorded run checks **100/100 labeled decisions**: 50 controls for each
variant. Both fill 14 patient, 10 job, and 12 insurance text fields, dispatch the
expected input/change events, preserve skipped fields, and never submit
implicitly. Separate integration checks reject stale observations and verify
explicit local button clicks. The [complete trace](Reports/browser-validation.json)
contains actual contexts, candidates, choices, event counts, and DOM-verified
actions from public examples only.

This demonstrates bounded browser form automation in a Swift app. It does not
inspect arbitrary desktop apps, parse PDFs, or establish accuracy on unseen forms.
The original 196-row parity check remains the conversion validation; this demo
adds evidence that the observe/decide/act loop works in a real browser engine.

## Matched Swift benchmark

Measured on an Apple M5 Pro, 24 GB, macOS 27.0 (26A428), release Swift, with
`cpuAndNeuralEngine`. Both models stay loaded. Each receives a 50-control warmup,
then ABBA blocks with two full passes per block: **200 timed calls per variant**.
The timer covers `CuaS1FormsManager.score`, including Swift byte encoding and
output decoding. No browser, animation, screenshot capture, or model loading
is included. Correctness checks occur outside the timer; no slow calls are removed.

| Export | Correct decisions | Median | p95 | ANE / CPU operations¹ |
| --- | ---: | ---: | ---: | ---: |
| Original | 200/200 | 0.912 ms | 0.933 ms | 149 / 24 |
| ANE gather | 200/200 | 0.961 ms | 0.984 ms | 162 / 3 |

ANE gather is about **5.4% slower by median call time** in this run. More ANE
operations did not improve latency. Keep the original as the default.

¹ Placement is from the earlier [compute-plan profile](../../Documentation/Decision/CuaS1Forms.md#ane-profile)
on the same machine and exact artifacts, not a utilization or power measurement.
The full benchmark report records file hashes, raw samples, per-form summaries,
load times (600/583 ms), and first calls (2.29/1.55 ms); loads can use system caches.
This small fixture comparison is exploratory, not a held-out generalization score.
Live UI call times can be higher because rendering and scheduling are active.

```bash
swift run --package-path Examples/CuaS1FormsDemo -c release CuaS1FormsDemo \
  --benchmark --report /absolute/path/to/variant-comparison.json \
  --hardware "Describe the measured Mac"
```

Add the same `--model` and `--ane-model` overrides to measure local packages.
For reproducible provenance, benchmark inputs must be portable `.mlpackage` files.

To capture the actual browser run with public examples, use an empty directory:

```bash
Examples/CuaS1FormsDemo/run.sh --browser \
  --showcase /absolute/path/to/capture --exit-after-showcase
ffmpeg -framerate 4 -i /absolute/path/to/capture/frame-%04d.png \
  -vf "scale=1440:-2" -c:v libx264 -crf 20 -pix_fmt yuv420p \
  -movflags +faststart browser-demo.mp4
```

The recording is a paced sequence of actual app/WebKit snapshots, held at four
frames per second to make actions readable. Capture is separate from benchmarking.
Only this app is captured; screen-recording permission is unnecessary. The output
also contains per-form screenshots and `browser-validation.json`. A failed
verification writes `failure.txt` instead of a success report.

## Try it

1. Enter your details in the left panel. It starts empty; **Use example** loads the
   selected form's public sample data for a quick trial. The panel names the sample.
2. Choose **Patient registration**, **Job application**, or **Auto insurance claim**.
   Untouched examples automatically change to match the selected form. Once you edit,
   add, or remove a detail, your profile stays the same across forms, so you can see
   how the model matches different field labels to the same information. Press
   **Use example** again to return to the selected form's sample.
3. Press **Fill form** for a paced pass, or **Step** for one live prediction.
4. Inspect the selected option, candidate probabilities, exact input context, and
   measured time. Click a field or a past decision to inspect it again.
5. Edit a target field and press **Recheck form** after a complete pass. The model
   sees current field values and can choose `skip` for already-filled fields.
6. Press **Submit demo** to produce a local receipt. A model-selected `click`
   remains a proposal; the scoring loop never submits the form itself.

Each example includes data for its form: patient contact, address, insurance, and
emergency contact details; job contact details, LinkedIn, employer, title,
experience, salary, start date, and cover letter; or insurance policy, vehicle,
incident, and claim details. The examples fill 14, 10, and 12 text fields respectively.
Promo, referral, and agent codes are left empty as specified by the original tasks.

Use **Add another detail** for a labeled value such as insurance provider or
current employer. Blank values are excluded from the candidates. An explicit
first and last name also produce a combined full-name choice, unless a full name
is already supplied. The model sees only these current choices plus check/click/skip;
there is no fallback to hidden sample information.

**Stop** cancels a sequence. **Reset** clears the target form and keeps your details.
Editing, adding, or removing source details clears old predictions and resets the
target preview. **Clear** removes personal information from the session. Entered
values stay in memory; the app does not save them or send them over the network.
The 32-option and 96-byte-per-option model limits are validated before inference.

Animation uses a 240 ms presentation delay per control; displayed scoring times
exclude that delay and include the Swift manager call. The default SwiftUI mode uses local controls. Browser automation is available in
the explicit `--browser` mode; neither mode submits to an external service.

## What this demonstrates

The default mode contains native SwiftUI controls and changes them using actual model
predictions. Its target form descriptions come from the original Cua demo. Source candidates
come from the editable profile. **Use example** preserves the original document
entities and distractors for reproducible sample checks.
Answer labels are not decoded by the UI catalog or passed to inference; only the
verification command reads them to score correctness.

This is a bounded example of integrating the decision component. It does not
parse a PDF or inspect arbitrary applications. Users enter labeled values themselves; the optional examples supply the original
pre-extracted document entities. These three sample forms do not establish
accuracy on new forms, new languages, or real-world automation tasks.

## Validation

A verification mode uses the same controller and real Swift model manager:

```bash
swift run --package-path Examples/CuaS1FormsDemo CuaS1FormsDemo --verify
```

It checks the explicitly selected initial-form rows **0–17, 68–82, and 130–146**:
50 decisions across the three upstream forms, plus a filled-field recheck,
single-step execution, cancellation/reset, and explicit local submission.
A local run selected all 50 labeled options correctly and passed the state checks.
The check loads an example once, then switches forms to verify that each form gets
its own sample. Unchanged editor updates keep examples following forms; edited
profiles persist, and clearing details prevents samples from returning automatically.
An additional pass enters four public sample values into a blank profile and reuses
that profile across all three forms: supplied names/email/phone are filled, missing
values remain empty, and source edits invalidate the old choices and predictions.
These changed-profile checks are demo regressions, not the original benchmark score.
`--model /absolute/path/to/model.mlpackage` also works in verification mode.

Unit tests cover the pinned fixture, candidate retention, context updates,
fill-value parsing, checkbox/click semantics, invalid-action rejection, empty-profile
handling, source-only candidates, name combination, option/byte limits, example
switching, and preserving edited profiles:

```bash
swift test --package-path Examples/CuaS1FormsDemo
```

The repository CI builds the native demo and runs these tests with Xcode. Locally,
the standalone `--verify` mode can run even when XCTest is unavailable.

For visual regression checks in a logged-in Mac session, the app can save only
its own content view, without screen-recording permission:

```bash
Examples/CuaS1FormsDemo/run.sh \
  --snapshot /absolute/path/to/demo.png --snapshot-filled --exit-after-snapshot
```

Add `--snapshot-form job-application` or `--snapshot-form auto-claim` to check
switching from the patient example to either other form before filling it.

## Source and license

`Sources/CuaDemoCore/Resources/demo.jsonl` is the unmodified MIT-licensed upstream
[demo dataset](https://huggingface.co/datasets/cua-ai/cua-s1-forms/tree/8273f34778b99ac2e12d9f6e7d57dad99ae20845).
Its byte hash is `4f43b442e79ba2e2ce731e27e9b8e340c2b5dfcaffc92d8ff564c34f115ff1ca`.
The bundled source fixture has 196 rows, but the interactive app uses only the
50 original controls before browser chrome and the second UI state. Dataset
revision and hash are validated when the app loads. See [UPSTREAM-LICENSE](UPSTREAM-LICENSE).

The model, model license, and conversion provenance are in
[HF model PR #1](https://huggingface.co/FluidInference/cua-s1-forms-coreml/discussions/1).
Model binaries are downloaded into the local cache and are not committed here.
See [the manager guide](../../Documentation/Decision/CuaS1Forms.md) and
[Mobius PR #97](https://github.com/FluidInference/mobius/pull/97) for the complete
196-row conversion parity checks and reproducible export pipeline.
