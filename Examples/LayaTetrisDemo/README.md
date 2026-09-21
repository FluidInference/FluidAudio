# laya plays Tetris (macOS)

SwiftUI app that exercises `LayaManager` from the parent package: every legal landing of the
current piece is described in one sentence, laya answers *"Is this a clean placement?"* on the
Neural Engine, and the landing with the highest P(true) is played. The console lists each scored
sentence with its probability and per-call latency; the tiles show milliseconds per decision,
decisions per minute, lines, and the bucket in use.

```bash
cd Examples/LayaTetrisDemo
swift run -c release LayaTetrisDemo
```

No Xcode project is needed. **Load model** downloads the 128-token bucket and tokenizer from
`FluidInference/laya-coreml` (614 MB + 34 MB) on first use; set `LAYA_MODEL_DIR` to a directory
holding the bundles to skip the download. **Play** scores every landing on the Neural Engine (about 5 ms per decision inside the app on an M5 Pro,
3.8 ms from the CLI without UI updates) and pauses briefly after each placed piece; the *delay per scored landing* slider slows the
scoring down so each candidate can be watched being evaluated on the board (orange outline),
and the chosen landing is drawn in green.

**Policy** switches to a feature-weighted heuristic or random play for comparison. Zero-shot laya
clears a few dozen lines before topping out; the heuristic plays indefinitely. The demo is about
decision latency on device, not Tetris skill.

Headless smoke test (used by CI-less verification):

```bash
LAYA_DEMO_AUTORUN=1 LAYA_DEMO_QUIT_AFTER=20 swift run -c release LayaTetrisDemo
```

The simulation is the same code as `fluidaudiocli laya-tetris`; the app keeps its own copy in
`TetrisSimulation.swift` because examples only link the library product.
