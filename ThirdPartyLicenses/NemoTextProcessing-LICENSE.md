# NemoTextProcessing (text-processing-rs)

The `NemoTextProcessing` binary target is the prebuilt xcframework of
[`text-processing-rs`](https://github.com/FluidInference/text-processing-rs)
(v0.3.0), used by `NemoTextNormalizer` for byte-exact NeMo text normalization.

- **Source:** https://github.com/FluidInference/text-processing-rs
- **License:** Apache License, Version 2.0

Built with the `fst-engine` feature, the xcframework includes and statically
links the following third-party works. All are permissive (Apache-2.0 / MIT)
and compatible.

## NVIDIA NeMo Text Processing

The compiled weighted-FST grammars and text-normalization fixtures are derived
from / copied from NVIDIA NeMo Text Processing (pinned commit
`1f1263579fe57ba7ed783cad3dddee710fcc5064`).

- **Source:** https://github.com/NVIDIA/NeMo-text-processing
- **License:** Apache License, Version 2.0
- **Copyright:** Copyright (c) NVIDIA CORPORATION & AFFILIATES.

## rustfst

Loads and executes the compiled OpenFST grammars.

- **Source:** https://github.com/Garvys/rustfst
- **License:** MIT OR Apache-2.0
- **Copyright:** Copyright (c) Alexandre Caulier and the rustfst contributors.

## flate2

Decompresses the bundled gzipped grammars at load time.

- **Source:** https://github.com/rust-lang/flate2-rs
- **License:** MIT OR Apache-2.0
- **Copyright:** Copyright (c) Alex Crichton and the flate2 contributors.

## Other Rust dependencies

The xcframework transitively links additional permissive-licensed Rust crates
(MIT and/or Apache-2.0) via `rustfst` and `flate2` — e.g. `nom`,
`miniz_oxide`, `bitflags`, `anyhow`. See the crate's `THIRD-PARTY-LICENSES.md`
for detail.

Full texts: Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>,
MIT <https://opensource.org/license/mit>.
