# Kokoro English pronunciation lookup and G2P fallback

Kokoro's English frontend uses pronunciation overrides and the Misaki lexicon before
falling back to the BART grapheme-to-phoneme (G2P) model. A 40-word local comparison
illustrates why: the dictionary handles several irregular spellings that forced
BART inference mispronounces, while BART supplies pronunciations for dictionary misses.
Returning a pronunciation does not establish that it is correct.

This is a selected word-level diagnostic, not an overall accuracy or TTS quality
benchmark. It compares **dictionary-only lookup with forced BART inference**.
It does not compare the current complete frontend against BART: current `main`
already adds initialism, compound and possessive rules before fallback.

## Results

| Measure | Result |
| --- | --- |
| Words selected before inference | 40 |
| Dictionary entries found | 34 of 40 |
| Nonempty BART outputs | 40 of 40 |
| Identical raw outputs on dictionary-covered words | 25 of 34 |
| Words with a CMU reference | 35 of 40 |
| BART inference failures | 0 |

The six dictionary misses were `API`, `CPU`, `GPU`, `GitHub`, `Neuralink` and
`Kokoro`. These are misses in the tested lexicon cache, not necessarily in every
Misaki release. Full inputs, outputs, reference alternatives and artifact hashes
are in [the result snapshot](Results/KokoroEnglishG2P.json).

Both methods returned the same phonemes for words such as `phone`, `world`,
`people`, `water`, `computer`, `yacht`, `choir`, `queue`, `island`, `debt` and
`algorithm`. Raw agreement is not accuracy; valid stress or reduced-vowel
differences can also produce different strings.

### Selected differences

These are actual output symbols, not listening-test transcriptions. Misaki uses
`A` for /eɪ/, `I` for /aɪ/ and `O` for /oʊ/. The `ˈ` mark denotes primary stress.

| Word | Dictionary-only output | Forced BART output | Reference comparison |
| --- | --- | --- | --- |
| colonel | `kˈɜɹnᵊl` | `kˈɑlənᵊl` | Dictionary matches CMU's “kernel” pronunciation; BART introduces an extra syllable and changes the stressed vowel. |
| Wednesday | `wˈɛnzdˌA` | `wˈɛdnzdˌA` | BART inserts a `d` absent from both CMU alternatives. |
| inference | `ˈɪnfəɹəns` | `ɪnfˈɪɹəns` | BART shifts primary stress and changes the middle vowel relative to CMU. |
| NASA | `nˈæsə` | `nˈɑsə` | BART changes the stressed vowel relative to CMU. |
| API | Missing | `ˈæpi` | BART does not produce CMU's A-P-I initialism. |
| CPU | Missing | `spjˈu` | BART does not produce CMU's C-P-U initialism. |
| GitHub | Missing | `ɡˈɪθʌb` | BART produces /θ/ rather than CMU's /t h/ sequence. |

References come from the [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict/tree/74790861f652b15e4ac49015a90074ad62a27690),
maintained by Carnegie Mellon University's Speech Group (see the retained
[CMUdict license](Results/CMUDict-LICENSE.txt)). All listed alternatives
are retained in the snapshot. Its lowercase `ai` entry includes both /aɪ/ and
the letter-name reading, so acronym intent must be considered separately.

### What current callers receive

The [current English frontend](../../Sources/FluidAudio/TTS/KokoroAne/G2P/English/KokoroAneEnglishPhonemizer.swift)
does more than raw dictionary lookup:

- Custom pronunciations take precedence.
- Explicit letter-name overrides handle uppercase `AI` and `US`.
- Known words and acronyms such as `NASA` resolve from the lexicon.
- Unknown ASCII all-caps tokens of two to five letters are spelled using
  per-letter lexicon entries, when those entries are available.
- Compound and possessive handling precedes BART fallback.

Consequently, the forced-BART `API` and `CPU` rows are **not evidence that current
normal text synthesis mispronounces those initialisms**. With the required letter
entries loaded, the current rules also cover `GPU`. This behavior comes from code
inspection; the 40-word experiment did not run the current full frontend.

For an uncovered application-specific name, use `setEnglishCustomLexicon(_:)` with
the intended pronunciation in Kokoro's supported phoneme vocabulary. Keep BART as
coverage for unresolved words rather than assuming every returned sequence is correct.

## Method

The frozen selection contains ten common words, ten irregular spellings, ten technical
words and ten names/acronyms. One real BART prediction was made per word; no TTS
waveforms were synthesized. Execution used an Apple M5 Pro, macOS 27.0, Swift 6.2.3
debug build and the production CPU-only G2P configuration.

The source checkout was `c9cf87a6e458cf0afb34aabdfa2b1faede1f7360`, with unrelated
working-tree changes. The three evaluated source files matched that commit:

1. `LexiconAssetCache` loaded the actual cached `us_lexicon_cache.json`, filtering
   entries to the English `ANE/vocab.json` token set.
2. That revision's `KokoroAneEnglishPhonemizer` performed dictionary lookup with no
   custom overrides and a fallback returning `nil`. A failed lookup was recorded
   as missing, rather than using a generated pronunciation as a reference.
3. `G2PModel.shared.phonemize(word:)` ran for every word, using the same lowercase
   normalization as the deployed fallback, including for uppercase inputs.

At comparison base `0b0fa2ad710843d3f40af885caf9a86d1fa49c26`, `G2PModel.swift`
is byte-identical to the evaluated file. The English frontend has changed since
the evaluated revision, as described above. Cached asset bytes are identified by
SHA-256; their upstream release revision was not established.

To repeat the comparison, freeze a word manifest, load matching lexicon/vocabulary
and model assets, run dictionary-only lookup and forced BART on each word, and retain
both outputs. Use an internal debug test/probe to access these types; this is not a
new public benchmark command. The existing `g2p-benchmark` command evaluates the
separate Charsiu ByT5 model and does not reproduce this English BART comparison.

## Limits and interpretation

- Coverage and raw agreement are reported separately. No phoneme error rate or
  overall pronunciation accuracy percentage was computed.
- Words were deliberately selected, not sampled from measured user traffic.
  Training-data overlap with Misaki, BART or CMU was not ruled out.
- `quantization`, `spectrogram`, `GPU`, `Neuralink` and `Kokoro` have no reference
  in the pinned CMU dictionary. Nonempty output does not validate their pronunciation.
- Word-level results do not measure sentence context, homographs, connected-speech
  prosody, intelligibility of generated audio, naturalness or voice identity.
- Differences may originate in the trained model, conversion or decoding. The
  original PyTorch checkpoint was not evaluated, so this is not a claim about
  BART architectures in general or a demonstrated conversion defect.
- Single debug-call timings do not support a latency ranking. For text-to-audio
  measurements, see [TTS Benchmarks](Benchmarks.md).
