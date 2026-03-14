# Kokoro English Voice Quality Report

Quality assessment for all 28 English Kokoro TTS voices (11 AF, 9 AM, 4 BF, 4 BM). Includes upstream Kokoro grades, training data, pitch analysis, character traits, and our own quality ratings. Voice descriptions sourced from VoiceRankings, grades from Kokoro VOICES.md.

## Key Findings

- Only af_heart (A) and af_bella (A-) are top-graded, but both have sibilance issues
- Male voices have fewer sibilance problems but several have subtle background noise
- British English voices are the strongest group overall, with bf_emma rated "quite good"
- Kokoro's own grades don't always match perceived quality (e.g. am_adam graded F+ but sounds usable)

## Methodology

**Test Sentence:** "The quick brown fox jumps over the lazy dog near the riverbank, while the morning sun casts golden light across the meadow."

**Data Sources:**
- Upstream grades from [Kokoro VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
- Voice descriptions from [VoiceRankings](https://voicerankings.com)

**Quality Scale:**
- **Unusable**: Significant artifacts, distortion, or background noise that interferes with comprehension
- **Slightly unusable**: Noticeable quality issues (sibilance, noise) that may distract but don't prevent understanding
- **Borderline usable**: Minor quality concerns; acceptable for some use cases
- **Usable**: Good quality with minimal issues; suitable for most applications
- **Quite good**: High quality with no significant artifacts

## American English Female (af_*)

| Voice | Kokoro Grade | Training Data | Pitch | Character | Our Quality | Notes |
|-------|-------------|---------------|-------|-----------|-------------|-------|
| af_heart | A | — | 200 Hz, High | Warm, conversational, breathy, relaxed. Young adult (20s-30s). Energy 6/10 | Slightly unusable | Sibilant, stronger than af_bella |
| af_bella | A- | 10-100 hours | 205 Hz, High | Natural, intimate, slightly husky, vocal fry. Young adult. Energy 6/10 | Slightly unusable | Subtle sibilant |
| af_nicole | B- | 10-100 hours | 171 Hz, Medium-High | Whisper-soft, deeply intimate, ASMR-like, very breathy. Mature adult (30-50). Very slow. Energy 2/10 | Usable | |
| af_aoede | C+ | 1-10 hours | 184 Hz, Medium-High | Warm, velvety, intimate, empathetic. Young adult. Mild breathiness, subtle vocal fry. Energy 6/10 | Slightly unusable | |
| af_kore | C+ | 1-10 hours | 160 Hz, Medium | Warm, empathetic, calm, professional. Mature adult (30-50). Subtle vocal fry. Energy 6/10 | Borderline usable | |
| af_sarah | C+ | 1-10 hours | 200 Hz, High | Polite, approachable, clean, mild breathiness. Young adult. Energy 6/10 | Usable | |
| af_alloy | C | 10-100 min | 146 Hz, Medium | Bright, cheerful, crisp, minimal breathiness. Mature adult. Fast (192 WPM). Energy 6/10 | Unusable | |
| af_nova | C | 10-100 min | 164 Hz, Medium | Polished, professional, clear, no breathiness. Young adult. Fast (194 WPM). Energy 6/10 | Unusable | Notable background noise, subtle sibilant |
| af_sky | C- | 1-10 min | 159 Hz, Medium | Professional, smooth, consistent. Mature adult. Mild breathiness. Fast (184 WPM). Energy 6/10 | Unusable | Distortion, background noise |
| af_jessica | D | 10-100 min | 223 Hz, High | Bright, energetic, breathy, fast (206 WPM). Young adult (20s-30s). Energy 8/10 | Unusable | Noticeably low quality |
| af_river | D | 10-100 min | 177 Hz, Medium-High | Relaxed, textured, casual, vocal fry, no breathiness. Young adult. Fast (208 WPM). Energy 8/10 | Usable | |

## American English Male (am_*)

| Voice | Kokoro Grade | Training Data | Pitch | Character | Our Quality | Notes |
|-------|-------------|---------------|-------|-----------|-------------|-------|
| am_fenrir | C+ | 1-10 hours | 145 Hz, Medium | Confident, energetic, velvety, upbeat. Young adult (20s-30s). Energy 6/10 | Usable | |
| am_michael | C+ | 1-10 hours | 126 Hz, Low | Deep, warm, grounded, calm, authoritative. Mature adult (30-50). Deliberate pace (158 WPM). Energy 5/10 | Usable | |
| am_puck | C+ | 1-10 hours | 125 Hz, Low | Youthful, enthusiastic, bouncy, eager. Young adult (20s-30s). Energy 8/10 | Slightly unusable | Subtle background noise |
| am_echo | D | 10-100 min | 114 Hz, Low | Soft, intimate, breathy baritone, gentle, reassuring. Young adult. Energy 6/10 | Slightly unusable | Subtle background noise |
| am_eric | D | 10-100 min | 157 Hz, Medium | Relaxed, modern, slight huskiness, authentic. Young adult. Fast (207 WPM). Energy 8/10 | Slightly unusable | Subtle background noise |
| am_liam | D | 10-100 min | 134 Hz, Medium | Cheerful, articulate, crisp, minimal breathiness. Young adult (20s-30s). Very fast (201 WPM). Energy 6/10 | Usable | |
| am_onyx | D | 10-100 min | 91 Hz, Very Low | Deep, resonant, patient, calm. Mature adult (30-50). Energy 6/10 | Slightly unusable | Subtle background noise |
| am_adam | F+ | 1-10 hours | 116 Hz, Low | Polished, trustworthy, neighborly, clean, no breathiness. Mature adult (30-50). Fast (184 WPM). Energy 6/10 | Usable | |
| am_santa | D- | 1-10 min | 162 Hz, Medium | Rich, jolly, theatrical, grandfatherly. Senior. Energy 6/10 | Slightly unusable | Subtle background noise |

## British English Female (bf_*)

| Voice | Kokoro Grade | Training Data | Pitch | Character | Our Quality | Notes |
|-------|-------------|---------------|-------|-----------|-------------|-------|
| bf_emma | B- | 10-100 hours | 187 Hz, Medium-High | Polished, inviting, clear, friendly. Young adult. Fast (185 WPM). Energy 6/10 | Quite good | |
| bf_isabella | C | 10-100 min | 214 Hz, High | Warm, articulate, gentle breathiness, lively. Young adult. Energy 7/10 | Usable | Some sibilant |
| bf_alice | D | 10-100 min | 218 Hz, High | Polished, articulate, RP accent, glass-like texture. Young adult (20s-30s). Energy 6/10 | Unusable | Quality quite bad |
| bf_lily | D | 10-100 min | 196 Hz, Medium-High | Polished, articulate, calm, subtle vocal fry. Young adult. Energy 5/10 | Usable | |

## British English Male (bm_*)

| Voice | Kokoro Grade | Training Data | Pitch | Character | Our Quality | Notes |
|-------|-------------|---------------|-------|-----------|-------------|-------|
| bm_fable | C | 10-100 min | 124 Hz, Low | Refined, velvety, storytelling cadence, intimate, sophisticated. Young adult. Energy 6/10 | Unusable | Sibilant too strong |
| bm_george | C | 10-100 min | 138 Hz, Medium | Distinguished, polished, reassuring, calm, authoritative. Mature adult. Steady (165 WPM). Energy 5/10 | Usable | |
| bm_lewis | D+ | 1-10 hours | 102 Hz, Low | Deep, sophisticated, calm, composed. Mature adult. Energy 4/10 | Usable | |
| bm_daniel | D | 10-100 min | 121 Hz, Low | Crisp, articulate, modern, professional. Young adult. Fast (195 WPM). Energy 6/10 | Usable | |
