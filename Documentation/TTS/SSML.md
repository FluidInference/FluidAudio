# SSML Tag Support

FluidAudio TTS supports a subset of Speech Synthesis Markup Language (SSML) tags for controlling pronunciation and text interpretation.

## Supported Tags

### `<phoneme>` - Custom Pronunciation

Override the automatic pronunciation with explicit phonemes.

```xml
<phoneme alphabet="ipa" ph="kəˈkɔɹo">Kokoro</phoneme>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `alphabet` | No | Phonetic alphabet (defaults to "ipa") |
| `ph` | Yes | Phoneme string in IPA notation |

**Examples:**
```xml
<!-- Custom pronunciation for a name -->
<phoneme ph="ˈniːkəl">Nicol</phoneme>

<!-- Technical term pronunciation -->
<phoneme ph="ˈdʒɪf">GIF</phoneme>

<!-- Foreign word -->
<phoneme alphabet="ipa" ph="bɔ̃ʒuʁ">bonjour</phoneme>
```

### `<sub>` - Text Substitution

Replace displayed text with spoken alternative.

```xml
<sub alias="World Wide Web">WWW</sub>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `alias` | Yes | Text to speak instead of content |

**Examples:**
```xml
<!-- Acronym expansion -->
<sub alias="Artificial Intelligence">AI</sub>

<!-- Abbreviation -->
<sub alias="Doctor">Dr.</sub>

<!-- Symbol replacement -->
<sub alias="at">@</sub>
```

### `<say-as>` - Interpret Content Type

Control how specific content types are pronounced.

```xml
<say-as interpret-as="cardinal">123</say-as>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `interpret-as` | Yes | Content type (see table below) |
| `format` | No | Type-specific format (e.g., date format) |

## Say-As Interpret Types

### `characters` / `spell-out`

Spell out each character individually.

```xml
<say-as interpret-as="characters">ABC</say-as>
<!-- Output: "A B C" -->

<say-as interpret-as="spell-out">hello</say-as>
<!-- Output: "h e l l o" -->
```

### `cardinal` / `number`

Read as a cardinal number.

```xml
<say-as interpret-as="cardinal">123</say-as>
<!-- Output: "one hundred twenty-three" -->

<say-as interpret-as="number">1000000</say-as>
<!-- Output: "one million" -->
```

### `ordinal`

Read as an ordinal number.

```xml
<say-as interpret-as="ordinal">1</say-as>
<!-- Output: "first" -->

<say-as interpret-as="ordinal">21</say-as>
<!-- Output: "twenty-first" -->

<say-as interpret-as="ordinal">100</say-as>
<!-- Output: "one hundredth" -->
```

### `digits`

Spell each digit individually.

```xml
<say-as interpret-as="digits">123</say-as>
<!-- Output: "one two three" -->

<say-as interpret-as="digits">1024</say-as>
<!-- Output: "one zero two four" -->
```

### `date`

Read as a date. Supports format attribute for date order.

| Format | Description | Example Input | Example Output |
|--------|-------------|---------------|----------------|
| `mdy` | Month-Day-Year (default) | `12/25/2024` | "December twenty-fifth twenty twenty-four" |
| `dmy` | Day-Month-Year | `25/12/2024` | "twenty-fifth December twenty twenty-four" |
| `ymd` | Year-Month-Day | `2024-01-15` | "twenty twenty-four January fifteenth" |
| `md` | Month-Day | `12/25` | "December twenty-fifth" |
| `dm` | Day-Month | `25/12` | "twenty-fifth December" |
| `y` | Year only | `2024` | "twenty twenty-four" |
| `m` | Month only | `12` | "December" |
| `d` | Day only | `25` | "twenty-fifth" |

```xml
<say-as interpret-as="date" format="mdy">12/25/2024</say-as>
<!-- Output: "December twenty-fifth twenty twenty-four" -->

<say-as interpret-as="date" format="dmy">25/12/2024</say-as>
<!-- Output: "twenty-fifth December twenty twenty-four" -->
```

### `time`

Read as time or duration.

**Clock time:**
```xml
<say-as interpret-as="time">2:30</say-as>
<!-- Output: "two thirty" -->

<say-as interpret-as="time">3:00</say-as>
<!-- Output: "three o'clock" -->
```

**Duration:**
```xml
<say-as interpret-as="time">1'21"</say-as>
<!-- Output: "one minute twenty-one seconds" -->
```

### `telephone`

Read as a telephone number (digit by digit).

```xml
<say-as interpret-as="telephone">555-1234</say-as>
<!-- Output: "five five five one two three four" -->

<say-as interpret-as="telephone">(555) 123-4567</say-as>
<!-- Output: "five five five one two three four five six seven" -->
```

### `fraction`

Read as a fraction.

```xml
<say-as interpret-as="fraction">1/2</say-as>
<!-- Output: "one half" -->

<say-as interpret-as="fraction">3/4</say-as>
<!-- Output: "three quarters" -->

<say-as interpret-as="fraction">2/9</say-as>
<!-- Output: "two ninths" -->

<!-- Mixed fractions -->
<say-as interpret-as="fraction">3+1/2</say-as>
<!-- Output: "three and one half" -->
```

## Usage with TtsManager

SSML tags are processed automatically when you call `synthesize()`:

```swift
import FluidAudioTTS

let ttsManager = TtsManager()
try await ttsManager.initialize()

// SSML is processed automatically
let text = """
    The price is <say-as interpret-as="cardinal">42</say-as> dollars.
    Call us at <say-as interpret-as="telephone">555-1234</say-as>.
    """

let audio = try await ttsManager.synthesize(text: text, voice: .afHeart)
```

## Coexistence with Markdown Syntax

SSML tags coexist with the existing markdown-style phonetic syntax:

```swift
// Both syntaxes work together
let text = """
    <phoneme ph="kəˈkɔɹo">Kokoro</phoneme> and [Misaki](/mɪˈsɑːki/)
    """
```

## Edge Cases

- **No SSML tags:** Text passes through unchanged (fast path)
- **Malformed tags:** Invalid SSML tags pass through as literal text
- **Unknown interpret-as:** Content is returned unchanged
- **Invalid numbers:** Content is returned unchanged
- **Empty content:** Handled gracefully

## Reference

Based on [W3C SSML 1.1](https://www.w3.org/TR/speech-synthesis11/) and [Amazon Polly SSML](https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html).

[Interactive IPA Chart](https://www.ipachart.com/#google_vignette)
