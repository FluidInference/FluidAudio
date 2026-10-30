# Explicit sliding-window token revisions

`SlidingWindowAsrManager.tokenUpdates` is an opt-in throwing stream of raw
`TokenTiming` edits. It does not change model weights, decoder decisions,
vocabulary rescoring, or the legacy `transcriptionUpdates` API. Subscribe once,
before feeding audio, and use a new manager for each job.

Each event identifies its job and contiguous ordinal. Replace everything after
`retainedTokenCount` in the previous token sequence, then append `tokens`.
Only positions before `stableTokenCount` are irrevocable. The producer uses the
existing decoder's trailing-word retirement boundary; confidence confirmation
is not an immutability guarantee. The sample ranges describe the absolute
16 kHz window and center that produced the edit, not timestamp deduplication
thresholds. Token IDs, pieces and timings remain one-to-one, in decoder order. They include
SentencePiece `<unk>` (id 0; Bark 3.0 emits it for quotation marks), which the display text omits.
This is raw-token authority, not vocabulary-rescored display text.

A consumer can commit the newly stable prefix exactly once and replace its
provisional suffix. Equal timestamps, subwords and repeated words are not
identity heuristics. Empty events still carry authority and must be processed.

Successful `finish()` publishes one terminal event, even for empty input or
when the last inference emitted no tokens. The terminal event commits the
remaining suffix and closes the stream. EOF alone is not permission to commit.
`cancel()` throws cancellation on this stream without publishing a terminal.
Conversion, inference or publication failure closes the strict stream with an
error; subsequent legacy recovery cannot resume its lineage. A second or late
subscriber cannot steal or reconstruct the missing prefix. Strict streams are
single-use: legacy reset does not reopen them, and a job that already accepted
audio, finished or cancelled cannot acquire a new strict subscriber.

The contract suite runs with the package tests:

```sh
swift test --filter SlidingWindowTokenRevisionTests
```

It characterizes the actual manager publication boundary, cursor validation,
retirement, empty drain, failed publication, cancellation and subscriber
ownership without downloading models. Real-model and consumer qualification
remain necessary; these tests do not establish recognition quality or latency.
