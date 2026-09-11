# UI Design Requirements — Acoustic Key Transfer

A web tool that transfers short secrets (API keys, tokens, ≤120 bytes) between
two devices **through sound** — device speaker → air → device microphone, and
optionally through a phone call held between the devices. No network, no
server, no accounts: the only channel is audio, and nothing leaves the two
devices except sound.

The underlying modem "sings": data is encoded as vowel-like tones on a D
lydian arpeggio with vibrato. It is pleasant and distinctive — the UI should
lean into this (the sound IS the brand) rather than hide it.

## Core principle

**The receiver either shows the exact payload or an explicit failure — never
a corrupted one** (CRC-verified under the hood). The UI must never imply
partial success. Robustness comes from repetition: the sender loops the
transmission until the receiver confirms; the receiver merges information
across repeats, so every additional loop strictly increases success odds.

## Roles

One screen, two big entry actions: **Send** and **Receive**. No setup, no
pairing step.

### Send flow

1. **Input**: single text field for the secret. Masked by default with a
   show/hide toggle. Max 120 bytes (UTF-8) — live counter, hard stop with
   clear message beyond it.
2. **Fingerprint display**: after input, show a short human-comparable
   fingerprint of the payload (8 hex chars derived from its CRC-32, or an
   equivalent word/emoji encoding). The receiver shows the same fingerprint
   on success; users cross-check visually. This is the trust moment — make it
   prominent on both sides.
3. **Transmit**: one large Start button → the device plays the transmission
   **in a loop** (repeats separated by ~500 ms of silence) until the user
   stops. Display:
   - copy counter ("transmission 3…")
   - a playhead/progress indicator within the current copy
   - estimated duration per copy (see timing table)
   - volume slider (maps to modem volume 0–100; default ~50)
4. **Stop** is always available. Sender has no feedback channel — the
   *receiving human* says "got it"; design for that social loop (e.g. large
   states readable across a desk).
5. Guidance copy (shown compactly): quiet room helps; devices within ~1 m or
   phone held near the speaker; don't cover the speaker.

### Receive flow

1. **Mic permission**: request with `echoCancellation: false`,
   `noiseSuppression: false`, `autoGainControl: false`. If the platform
   ignores these, proceed anyway (the modem tolerates processing) but do not
   surface scary warnings.
2. **Level meter**: always-visible input meter with three zones — too quiet /
   good / clipping — plus a one-line hint ("move closer", "lower the sender's
   volume"). This is the #1 user-fixable failure cause.
3. **States** (drive distinct, glanceable visuals):
   - **Listening** — idle animation; suggestion to start the sender.
   - **Receiving** — the moment the signal locks (preamble detected), switch
     to an active state with progress. Progress can be time-based against the
     expected copy duration.
   - **Between copies** — after a failed copy, show "copy N didn't decode
     cleanly — combining with the next repeat…" with an attempts counter.
     This state is *normal operation*, not an error. Never make the user
     restart; the receiver keeps listening and merging automatically.
   - **Success** — the decoded secret (masked, with reveal toggle), a copy
     button, and the fingerprint for cross-checking with the sender. Strong
     positive cue (visual + optional short sound).
   - **Stuck** — after ~5 failed copies, keep listening but surface
     escalating guidance: raise volume, reduce distance, quiet the room; for
     phone-call transfers: disable the call app's noise suppression if
     possible, use the loudest earpiece/speaker.
4. Decoded secrets stay client-side; provide an explicit Clear/Done that
   wipes the value from the page.

## Timing expectations (design the states around these)

Per-copy air time at the transfer profile (~100% error-correction overhead):

| payload | per copy |
|---|---|
| 20 bytes | ~13 s |
| 40 bytes | ~23 s |
| 64 bytes | ~34 s |
| 120 bytes | ~61 s |

Formula: `(8·len + 60) × 60 ms`. Clean channels succeed on copy 1; rough
channels typically need 2–3 copies (the receiver's cross-copy merging needs
3+ for the hardest cases). The UI should feel calm about multi-minute
transfers — this is a deliberate robustness-over-speed design.

## Audio integration contract (for the WASM layer, port pending)

The UI should be built against this interface; audio plumbing runs at
**48 kHz mono f32** (request `sampleRate: 48000` on the AudioContext;
resample capture if the platform refuses):

- `transfer_encode(payload: bytes, volume: u8) -> Float32Array` — one copy of
  the transmission; the UI handles looping and gaps (or
  `transfer_encode_repeated(payload, volume, copies, gap_ms)`).
- `TransferRx.ingest(chunk: Float32Array) -> bytes | null` — feed mic audio
  continuously (AudioWorklet); returns the verified payload exactly once on
  success.
- `TransferRx.attempts() -> number` — failed-copy counter for the UI.
- Constants: `MAX_TRANSFER_PAYLOAD = 120`, sample rate 48000.

## Nice-to-have (post-MVP)

- Live spectrogram or stylized "singing notes" visualization while
  sending/receiving (the melody is real — D3/G#3/C#4 arpeggio around
  147–277 Hz fundamentals, energy concentrated 1000–2500 Hz).
- A "test tone" button on Receive that plays a sample transmission from a
  bundled clip so users can verify their setup solo.
- Phone-call mode preset: a dedicated help sheet for the two-humans-on-a-call
  scenario.

## Non-goals

- **No encryption/PIN UI.** Deliberate decision: same-room threat model;
  anyone recording the audio is in scope of physical security, not this tool.
  Do not add lock icons or security theater implying otherwise — the honest
  claim is "verified exact transfer", not "private transfer".
- No history/persistence of transferred secrets.
- No server communication of any kind.
