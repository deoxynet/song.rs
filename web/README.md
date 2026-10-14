# AKT — Acoustic Key Transfer (web)

Static web UI over the `song-rs` secret-transfer profile, compiled to WASM.
Transfers short secrets (≤120 bytes UTF-8) between two devices through sound.
No framework, no bundler, no server code — plain ES modules with a dedicated
classic worker for decoding.

Single-screen layout with two explicit, side-by-side roles: **Send** and
**Listen**. The receiving device starts microphone capture from a clear user
action, then keeps listening and merging incomplete copies automatically.
A verified result appears below both roles and persists until cleared.
Internal decoder states become calm, human-readable status copy rather than
screens or protocol state ids. Mic denied → the device remains sender-capable,
with a contextual retry action.

Debug hook: open with `?debug=1` and drive `window.__akt`
(`setPayload`, `setLine`, `setPlaying`, `setAttempts`, `setMicDenied`,
`injectLevel`, `showResult`, `clearResult`) to force zone contents without
audio or mic — tolerates a missing wasm build.

## Run

The page must be served over HTTP (ES modules + WASM + AudioWorklet do not work
from `file://`). From this directory:

```bash
python3 -m http.server 8080
# then open http://localhost:8080
```

Any static server works (`npx serve`, `caddy file-server`, …). Use
`localhost` or HTTPS — microphone capture and the clipboard API require a
secure context.

## Rebuild the WASM

Build both bindings from the same Rust source. From the repo root:

```bash
node web/build.mjs
# or, from web/: npm run build:wasm
```

This runs wasm-pack with `--target web` into `pkg/` for encoding/fingerprints,
and `--target no-modules` into `pkg-worker/` for background reception. Serve
both generated directories alongside the JS sources. Building only `pkg/`
leaves sending available but causes reception to report a decoder startup
failure. Additional wasm-pack flags can be passed to `build.mjs` (for example,
`--mode no-install --no-opt` when the toolchain is already installed).

## Background reception

`WasmTransferRx.ingest` runs only in `js/receiver-worker.js`, not in the page's
audio callback. The same decoder, integrity checks, and recovery search run
there; no pitches, note durations, synthesis, or received samples are changed
by offloading.

`ReceiverWorker` copies capture views into packets of at most 2,048 samples.
One packet is in flight at a time, and an acknowledgement advances the FIFO.
The caller's buffer is never detached. Queued plus in-flight audio is bounded
at 120 seconds (~22 MiB of samples): overflow stops reception visibly and
offers retry, rather than silently skipping samples or growing without limit.
The UI indicates a backlog above one second while capture continues.

Clear, pause/resume around local transmission, and retry are explicit session
boundaries. They discard queued old audio and reject late results from the
previous generation. A verified delivery does not reset the decoder or discard
queued input; the UI separately latches the first result until cleared. A
still-looping sender can be received again after a clear as new audio arrives.
Startup, load, message, and runtime failures stop capture and preserve sending
and retry. Cancelled microphone setup closes any stream that arrives late.

## Files

- `index.html` — intro, Send/Listen workspace, verified result + help footer
- `app.css` — responsive app layout on top of the design system
- `modernist-styles.css` — Modernist design system (copied from
  `docs/design/`, used as-is)
- `js/app.js` — zone rendering, decoder polling → zone-2 sentence, wiring,
  `?debug=1` hook
- `js/audio.js` — AudioContext plumbing: TX copy scheduling (500 ms gaps,
  background-tab safe), mic capture via AudioWorklet, linear resampler to
  48 kHz, 3-zone level metering
- `js/worklet.js` — AudioWorklet processor (batches mic frames into
  2048-sample chunks)
- `js/receiver.js` — main-thread FIFO, backpressure, reset generations, status
- `js/receiver-worker.js` — classic worker hosting the actual WASM receiver
- `js/fingerprint.js`, `js/words.js` — crc32 → 8-hex + four-word fingerprint
  (fixed 256-word list; one word per crc byte, big-endian)
- `build.mjs` — builds both WASM binding styles without a JS bundler
- `pkg/`, `pkg-worker/` — wasm-pack output (`song_rs_wasm.js` + `.wasm`)
- `tests/` — transport/capture unit tests and optional real-Chrome smoke test

## Browser support

The API targets below are not version-by-version verification. Reception uses
a classic worker plus `importScripts`, so it does not require module-worker
support, SharedArrayBuffer, or cross-origin isolation headers.

- **Chrome / Edge / Chromium ≥ 90** — full support; `AudioContext({sampleRate:
  48000})` honored, so no resampling path is exercised.
- **Firefox ≥ 76** — supported. Firefox may ignore the requested context
  sample rate on some devices; capture is then linear-resampled to 48 kHz
  before `ingest` (already wired, nothing to configure).
- **Safari ≥ 14.1** — AudioWorklet supported. Safari/iOS may silently ignore
  the `echoCancellation/noiseSuppression/autoGainControl: false` constraints;
  per spec the app proceeds anyway (the modem tolerates processing). iOS
  forces the hardware sample rate (44.1 kHz on many devices) — the resampler
  covers this.
- Autoplay policy: audio only starts from a user gesture (`Start sending` or
  `Start listening`), which satisfies all engines and gives microphone
  permission a clear explanation before the browser prompt.

## Browser verification

With Node.js 22 or newer, from `web/`:

```bash
npm test
npm run test:browser
# CHROME_BIN=/path/to/chrome npm run test:browser
```

The unit suite requires no WASM build or browser. It tests exact FIFO sample
transport, buffer ownership, startup/reset/overflow/error handling, cancelled
permission and worklet setup, retry, stale capture callbacks, and continuous
44.1→48 kHz resampling.

The optional Chrome test requires both WASM builds. It starts a loopback-only
static server under a non-root URL path, uses a new temporary browser profile,
muted output, and a fake microphone, and closes its own browser/server when
done. It does not touch an existing browser profile or real microphone. The
temporary profile path is printed; this invocation's profile is removed after
its browser exits, including when a check fails.

Sender checks require both WASM encoding exports to reject empty payloads with
a JavaScript error, then verify that a valid one-byte payload still produces
finite, identical single-copy and repeated-copy audio. Empty transfer payloads
are rejected before checksum framing; valid payloads remain 1–120 bytes.

Real-worker checks cover 1/64/120-byte transfers with 128/511/2048-sample
callbacks, ±2% sample-clock mismatch on 64/120-byte frames with fractional-slot
starts after idle background, NaN/infinite sample dropouts, four queued sequential
and repeated transmissions, three back-to-back transfers with different clocks
and a missing first end marker, a damaged-header/missing-marker transfer completed
within a three-slot silent gap before the next sender, reset, an end-shaped background fragment before a
maximum-length transfer, and recovery after 72 seconds of periodic background.
Unanchored background must not count as a failed copy. A main-thread heartbeat
checks for gaps of 500 ms or longer during
the heavy background case. The page checks missing worker assets, microphone
retry, result latching, a delayed verified response after clear, and stopping
tracks after a worker failure. These run against the real production JS and
WASM, not a fake decoder. Timing on one desktop is not a low-end-device or
physical-channel guarantee; the smoke test also relaxes autoplay policy.

Eight additional worker cases combine half-amplitude echoes with clock mismatch:
4/16/64/120-byte transfers, 25 ms echoes at ratio 0.98 and 60 ms echoes at ratio 1.02.
They use fractional capture phase, preserve the real echo tail, and require
exact payload delivery through the same 60-second queue-drain limit. All passed
in Chrome 153; short cases took 0.14–0.78 seconds, but the hardest 120-byte case
took about 25 seconds in this run. This is
recovery coverage, not a hard real-time claim. Native callback/queue-debt checks
are available with `cargo run --release --example receiver_latency -- --transfer --echo`.

All four clock cases pass the unchanged 60-second worker queue-drain limit in
isolated Chrome 153. The formerly timing-out 120-byte transfer at ratio 0.98,
after six seconds of idle audio, now drains in a few seconds. The decoder
retains intact header lengths across timing offsets instead of repeating
speculative-length searches. Clear clock measurements postpone premature probes
for slow frames or add earlier probes for fast ones, retaining the original
worst-case fallback (and the fast frame's nominal deadline). Smoothed guard
corrections resist noisy measurements; bounded, CRC-checked quiet-tail probes
can finish filtered recordings without discarding quiet phrases. One locally verified
clock phase can also be tried before ordinary analysis; on rejection, the
original timing search remains available. A later frame's hint cannot jump
ahead of an earlier retained candidate. The original fallback deadline and
integrity gates are unchanged. The consecutive-clock test delivers
all three frames in order, including the repeated short payload; successful
recovery preserves already-recorded lookahead for the next sender. Two slots
of preceding audio are also retained as separate analysis-only context, so a
noisy fast next frame can align slightly before that boundary without replaying
old notes through preamble detection. If a fully verified sync survives but its
length header does not, a bounded CRC-checked probe can use a three-slot quiet
gap, including true silence. The worker test checks that the first payload
arrives after exactly the third silent slot, before any samples from the next
sender are queued. Failed probes keep receiving and never fabricate a length;
the transfer profile still finalizes failed copies on the fourth silent slot.
Clock cases run individually after the other worker/UI checks.
Long callbacks remain; these are desktop regression results,
not a hard real-time guarantee. Equivalent native workloads are available through
`cargo run --release --example receiver_latency -- --transfer --clock`.

## Needs real-device testing

Everything acoustic — the browser plumbing is testable on one machine, the
product isn't:

- Actual speaker → air → mic transfer between two physical devices (start
  with ~20-byte payloads at ≤1 m, quiet room).
- iOS Safari capture: 44.1 kHz resample path + whether mic processing that
  can't be disabled still decodes.
- Phone-call transfer (device A on a call, device B held to the earpiece).
- Level-meter zone calibration against real mic gain ranges (`quiet` /
  `good` / `clip` thresholds are first-pass values in `js/audio.js`).
- Clipboard `copy` button on non-localhost HTTP (requires secure context —
  expected to no-op there).
- Long RX sessions on low-end phones: verify sustainable worker throughput,
  capture scheduling, queue limits, and foreground/background transitions.
  Heavy RS/erasure passes no longer run on the main thread, but they still
  consume CPU and can delay delivery. Worker offloading does not make the
  decoder itself hard real-time.
