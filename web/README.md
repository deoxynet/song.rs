# AKT — Acoustic Key Transfer (web)

Static web UI over the `song-rs` secret-transfer profile, compiled to WASM.
Transfers short secrets (≤120 bytes UTF-8) between two devices through sound.
No framework, no bundler, no server code — plain ES modules.

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

`pkg/` is the wasm-pack output. To regenerate it from the repo root:

```bash
wasm-pack build wasm/ --target web --out-dir ../web/pkg
```

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
- `js/fingerprint.js`, `js/words.js` — crc32 → 8-hex + four-word fingerprint
  (fixed 256-word list; one word per crc byte, big-endian)
- `pkg/` — wasm-pack output (`song_rs_wasm.js` + `.wasm`)

## Browser support

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
- Long RX sessions: `ingest` runs on the main thread; verify no dropped
  chunks on low-end phones during the heavier RS/erasure analysis passes.
