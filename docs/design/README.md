# Handoff: Acoustic Key Transfer — UI states

## Overview

A web tool that transfers short secrets (API keys, tokens, ≤120 bytes UTF-8) between two
devices **through sound**: speaker → air → microphone, optionally down a phone call held
between the two devices. No network, no server, no accounts, no persistence.

The core guarantee the UI must preserve: **the receiver shows the exact payload or an
explicit failure — never a corrupted one** (CRC-32 verified in the audio layer). Robustness
comes from repetition: the sender loops the transmission until a human on the receiving end
says "got it"; the receiver merges information across repeats, so every extra copy strictly
improves the odds.

This design is a **state board**: one tile per state of the machine, static, no interaction.
It is the visual + copy specification for the real tool, not the tool itself.

## About the design files

The files in this bundle are **design references authored in HTML** — a prototype of the
intended look, information density and copy. They are not production code to copy.

The task is to **recreate these states in the target codebase's existing environment**
(React, Vue, SwiftUI, native, whatever is already there) using its established component
library, styling approach and state patterns. If no environment exists yet, pick the
appropriate framework for the product and implement the states there. The Web Audio /
WASM plumbing described under "Audio contract" is the functional half and is not
implemented in the prototype at all.

`Acoustic Key Transfer.dc.html` is authored in a component format specific to the design
tool; read it as markup + inline styles. `modernist-styles.css` is the design system
stylesheet it consumes — token names referenced throughout this document come from it.

## Fidelity

**High fidelity.** Colors, type, spacing, rules and copy are final and come from the bound
*Modernist* design system. Recreate the density and the ruled-grid structure faithfully; do
not soften the 2px rules, do not round corners, do not center labels.

Two deliberate departures from the design system, applied on purpose because this is an
engineer tool: numeric readouts and status lines are set in a **monospace stack**
(`ui-monospace, Menlo, monospace`) rather than Archivo, and body sizes run small (11–13px).
Archivo 800 is used only for the fingerprint words.

## Layout

Three bands, top to bottom, on `--color-bg`:

1. **Header bar** — `2px solid var(--color-divider)` bottom border, `padding: 8px 16px`,
   baseline-aligned flex row, `gap: 16px`:
   - `AKT` — Archivo 800, 13px, `letter-spacing: 0.06em`
   - `audio · 48000 Hz mono f32 · max 120 B · crc32` — mono 11px, muted
   - `states` — mono 11px, muted, `margin-left: auto`
2. **State grid** — `display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 0`.
   Every tile is single-width; each carries `border-right: 2px solid var(--color-divider)`
   and `border-bottom: 2px solid var(--color-divider)`, `padding: 16px`,
   `min-height: 230px`, `display: flex; flex-direction: column; gap: 12px`.
   **Do not** paint the rules with a container background showing through gaps — a short
   last row then renders as a solid gray block. Rules belong on the tiles.
3. **Reference footer** — `2px` top border, `padding: 16px`,
   `grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 32px`: the air-time
   table (design-system `.table`, mono 12px) and the audio interface signatures (mono 12px).

Every tile begins with the same header row: state id on the left, ordinal on the right,
mono 11px, `border-bottom: 1px solid var(--color-divider)`, `padding-bottom: 5px`. Live
states (`TX.LOOP`, `RX.LOCK`, `RX.OK`) set the id in `var(--color-accent)`.

Every tile's terminal action is pinned to the bottom with `margin-top: auto`.

Responsive: single layout, works phone → desktop. The grid collapses to one column below
~740px; the two-pane tiles (`TX.LOOP`, `RX.OK`) use an inner
`repeat(auto-fit, minmax(240px, 1fr))` grid that stacks inside a narrow tile.

## States

Ids are the state machine's names; use them in code.

### 01 `IDLE`
Purpose: choose a role. No setup, no pairing.
Two buttons in a `1fr 1fr` grid, `padding: 12px`, labels flush left: **send**
(`.btn .btn-primary`) and **receive** (`.btn .btn-secondary`). Below, pinned bottom, two
muted mono 11px lines: `tx: encode → loop until stopped` /
`rx: ingest → merge copies → verified payload`.

### 02 `TX.INPUT`
Purpose: enter the secret.
- `.input` (readonly in the mock) showing the masked value `••••…` in mono, plus a
  `show` secondary button. **Masked by default**; toggle reveals plaintext.
- Row: muted `payload` / `38 / 120 B` (mono 11px). Live byte counter, UTF-8 bytes not
  characters.
- 2px meter: `--color-neutral-300` track, `--color-text` fill at `length/120`.
- Row: muted `copy time` / `22.0 s`, computed from the formula below.
- **next** — `.btn .btn-primary .btn-block`, `padding: 12px`.

### 02b `TX.INPUT — over limit`
Hard stop past 120 bytes. Same tile with: input border `var(--color-accent)`, value shown
in plaintext, counter row both sides in `var(--color-accent-700)` — `over by 4 B` /
`124 / 120 B` — meter filled 100% in `--color-accent`, and **next** genuinely `disabled`
(45% opacity from `.btn:disabled`). No transmission may start while over the limit.

### 03 `TX.READY`
Purpose: the trust moment. Muted mono label `fingerprint · crc32`, then the fingerprint —
by default four words, Archivo 800, 20px, two lines: `RIVER CLAMP / OTTER VAULT`
(hex variant: mono 20px, `letter-spacing: 0.14em`, `3fa291c7`). Below a 1px rule, a mono
11px key/value block: `payload 38 B`, `copy 22.0 s`, `gap 500 ms`. Terminal action
**start**.

### 04 `TX.LOOP`
Purpose: transmit on repeat until stopped. Two panes.

Left:
- `copy 3` — mono, `clamp(28px, 4vw, 44px)`, `line-height: 1`. Increments per repeat.
- `playhead` / `14.0 / 22.0 s` over a 10px bar, `1px solid var(--color-text)`,
  `--color-accent` fill at position-in-copy.
- `volume` / `50` over a 10px bar, `1px solid var(--color-divider)` on
  `--color-surface`, `--color-text` fill. Maps to the modem's `volume` 0–100, default 50.
- Rule, then mono 11px: `elapsed 68.4 s`, `fingerprint 3fa291c7`,
  `no rx feedback channel —` (states plainly that the sender gets no ack; the receiving
  human is the ack).
- **stop** — `.btn .btn-secondary .btn-block`. Stop is always available.

Right (the "singing notes" band): muted mono 11px `1000–2500 Hz` / `d3 g#3 c#4`, then a
bar band — `display: flex; align-items: flex-end; gap: 2px; min-height: 120px;
border-bottom: 1px solid var(--color-text)`, 24 bars of `flex: 1`. Bars above ~55% of
current energy in `--color-accent`, mid bars `--color-neutral-400`, decayed tail
`--color-neutral-300`. In production drive heights from an `AnalyserNode`; energy sits
1000–2500 Hz over D3/G♯3/C♯4 fundamentals (147–277 Hz). Footer line, muted mono 11px:
`quiet room · ≤1 m · speaker clear`.

### 05 `RX.PERM`
Purpose: microphone permission, stated as constraints rather than prose. Mono 11px
key/value list: `echoCancellation false`, `noiseSuppression false`, `autoGainControl false`,
`sampleRate 48000`. Actions: **grant microphone** (primary, flush-left label) and **back**
(secondary). If the platform silently ignores the constraints, proceed anyway — the modem
tolerates processing. Do **not** surface a warning.

### 06 `RX.LISTEN`
Purpose: idle listening. `no signal` — mono `clamp(20px, 3vw, 30px)`.
Level meter — 12px tall, `1px solid var(--color-text)`, three zones by flex ratio
2 / 5 / 2 (`--color-neutral-300`, `--color-neutral-200`, `--color-neutral-300`), a
`--color-text` fill overlaid at the current level, zone captions `quiet` / `good` / `clip`
in mono 10px below. One actionable line in `--color-accent-700`:
`reduce distance or raise tx volume`. This meter is the #1 user-fixable failure cause —
it is always visible in every RX state. **stop** pinned bottom.

### 07 `RX.LOCK`
Purpose: preamble detected, signal locked. Tile id in accent.
`preamble · copy 1` — mono `clamp(20px, 3vw, 30px)`, `--color-accent-700`.
`progress` / `9.0 / 22.0 s` over a 10px bar bordered and filled in `--color-accent`
(time-based against the expected copy duration). A 52px, 12-bar version of the notes band.
**stop** pinned bottom.

### 08 `RX.MERGE`
Purpose: a copy failed CRC; the receiver keeps listening and merges. **This is normal
operation, not an error** — styled in ink, never red, and the user is never asked to
restart. `copy 2 crc fail / merging into copy 3` — mono `clamp(17px, 2.4vw, 22px)`, two
lines. A 3-segment copy strip, 8px tall: heard copies `--color-neutral-400`, the copy in
flight `--color-accent`. Rule, then mono 11px: `attempts 2`, `listening 64.2 s`,
`level good`. **stop** pinned bottom.

### 09 `RX.STALL`
Purpose: after ~5 failed copies, escalate guidance while still listening.
`5 copies · 0 verified / still listening` — mono `clamp(17px, 2.4vw, 22px)`.
Rule, then a mono 11px arrow list: `→ tx volume 80`, `→ distance ≤ 30 cm`,
`→ silence fans, notifications`, `→ speaker facing mic, unobstructed`. Rule, then
`level` / `clip · lower tx volume` in `--color-accent-700`. Actions: **keep listening**
(secondary, flush left) and **reset** (secondary). The receiver never gives up on its own.

### 10 `RX.OK`
Purpose: verified payload, cross-checked. Tile id in accent. Two panes.

Left: `verified · copy 3` — mono `clamp(22px, 3vw, 34px)`. Then the decoded value in a
`.input`, masked, with **reveal** (secondary) and **copy** (primary). Rule, then mono 11px:
`length 38 B`, `crc32 3fa291c7`, `attempts 3`. **clear** pinned bottom — wipes the value
from memory and the DOM; nothing is stored anywhere.

Right: muted mono 11px `cross-check with sender`, the same fingerprint as `TX.READY` in
`--color-accent-700` (Archivo 800, `clamp(20px, 2.6vw, 26px)`), then muted
`mismatch → discard, retransmit`.

## Interactions & behavior

Flows: `IDLE → TX.INPUT → TX.READY → TX.LOOP → (stop) → IDLE` and
`IDLE → RX.PERM → RX.LISTEN → RX.LOCK → (RX.MERGE → RX.LOCK)* → RX.OK → (clear) → IDLE`.

- `TX.LOOP` plays copies back to back with 500 ms of silence between them, indefinitely,
  until the user stops. Copy counter increments; playhead resets each copy.
- `RX` runs continuously from a single mic stream; `RX.LISTEN → RX.LOCK` on preamble
  detection, `RX.LOCK → RX.MERGE` when a copy fails CRC, `→ RX.OK` when `ingest` returns
  bytes. `RX.STALL` is `RX.MERGE` with `attempts >= 5` — same machine, escalated copy.
- Reveal toggles are per-field and default to masked in both `TX.INPUT` and `RX.OK`.
- Byte counter and copy-time estimate update on every keystroke; `next` disables while
  over 120 bytes.
- Level meter and notes band update at animation-frame rate from an `AnalyserNode`.
- No animation beyond the meters and progress bars. No transitions on state changes —
  states must be instantly legible across a desk.
- Focus: `:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px }`,
  never the browser default.

### Explicit non-goals (from the requirements — do not add)

- **No encryption or PIN UI, no lock/shield iconography, no security theater.** Same-room
  threat model. The honest claim is "verified exact transfer", never "private transfer".
- No history or persistence of transferred secrets. No server communication of any kind.
- No state may imply partial success.
- Failed copies must never be presented as errors.
- Red (`--color-accent`) is reserved for: live audio, the primary action, and the one
  fixable problem on screen. Nothing else.

## State management

```
mode        : 'idle' | 'tx' | 'rx'
txState     : 'input' | 'ready' | 'loop'
rxState     : 'perm' | 'listen' | 'lock' | 'merge' | 'ok'
payload     : string          // plaintext, memory only, never persisted
byteLength  : number          // TextEncoder().encode(payload).length
overLimit   : byteLength > 120
fingerprint : string          // crc32 hex → four-word encoding
volume      : 0..100          // default 50
copyIndex   : number          // tx: copies played; rx: copies heard
copyElapsed : number          // seconds within the current copy
copyDuration: number          // (8*len + 60) * 0.06 seconds
attempts    : number          // TransferRx.attempts()
level       : number          // 0..1 rms → quiet | good | clip
revealed    : boolean         // per field, resets on state change
decoded     : string | null   // wiped by `clear`
```

No data fetching. All state is per-session and dies with the page.

## Audio contract (not implemented in the prototype)

Plumbing runs at **48 kHz mono f32**; request `sampleRate: 48000` on the `AudioContext`
and resample capture if the platform refuses. Capture via `AudioWorklet`, with
`echoCancellation`, `noiseSuppression` and `autoGainControl` all `false`.

```
transfer_encode(payload: bytes, volume: u8) -> Float32Array          // one copy
transfer_encode_repeated(payload, volume, copies, gap_ms) -> Float32Array
TransferRx.ingest(chunk: Float32Array) -> bytes | null               // verified once
TransferRx.attempts() -> number                                      // failed copies
MAX_TRANSFER_PAYLOAD = 120
```

Air time per copy — `(8·len + 60) × 60 ms`:

| bytes | per copy |
|---|---|
| 20 | ~13 s |
| 40 | ~23 s |
| 64 | ~34 s |
| 120 | ~61 s |

Clean channels succeed on copy 1; rough channels typically need 2–3 (cross-copy merging
needs 3+ for the hardest cases). The UI must feel calm about multi-minute transfers — this
is robustness over speed by design.

## Design tokens

From `modernist-styles.css` (*Modernist*: flat, architectural, near-mono red on white,
zero radius, 2px rules, Archivo throughout). Use the variables, never the literals.

Colors used here: `--color-bg #f3f2f2`, `--color-surface #eae9e9`, `--color-text #201e1d`,
`--color-accent #ec3013`, `--color-divider` = `color-mix(in srgb, #201e1d 40%, transparent)`,
`--color-accent-700 #ae1800` (accent at paragraph size — the base accent is only 3:1 on
this ground, so small red text must use 700), `--color-accent-600 #dd2b0f` (hover),
`--color-neutral-200 #eae7e7`, `--color-neutral-300 #d7d3d3`, `--color-neutral-400 #bab6b6`.

Spacing: 4 / 8 / 12 / 16 / 24 / 32 px (`--space-1…8`). Radius: **0 everywhere**
(`--radius-sm/md/lg`). Shadows: unused in this design.

Type: `--font-heading` / `--font-body` both `Archivo` (400 / 600 / 800), body 13px on this
page; monospace `ui-monospace, 'SF Mono', Menlo, monospace` at 10–13px for all readouts,
scaling up via `clamp()` for headline status lines as noted per state.

Design-system classes reused as-is: `.btn` + `.btn-primary` / `.btn-secondary` /
`.btn-block`, `.input`, `.table`, `.hr`, `.text-muted`. Button labels are **flush left**
in wide buttons — do not center them.

## Assets

None. No images, no icon set in this design (the system specifies Lucide if icons are ever
added). Archivo loads from Google Fonts via the stylesheet's `@import`.

## Files

- `Acoustic Key Transfer.dc.html` — the state board; all 11 tiles plus the reference footer.
- `modernist-styles.css` — the design system stylesheet the board consumes (tokens + classes).
- `ui-requirements.md` — the original product requirements this design implements.

Tweakable props on the design (`fingerprintFormat: words | hex | both`,
`showNotesBand: boolean`, `volume: 0–100`) exist for review and are not product features.
