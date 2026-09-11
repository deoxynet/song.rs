# sóng
*sống* 🌊

Data as song. Song as survival.

The name is a triple pun, and every reading is a spec:

- **sóng** *(Vietnamese: wave)* 🌊 — it's an acoustic modem. Sound waves.
- **sống** *(Vietnamese: to live, to survive)* — the waveform survives what kills other modems: VoIP codecs, neural noise cancellation, spectral mangling, packet loss.
- **song** *(English)* — it doesn't beep. It *sings*. The default transmission performs the head of Coltrane's Giant Steps while your bytes ride the vowels.

## The Problem

Acoustic modems like ggwave use pure sine tones. These get destroyed by phone call codecs (AMR, Opus), noise cancellation (Krisp, RNNoise, WebRTC), and speaker-air-microphone chains.

VoIP platforms like Teams and Zoom are worse still: their "voice optimization" crushes the F1 formant band (400–850 Hz) by up to 60 dB and tilts the spectrum aggressively even inside the surviving bands.

## The Insight

Noise cancellation preserves human voice. So data is encoded as voice-like signals — formant positions instead of pure tones, harmonic structure instead of sine waves, and rhythmic silences that convince speech detectors this is a voice worth keeping. If it sounds like singing, it survives.

## How It Works

Data is a sequence of **vowel-like sounds**; pitch is a free artistic channel:

- **Vowel channel** (2 bits/symbol): the F2 formant position (1000–2500 Hz — above everything VoIP crushes) selects one of 4 vowels
- **Pitch (melody)**: F0 carries **no data**. Detection is F0-agnostic, so the encoder may sing anything in its compass (B2–F#4). By default it sings Giant Steps; `encode_with_pitch` accepts any per-byte pitch track; `encode_armored` uses the margin-verified D lydian arpeggio (D3 · G#3 · C#4) for worst-case channels

| Vowel | F1 (Hz) | F2 (Hz) | Note |
|-------|---------|---------|------|
| 0 | 416 | 1074 | low F1, low F2 |
| 1 | 624 | 1420 | mid F1, mid-low F2 |
| 2 | 520 | 1663 | mid F1, mid-high F2 |
| 3 | 832 | 2252 | high F1, high F2 |

**Detection is a retry ladder.** The receiver never trusts one method: default centroids → pilot-calibrated centers → pilot-derived **channel EQ** (the 8 known pilot symbols measure the live channel's spectral tilt, which is then inverted) → **matched-filter templates** against per-note synthesis spectra → **F0-adaptive centroids** (detect each window's note, classify against that note's expected values) → an **ensemble vote** across all of them. Every risky decode passes an anti-miscorrection gate (re-encode and demand agreement with what was actually received), Chase second-choice retries flip the weakest symbols, and RS erasure decoding mops up. A wrong payload is treated as worse than no payload.

## The Songbook

```
cargo run --release --example duet         # two robots chat (sings Giant Steps — the default)
cargo run --release --example minuet       # Minuet in G, BWV Anh. 114 — the rice cooker song
cargo run --release --example amaryllis    # Amaryllis (1581) — the other rice cooker song
cargo run --release --example giant_steps  # the default melody, verified through hostile channels
cargo run --release --example channels     # the same message through 7 increasingly hostile channels
cargo run --release --example midi_melody <file.mid>   # sing ANY MIDI file
```

`midi_melody` takes arbitrary MIDI (including messy [basic-pitch](https://github.com/spotify/basic-pitch) transcriptions of real recordings): it extracts a melody line with contour-preserving octave selection, quantizes rhythm to the tune's own pulse, auto-transposes into the compass, and verifies the result still decodes.

## Speed

| Method | Throughput |
|--------|------------|
| Morse code (20 WPM) | ~10 bps |
| Morse code (expert, 40 WPM) | ~20 bps |
| **This modem** | **~30 bps** |

2–3× a skilled Morse operator, through channels that would erase Morse entirely (VAD, noise suppression, AGC) — while playing jazz.

## Usage

### Single-message API

```rust
// Encode (sings Giant Steps by default)
let audio = song_rs::encode(b"hello", 25).unwrap(); // Vec<f32> at 48 kHz

// Maximum channel margin (D lydian arpeggio, margin-verified notes)
let audio = song_rs::encode_armored(b"hello", 25).unwrap();

// Any melody you like (one F0 per payload byte, B2..F#4)
let track = vec![174.61; 5]; // F3 drone, why not
let audio = song_rs::encode_with_pitch(b"hello", 25, Some(&track)).unwrap();

// Decode (streaming — feed chunks of any size)
let mut decoder = song_rs::Decoder::new();
let payload = decoder.decode(&audio).unwrap().unwrap();
assert_eq!(&payload, b"hello");
```

### Secret transfer (API keys)

CRC-32 gate + ~100% RS parity + repeat transmission with cross-copy soft combining. Always armored — secrets don't trade margin for style:

```rust
let audio = song_rs::transfer::encode_repeated(b"sk-live-...", 50, 3, 500).unwrap();

let mut rx = song_rs::transfer::TransferRx::new();
// feed mic audio; returns Some(payload) only with a valid CRC
if let Some(key) = rx.ingest(&audio) { /* delivered, verified */ }
```

### Streaming FEC and PHY

Continuous RS-coded streams (`StreamTx`/`StreamRx`, RS(15,k) over GF(2⁴) with block interleaving) and a reliable-delivery PHY (`Phy`: stop-and-wait ARQ, fragmentation to 2,224 bytes, ACK/NAK, retransmit). Both are pure state machines — no threads, no I/O, WASM-compatible.

## Protocol

| Parameter | Value |
|---|---|
| Sample rate | 48,000 Hz (8 kHz mode via feature flag) |
| Melodic compass | B2–F#4, chromatic (123.5–370 Hz) |
| Default melody | Giant Steps head, 1 note per byte (~250 bpm) |
| Armored melody | D3 · G#3 · C#4 arpeggio (margin-verified) |
| Symbol | 50 ms voiced + 10 ms guard, 2 bits |
| Sync | 4-symbol preamble + 8-symbol calibration pilot |
| Max payload | 140 bytes single message |
| Error correction | RS over GF(2⁸), adaptive parity; ~100% parity paranoid profile |

## Robustness

**219 tests at 48 kHz, 141 at 8 kHz.** Channel simulations include:

- Teams/VoIP spectral shaping measured from real calls — decodes through **50 dB of in-band tilt** (via channel EQ + template matching)
- Neural noise cancellation (RNNoise), including *after* the tilt
- Opus at VoIP bitrates, with **packet loss + PLC** (scattered and 200 ms bursts)
- Jitter-buffer behavior: ±2% time warp, 20 ms frame drops and inserts
- Competing speech (babble) to 10 dB SIR — the interference denoisers deliberately pass
- Phone line (bandpass + 8 kHz resample + μ-law), AM radio, clipping, reverb, fading, drift, 6-bit ADC, and stacked nightmare combinations
- White-noise floor: decodes to ≈ **−20 dB SNR** (transfer profile)

The decoder never returns plausible garbage: erasure decodes, brute-forced lengths, and Chase trials are all gated by re-encode agreement, a length lock from pristine headers, and (for transfers) CRC-32. Failure is loud, success is verified.

## Architecture

| Module | Purpose |
|---|---|
| `src/lib.rs` | Public API (`encode`, `encode_armored`, `Decoder`) |
| `src/protocol.rs` | Constants, melodies (`GIANT_STEPS`, `MELODY`), ECC profiles |
| `src/formant.rs` | Vowel alphabet, synthesis, detection ladder (centroids, EQ, templates, F0-adaptive), note bank |
| `src/encoder.rs` | Payload → RS → symbols → sung audio |
| `src/decoder.rs` | Streaming state machine, retry ladder, anti-miscorrection gates |
| `src/transfer.rs` | Secret-transfer profile: CRC gate, paranoid ECC, soft combining |
| `src/stream.rs` | Streaming FEC codec with block interleaving |
| `src/phy.rs` | Stop-and-wait ARQ, fragmentation, ACK/NAK |
| `src/reed_solomon.rs`, `src/rs4.rs` | RS codecs over GF(2⁸) / GF(2⁴) |
| `src/fft.rs` | FFT wrapper |

## Dependencies

- [`rustfft`](https://crates.io/crates/rustfft) — FFT computation
- [`thiserror`](https://crates.io/crates/thiserror) — error types
- [`hound`](https://crates.io/crates/hound), [`criterion`](https://crates.io/crates/criterion), [`nnnoiseless`](https://crates.io/crates/nnnoiseless), [`opus`](https://crates.io/crates/opus) — dev only (WAV, benches, RNNoise, real Opus)

## License

MIT

---

*ăn cơm chưa?* — the traditional greeting asks if you've eaten rice yet, because eating means you're surviving. This modem asks the same of your data, and it asks in song.
