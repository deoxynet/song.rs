# ggwave-voice

Voice-formant acoustic modem. Encodes data as vowel-like audio that survives phone codecs and noise cancellation.

## Build & Test

```bash
cargo build
cargo test                   # unit + integration tests
cargo test --test robustness # channel simulation tests (phone, AM radio, etc.)
cargo test --test opus        # real libopus roundtrip (needs system libopus)
cargo bench                  # criterion benchmarks (encode/decode/roundtrip)
```

## Architecture

| Module | Purpose |
|---|---|
| `src/lib.rs` | Public API (`encode`, `Decoder`), top-level tests |
| `src/protocol.rs` | Constants (sample rate, symbol timing, FFT size, formant bands) |
| `src/formant.rs` | Vowel alphabet, melodic synthesis, F0-agnostic detection, 2-bit packing |
| `src/encoder.rs` | Payload → RS-encode → symbols → synthesized audio |
| `src/decoder.rs` | Streaming state machine (Listening → Receiving → Analyzing) |
| `src/fft.rs` | FFT wrapper (Hann-windowed and raw power spectrum) |
| `src/reed_solomon.rs` | RS codec over GF(2^8) with erasure support, ported from ggwave C++ |
| `src/phy.rs` | PHY layer: stop-and-wait ARQ, fragmentation (up to 2,224 bytes), ACK/NAK, timeout/retransmit |
| `src/dss.rs` | Direct Sequence Spread (XOR whitening, currently unused) |
| `src/transfer.rs` | Secret-transfer profile: CRC-32 gate, paranoid ECC, repeat + soft combining (`TransferRx`) |

## Key Design

- **Vowel-only encoding**: each symbol carries 2 bits via vowel identity (F2 formant, 4 vowels). **Pitch carries no data** — detection is F0-agnostic, so pitch distortion cannot corrupt data.
- **Default melody: Giant Steps.** `encode()` sings the payload to the 16-bar head of Coltrane's Giant Steps (`protocol::GIANT_STEPS`, one note per payload byte ≈ quarter notes at ~250 bpm, transposed to the modem's B2–F#4 compass so the true contour fits with no octave folds; the melody came verbatim from a MIDI transcription of the head). Sync (preamble + pilot), header, and RS parity always stay on the margin-verified D lydian arpeggio (`protocol::MELODY`, per-symbol). `encode_armored()` uses the arpeggio for the payload too — maximum channel margin — and the transfer profile is always armored: the worst measured channel (heavy tilt + denoiser) destroys certain note+vowel windows for the wider note set (measured: ~7% symbol errors, identical across every classifier rung = channel information loss, not decoder slack). Cost of the melodic default elsewhere is a sliver: e.g. 10% periodic Opus packet loss survives 9/10 phase alignments (armored: 10/10) — see `opus_packet_loss_10pct`.
- **Armored melody notes chosen numerically** (D3, G#3, C#4): each note must keep all four vowel F2-band centroids separable, including under reverb combs — a note needs ≥3 harmonics in vowel 0's F2 Gaussian core so one comb null can't flip it (E3/G3/A3/B3 fail this and are excluded). Classification compares against `EXPECTED_F2_CENTROIDS` (measured synthesis centroids), not nominal F2. Expressive/default melodies draw from the wider chromatic bank (`formant::TEMPLATE_NOTES`, B2–F#4): below B2 the 26-harmonic synthesis can't reach vowel 3's formant; above F#4 the harmonic grid is too sparse to separate vowels 0/1. The F0-adaptive rung classifies against per-note expected centroids, so the decoder needs no knowledge of the tune.
- **Symbol timing**: 50 ms voiced + 10 ms guard = 60 ms per symbol (2880 samples at 48 kHz)
- **Preamble**: 4 symbols start ([0, 3, 0, 3]) and end ([3, 0, 3, 0]), using max-distance vowel pair
- **Streaming codec**: RS(15,k) over GF(2^4), one GF(16) symbol = two vowel symbols, byte = 2 nibbles = 4 vowels (no bit straddling). End-of-stream is detected by trailing silence (~10 dB below running RMS average, 4 windows), never by symbol pattern — data cannot alias silence.
- **RS ECC**: adaptive — `len < 4 ? 2 : max(4, 2*(len/5))` parity bytes
- **Decoder**: streaming, processes symbol-sized windows. History ring buffer for alignment recovery. Multi-offset analysis as fallback. RS erasure decoding via symbol confidence as fallback when hard-decision fails.
- **Expressive pitch API**: `encode_with_pitch` accepts a per-payload-byte F0 track (pitch carries no data, so callers may drive it semantically); notes should come from the chromatic bank compass (`formant::TEMPLATE_NOTES`, B2–F#4). Only the armored arpeggio notes are margin-verified for worst-case channels.
- **Calibration pilot**: 8 known symbols after the start preamble; the receiver measures each vowel's F2 centroid through the live channel. Measured centers are used strictly as a decode RETRY after the default centers fail (sanity-gated: ascending, ≥120 Hz apart, within ±150 Hz of defaults) — never as primary, because a pilot measured through a denoiser's warm-up ramp can pass the gate yet be worse. Block path only.
- **Channel EQ + template retry ladder** (block path): the pilot also yields a per-bin F2-band channel gain estimate (`formant::estimate_eq`, log-domain bucketed vs known transmit spectra, inversion floored at −60 dB). Decode retries run default centers → calibrated centers → EQ'd centroid → matched-filter templates (`classify_symbol_template`: correlate the equalized band against per-(vowel, melody-note) synthesis templates — rejects leakage on the *neighbor's* F0 grid that drags a centroid). This ladder decodes through Teams-style in-band tilt up to ~55 dB; 60+ dB buries vowel 0 below the leakage floor and is a known information limit (`teams_f1_attenuation_extreme` stays should_panic).
- **Structural preamble detection**: exact vowel matching fails under tilt, but the preamble's lo-hi-lo-hi centroid *alternation* survives any monotone spectral shaping (`structural_start_match`, swing ≥250 Hz, spread <200 Hz). The Listening lock accepts either; a late lock (the pilot's own [2,3,2,3] run is preamble-shaped) is covered by a 16-window history ring that always retains preamble + pilot. Decode-time candidates come from the recorded lock position (`preamble_pos_hint` = history_len − 4), exact matches, loose structural matches, and blind head positions — all scanned only in the first 24 windows (deeper "preambles" are data patterns) and each verified by RS-gated decode.
- **Chase second-choice retry**: when hard-decision and erasure decoding fail, the ≤6 weakest symbols are flipped to their runner-up classification (2^k hypotheses, cheapest first). Every classification pass carries per-symbol runner-ups for this.
- **Anti-miscorrection gates** (Standard profile): risky decodes — erasure fallback (weak residual syndrome once parity is spent on erasures), derived lengths, Chase trials — must pass `reencode_agreement`: re-encode the candidate payload and demand ≥0.80 symbol agreement with the received stream AND ≤6% disagreement among high-confidence symbols (a true decode disagrees only at low-confidence, channel-corrupted positions; a miscorrection "fixes" clean bytes). Competing risky candidates are ranked by agreement. Paranoid skips the gate — the transfer CRC is stronger.
- **Length lock**: a zero-correction, high-confidence header decode from the default classification locks the payload length for the whole trial — every other-length interpretation from any rung/candidate is inadmissible, which closes the wrong-length miscorrection class and collapses the failure-path search (~12× test-suite speedup vs unlocked).
- **Trailing-silence finalize**: both profiles finalize a reception on sustained relative silence (Paranoid 4 windows for repeat-flow turnaround; Standard 6, tolerating denoiser gain dips), so a corrupted end preamble can't strand the decoder in Receiving. Robustness-test decode helpers append ~10 slots of silence — the real-channel condition.
- **Payload validator hook** (`Decoder::set_payload_validator`): TransferRx installs its CRC check so a mid-reception trial decode of plausible garbage is treated as *no decode* (reception continues) instead of terminating and splitting the copy — without it, Chase×brute-force false positives fragment receptions and poison soft combining.
- **Soft-combining alignment**: TransferRx aligns harvested attempts to the first by maximal symbol agreement over ±6 shifts before majority voting (attempts can be harvested on slightly different grids), votes over the full reference length, and only harvests streams anchored by the lock hint or an exact preamble match. Note: solo decoding is now strong enough (erasures + Chase + brute-forced lengths + CRC-validated retry) that white noise cannot separate solo-fail from combine-pass; the voting mechanism is unit-tested with injected confident errors and shifts.
- **StreamRx fuzzy lock**: streaming preamble accepts exact, 3-of-4, or structural centroid matches — fuzzy paths gated by all four windows standing ≥3× above a slow-rising/fast-falling ambient noise floor (absolute 1e-4 on cold start), so one corrupted preamble symbol doesn't cost the stream but idle noise can't false-lock.
- **Transfer profile** (`src/transfer.rs`, for API-key-style secrets): CRC-32 inside the payload makes RS miscorrection an explicit failure (correctness, not security — encryption deliberately out of scope, same-room threat model); `EccProfile::Paranoid` = ~100% parity with brute-forced length when the weak 3-byte header is corrupted; `TransferRx` majority-votes symbols across repeated receptions, demoting contested positions to confidence-margin so they become RS erasures. Decoder also ends a reception on sustained silence (4 windows, relative RMS) so corrupted copies finalize.
- **PHY layer**: stop-and-wait ARQ over the acoustic channel. Fragments messages into 139-byte frames, ACK/NAK per fragment, timeout/retransmit (configurable). Test harness in `tests/phy.rs` with loopback, noise, lost-packet, and fragmentation scenarios.

## Conventions

- Audio is `f32` samples at 48 kHz (mono)
- Internal synthesis uses `f64` for precision, casts to `f32` at output
- Volume is `u8` 0..=100
- Max payload 140 bytes
- Tests use deterministic LCG PRNG (no randomness)
