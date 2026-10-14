# sóng
*sống* 🌊

Data as song. Song as survival.

The name is a triple pun, and every reading is a spec:

- **sóng** *(Vietnamese: wave)* 🌊 — it's an acoustic modem. Sound waves.
- **sống** *(Vietnamese: to live, to survive)* — the waveform survives what kills other modems: VoIP codecs, neural noise cancellation, spectral mangling, packet loss.
- **song** *(English)* — it doesn't beep. It *sings*. Every transmission is a jam session with a strict form: **The Lick** counts it in, your bytes sing the head of **Giant Steps**, and then **John Coltrane personally solos over your Reed–Solomon parity** until the frame ends and the band cuts him off.

Giant Steps has been terrorizing musicians since 1959. It is now available to terrorize your network stack.

## The Problem

Acoustic modems like ggwave use pure sine tones. These get destroyed by phone call codecs (AMR, Opus), noise cancellation (Krisp, RNNoise, WebRTC), and speaker-air-microphone chains.

VoIP platforms like Teams and Zoom are worse still: their "voice optimization" crushes the F1 formant band (400–850 Hz) by up to 60 dB and tilts the spectrum aggressively even inside the surviving bands.

## The Insight

Noise cancellation preserves human voice. The denoiser is a bouncer, and the modem gets in because it sings: formant positions instead of pure tones, harmonic structure instead of sine waves, and rhythmic breaths that convince every speech detector this is a voice worth keeping. If it sounds like singing, it survives. Krisp will mute your mechanical keyboard, your dog, and your smoke alarm, and it will hold the door open for 140 bytes of vowels doing Coltrane.

## 1959

In May 1959, John Coltrane recorded Giant Steps: a chord progression that moves the tonal center by major thirds, splitting the octave into three equal parts — the maximum harmonic distance you can travel per chord and still resolve.

In 1960, Irving Reed and Gustave Solomon published *Polynomial Codes over Certain Finite Fields*: error-correcting codes that are **maximum distance separable** — the maximum correction distance you can buy per parity symbol, provably, forever.

Two answers to the same question — *how far apart can structure hold things?* — arriving within months of each other, one in B–G–E♭, one in GF(2⁸). They never met. This repository is their meeting: every default transmission protects Coltrane's changes with Reed–Solomon parity, and sings that parity as Coltrane's solo. Maximum distance, both directions.

## How It Works

Data is a sequence of **vowel-like sounds**; pitch is a free artistic channel:

- **Vowel channel** (2 bits/symbol): the F2 formant position (1000–2500 Hz — above everything VoIP crushes) selects one of 4 vowels
- **Pitch (melody)**: F0 carries **no data**. Detection is F0-agnostic, so the encoder may sing anything in its compass (B2–F#4). By default it sings Giant Steps; `encode_with_pitch` accepts any per-byte pitch track; `encode_armored` uses the margin-verified D lydian arpeggio (D3 · G#3 · C#4) for the payload

Every default transmission is therefore a complete, correctly structured jazz performance:

```
┌──────────────┬───────────────────────┬──────────────────────────────┬──────────┐
│   THE LICK   │    GIANT STEPS head   │       COLTRANE'S SOLO        │ final E  │
│  E F# G A    │                       │  (1959, transcribed, note    │          │
│  F#  D  E    │      your bytes       │   for note, ~250 bpm)        │  (held)  │
├──────────────┼───────────────────────┼──────────────────────────────┼──────────┤
│ sync, pilot, │        payload        │  CRC-32 + Reed–Solomon       │   end    │
│ length hdr   │                       │  parity                      │ preamble │
└──────────────┴───────────────────────┴──────────────────────────────┴──────────┘
   1.44 s          240 ms per byte        the parity budget decides       240 ms
                                          how much solo you get
```

Yes, that means:

- **The most recognizable phrase in jazz is a synchronization preamble.** Musicians have been using The Lick as a sync marker for decades; this is merely the first time it carries a calibration pilot.
- **Your checksum has a discography.** The four CRC-32 bytes are sung, note for note, as the opening bar of Coltrane's recorded solo. The Reed–Solomon parity continues the solo from there.
- **Stronger error correction = more jazz.** This is the only forward-error-correction scheme where turning up the parity gives you more of the 1959 solo. The paranoid transfer profile (~100% parity) buys almost two full choruses to protect your API key. Frames under ~11 bytes don't get a solo at all — the band does not hand out a chorus on a jingle. Instead the frame itself plays the changes: the intro Lick is home (E minor over G), and the outro answers **in C minor over E♭** — the first Giant Steps modulation, performed by the frame structure — resolving to a held home E. A one-byte paranoid transfer's outro even reaches G♯ minor over B: the full Coltrane cycle in four seconds of modem. This survives because (measured) the short-echo clock recovery needs the phrase *rhythm* to repeat, and transposition repeats the rhythm while moving the key. Even the jingles play the changes.
- **The band cuts him off.** The end preamble is a sync marker, so the solo stops mid-line wherever the parity budget ends and the frame holds the final E — home. Coltrane gets exactly as many bars as your data can pay for. This is the most honest depiction of a jam session ever encoded in a checksum, and (measured, not vibes) pinning that final note is load-bearing for damaged-header recovery on hostile channels.
- **The solo's entrance is a backup length header.** The bar where Coltrane comes in marks where your payload ends, and the decoder actually uses it: when the real length header is destroyed, final analysis locates the solo's start by template alignment and feeds that length in as a CRC-gated candidate — measured exact on 70/70 lab cases through channels where two-thirds of the bytes were corrupt. When the header dies, the receiver finds your data by listening for where the band changes tunes.

| Vowel | F1 (Hz) | F2 (Hz) | Note |
|-------|---------|---------|------|
| 0 | 416 | 1074 | low F1, low F2 |
| 1 | 624 | 1420 | mid F1, mid-low F2 |
| 2 | 520 | 1663 | mid F1, mid-high F2 |
| 3 | 832 | 2252 | high F1, high F2 |

**Detection is a retry ladder.** The receiver never trusts one method: default centroids → pilot-calibrated centers → pilot-derived **channel EQ** (the 8 known pilot symbols measure the live channel's spectral tilt, which is then inverted) → **matched-filter templates** against per-note synthesis spectra → **F0-adaptive centroids** (detect each window's note, classify against that note's expected values) → an **ensemble vote** across all of them. Chase second-choice retries flip the weakest symbols, and RS erasure decoding mops up. Every default-profile result must pass CRC-32 validation, including intermediate retries: a wrong payload is treated as worse than no payload.

## The Songbook

There is one tune. We committed.

```
cargo run --release --example duet         # two robots trade choruses of Giant Steps
cargo run --release --example giant_steps  # the default performance, verified through six hostile channels
cargo run --release --example channels     # the same message through 7 increasingly hostile channels
cargo run --release --example midi_melody <file.mid>   # sing ANY MIDI file (the payload melody is a free channel)
```

`midi_melody` takes arbitrary MIDI (including messy [basic-pitch](https://github.com/spotify/basic-pitch) transcriptions of real recordings): it extracts a melody line with contour-preserving octave selection, quantizes rhythm to the tune's own pulse, auto-transposes into the compass, and verifies the result still decodes.

## Speed

| Method | Throughput | Survives a Teams call |
|--------|------------|----------------------|
| Morse code (20 WPM) | ~10 bps | no |
| Morse code (expert, 40 WPM) | ~20 bps | no |
| ggwave (audible) | ~64–128 bps | no — the denoiser eats it |
| John Coltrane (Giant Steps, 1959) | 0 bps | famously |
| **This modem** | **33 bps raw; ~23 bps payload at 140 bytes** | **yes — that's the whole point** |

Payload throughput includes sync, CRC, and error-correction overhead; short messages have more overhead. The waveform is designed for voice channels with VAD, noise suppression, and AGC — while playing jazz. It is slower than ggwave the way a submarine is slower than a speedboat: the speedboat does not work underwater.

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

Default messages include CRC-32 inside the Reed–Solomon-protected block. **This is a wire-format change:** both peers must use the new default. To read an older checksum-free recording, explicitly select `song_rs::Decoder::with_profile(song_rs::protocol::EccProfile::Legacy)`. To send to an old peer, use `song_rs::encoder::encode_with_profile(payload, volume, track, EccProfile::Legacy)`. There is no automatic downgrade to unchecked decoding. The transfer format is unchanged; PHY frames use the new default. Streaming FEC has its own CRC-protected default, described below.

Custom pitch tracks must have one finite, positive frequency per payload byte. Invalid values return `Error::InvalidPitch { index, hz }` before encoding/synthesis in every block profile, even at zero volume. Valid pitches are never rounded, clamped, or replaced. Channel coverage remains the B2–F#4 singing register, including ±30-cent detuning; accepting another positive frequency does not guarantee its recovery.

### Secret transfer (API keys)

CRC-32 gate + ~100% RS parity + repeat transmission with cross-copy soft combining. Armored payload melody; The Lick opens, and the ~100% parity means Coltrane plays nearly two full choruses over your secret. Nothing about an API key transfer says "serious infrastructure" like a saxophone solo the length of the key itself:

Payloads must contain 1–120 bytes. Both transfer encoders return `Error::EmptyPayload` for an empty input before adding CRC bytes; the WASM bindings expose this as a recoverable JavaScript error.

```rust
let audio = song_rs::transfer::encode_repeated(b"sk-live-...", 50, 3, 500).unwrap();

let mut rx = song_rs::transfer::TransferRx::new();
// feed mic audio; returns Some(payload) only with a valid CRC
if let Some(key) = rx.ingest(&audio) { /* delivered, verified */ }
```

### Streaming FEC and PHY

Continuous RS-coded streams (`StreamTx`/`StreamRx`, RS(15,k) over GF(2⁴) with block interleaving) and a reliable-delivery PHY (`Phy`: stop-and-wait ARQ, fragmentation to 2,224 bytes, ACK/NAK, retransmit). Both are pure state machines — no threads, no I/O, WASM-compatible. PHY's default ACK timeout is six seconds, including the CRC-protected ACK's 3.84-second airtime plus room for channel latency.

`StreamTx::new` and `StreamRx::new` use `StreamProfile::Standard`: one length byte, up to 255 payload bytes, and four little-endian IEEE CRC-32 bytes, all protected by RS. The checksum covers the **length byte plus payload**. **Both peers must use this new streaming format.** To use older checksum-free peers or recordings, explicitly construct both sides with `StreamTx::with_profile(config, StreamProfile::Legacy)` / `StreamRx::with_profile(config, StreamProfile::Legacy)`. Standard never falls back to unchecked Legacy decoding. CRC detects accidental corruption; it is not authentication.

Call `feed` once with the payload; it supplies the prefix and checksum. `feed_raw` instead accepts caller-framed bytes, in arbitrary chunks, and adds neither. The musical pitch schedule, accents, vibrato, and 60 ms slots are unchanged. With default parity 4 and depth 2, CRC occupies existing padding or adds one 3.6-second interleaver block. Other configurations have different padding/airtime costs. The block-codec Giant Steps melody and The Lick are unaffected by this streaming change.

Raw-input and output-callback boundary tests compare complete audio sample bits with a single-feed transmission in both profiles, including pauses for more input, empty calls, partial interleaver blocks, and a maximum payload. These pauses do not insert silence into the returned audio; an application must still buffer enough data to avoid playback underruns.

`StreamRx::read` and `read_all` consume completed payload bytes in reception order. Unread bytes survive a later stream, including an incomplete one; a large capture batch can therefore yield bytes from several completed streams. This is a byte queue, not a message-boundary API. Callers should drain it regularly, since storage grows with unread delivered data.

Capture does not have to start on a symbol boundary. `StreamRx` retains six slots (360 ms), scans preamble offsets every 15 ms, and refines timing against the silent guards every 5 ms. Acquisition uses up to two slots of lookahead; buffered payload samples are replayed exactly once. The cached measurements move with the retained audio and remain bounded during idle capture. Each new song reacquires timing independently.

During reception, guard measurements adjust the next symbol boundary by at most 2.5 ms, in 0.5 ms steps. Timing corrections require a substantial improvement over the current position and a guard well below the local voice; an already-clear guard is left alone, including a faint filter tail. A smoothed slot duration carries the clock through unvoiced dropouts and trailing silence, and resets at the next preamble. Positive corrections wait for real lookahead samples; negative corrections retain the exact unconsumed suffix. The receiver changes its analysis grid, not the transmitted notes, rhythm, or audio. Tests cover maximum-length streams with constant ±0.1%, ±0.5%, and ±2% clock mismatch, wandering ±2% rates, ±1.5% rate switches, noise/phone channels, 40 dB quiet phrases, and neighboring senders with different clocks.

Failed RS codewords retain their nibble positions: delivery requires a decoded length header and every claimed payload/checksum nibble, while failures confined to trailing padding do not discard complete data. Standard verifies CRC after error correction, so successful RS decoding alone cannot authorize delivery. Legacy remains unchecked.

If the first streaming result fails integrity, Standard retries confidence-ranked erasure prefixes and combinations across codewords. Every alternative must pass the same length/payload/CRC checks; there is no unchecked fallback. Search is capped at 512 complete candidates and 512 queued states, with frame history limited to the largest legal payload (plus its final partial codeword). Padding words outside each candidate's claimed frame do not consume CRC trials. Valid first-pass frames skip this search; Legacy keeps its original decoding policy. These receiver-only retries add no transmitted notes or airtime. Invalid parity above 14 or zero interleaver depth is rejected at construction.

## Protocol

| Parameter | Value |
|---|---|
| Sample rate | 48,000 Hz (8 kHz mode via feature flag) |
| Melodic compass | B2–F#4, chromatic (123.5–370 Hz) |
| Default melody | Giant Steps head, 1 note per byte (~250 bpm) |
| Armored payload melody | D3 · G#3 · C#4 arpeggio (margin-verified) |
| Block intro | The Lick in E minor, fitting the opening Gmaj7 |
| Block outro | Coltrane's 1959 solo, eighth notes at ~250 bpm, cut to the parity budget; end marker holds E |
| Symbol | 50 ms voiced + 10 ms guard, 2 bits |
| Sync | 4-symbol preamble + 8-symbol calibration pilot |
| Max payload | 140 bytes single message |
| Integrity | CRC-32 (IEEE), 4 little-endian bytes after the payload, inside RS |
| Error correction | RS over GF(2⁸), adaptive parity; ~100% parity paranoid profile |

The length header still describes caller-visible payload bytes. Default RS parity is computed over payload + CRC; use `EccProfile::Standard.encoded_bytes(len)` for frame-size calculations. The checksum adds 0.96–1.44 seconds to the musical outro (four CRC bytes and, where needed, two extra parity bytes), without changing the requested melody's notes or timing. Explicit `Legacy` frames retain the original arpeggio and its pilot calibration; streaming FEC is a separate framing format.

## Robustness

**383 tests at 48 kHz, 283 at 8 kHz**, with no ignored cases. The full channel simulations below run at 48 kHz; the 8 kHz suite covers the core codec, PHY, streaming FEC, and transfer profile. Channel simulations include:

- Every chromatic pitch B2–F#4 carrying all four vowels, sustained vowels, and alternating extremes through six channels: clean, room echo, noisy phone, spectral tilt, RNNoise, and tilt followed by RNNoise
- Ascending, descending, wide-leap, and shuffled melodies through tilt + RNNoise; detuned melodies between the receiver's template pitches
- 108 random-melody cases across 12 payload sizes (1–140 bytes), three seeds, varied receiver phase and chunk sizes, through clean, noisy-phone, and tilt + RNNoise channels
- Teams/VoIP spectral shaping measured from real calls — decodes through **50 dB of in-band tilt** (via channel EQ + template matching)
- Neural noise cancellation (RNNoise), including *after* the tilt
- Opus at VoIP bitrates, with **packet loss + PLC** (scattered and 200 ms bursts)
- 84 continuous-call cases with background noise before and after the song, single-pass RNNoise, Opus latency, varied pitch/volume, and up to five seconds of idle background before acquisition. Cases include damaged length headers and maximum-length frames. Maximum-length and quiet one-byte frames also run consecutively without receiver/codec resets, with zero or 180 ms between songs
- Jitter-buffer behavior: ±2% time warp, 20 ms frame drops and inserts
- Block clock mismatch through maximum-length melodies, all three profiles, detuned contours and 40 dB quiet phrases; ±2% drift through noise, phone filtering, and tilt + RNNoise. Timed completion also covers a missing end marker with continuous voiced background
- 64 combined melody/clock/channel cases: 1/17/64/140-byte payloads, ascending and wide-leap contours, shuffled ±30-cent pitches, and Giant Steps, at ±2% clock mismatch through noisy phone and tilt + RNNoise channels
- 80 matching transfer-profile cases through the real CRC-checking `TransferRx`, including custom melodies and the browser's default armored melody, through its 120-byte maximum
- 16 continuous-channel sequences across both block profiles: four neighboring songs with changing clocks and melodies, quieter one-byte frames and repeated payloads, zero/180 ms gaps, and an intact or missing first end marker. Each phone/denoiser channel processes the whole sequence without resets
- Competing speech (babble) to 10 dB SIR — the interference denoisers deliberately pass
- Phone line (bandpass + 8 kHz resample + μ-law), AM radio, clipping, reverb, fading, drift, 6-bit ADC, and stacked nightmare combinations
- White-noise floor: decodes to ≈ **−20 dB SNR** (transfer profile)

The 180 Hz F2 formant width improves vowel separation without changing notes, rhythm, accents, or vibrato. Quiet periodic notes no longer trigger silence finalization. Frame-length locks require a matching sync sequence and an intact header, including after channel EQ. Final analysis ranks candidates by codeword agreement. Tests verify that the framing changes preserve the requested payload melody's waveform sample for sample, and that the actual intro audio follows The Lick with matching pilot calibration while the outro follows the solo table with the end preamble held on E. That held E is verified load-bearing: the damaged-header maximum-length Opus case at 40 dB tilt recovers with a pitch-stable end marker and fails without one, independent of the outro melody's register.

Block decoding also retries supported frame boundaries with guard-based clock tracking after failed ordinary alignments. At most 32 fallback timing grids adjust each next boundary by up to 2.5 ms. They share the streaming receiver's guard measurements, but smooth phase corrections and hold the measured rate when filtering obscures a guard. This prevents noisy guards from accumulating a whole-symbol alignment error. Recording deadlines and validated completion probes include bounded slow-clock margin. A supported header can supply an exact endpoint when trailing background exceeds ordinary alignment slop; delivery still requires CRC or caller validation. Each retry carries only its own frame's length lock, and failed completion probes resume scanning for later senders. This changes neither the transmitted music nor its integrity checks.

Clock mismatch also shifts received formant frequencies. A failed clocked candidate can reuse its FFT cache for a frequency-normalized F2-template retry when both halves support a consistent rate. An amplitude-envelope estimate handles echoes that obscure the guards: an eight-slot coarse correlation bounds a 24-slot refinement, with agreement between recording halves. This enables at most 32 additional phase hypotheses over the first two nominal recording slots, before the unchanged nominal/guard searches. The estimate supplies neither synchronization nor a payload length. Each hypothesis must pass the local header check or full preamble-and-pilot check, then CRC/caller validation; known conflicting lengths are not overridden. Unchecked profiles do not use these retries. No audio is resampled or replaced, and successful endpoints still identify original recorded samples.

Intact length headers found at other timing offsets are retained with their frame starts. Provisional and final decodes reuse this checked evidence instead of repeatedly searching lengths for an incomplete frame. Conflicting retained hints cannot supply a length lock, later frames keep their own lengths, and new recordings clear all hints. Clock retries still require sync/header evidence at the candidate's own timing offset. This fixes the maximum-length slow-clock browser timeout without extending its limit or changing synthesis.

Consistent clock measurements can postpone an otherwise-premature probe for slow frames or add an earlier probe for fast frames. The estimate requires clear guards and an intact header on the tracked grid; both measured halves must agree in direction and rate. Shifts stay within the original clock-slack bound. The original worst-case fallback remains available, as does the nominal deadline for fast frames. Short or ambiguous measurements keep the original timing. Tests cover gain changes, wrong-length rejection, acquisition transients, and senders that speed up or slow down after the header.

The best locally verified drifting-clock phase can supply one prioritized decoding attempt. It rechecks its header, cannot jump ahead of an earlier retained frame, and cannot use conflicting length evidence. Every success still requires CRC or caller validation. On failure, all 32 original nominal offsets and up to 32 fallback clock grids remain available in their original order, with unchanged combining observations. Clean nominal-rate prefixes add no priority grid. The single hint is cleared on delivery, reset, or reacquisition; tests also reject incomplete bodies, stale sync evidence, and bad-CRC frames before a later sender.

Filtered tails and phone noise can mask strict silence. With a supported length, four windows below 60% of the recent voice level may prompt a non-finalizing completion probe once the frame could fit. When a complete sync sequence survives but no trustworthy length is available, three quiet windows can prompt analysis before another sender starts. This fallback never invents a length or overrides known/conflicting header evidence. Both paths require CRC or caller validation and share a four-probe budget per recording. A failed probe preserves the audio and keeps receiving; it cannot turn a quiet musical phrase into an endpoint. Tests exercise 40 dB phrases with intact and damaged headers, unchecked profiles, budget limits, and lifecycle. The original strict silence/periodicity gate is unchanged.

The transfer profile runs this checked probe before skipping silent windows in its preview. A damaged-header frame can therefore complete during a truly silent three-slot gap, not just a periodic denoiser tail. Rejection still preserves the copy and the original fourth-slot failure finalization. Native tests at both sample rates and a real-browser worker regression check delivery before the next sender, without extra silence or fabricated length hints.

A successful block decode preserves audio already recorded after its codeword, ahead of unread callback samples. The boundary comes from the winning timing grid's last voiced codeword window, not an assumed end marker or nominal duration. This prevents delayed recovery from swallowing the next song's preamble. Tests cover missing end markers, back-to-back maximum/short frames with different clocks and melodies, repeated payloads, whole-recording callbacks, and bit-exact suffix ordering. Failed completion probes do not discard or replay their recording.

Validated delivery also retains two slots of real preceding audio as separate analysis context. A noisy, fast next song can need an alignment that starts slightly before the previous codeword's boundary. This context is not replayed through live preamble detection: it is used once at the next contiguous acquisition and discarded when listening history slides, clears on silence, or resets. Unchecked profiles do not use it. Tests check exact samples, size bounds, header positions, lifecycle, and recovery of short neighboring melodies.

The streaming receiver shares the block decoder's normalized, DC-rejected periodicity check: a quiet voiced phrase stays data rather than becoming silence or erasures. Streaming regressions cover two-second phrases 20/40/60 dB below the surrounding audio, every vowel across the chromatic register and ±30-cent detuning, and a finite capture spike followed by quiet notes. Real silence and low-level nonperiodic noise still end a stream after four slots. No transmitter gain, notes, rhythm, or framing change is required.

Streaming timing tests cover fractional-slot starts, changing gaps between messages, interleaver depths 1/2/4, phone filtering, noise, and a damaged sync symbol. Both wire profiles recover across all 20 chromatic pitches and ±30-cent detuning. When F1 overrides the F2 vowel decision, a spectral-template check can support the F2 reading; this fixes a clean detuned D♯4 misclassification without changing the sung note. Silence completion needs the fourth complete 50 ms analysis window, not its unused final guard, so a short capture tail does not stall an otherwise complete message. A partial quiet voiced window remains buffered and is never consumed as silence.

Block and streaming receivers replace NaN/infinite input samples with zero before buffering, level tracking, or spectral analysis. These are sample-length dropouts: no audio is removed, and every finite sample is preserved bit-for-bit, without clipping or normalization. This also protects the transfer/WASM and PHY paths that use the block decoder. Regressions cover invalid idle audio, sparse invalid samples throughout every note/vowel and detuned melodies, fragmented callbacks, consecutive streams, and data/ACK exchange. Larger dropouts still depend on the available error-correction capacity.

An idle-noise acquisition cannot discard an arriving song just because its recording deadline expires: a failed deadline retains three seconds of overlap for reacquisition, with a bounded recording size and the same integrity checks. Both sample rates test maximum-length melodies straddling that deadline, repeated bounded rollovers, and release on subsequent silence.

When the length header is damaged, a complete preamble/pilot match can still preserve a late start and enough room for a maximum-length song. This does **not** lock or invent a payload length: later headers are still scanned, and delivery still requires CRC validation. A regression rejects an RS-valid, bad-CRC frame and recovers the following valid frame without resetting.

The decoder also reads the length from the music itself. The framing melody after the payload is fixed, so **the bar where Coltrane's solo enters is a headerless length field**: final analysis scores every candidate length by how well the outro's expected notes align (measured exact on 70/70 lab cases through channels where 137 of 200 bytes were corrupt), and feeds the winner in as one prioritized, CRC-gated candidate. A frame whose length header is destroyed, followed by trailing voiced background, now delivers — the receiver finds your data's end by listening for where the solo starts. The hint never locks a length, never outranks a real header, and never bypasses CRC; Legacy frames (no melody transition) are excluded. The same lab measured melody *mismatch* flags as RS erasures and rejected them: pitch survives what flips vowels — the melody witnesses structure, not damage.

Transfer CRC validation runs inside each hard-decision, erasure, alternate-length, and Chase attempt, including soft-combined receptions. A plausible but CRC-invalid RS result therefore cannot hide a later correct recovery. `decode_symbol_stream_validated` exposes the same behavior for external combiners. Regressions cover hard-RS miscorrection followed by erasure or Chase recovery, a valid-looking wrong-length prefix, and preservation of the internal CRC and length lock.

Transfer combining retains the first classified stream for alignment plus each position's vowel counts and maximum confidences. Retry storage and per-copy alignment work do not grow with the number of copies; every failed copy still contributes to the original vote. Observations are harvested once, and a CRC-verified solo or combined delivery retires its history without discarding buffered audio for later messages. `attempts()` remains a lifetime failed-copy counter for UI progress. Reference tests compare accumulated votes and decisions bit-for-bit with full-history voting, and a 4,096-copy test checks fixed storage without lost votes. Copies before delivery are assumed to belong to one transfer; create a fresh receiver to abandon an unfinished transfer. Repeated transmissions may deliver the same payload again—the web UI separately latches the first verified result.

Speculative trials retain combining observations only when a start marker or recorded start hint anchors their data region. An ending-like noise fragment alone cannot establish a short reference that truncates later copies, overwrite an aligned observation, or increment the failed-copy count. The same blind candidates still get integrity-checked solo decode attempts, so a damaged start does not remove that recovery path. Regressions cover noise before complementary failed copies, preservation of existing observations, and start hints beyond the currently classified prefix.

The robustness tests reject incorrect payloads even in channels where recovery is optional. Standard frames have an independent CRC-32 check; Legacy frames retain heuristic agreement gates and can miscorrect beyond their correction capacity. CRC detects accidental corruption, not deliberate forgery. These deterministic simulations are regression coverage, not a guarantee for every melody or real-world channel.

Short recordings can have too few repeated beats for the 24-slot clock estimate. If that estimate fails and the recording has fewer than four long-lag repeats per half, the receiver tries 16-, 12-, then eight-slot correlations with the same stability thresholds. The first supported estimate supplies the same bounded 32-phase search; longer recordings keep the original calculation. Tests cover every 1–16-byte length, wide leaps, detuning and Giant Steps through 5/25/50/60/65/120 ms half-amplitude echoes at nominal and ±2% clocks, plus integrity and next-song checks at both sample rates.

Exact digital silence now releases a short false acquisition even when no post-pilot voice initialized its level estimate. Pilot collection and the four-/six-window silence thresholds are unchanged. An acquisition with no room beyond sync and a length header returns to listening without adding a failed-copy observation or discarding unread audio. Tests check all profiles and recovery of an already-buffered next song.

## Receiver compute

Severe echoes still cause long synchronous calls. A 48 kHz native run on the development Mac measured a roughly 10-second callback stall for a 120-byte transfer with a 25 ms half-amplitude echo and slow clock, even though total compute was less than the recording's duration. Recovery success and average throughput do not establish real-time callback safety.

`cargo run --release --example receiver_latency` measures native Standard-decoder cost in 2048-sample callbacks for a clean melody, a maximum-length melody after idle, and periodic background alone. Add `-- --transfer` to measure the web app's TransferRx profile, `-- --transfer --clock` for its 64/120-byte clock-drift workloads, or `--features 8khz` for the lower sample rate. `-- --echo` combines 25/60 ms half-amplitude echoes with clock drift for the default Giant Steps melody; `-- --transfer --echo` measures the armored transfer melody. It reports total compute, callback p99/maximum, modeled queue debt, and verified deliveries. These are native measurements, not browser timing guarantees; long background-finalization and clock-retry callbacks still need improvement.

The [web receiver](web/README.md) runs its unchanged WASM decoder in a dedicated classic worker. An acknowledged FIFO preserves capture order and caller-owned buffers, with a 120-second queued/in-flight sample bound and visible failure on overload. Explicit resets reject stale results; successful reception does not discard queued audio. Unit and real-Chrome tests cover capture cancellation, sequential delivery, delayed results after clear, failure/retry, and main-thread responsiveness during a 72-second periodic-background recovery case. This isolates heavy decoding from the UI; the native/WASM API remains synchronous, and physical-device throughput still needs testing.

FFT planning, Hann coefficients, and scratch buffers are reused in one replaceable workspace per thread. Tests compare spectra bit-for-bit with the original calculation. A frame-size preflight also avoids classifier/Chase retries when no supported frame could fit the observed region; it preserves the existing length and trailing-slop rules, without adding signal-level or pitch restrictions.

Within each alignment, header checks, pilots, and classifier retries share an `AlignedSpectra` view of the immutable recording. Every requested voiced window gets one full-size FFT; only the 194 bins used by the classifiers are retained, including the harmonic detector's neighboring bins. The cache is discarded on the next alignment, keeping memory bounded. Tests compare every classification field, pilot EQ, and calibration bit-for-bit against the original full-spectrum methods at both sample rates, including detuning, quiet notes, silence, noise, and partial-window boundaries.

Template matching precomputes fixed amplitudes and prefix norms and stores the four vowels together at each bin. On AArch64 targets with NEON, independent vowel comparisons use vector multiply/add operations; other targets retain the portable implementation. Every vowel keeps its original increasing-bin arithmetic order, without fused operations or reordered sums, and all 20 pitches and four vowels remain in the search. Bit-identical tests cover every cached amplitude and prefix norm, both dot-product implementations, mixed notes, detuning, EQ, and partial spectra.

Reed–Solomon uses one fixed 64 KiB multiplication table throughout encoding, correction, and syndrome evaluation, shares initial syndrome checks between hard and erasure retries of the same received bytes, and builds encoder-only generator polynomials lazily. The error-locator loop reserves two work buffers once, then swaps, scales, and adds in place instead of allocating temporary polynomials per iteration. Tests compare every field product with both the original logarithm/exponent calculation and independent carry-less polynomial multiplication, and every parity/erasure budget with the original locator loop. End-to-end reference checks cover correction/retry behavior at every supported profile length. The transmitted bytes, musical waveform, integrity gates, and recovery search are unchanged by these compute optimizations.

When more than one candidate body needs RS decoding, a hypothesis-local `PrefixDecoder` reuses intermediate syndrome values across lengths. Its first request follows the ordinary path with no prefix table; subsequent requests compute each needed root/prefix pair once, over at most 255 bytes and 255 roots. The received bytes are borrowed immutably, and corrected codewords still receive a full independent syndrome verification. Reference tests compare every valid prefix length and parity width, plus hard/erasure results at every supported profile length.

An analysis-local failure cache avoids correcting an unchanged payload again during header-only Chase flips or classifier/alignment retries. It compares the full codeword, profile, payload length, and ordered erasure positions, retains at most one failed body per supported length, and is dropped after the analysis (or standalone symbol-stream call). Speculative end-marker probes also share one cache across their classifier rungs, discarded after each probe. Profile-valid bodies and header/agreement or caller-validator rejections are not cached. Tests compare cached and uncached decoding, including changed erasures, changed payload decisions, Legacy acceptance gates, and a stateful validator that accepts an earlier rejection on retry. Probe-level reference checks preserve delivery bytes/endpoints, combining observations, and length-search counts while requiring fewer repeated corrections.

Erasure selection shares one lazy confidence ordering across all candidate lengths within a fixed hypothesis. Each length selects only its own weak bytes and retains the same correction budget and stable tie order. The view covers at most 255 bytes and is rebuilt when a Chase hypothesis changes confidences; clean hard-decision decodes never sort it. Reference comparisons cover every body length and parity budget from 0 to 255, including missing confidence values, threshold boundaries, ties, NaNs, and infinities.

Chase retries reuse exact Reed–Solomon syndrome values from the original received body, updating only the coefficients changed by each hypothesis. This failure-path cache covers at most 255 bytes and 255 roots, is allocated only after the initial decode fails and Chase candidates exist, and is discarded after that symbol-stream call. Each hypothesis still uses its actual bytes and erasure positions, and corrected codewords receive the original full syndrome verification. Reference tests compare every prefix/parity width, correction and erasure results at all profile lengths, and complete retry results plus stateful validator-call order. No recovery candidates, integrity checks, or transmitted samples are removed.

## Architecture

| Module | Purpose |
|---|---|
| `src/lib.rs` | Public API (`encode`, `encode_armored`, `Decoder`) |
| `src/protocol.rs` | Constants, ECC profiles, and the setlist (`GIANT_STEPS`, `GIANT_STEPS_SOLO`, `FRAMING_LICK`, `MELODY`) |
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

And when your bytes arrive — through the codec, through the denoiser, through the tilt — the last thing the channel hears is Coltrane, cut off mid-phrase, resolving to E. He'd have kept going. The parity ran out. *sống.*
