//! Sing any MIDI file: extract a monophonic melody line, fit it into the
//! modem's B2..F#4 compass, and encode a payload to it.
//!
//! Handles messy real-world MIDI (e.g. Spotify basic-pitch transcriptions:
//! octave-doubled notes, unquantized timing, extreme registers) by sampling
//! the most salient sounding note on the byte grid and auto-transposing.
//!
//! Usage: cargo run --release --example midi_melody <file.mid> [message] [out_dir]

use song_rs::Decoder;

// ── Minimal SMF parser ───────────────────────────────────────────────

struct MidiNote {
    start: f64, // seconds
    dur: f64,
    note: u8,
    vel: u8,
}

fn read_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn read_varlen(b: &[u8], pos: &mut usize) -> u32 {
    let mut val = 0u32;
    loop {
        let byte = b[*pos];
        *pos += 1;
        val = (val << 7) | (byte & 0x7F) as u32;
        if byte & 0x80 == 0 {
            return val;
        }
    }
}

/// Parse all note events across tracks, applying the tempo map.
fn parse_midi(data: &[u8]) -> Vec<MidiNote> {
    assert_eq!(&data[0..4], b"MThd", "not a MIDI file");
    let ntracks = u16::from_be_bytes([data[10], data[11]]) as usize;
    let division = u16::from_be_bytes([data[12], data[13]]) as f64;

    // First pass: tempo map (tick, us_per_qn) from all tracks.
    let mut tempos: Vec<(u64, f64)> = vec![(0, 500_000.0)];
    // Second pass fills notes.
    let mut notes = Vec::new();

    for pass in 0..2 {
        let mut off = 14usize;
        for _ in 0..ntracks {
            assert_eq!(&data[off..off + 4], b"MTrk");
            let len = read_u32(&data[off + 4..]) as usize;
            let trk = &data[off + 8..off + 8 + len];
            off += 8 + len;

            let mut pos = 0usize;
            let mut tick = 0u64;
            let mut running = 0u8;
            let mut active: Vec<(u8, u64, u8)> = Vec::new(); // note, start_tick, vel

            let tick_to_sec = |t: u64, tempos: &[(u64, f64)]| -> f64 {
                let mut sec = 0.0;
                let mut prev_tick = 0u64;
                let mut prev_tempo = 500_000.0;
                for &(tt, us) in tempos {
                    if tt >= t {
                        break;
                    }
                    sec += (tt - prev_tick) as f64 * prev_tempo / (division * 1e6);
                    prev_tick = tt;
                    prev_tempo = us;
                }
                sec + (t - prev_tick) as f64 * prev_tempo / (division * 1e6)
            };

            while pos < trk.len() {
                tick += read_varlen(trk, &mut pos) as u64;
                let b = trk[pos];
                if b == 0xFF {
                    let meta = trk[pos + 1];
                    pos += 2;
                    let mlen = read_varlen(trk, &mut pos) as usize;
                    if meta == 0x51 && pass == 0 {
                        let us = ((trk[pos] as u32) << 16
                            | (trk[pos + 1] as u32) << 8
                            | trk[pos + 2] as u32) as f64;
                        tempos.push((tick, us));
                    }
                    pos += mlen;
                    running = 0;
                } else if b == 0xF0 || b == 0xF7 {
                    pos += 1;
                    let mlen = read_varlen(trk, &mut pos) as usize;
                    pos += mlen;
                    running = 0;
                } else {
                    let status = if b & 0x80 != 0 {
                        running = b;
                        pos += 1;
                        b
                    } else {
                        running
                    };
                    match status & 0xF0 {
                        0x80 | 0x90 | 0xA0 | 0xB0 | 0xE0 => {
                            let d1 = trk[pos];
                            let d2 = trk[pos + 1];
                            pos += 2;
                            if pass == 1 {
                                let kind = status & 0xF0;
                                if kind == 0x90 && d2 > 0 {
                                    active.push((d1, tick, d2));
                                } else if kind == 0x80 || (kind == 0x90 && d2 == 0) {
                                    if let Some(i) =
                                        active.iter().position(|&(n, _, _)| n == d1)
                                    {
                                        let (n, st, vel) = active.remove(i);
                                        notes.push(MidiNote {
                                            start: tick_to_sec(st, &tempos),
                                            dur: tick_to_sec(tick, &tempos)
                                                - tick_to_sec(st, &tempos),
                                            note: n,
                                            vel,
                                        });
                                    }
                                }
                            }
                        }
                        0xC0 | 0xD0 => pos += 1,
                        _ => panic!("bad MIDI status {status:#x} at {pos}"),
                    }
                }
            }
        }
        if pass == 0 {
            tempos.sort_by_key(|&(t, _)| t);
        }
    }
    notes.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    notes
}

// ── Melody extraction + compass fitting ──────────────────────────────

/// Modem compass in MIDI note numbers: B2..F#4 (the chromatic note bank).
const COMPASS_LO: i32 = 47;
const COMPASS_HI: i32 = 66;

fn midi_to_hz(m: i32) -> f64 {
    440.0 * 2.0f64.powf((m - 69) as f64 / 12.0)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: midi_melody <file.mid> [message] [out_dir]");
    let stem = std::path::Path::new(&path)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let message = args.next().unwrap_or_else(|| format!("{stem} via the song.rs vowel modem"));
    let dir = args.next().unwrap_or_else(|| "/tmp".to_string());

    let notes = parse_midi(&std::fs::read(&path).expect("read midi"));
    assert!(!notes.is_empty(), "no notes in MIDI");
    let song_end = notes.iter().map(|n| n.start + n.dur).fold(0.0, f64::max);
    println!("{}: {} notes, {:.1}s", stem, notes.len(), song_end);

    // Rhythm quantization: sampling an unquantized performance on a rigid
    // 240 ms grid aliases a steady pulse into irregular long-short
    // stutters. Instead, detect the tune's own beat unit (median
    // inter-onset interval, octave-double onsets deduplicated) and map ONE
    // unit to ONE byte slot — rhythm relationships survive exactly, and
    // the tempo lands wherever 1 unit = 240 ms puts it.
    let mut onsets: Vec<f64> = Vec::new();
    for n in &notes {
        if onsets.last().map_or(true, |&o| n.start - o > 0.03) {
            onsets.push(n.start);
        }
    }
    let mut iois: Vec<f64> = onsets.windows(2).map(|w| w[1] - w[0]).collect();
    iois.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let unit = if iois.is_empty() { 0.24 } else { iois[iois.len() / 2] }.max(0.05);
    let t0 = onsets[0];
    println!("beat unit {:.0} ms ({} onsets)", unit * 1000.0, onsets.len());

    // Quantize every note to the unit grid.
    let quant: Vec<(usize, usize, u8, u8)> = notes
        .iter()
        .map(|n| {
            let q_start = (((n.start - t0) / unit).round().max(0.0)) as usize;
            let q_dur = ((n.dur / unit).round() as usize).max(1);
            (q_start, q_dur, n.note, n.vel)
        })
        .collect();
    let n_slots = quant
        .iter()
        .map(|&(s, d, _, _)| s + d)
        .max()
        .unwrap()
        .clamp(1, song_rs::protocol::MAX_LENGTH_VARIABLE);

    // Per-slot candidates (basic-pitch transcriptions sound the same pitch
    // class in TWO octaves at once, with the stronger velocity flipping
    // between layers — picking by velocity shreds the contour into ±octave
    // zigzags). Choose one note per slot by dynamic programming that
    // minimizes the total melodic interval across the whole line, with
    // velocity only as a gentle preference. Empty slots sustain the
    // previous pitch — the modem cannot rest.
    let mut cands: Vec<Vec<(i32, u8)>> = vec![Vec::new(); n_slots];
    for &(s, d, note, vel) in &quant {
        for slot_cands in cands.iter_mut().skip(s).take(d) {
            if !slot_cands.iter().any(|&(p, _)| p == note as i32) {
                slot_cands.push((note as i32, vel));
            }
        }
    }
    for i in 0..n_slots {
        if cands[i].is_empty() {
            let prev = if i > 0 { cands[i - 1].clone() } else { vec![(notes[0].note as i32, 64)] };
            cands[i] = prev;
        }
    }
    // DP: cost[i][k] = best total cost ending at candidate k of slot i.
    let mut cost: Vec<Vec<f64>> = Vec::with_capacity(n_slots);
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(n_slots);
    cost.push(cands[0].iter().map(|&(_, v)| -0.02 * v as f64).collect());
    back.push(vec![0; cands[0].len()]);
    for i in 1..n_slots {
        let mut c_row = Vec::with_capacity(cands[i].len());
        let mut b_row = Vec::with_capacity(cands[i].len());
        for &(p, v) in &cands[i] {
            let (bk, bc) = cost[i - 1]
                .iter()
                .enumerate()
                .map(|(k, &c)| (k, c + (p - cands[i - 1][k].0).abs() as f64))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            c_row.push(bc - 0.02 * v as f64);
            b_row.push(bk);
        }
        cost.push(c_row);
        back.push(b_row);
    }
    let mut k = cost[n_slots - 1]
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let mut line = vec![0i32; n_slots];
    for i in (0..n_slots).rev() {
        line[i] = cands[i][k].0;
        k = back[i][k];
    }

    // Global transposition: try semitone shifts, keep the one placing the
    // most melody time inside the compass; ties prefer centering the
    // median note (octave-preserving contour beats register vanity).
    let center = (COMPASS_LO + COMPASS_HI) as f64 / 2.0;
    let mut best_shift = 0i32;
    let mut best_score = (-1i64, f64::MAX);
    for shift in -60..=24 {
        let inside = line.iter().filter(|&&m| {
            let s = m + shift;
            (COMPASS_LO..=COMPASS_HI).contains(&s)
        }).count() as i64;
        let mut sorted: Vec<i32> = line.iter().map(|&m| m + shift).collect();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2] as f64;
        let dist = (median - center).abs();
        if (inside, -dist) > (best_score.0, -best_score.1) {
            best_score = (inside, dist);
            best_shift = shift;
        }
    }
    println!(
        "transposing {} semitones ({} octaves{}), {}/{} slots in compass",
        best_shift,
        best_shift / 12,
        if best_shift % 12 != 0 { format!(" {:+} st", best_shift % 12) } else { String::new() },
        best_score.0,
        n_slots
    );

    // Apply shift; octave-fold any stragglers into the compass.
    let track: Vec<f64> = line
        .iter()
        .map(|&m| {
            let mut s = m + best_shift;
            while s < COMPASS_LO {
                s += 12;
            }
            while s > COMPASS_HI {
                s -= 12;
            }
            midi_to_hz(s)
        })
        .collect();

    // Payload sized to the melody.
    let mut msg = message.into_bytes();
    msg.truncate(track.len());
    msg.resize(track.len(), b' ');

    let audio = song_rs::encode_with_pitch(&msg, 50, Some(&track)).unwrap();

    // Verify the melody didn't cost the data (clean channel).
    let ok = {
        let mut dec = Decoder::new();
        match dec.decode(&audio) {
            Ok(Some(ref d)) => d == &msg,
            Err(_) => false,
            Ok(None) => {
                let silence = vec![0.0f32; 10 * song_rs::protocol::SYMBOL_TOTAL_SAMPLES];
                matches!(dec.decode(&silence), Ok(Some(ref d)) if d == &msg)
            }
        }
    };

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max).max(1e-9);
    let gain = 0.6 / peak;
    let out_path = format!("{dir}/{stem}_modem.wav");
    let mut w = hound::WavWriter::create(&out_path, spec).unwrap();
    for &s in &audio {
        w.write_sample(((s * gain) * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .unwrap();
    }
    w.finalize().unwrap();

    println!(
        "{} bytes as {:.1}s of audio, decode {} -> {}",
        msg.len(),
        audio.len() as f64 / 48000.0,
        if ok { "OK" } else { "FAILED" },
        out_path
    );
    assert!(ok, "melody broke the clean-channel decode");
}
