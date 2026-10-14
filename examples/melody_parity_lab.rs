//! Melody-as-parity feasibility lab. VERDICTS (measured 2026-09-11):
//!
//! 1. Melody-mismatch ERASURE flags: NOT viable. Recall is 5–10% because
//!    pitch is deliberately the most robust part of the signal — a window
//!    corrupted enough to flip its vowel still verifies its note, so the
//!    flag only fires on total destruction, which low classifier
//!    confidence already marks. 41 transition-band cases, 0 wins over
//!    confidence-ordered erasures.
//!
//! 2. Length-from-melody: STRONGLY viable — and the signal is the OUTRO,
//!    not the head. Scoring candidate payload lengths by how well the
//!    fixed framing melody (Coltrane's solo + pinned end E) aligns after
//!    the hypothesized payload recovers the exact length 70/70 across all
//!    channels tested, including ones where 137/200 body bytes are wrong
//!    and no decode is possible. Works for ANY payload melody (custom
//!    tracks included) because the framing melody is deterministic per
//!    profile: the solo's start position IS a headerless length field.
//!    Set LAB_OUTRO_ONLY=1 to score the outro term alone (sharper:
//!    0.87–0.93 match scores through tilt+RNNoise).
//!
//! It changes no production code. Usage:
//!   cargo run --release --example melody_parity_lab

use nnnoiseless::DenoiseState;
use rustfft::{num_complex::Complex, FftPlanner};
use song_rs::formant;
use song_rs::protocol::{
    self, EccProfile, ENCODED_DATA_OFFSET, PILOT_LEN, PREAMBLE_LEN, SAMPLES_PER_SYMBOL,
    SYMBOL_TOTAL_SAMPLES,
};
use song_rs::reed_solomon::ReedSolomon;

const SAMPLE_RATE: f64 = 48000.0;

/// Same chromatic bank as formant::TEMPLATE_NOTES (B2..F#4).
const NOTE_BANK: [f64; 20] = [
    123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00,
    233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99,
];

fn nearest_note(hz: f64) -> usize {
    NOTE_BANK
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - hz).abs().partial_cmp(&(b.1 - hz).abs()).unwrap())
        .unwrap()
        .0
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

// ── Channels (copied from examples/giant_steps.rs) ───────────────────

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    (*state >> 16) as f32 / 65535.0
}

fn normal(state: &mut u32) -> f32 {
    let u1 = lcg(state).max(1e-10);
    let u2 = lcg(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn add_noise(samples: &[f32], snr_db: f64, seed: u32) -> Vec<f32> {
    let sig_power: f64 =
        samples.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / samples.len() as f64;
    let noise_rms = sig_power.sqrt() / 10.0f64.powf(snr_db / 20.0);
    let mut st = seed;
    samples.iter().map(|&s| s + normal(&mut st) * noise_rms as f32).collect()
}

fn teams_spectral_filter(samples: &[f32], f1_atten_db: f64, f0_boost_db: f64) -> Vec<f32> {
    let n = samples.len();
    let hz_per_bin = SAMPLE_RATE / n as f64;
    let mut buf: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();
    let mut planner = FftPlanner::new();
    planner.plan_fft_forward(n).process(&mut buf);
    for (i, c) in buf.iter_mut().enumerate() {
        let freq = if i <= n / 2 { i as f64 * hz_per_bin } else { (n - i) as f64 * hz_per_bin };
        let gain_db = if freq < 250.0 {
            f0_boost_db * (1.0 - (freq - 208.0).abs() / 50.0).max(0.0)
        } else if (400.0..=850.0).contains(&freq) {
            -f1_atten_db
        } else if freq > 850.0 && freq < 2000.0 {
            -f1_atten_db * (1.0 - (freq - 850.0) / 1150.0)
        } else if (2000.0..=2500.0).contains(&freq) {
            -f1_atten_db * 0.3
        } else {
            0.0
        };
        let gain = 10.0f64.powf(gain_db / 20.0) as f32;
        *c = Complex::new(c.re * gain, c.im * gain);
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f32).collect()
}

fn apply_rnnoise(samples: &[f32]) -> Vec<f32> {
    let frame_size = DenoiseState::FRAME_SIZE;
    let mut state = DenoiseState::new();
    let mut out = Vec::with_capacity(samples.len());
    let scale_in = 32767.0f32;
    let scale_out = 1.0 / 32767.0f32;
    let mut input_frame = vec![0.0f32; frame_size];
    let mut output_frame = vec![0.0f32; frame_size];
    let first_len = frame_size.min(samples.len());
    input_frame[..first_len].copy_from_slice(&samples[..first_len]);
    for s in &mut input_frame[..first_len] {
        *s *= scale_in;
    }
    state.process_frame(&mut output_frame, &input_frame);
    state.process_frame(&mut output_frame, &input_frame);
    for s in &output_frame[..first_len] {
        out.push(s * scale_out);
    }
    let mut pos = first_len;
    while pos < samples.len() {
        let chunk_len = frame_size.min(samples.len() - pos);
        input_frame[..chunk_len].copy_from_slice(&samples[pos..pos + chunk_len]);
        for s in &mut input_frame[chunk_len..] {
            *s = 0.0;
        }
        for s in &mut input_frame[..chunk_len] {
            *s *= scale_in;
        }
        state.process_frame(&mut output_frame, &input_frame);
        for s in &output_frame[..chunk_len] {
            out.push(s * scale_out);
        }
        pos += chunk_len;
    }
    out
}


/// Amplitude-spectrum references for every (note, vowel), from the real
/// synthesizer. Correlation happens in the F2 band the channel preserves.
fn reference_spectra() -> Vec<[Vec<f32>; 4]> {
    NOTE_BANK
        .iter()
        .map(|&hz| {
            std::array::from_fn(|v| {
                let slot = formant::synthesize_symbol(v, hz, 0.5, 0);
                let voiced: Vec<f32> =
                    slot[..SAMPLES_PER_SYMBOL].iter().map(|&s| s as f32).collect();
                let mut spec = vec![0.0f32; SAMPLES_PER_SYMBOL];
                song_rs::fft::power_spectrum(&voiced, &mut spec);
                spec.iter().map(|&p| p.max(0.0).sqrt()).collect()
            })
        })
        .collect()
}

const BAND_LO: usize = 50; // 1000 Hz at 20 Hz/bin
const BAND_HI: usize = 125; // 2500 Hz

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in BAND_LO..BAND_HI {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

// ── The experiment ───────────────────────────────────────────────────

struct CaseResult {
    wrong_bytes: usize,
    est_len: usize,
    est_score: f64,
    ecc: usize,
    flagged_bytes: usize,
    true_flags: usize,
    conf_captured: usize,
    hard: bool,
    conf: bool,
    melody: bool,
    combined: bool,
}

fn run_case(refs: &[[Vec<f32>; 4]], payload: &[u8], volume: u8, channel: impl Fn(&[f32]) -> Vec<f32>) -> CaseResult {
    run_case_with(refs, payload, volume, &channel, None)
}

fn run_case_with(refs: &[[Vec<f32>; 4]], payload: &[u8], volume: u8, channel: &dyn Fn(&[f32]) -> Vec<f32>, custom_track: Option<&[f64]>) -> CaseResult {
    let len = payload.len();
    let audio = song_rs::encoder::encode_with_profile(payload, volume, custom_track, EccProfile::Standard).unwrap();
    let received = channel(&audio);
    assert!(received.len() >= audio.len(), "channel must not shorten audio");

    // Replicate the encoder's body geometry to know ground truth.
    let mut protected = payload.to_vec();
    protected.extend_from_slice(&crc32(payload).to_le_bytes());
    let ecc = EccProfile::Standard.ecc_bytes(len);
    let rs = ReedSolomon::new(protected.len(), ecc);
    let sent_body = rs.encode(&protected);
    let sent_syms = formant::bytes_to_symbols(&sent_body);

    let payload_start = PREAMBLE_LEN + PILOT_LEN + 4 * ENCODED_DATA_OFFSET;
    let payload_end = payload_start + 4 * len;
    let total_symbols = audio.len() / SYMBOL_TOTAL_SAMPLES;
    let track: Vec<f64> = custom_track.map(|t| t.to_vec()).unwrap_or_else(|| protocol::default_byte_pitch(len));

    // Classify + note-detect each body symbol at true alignment.
    let mut got_syms = Vec::with_capacity(sent_syms.len());
    let mut confs = Vec::with_capacity(sent_syms.len());
    let mut note_ok = Vec::with_capacity(sent_syms.len());
    let mut deltas: Vec<isize> = Vec::with_capacity(sent_syms.len());
    for (k, _) in sent_syms.iter().enumerate() {
        let pos = payload_start + k;
        let start = pos * SYMBOL_TOTAL_SAMPLES;
        let window: Vec<f32> = received[start..start + SAMPLES_PER_SYMBOL].to_vec();
        let mut spectrum = vec![0.0f32; SAMPLES_PER_SYMBOL];
        song_rs::fft::power_spectrum(&window, &mut spectrum);
        let (vowel, conf) = formant::classify_symbol(&spectrum, &formant::EXPECTED_F2_CENTROIDS);
        let expected_hz = if pos < payload_end {
            track[(pos - payload_start) / 4]
        } else {
            protocol::outro_f0(pos - payload_end, total_symbols - payload_end)
        };
        got_syms.push(vowel);
        confs.push(conf);
        // Verification, not identification: does the expected note explain
        // this window about as well as the best (note, vowel) template?
        let amp: Vec<f32> = spectrum.iter().map(|&p| p.max(0.0).sqrt()).collect();
        let n_e = nearest_note(expected_hz);
        let best_expected = (0..4).map(|v| cosine(&amp, &refs[n_e][v])).fold(0.0f64, f64::max);
        let mut best_all = 0.0f64;
        let mut best_note = n_e;
        for (n, note_refs) in refs.iter().enumerate() {
            for r in note_refs {
                let c = cosine(&amp, r);
                if c > best_all {
                    best_all = c;
                    best_note = n;
                }
            }
        }
        deltas.push(best_note as isize - n_e as isize);
        let gamma: f64 = std::env::var("LAB_GAMMA").ok().and_then(|g| g.parse().ok()).unwrap_or(0.85);
        note_ok.push(best_expected >= gamma * best_all);
    }

    let got_body = formant::symbols_to_bytes(&got_syms, sent_body.len());

    // Byte-level ground truth and melody flags.
    let n = sent_body.len();
    let wrong: Vec<bool> = (0..n).map(|i| got_body[i] != sent_body[i]).collect();
    let flag_score: Vec<usize> =
        (0..n).map(|i| (0..4).filter(|s| !note_ok[4 * i + s]).count()).collect();
    let byte_conf: Vec<f64> = (0..n)
        .map(|i| (0..4).map(|s| confs[4 * i + s]).fold(f64::MAX, f64::min))
        .collect();

    let wrong_bytes = wrong.iter().filter(|&&w| w).count();
    let flagged: Vec<usize> = (0..n).filter(|&i| flag_score[i] > 0).collect();
    let true_flags = flagged.iter().filter(|&&i| wrong[i]).count();

    // A candidate correction must reproduce the payload CRC (the real gate).
    let valid = |decoded: &Option<Vec<u8>>| -> bool {
        decoded.as_ref().is_some_and(|d| {
            d.len() >= len + 4
                && crc32(&d[..len]) == u32::from_le_bytes(d[len..len + 4].try_into().unwrap())
                && d[..len] == *payload
        })
    };

    let hard = valid(&rs.decode(&got_body));

    // Budget sweep mimicking the ladder: any budget that lands a CRC-valid
    // payload counts as success.
    let sweep = |order: &[usize]| -> bool {
        let mut budget = 2;
        while budget <= ecc.min(order.len()) {
            let erase: Vec<usize> = order[..budget].to_vec();
            if valid(&rs.decode_with_erasures(&got_body, &erase)) {
                return true;
            }
            budget += 2;
        }
        false
    };

    let mut by_conf: Vec<usize> = (0..n).collect();
    by_conf.sort_by(|&a, &b| byte_conf[a].partial_cmp(&byte_conf[b]).unwrap());
    let conf_ok = sweep(&by_conf);
    // How many actually-wrong bytes sit inside the lowest-W confidence
    // positions? (W = wrong count). If this equals W, confidence ordering
    // already finds the damage and melody flags are redundant.
    let conf_captured =
        by_conf[..wrong_bytes.min(n)].iter().filter(|&&i| wrong[i]).count();

    let mut by_flag = flagged.clone();
    by_flag.sort_by(|&a, &b| {
        flag_score[b]
            .cmp(&flag_score[a])
            .then(byte_conf[a].partial_cmp(&byte_conf[b]).unwrap())
    });
    let melody_ok = sweep(&by_flag);

    let mut combined = by_flag.clone();
    for &i in &by_conf {
        if !combined.contains(&i) {
            combined.push(i);
        }
    }
    let combined_ok = sweep(&combined);

    // ── Length-from-melody: where does the head hand off to the solo? ──
    // For each candidate payload length L, score how well every body
    // byte's spectrum matches the melody that L would imply (head note for
    // bytes < L, solo schedule for bytes >= L). The argmax is a length
    // estimate that needs no header. Uses per-symbol best-vowel cosine to
    // the note's templates, i.e. the same verification signal as flags.
    let est = {
        let mut sym_note_sim = vec![[0.0f64; 20]; sent_syms.len()];
        for (k, sims) in sym_note_sim.iter_mut().enumerate() {
            let posn = payload_start + k;
            let start = posn * SYMBOL_TOTAL_SAMPLES;
            let window: Vec<f32> = received[start..start + SAMPLES_PER_SYMBOL].to_vec();
            let mut spectrum = vec![0.0f32; SAMPLES_PER_SYMBOL];
            song_rs::fft::power_spectrum(&window, &mut spectrum);
            let amp: Vec<f32> = spectrum.iter().map(|&p| p.max(0.0).sqrt()).collect();
            for (n2, note_refs) in refs.iter().enumerate() {
                sims[n2] = note_refs.iter().map(|r| cosine(&amp, r)).fold(0.0f64, f64::max);
            }
        }
        let full_track = protocol::default_byte_pitch(protocol::MAX_LENGTH_VARIABLE);
        let mut best_l = 0usize;
        let mut best_score = f64::MIN;
        for l in 1..=protocol::MAX_LENGTH_VARIABLE {
            // A candidate length L implies body length L+4+ecc(L); only
            // score candidates whose body fits the observed symbols.
            let cand_body = l + 4 + EccProfile::Standard.ecc_bytes(l);
            if cand_body * 4 > sent_syms.len() {
                break;
            }
            let outro_only = std::env::var("LAB_OUTRO_ONLY").is_ok();
            let mut score = 0.0f64;
            let mut terms = 0usize;
            for j in 0..cand_body {
                for s in 0..4 {
                    let sym = 4 * j + s;
                    if j < l {
                        if !outro_only {
                            score += sym_note_sim[sym][nearest_note(full_track[j])];
                            terms += 1;
                        }
                    } else {
                        let hz = protocol::outro_f0(sym - 4 * l, 4 * (cand_body - l) + PREAMBLE_LEN);
                        score += sym_note_sim[sym][nearest_note(hz)];
                        terms += 1;
                    }
                }
            }
            score /= terms.max(1) as f64;
            if score > best_score {
                best_score = score;
                best_l = l;
            }
        }
        (best_l, best_score)
    };
    let (est_len, est_score) = est;

    if std::env::var("LAB_DELTAS").is_ok() {
        let mut hist = std::collections::BTreeMap::new();
        for &d in &deltas {
            *hist.entry(d).or_insert(0usize) += 1;
        }
        eprintln!("  delta histogram (detected - expected semitones): {hist:?}");
    }

    CaseResult {
        wrong_bytes,
        est_len,
        est_score,
        ecc,
        flagged_bytes: flagged.len(),
        true_flags,
        conf_captured,
        hard,
        conf: conf_ok,
        melody: melody_ok,
        combined: combined_ok,
    }
}

fn main() {
    let refs = reference_spectra();
    let mut channels: Vec<(String, Box<dyn Fn(&[f32]) -> Vec<f32>>)> = vec![
        ("clean".into(), Box::new(|s: &[f32]| s.to_vec()) as Box<dyn Fn(&[f32]) -> Vec<f32>>),
    ];
    for snr10 in [-50i32, -55, -60, -63, -65, -68, -70, -75] {
        let snr = snr10 as f64 / 10.0;
        for seed in [7u32, 8, 9] {
            channels.push((
                format!("noise_{snr}db_s{seed}"),
                Box::new(move |s: &[f32]| add_noise(s, snr, seed)),
            ));
        }
    }
    for (tilt, snr) in [(40.0, 25.0), (40.0, 21.0), (45.0, 21.0), (45.0, 18.0), (50.0, 20.0)] {
        for seed in [7u32, 8] {
            channels.push((
                format!("tilt{tilt}_n{snr}_rnn_s{seed}"),
                Box::new(move |s: &[f32]| {
                    apply_rnnoise(&add_noise(&teams_spectral_filter(s, tilt, 15.0), snr, seed))
                }),
            ));
        }
    }

    println!(
        "{:<22} {:>4}  {:>5} {:>4} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>4} {:>4} {:>5}",
        "channel", "len", "wrong", "ecc", "flags", "hits", "confc", "hard", "conf", "mldy", "both", "estL"
    );
    let mut interesting = 0usize;
    let mut melody_wins = 0usize;
    let mut len_exact = 0usize;
    let mut len_total = 0usize;
    for (name, channel) in &channels {
        for &(len, vol) in &[(64usize, 50u8), (140, 50)] {
            let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
            let r = run_case(&refs, &payload, vol, channel);
            // Only print the transition band (and clean as sanity).
            let transition = r.wrong_bytes > r.ecc / 2 && r.wrong_bytes <= r.ecc + 10;
            if transition || name == "clean" || r.melody != r.conf || r.combined != r.conf {
                println!(
                    "{:<22} {:>4}  {:>5} {:>4} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>4} {:>4} {:>5}",
                    name, len, r.wrong_bytes, r.ecc, r.flagged_bytes, r.true_flags,
                    r.conf_captured, r.hard as u8, r.conf as u8, r.melody as u8, r.combined as u8,
                    r.est_len
                );
            }
            if r.est_len == len {
                len_exact += 1;
            }
            len_total += 1;
            if transition {
                interesting += 1;
                if (r.combined && !r.conf) || (r.melody && !r.conf) {
                    melody_wins += 1;
                }
            }
            if name == "clean" {
                assert_eq!(r.wrong_bytes, 0, "clean channel must classify perfectly");
            }
        }
    }
    println!("\ntransition-band cases: {interesting}; melody-erasure wins over confidence: {melody_wins}");
    println!("melody-derived length exact: {len_exact}/{len_total} across ALL cases (including ones far beyond decodability)");

    // Gate check: a custom-melody sender must be distinguishable, or the
    // hint would mislead. Compare match scores: default melody vs a
    // shuffled custom track, clean and through tilt+RNNoise.
    println!("\ngate check (argmax score; default melody should score high, custom low):");
    for &(custom, ch) in &[(false, false), (false, true), (true, false), (true, true)] {
        let len = 64usize;
        let payload: Vec<u8> = (0..len).map(|i| (i * 73 + 41) as u8).collect();
        let channel: Box<dyn Fn(&[f32]) -> Vec<f32>> = if ch {
            Box::new(|s: &[f32]| apply_rnnoise(&add_noise(&teams_spectral_filter(s, 40.0, 15.0), 25.0, 7)))
        } else {
            Box::new(|s: &[f32]| s.to_vec())
        };
        let r = if custom {
            let track: Vec<f64> = (0..len)
                .map(|i| NOTE_BANK[(i * 7 + 3) % 20])
                .collect();
            run_case_with(&refs, &payload, 50, &channel, Some(&track))
        } else {
            run_case_with(&refs, &payload, 50, &channel, None)
        };
        println!(
            "  melody={} channel={}  estL={} (true {len})  score={:.4}",
            if custom { "custom " } else { "default" },
            if ch { "tilt+rnnoise" } else { "clean       " },
            r.est_len, r.est_score
        );
    }
    println!("wrong: corrupted body bytes; ecc: parity budget; confc: wrong bytes inside lowest-W confidences");
    println!("flags: bytes with >=1 melody-mismatched symbol; hits: flags that were wrong bytes");
    println!("hard/conf/mldy/both: RS recovery by strategy (1 = CRC-valid payload recovered)");
}
