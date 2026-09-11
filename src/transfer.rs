//! Secret-transfer profile: zero-error-tolerance payloads over the air.
//!
//! For short high-entropy text (API keys) where a wrong byte is worse than no
//! decode. Three layers on top of the base modem:
//!
//! 1. **CRC-32 inside the payload** — the anti-miscorrection gate. RS
//!    hard-decision decoding can miscorrect and return plausible garbage as
//!    success; the CRC makes delivery of a wrong payload an explicit failure
//!    instead. Correctness, not security (encryption is out of scope by
//!    design: same-room threat model).
//! 2. **Paranoid ECC** (`EccProfile::Paranoid`): ~100% RS parity — corrects
//!    up to 50% byte corruption in the single codeword.
//! 3. **Repeat + soft combining**: the sender loops the transmission; the
//!    receiver merges symbol classifications across receptions by confidence
//!    (time diversity), so two individually-undecodable copies can combine
//!    into a good one.

use crate::decoder::{decode_symbol_stream, Decoder};
use crate::protocol::*;

/// Maximum transfer payload: payload + 4-byte CRC must fit the paranoid
/// RS(255) codeword (2·len ≤ 255 - header).
pub const MAX_TRANSFER_PAYLOAD: usize = 120;

const CRC_LEN: usize = 4;

/// CRC-32 (IEEE, reflected, poly 0xEDB88320) — bitwise, no table, no deps.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Encode a payload for transfer: CRC-32 appended, paranoid ECC.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    if payload.len() > MAX_TRANSFER_PAYLOAD {
        return Err(crate::Error::PayloadTooLarge {
            size: payload.len(),
            max: MAX_TRANSFER_PAYLOAD,
        });
    }
    let mut framed = Vec::with_capacity(payload.len() + CRC_LEN);
    framed.extend_from_slice(payload);
    framed.extend_from_slice(&crc32(payload).to_le_bytes());
    crate::encoder::encode_with_profile(&framed, volume, None, EccProfile::Paranoid)
}

/// Encode `copies` repetitions separated by `gap_ms` of silence.
/// The receiver soft-combines across copies, so more copies monotonically
/// increase the decode probability on bad channels.
pub fn encode_repeated(
    payload: &[u8],
    volume: u8,
    copies: usize,
    gap_ms: u32,
) -> Result<Vec<f32>, crate::Error> {
    let one = encode(payload, volume)?;
    let gap = vec![0.0f32; (SAMPLE_RATE as usize * gap_ms as usize) / 1000];
    let mut out = Vec::with_capacity((one.len() + gap.len()) * copies);
    for i in 0..copies.max(1) {
        if i > 0 {
            out.extend_from_slice(&gap);
        }
        out.extend_from_slice(&one);
    }
    Ok(out)
}

/// Verify and strip the CRC from a decoded frame.
fn check_crc(framed: &[u8]) -> Option<Vec<u8>> {
    if framed.len() < CRC_LEN + 1 {
        return None;
    }
    let (payload, tail) = framed.split_at(framed.len() - CRC_LEN);
    let expect = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if crc32(payload) == expect {
        Some(payload.to_vec())
    } else {
        None
    }
}

/// Receiver for repeated transfer transmissions with soft combining.
///
/// Feed audio continuously via `ingest`; returns the payload as soon as any
/// single reception decodes with a valid CRC, or when the confidence-merged
/// combination of failed receptions does.
pub struct TransferRx {
    decoder: Decoder,
    /// Aligned data-region classification attempts from failed receptions.
    attempts: Vec<(Vec<usize>, Vec<f64>)>,
}

impl Default for TransferRx {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferRx {
    pub fn new() -> Self {
        let mut decoder = Decoder::with_profile(EccProfile::Paranoid);
        // A decode whose CRC fails must not terminate the reception: keep
        // receiving so the full copy stays intact for soft combining.
        decoder.set_payload_validator(|framed| check_crc(framed).is_some());
        Self {
            decoder,
            attempts: Vec::new(),
        }
    }

    /// Number of reception attempts accumulated so far (for UI progress).
    pub fn attempts(&self) -> usize {
        self.attempts.len()
    }

    /// True while the underlying decoder is locked onto a transmission
    /// (mid-reception), false while idle-listening.
    pub fn is_locked(&self) -> bool {
        self.decoder.is_receiving()
    }

    /// Feed audio. Returns the verified payload once available.
    pub fn ingest(&mut self, samples: &[f32]) -> Option<Vec<u8>> {
        match self.decoder.decode(samples) {
            Ok(Some(framed)) => {
                if let Some(payload) = check_crc(&framed) {
                    return Some(payload);
                }
                // RS "success" with bad CRC = miscorrection; treat as a
                // failed attempt and fall through to combining.
                self.harvest_attempt();
                self.try_combine()
            }
            Ok(None) => None,
            Err(_) => {
                // Reception failed outright — harvest its aligned symbol
                // stream (if any) and try cross-reception combining.
                self.harvest_attempt();
                self.try_combine()
            }
        }
    }

    fn harvest_attempt(&mut self) {
        if let Some((syms, confs)) = self.decoder.last_attempt() {
            if !syms.is_empty() {
                self.attempts.push((syms.clone(), confs.clone()));
            }
        }
    }

    /// Merge attempts per-symbol by highest confidence, then re-decode.
    fn try_combine(&self) -> Option<Vec<u8>> {
        if self.attempts.len() < 2 {
            return None;
        }
        // Receptions can be harvested with small relative symbol shifts
        // (the analyzer's offset search does not always settle on the same
        // grid across copies). Align every attempt to the first by maximal
        // symbol agreement over a small shift range before voting —
        // position-wise voting over misaligned streams is worse than any
        // single stream.
        const MAX_SHIFT: isize = 6;
        let (ref_syms, _) = &self.attempts[0];
        let mut aligned: Vec<(isize, &(Vec<usize>, Vec<f64>))> =
            vec![(0, &self.attempts[0])];
        for att in &self.attempts[1..] {
            let (syms, _) = att;
            let mut best_shift = 0isize;
            let mut best_matches = -1isize;
            for shift in -MAX_SHIFT..=MAX_SHIFT {
                let mut matches = 0isize;
                for i in 0..ref_syms.len() {
                    let j = i as isize + shift;
                    if j >= 0 && (j as usize) < syms.len() && syms[j as usize] == ref_syms[i]
                    {
                        matches += 1;
                    }
                }
                if matches > best_matches {
                    best_matches = matches;
                    best_shift = shift;
                }
            }
            aligned.push((best_shift, att));
        }

        // Vote over the full reference length: positions past a shorter
        // (shifted) attempt's end simply draw on fewer copies. Truncating
        // to the shortest attempt would drop trailing symbols and starve
        // the RS codeword of its final bytes.
        let vote_len = ref_syms.len();
        let mut symbols = Vec::with_capacity(vote_len);
        let mut confs = Vec::with_capacity(vote_len);
        let mut seconds = Vec::with_capacity(vote_len);
        for i in 0..vote_len {
            // Majority vote first: with 3+ receptions, two clean copies
            // agreeing beat any single noisy one regardless of confidence.
            let mut votes = [(0usize, 0.0f64); NUM_SYMBOLS];
            for (shift, (s, c)) in &aligned {
                let j = i as isize + shift;
                if j < 0 || j as usize >= s.len() {
                    continue;
                }
                let (s, c) = (s[j as usize], c[j as usize]);
                if s < NUM_SYMBOLS {
                    votes[s].0 += 1;
                    votes[s].1 = votes[s].1.max(c);
                }
            }
            if votes.iter().all(|&(v, _)| v == 0) {
                symbols.push(0);
                confs.push(0.0);
                seconds.push(0);
                continue;
            }
            let (win_sym, &(win_votes, win_conf)) = votes
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    (a.1 .0, a.1 .1)
                        .partial_cmp(&(b.1 .0, b.1 .1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            // Runner-up by (votes, confidence) — also the Chase second
            // choice for the merged stream.
            let (second_sym, &(_, runner_up)) = votes
                .iter()
                .enumerate()
                .filter(|&(s, _)| s != win_sym)
                .max_by(|a, b| {
                    (a.1 .0, a.1 .1)
                        .partial_cmp(&(b.1 .0, b.1 .1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let contested = votes.iter().filter(|&&(v, _)| v > 0).count() > 1;
            // A clear majority keeps full confidence. A contested position
            // is demoted to its confidence *margin*, so it flows into RS
            // erasure handling (1 parity each) instead of being trusted as a
            // hard decision (2 parity when wrong) — noise can classify with
            // high confidence, so raw max-conf is not enough.
            let conf = if !contested {
                win_conf
            } else if win_votes * 2 > self.attempts.len() {
                win_conf.min(0.5 + win_conf - runner_up)
            } else {
                (win_conf - runner_up).max(0.0)
            };
            symbols.push(win_sym);
            confs.push(conf);
            seconds.push(second_sym);
        }
        let framed =
            decode_symbol_stream(&symbols, Some(&confs), Some(&seconds), EccProfile::Paranoid)?;
        check_crc(&framed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_known_value() {
        // Standard IEEE CRC-32 test vector
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_crc_check_rejects_flip() {
        let mut framed = b"secret".to_vec();
        framed.extend_from_slice(&crc32(b"secret").to_le_bytes());
        assert_eq!(check_crc(&framed).unwrap(), b"secret");
        framed[2] ^= 0x01;
        assert!(check_crc(&framed).is_none());
    }

    /// The transmitted data-region symbol stream for a transfer frame.
    fn true_symbols(key: &[u8]) -> Vec<usize> {
        let mut framed = key.to_vec();
        framed.extend_from_slice(&crc32(key).to_le_bytes());
        let len = framed.len();
        let rs_len = crate::reed_solomon::ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
        let mut enc = rs_len.encode(&[len as u8]);
        let n_ecc = EccProfile::Paranoid.ecc_bytes(len);
        let rs = crate::reed_solomon::ReedSolomon::new(len, n_ecc);
        enc.extend(rs.encode(&framed));
        crate::formant::bytes_to_symbols(&enc)
    }

    #[test]
    fn test_combine_mechanism_confident_errors_and_shifts() {
        // Voting must recover the payload from three attempts that each
        // carry 25% CONFIDENT wrong symbols (the case solo decoding cannot
        // fix: high confidence blocks erasure marking and Chase only
        // second-guesses a handful) — including attempts harvested with a
        // ±1 symbol shift, which the aligner must undo before voting.
        let key = b"mechanism-check-0123456789abcdef";
        let syms = true_symbols(key);
        let n = syms.len();

        let mut lcg = 0xC0FFEEu32;
        let mut rand = move || {
            lcg = lcg.wrapping_mul(1103515245).wrapping_add(12345);
            (lcg >> 16) as f64 / 65536.0
        };

        let mut rx = TransferRx::new();
        for shift in [0isize, 1, -1] {
            let mut s = syms.clone();
            let mut c = vec![0.9f64; n];
            for i in 0..n {
                if rand() < 0.25 {
                    s[i] = (s[i] + 1 + (rand() * 3.0) as usize) % NUM_SYMBOLS;
                    c[i] = 0.8; // confidently wrong
                }
            }
            match shift {
                1 => {
                    // Harvested one symbol early: junk prepended.
                    s.insert(0, 2);
                    c.insert(0, 0.9);
                }
                -1 => {
                    // Harvested one symbol late: first symbol lost.
                    s.remove(0);
                    c.remove(0);
                }
                _ => {}
            }
            rx.attempts.push((s, c));
        }

        let got = rx.try_combine().expect("combining must recover the key");
        assert_eq!(got, key);
    }
}
