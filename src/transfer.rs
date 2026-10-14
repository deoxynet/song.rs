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
//!    up to 50% erased bytes or 25% unknown byte errors per codeword at 1:1 parity.
//! 3. **Repeat + soft combining**: the sender loops the transmission; the
//!    receiver merges symbol classifications across receptions by confidence
//!    (time diversity), so two individually-undecodable copies can combine
//!    into a good one.

use crate::decoder::{decode_symbol_stream_validated, Decoder};
use crate::protocol::*;

/// Maximum transfer payload: payload + 4-byte CRC must fit the paranoid
/// RS(255) codeword (2·len ≤ 255 - header).
pub const MAX_TRANSFER_PAYLOAD: usize = 120;

const CRC_LEN: usize = 4;

/// CRC-32 (IEEE, reflected, poly 0xEDB88320) — bitwise, no table, no deps.
pub use crate::checksum::crc32;

/// Encode 1..=120 payload bytes for transfer: CRC-32 appended, paranoid ECC.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    if payload.is_empty() {
        return Err(crate::Error::EmptyPayload);
    }
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
/// The receiver soft-combines across copies; independent receptions can
/// improve recovery on bad channels without changing the melody.
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
    valid_crc(framed).then(|| framed[..framed.len() - CRC_LEN].to_vec())
}

/// Allocation-free acceptance predicate, shared by solo and merged retries.
fn valid_crc(framed: &[u8]) -> bool {
    if framed.len() < CRC_LEN + 1 {
        return false;
    }
    let (payload, tail) = framed.split_at(framed.len() - CRC_LEN);
    let expect = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    crc32(payload) == expect
}

/// Receiver for repeated transfer transmissions with soft combining.
///
/// Feed audio continuously via `ingest`; returns the payload as soon as any
/// single reception decodes with a valid CRC, or when the confidence-merged
/// combination of failed receptions does.
pub struct TransferRx {
    decoder: Decoder,
    combiner: AttemptCombiner,
    /// Lifetime failed-copy count, retained after delivery for UI progress.
    failed_attempts: usize,
}

/// Sufficient statistics for the original fixed-reference majority vote.
/// Later copies need only the first copy for alignment, and each position's
/// vote count and maximum confidence per vowel. Storage does not grow with
/// the number of receptions, and no old copy's contribution is evicted.
#[derive(Default)]
struct AttemptCombiner {
    reference: Vec<usize>,
    votes: Vec<[(usize, f64); NUM_SYMBOLS]>,
    copies: usize,
}

impl AttemptCombiner {
    fn observe(&mut self, symbols: &[usize], confs: &[f64]) {
        if symbols.is_empty() { return; }
        debug_assert_eq!(symbols.len(), confs.len());
        let shift = if self.copies == 0 {
            self.reference = symbols.to_vec();
            self.votes = vec![[(0, 0.0); NUM_SYMBOLS]; symbols.len()];
            0
        } else {
            // Preserve the old shift range, match count, and first-tie rule.
            const MAX_SHIFT: isize = 6;
            let mut best_shift = 0;
            let mut best_matches = -1isize;
            for shift in -MAX_SHIFT..=MAX_SHIFT {
                let mut matches = 0;
                for (i, &reference) in self.reference.iter().enumerate() {
                    let j = i as isize + shift;
                    if j >= 0 && (j as usize) < symbols.len() && symbols[j as usize] == reference {
                        matches += 1;
                    }
                }
                if matches > best_matches {
                    best_matches = matches;
                    best_shift = shift;
                }
            }
            best_shift
        };
        self.copies = self.copies.saturating_add(1);
        for (i, votes) in self.votes.iter_mut().enumerate() {
            let j = i as isize + shift;
            if j < 0 || j as usize >= symbols.len() { continue; }
            let (symbol, confidence) = (symbols[j as usize], confs[j as usize]);
            if symbol < NUM_SYMBOLS {
                votes[symbol].0 = votes[symbol].0.saturating_add(1);
                votes[symbol].1 = votes[symbol].1.max(confidence);
            }
        }
    }
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
        decoder.set_payload_validator(valid_crc);
        Self {
            decoder,
            combiner: AttemptCombiner::default(),
            failed_attempts: 0,
        }
    }

    /// Lifetime number of failed reception attempts (for UI progress).
    /// Delivery retires combining votes but preserves this counter.
    pub fn attempts(&self) -> usize {
        self.failed_attempts
    }

    /// True while the underlying decoder is locked onto a transmission
    /// (mid-reception), false while idle-listening.
    pub fn is_locked(&self) -> bool {
        self.decoder.is_receiving()
    }

    /// Feed audio. Returns each verified reception's payload once available.
    /// Solo or combined delivery retires old combining votes, without
    /// discarding buffered audio for the next reception. A new transmission
    /// of the same payload may be delivered again; this is not deduplication.
    pub fn ingest(&mut self, samples: &[f32]) -> Option<Vec<u8>> {
        match self.decoder.decode(samples) {
            Ok(Some(framed)) => {
                if let Some(payload) = check_crc(&framed) {
                    self.combiner = AttemptCombiner::default();
                    let _ = self.decoder.take_last_attempt();
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
        if let Some((syms, confs)) = self.decoder.take_last_attempt() {
            if !syms.is_empty() {
                self.failed_attempts = self.failed_attempts.saturating_add(1);
                self.combiner.observe(&syms, &confs);
            }
        }
    }

    /// Merge votes, validate, and retire history only after checked delivery.
    fn try_combine(&mut self) -> Option<Vec<u8>> {
        if self.combiner.copies < 2 {
            return None;
        }
        let (symbols, confs, seconds) = self.combiner.decisions();
        let framed = decode_symbol_stream_validated(
            &symbols, Some(&confs), Some(&seconds), EccProfile::Paranoid, None, valid_crc,
        )?;
        let payload = check_crc(&framed)?;
        self.combiner = AttemptCombiner::default();
        Some(payload)
    }
}

impl AttemptCombiner {
    fn decisions(&self) -> (Vec<usize>, Vec<f64>, Vec<usize>) {
        // Vote over the full reference length: positions past a shorter
        // (shifted) attempt's end simply draw on fewer copies. Truncating
        // to the shortest attempt would drop trailing symbols and starve
        // the RS codeword of its final bytes.
        let vote_len = self.votes.len();
        let mut symbols = Vec::with_capacity(vote_len);
        let mut confs = Vec::with_capacity(vote_len);
        let mut seconds = Vec::with_capacity(vote_len);
        for i in 0..vote_len {
            // Majority vote first: with 3+ receptions, two clean copies
            // agreeing beat any single noisy one regardless of confidence.
            let votes = &self.votes[i];
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
            } else if win_votes > self.copies / 2 {
                win_conf.min(0.5 + win_conf - runner_up)
            } else {
                (win_conf - runner_up).max(0.0)
            };
            symbols.push(win_sym);
            confs.push(conf);
            seconds.push(second_sym);
        }
        (symbols, confs, seconds)
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

    /// Each copy has confidently wrong bytes beyond the solo RS radius;
    /// two of three copies agree on the correct vowel at every position.
    fn complementary_failures(key: &[u8]) -> Vec<(Vec<usize>, Vec<f64>)> {
        let original = true_symbols(key);
        (0..3).map(|copy| {
            let mut symbols = original.clone();
            let mut confs = vec![0.95; symbols.len()];
            for byte in ENCODED_DATA_OFFSET..symbols.len() / 4 {
                if (byte - ENCODED_DATA_OFFSET) % 3 == copy {
                    for pos in 4 * byte..4 * byte + 4 {
                        symbols[pos] = (symbols[pos] + 1) % NUM_SYMBOLS;
                        confs[pos] = 0.9;
                    }
                }
            }
            assert!(decode_symbol_stream_validated(
                &symbols, Some(&confs), None, EccProfile::Paranoid, None, valid_crc,
            ).is_none(), "fixture must fail solo decoding");
            (symbols, confs)
        }).collect()
    }

    fn add_failed_fixture(rx: &mut TransferRx, symbols: &[usize], confs: &[f64]) {
        rx.failed_attempts += 1;
        rx.combiner.observe(symbols, confs);
    }

    #[test]
    fn unanchored_background_cannot_truncate_a_later_combining_reference() {
        // No start preamble or pilot. A late end-shaped fragment triggers a
        // speculative trial from an unknown offset, followed by finalization.
        // Its 40-symbol prefix is too short to contain even one transfer.
        let mut background = Vec::new();
        for (pos, vowel) in std::iter::repeat(2).take(40)
            .chain(crate::formant::PREAMBLE_END).enumerate()
        {
            background.extend(crate::formant::synthesize_symbol(vowel, 185.0, 0.2, pos)
                .into_iter().map(|sample| sample as f32));
        }
        background.extend(vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]);
        let key = b"the actual melody needs its whole combining reference";
        let failures = complementary_failures(key);
        for chunk_size in [437, 2048, background.len()] {
            let mut rx = TransferRx::new();
            for chunk in background.chunks(chunk_size) {
                assert!(rx.ingest(chunk).is_none(), "background is not a transfer");
            }
            for (symbols, confs) in &failures {
                add_failed_fixture(&mut rx, symbols, confs);
            }
            assert_eq!(rx.try_combine().as_deref(), Some(key.as_slice()),
                "background must not fix the reference length or vote in a later melody ({chunk_size}-sample chunks)");
            assert_eq!(rx.attempts(), 3, "only anchored receptions count as failed copies");
        }
    }

    #[test]
    fn solo_delivery_does_not_truncate_the_next_messages_combining_reference() {
        let first = b"first";
        let second = b"the next melody needs a longer combining reference";
        let mut rx = TransferRx::new();
        let (symbols, confs) = complementary_failures(first).remove(0);
        add_failed_fixture(&mut rx, &symbols, &confs);
        let mut audio = encode(first, 50).unwrap();
        audio.extend(vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]);
        let mut deliveries = Vec::new();
        for chunk in audio.chunks(437) {
            if let Some(bytes) = rx.ingest(chunk) { deliveries.push(bytes); }
        }
        assert_eq!(deliveries, vec![first.to_vec()]);
        assert!(rx.decoder.last_attempt().is_none(), "delivered audio must not be harvested later");
        assert_eq!(rx.attempts(), 1, "UI count survives delivery");
        assert!(rx.combiner.reference.is_empty());
        for (symbols, confs) in complementary_failures(second) {
            add_failed_fixture(&mut rx, &symbols, &confs);
        }
        assert_eq!(rx.try_combine().as_deref(), Some(second.as_slice()));
        assert_eq!(rx.attempts(), 4);
    }

    #[test]
    fn combined_delivery_retires_votes_instead_of_replaying_the_old_payload() {
        let key = b"a verified combined melody";
        let mut rx = TransferRx::new();
        for (symbols, confs) in complementary_failures(key) {
            add_failed_fixture(&mut rx, &symbols, &confs);
        }
        assert_eq!(rx.try_combine().as_deref(), Some(key.as_slice()));
        assert_eq!(rx.attempts(), 3, "UI count survives combined delivery");
        assert!(rx.try_combine().is_none(), "old votes must not deliver the same payload again");
    }

    #[test]
    fn a_failed_reception_can_only_be_harvested_once() {
        let mut rx = TransferRx::new();
        rx.decoder.set_payload_validator(|_| false);
        let mut audio = encode(b"harvest one failed reception only once", 50).unwrap();
        audio.extend(vec![0.0; 10 * SYMBOL_TOTAL_SAMPLES]);
        assert!(rx.decoder.decode(&audio).is_err());
        assert!(rx.decoder.last_attempt().is_some());
        rx.harvest_attempt();
        assert_eq!(rx.attempts(), 1);
        rx.harvest_attempt();
        assert_eq!(rx.attempts(), 1, "a stale observation is not a new copy");
        assert_eq!(rx.combiner.copies, 1);
    }

    #[test]
    fn unsuccessful_combining_keeps_every_vote_for_later_copies() {
        let key = b"do not retire the votes until the CRC passes";
        let failures = complementary_failures(key);
        let mut rx = TransferRx::new();
        for _ in 0..2 { add_failed_fixture(&mut rx, &failures[0].0, &failures[0].1); }
        assert!(rx.try_combine().is_none());
        assert_eq!(rx.combiner.copies, 2);
        for (symbols, confs) in &failures[1..] {
            for _ in 0..2 { add_failed_fixture(&mut rx, symbols, confs); }
        }
        assert_eq!(rx.try_combine().as_deref(), Some(key.as_slice()));
        assert_eq!(rx.attempts(), 6);
        assert_eq!(rx.combiner.copies, 0);
    }

    // Independent full-history reference: align all old copies again and
    // recalculate each vote from scratch, as the original receiver did.
    fn reference_votes(attempts: &[(Vec<usize>, Vec<f64>)]) -> Vec<[(usize, f64); NUM_SYMBOLS]> {
        let reference = &attempts[0].0;
        let mut aligned = vec![(0isize, &attempts[0])];
        for attempt in &attempts[1..] {
            let mut best_shift = 0;
            let mut best_matches = -1isize;
            for shift in -6..=6 {
                let mut matches = 0;
                for i in 0..reference.len() {
                    let j = i as isize + shift;
                    if j >= 0 && (j as usize) < attempt.0.len() && attempt.0[j as usize] == reference[i] {
                        matches += 1;
                    }
                }
                if matches > best_matches {
                    best_matches = matches;
                    best_shift = shift;
                }
            }
            aligned.push((best_shift, attempt));
        }
        (0..reference.len()).map(|i| {
            let mut votes = [(0usize, 0.0f64); NUM_SYMBOLS];
            for &(shift, (ref symbols, ref confs)) in &aligned {
                let j = i as isize + shift;
                if j < 0 || j as usize >= symbols.len() { continue; }
                let (symbol, confidence) = (symbols[j as usize], confs[j as usize]);
                if symbol < NUM_SYMBOLS {
                    votes[symbol].0 += 1;
                    votes[symbol].1 = votes[symbol].1.max(confidence);
                }
            }
            votes
        }).collect()
    }

    #[test]
    fn incremental_combining_matches_full_history_at_every_prefix() {
        let mut random = 0x4321abcdu32;
        let mut next = || {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            random as usize
        };
        let confidences = [0.0, -0.0, 0.2, 0.5, 0.9, 1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for len in [1, 7, 13, 76, 300, 1024] {
            for pattern in 0..4 {
                let original: Vec<_> = (0..len).map(|i| match pattern {
                    0 => NUM_SYMBOLS, // abstentions, tied alignments, no votes
                    1 => 0,           // ties among alignments and confidences
                    2 => i % NUM_SYMBOLS,
                    _ => next() % NUM_SYMBOLS,
                }).collect();
                let mut combiner = AttemptCombiner::default();
                let mut history = Vec::new();
                for copy in 0..24 {
                    let mut symbols = original.clone();
                    if copy > 0 {
                        for symbol in &mut symbols {
                            if next() % 5 == 0 { *symbol = next() % (NUM_SYMBOLS + 1); }
                        }
                        let shift = (copy % 13) as isize - 6;
                        if shift > 0 {
                            symbols.splice(0..0, std::iter::repeat(NUM_SYMBOLS).take(shift as usize));
                        } else if shift < 0 {
                            symbols.drain(..((-shift) as usize).min(symbols.len() - 1));
                        }
                        if copy % 3 == 0 { symbols.truncate((symbols.len() / 2).max(1)); }
                    }
                    let confs: Vec<_> = (0..symbols.len()).map(|_| confidences[next() % confidences.len()]).collect();
                    combiner.observe(&symbols, &confs);
                    history.push((symbols, confs));
                    let expected = reference_votes(&history);
                    let actual = combiner.decisions();
                    assert_eq!(combiner.copies, history.len());
                    assert_eq!(actual.0.len(), expected.len());
                    for (i, votes) in expected.iter().enumerate() {
                        for symbol in 0..NUM_SYMBOLS {
                            assert_eq!(combiner.votes[i][symbol].0, votes[symbol].0);
                            assert_eq!(combiner.votes[i][symbol].1.to_bits(), votes[symbol].1.to_bits());
                        }
                        if votes.iter().all(|&(count, _)| count == 0) {
                            assert_eq!((actual.0[i], actual.1[i].to_bits(), actual.2[i]), (0, 0, 0));
                            continue;
                        }
                        // Stable ascending ranking gives the original last-
                        // vowel tie behavior independently of max_by().
                        let mut rank: Vec<_> = (0..NUM_SYMBOLS).collect();
                        rank.sort_by(|&a, &b| votes[a].partial_cmp(&votes[b]).unwrap());
                        let winner = rank[NUM_SYMBOLS - 1];
                        let runner = rank[NUM_SYMBOLS - 2];
                        let (count, confidence) = votes[winner];
                        let contested = votes.iter().filter(|&&(count, _)| count > 0).count() > 1;
                        let expected_conf = if !contested {
                            confidence
                        } else if count * 2 > history.len() {
                            confidence.min(0.5 + confidence - votes[runner].1)
                        } else {
                            (confidence - votes[runner].1).max(0.0)
                        };
                        assert_eq!((actual.0[i], actual.2[i]), (winner, runner));
                        assert_eq!(actual.1[i].to_bits(), expected_conf.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn thousands_of_copies_keep_fixed_storage_without_dropping_votes() {
        let len = 4 * EccProfile::Paranoid.encoded_bytes(MAX_TRANSFER_PAYLOAD + CRC_LEN);
        let symbols = vec![1; len];
        let confs = vec![0.9; len];
        let mut combiner = AttemptCombiner::default();
        combiner.observe(&[], &[]);
        assert_eq!(combiner.copies, 0);
        combiner.observe(&symbols, &confs);
        let reference = (combiner.reference.as_ptr(), combiner.reference.capacity());
        let votes = (combiner.votes.as_ptr(), combiner.votes.capacity());
        for _ in 1..4096 {
            combiner.observe(&symbols, &confs);
            assert_eq!((combiner.reference.as_ptr(), combiner.reference.capacity()), reference);
            assert_eq!((combiner.votes.as_ptr(), combiner.votes.capacity()), votes);
        }
        assert_eq!(combiner.copies, 4096);
        assert_eq!(combiner.reference.len(), len);
        assert_eq!(combiner.votes.len(), len);
        for votes in &combiner.votes {
            assert_eq!(*votes, [(0, 0.0), (4096, 0.9), (0, 0.0), (0, 0.0)]);
        }
    }

    #[test]
    fn test_combine_crc_rejection_still_allows_erasure_recovery() {
        let key = b"melodies";
        let original_symbols = true_symbols(key);
        let original_frame = crate::formant::symbols_to_bytes(
            &original_symbols, EccProfile::Paranoid.encoded_bytes(key.len() + CRC_LEN),
        );
        let original = &original_frame[ENCODED_DATA_OFFSET..];
        let len = key.len() + CRC_LEN;
        let rs = crate::reed_solomon::ReedSolomon::new(len, EccProfile::Paranoid.ecc_bytes(len));
        let mut wrong_payload = original[..len].to_vec();
        wrong_payload[0] ^= 0x40; // Deliberately preserve the original CRC.
        assert!(!valid_crc(&wrong_payload));
        let wrong = rs.encode(&wrong_payload);
        let differences: Vec<usize> = original.iter().zip(&wrong).enumerate()
            .filter_map(|(i, (a, b))| (a != b).then_some(i)).collect();
        assert_eq!(differences.len(), 13);
        let erased = &differences[..7];
        let mut damaged = original.to_vec();
        let mut confs = vec![1.0; original_symbols.len()];
        for &pos in erased {
            damaged[pos] = wrong[pos];
            let start = 4 * (ENCODED_DATA_OFFSET + pos);
            confs[start..start + 4].fill(0.0);
        }
        assert_eq!(rs.decode(&damaged).as_deref(), Some(wrong_payload.as_slice()));
        assert_eq!(rs.decode_with_erasures(&damaged, erased).as_deref(), Some(&original[..len]));
        let mut frame = original_frame[..ENCODED_DATA_OFFSET].to_vec();
        frame.extend(damaged);
        let syms = crate::formant::bytes_to_symbols(&frame);
        let mut rx = TransferRx::new();
        // Identical weak observations leave the same seven errors after
        // voting. The combined decoder must reject hard RS and try erasures.
        add_failed_fixture(&mut rx, &syms, &confs);
        add_failed_fixture(&mut rx, &syms, &confs);
        assert_eq!(rx.try_combine().as_deref(), Some(key.as_slice()));
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
            add_failed_fixture(&mut rx, &s, &c);
        }

        let got = rx.try_combine().expect("combining must recover the key");
        assert_eq!(got, key);
    }
}
