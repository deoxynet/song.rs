//! Shared receiver timing evidence and position-preserving input repair.

use crate::protocol::{GUARD_SAMPLES, SAMPLES_PER_SYMBOL, SYMBOL_TOTAL_SAMPLES};

pub(crate) const CLOCK_RADIUS: usize = GUARD_SAMPLES / 4;
const CLOCK_STEP: usize = GUARD_SAMPLES / 20;

/// Guard-only timing evidence shared by live streaming and block retries.
/// The current 50 ms voiced window is always before the returned boundary.
pub(crate) fn guard_clock_step(window: &[f32]) -> usize {
    guard_clock_evidence(window).unwrap_or(SYMBOL_TOTAL_SAMPLES)
}

/// None means the guard is obscured, not evidence of a nominal sample clock.
pub(crate) fn guard_clock_evidence(window: &[f32]) -> Option<usize> {
    let probe_len = GUARD_SAMPLES / 2;
    let probe_start = SAMPLES_PER_SYMBOL + CLOCK_RADIUS;
    let energy = |start: usize, len: usize| {
        window[start..start + len].iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
    };
    let original = energy(probe_start, probe_len);
    if original == 0.0 { return Some(SYMBOL_TOTAL_SAMPLES); }
    let voice = energy(0, SAMPLES_PER_SYMBOL);
    // Ignore an already-clear guard, including an asymptotic filter tail.
    // A local power ratio, not an absolute threshold, protects quiet notes.
    if original * (SAMPLES_PER_SYMBOL as f64) < 0.0001 * voice * probe_len as f64 {
        return Some(SYMBOL_TOTAL_SAMPLES);
    }
    let mut best = original;
    let mut offset = 0isize;
    for distance in (CLOCK_STEP..=CLOCK_RADIUS).step_by(CLOCK_STEP) {
        for delta in [-(distance as isize), distance as isize] {
            let candidate = energy((probe_start as isize + delta) as usize, probe_len);
            if candidate < best {
                best = candidate;
                offset = delta;
            }
        }
    }
    if best < 0.7 * original && best * (SAMPLES_PER_SYMBOL as f64) < 0.15 * voice * probe_len as f64 {
        Some((SYMBOL_TOTAL_SAMPLES as isize + offset) as usize)
    } else {
        None
    }
}

/// Recording slack for the slowest grid this tracker can follow.
pub(crate) fn clock_slack_windows(symbols: usize) -> usize {
    (symbols * CLOCK_RADIUS).div_ceil(SYMBOL_TOTAL_SAMPLES)
}

/// Keep finite samples bit-for-bit, including signed zero and quiet tails.
/// Replace only NaN/infinities before they reach RMS, FFT, or history state.
/// Do not filter samples out: every input sample must retain its time slot.
pub(crate) fn append_finite_samples(buffer: &mut Vec<f32>, samples: &[f32]) {
    buffer.extend(
        samples
            .iter()
            .map(|&sample| if sample.is_finite() { sample } else { 0.0 }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_repair_preserves_finite_bits_and_sample_count_at_every_chunk_size() {
        let samples: Vec<_> = [0u32, 0x8000_0000]
            .into_iter()
            .flat_map(|sign| {
                (0..=255).flat_map(move |exponent| {
                    [0, 1, 0x3f_ffff, 0x40_0000, 0x7f_ffff]
                        .map(|mantissa| f32::from_bits(sign | (exponent << 23) | mantissa))
                })
            })
            .collect();
        let expected: Vec<_> = samples
            .iter()
            .map(|sample| {
                let bits = sample.to_bits();
                if bits & 0x7f80_0000 == 0x7f80_0000 {
                    0
                } else {
                    bits
                }
            })
            .collect();
        for chunk_size in [1, 2, 3, 17, 437, 2048, samples.len()] {
            let mut buffer = vec![17.0, -0.0];
            for chunk in samples.chunks(chunk_size) {
                append_finite_samples(&mut buffer, &[]);
                append_finite_samples(&mut buffer, chunk);
            }
            assert_eq!(buffer.len(), samples.len() + 2);
            assert_eq!(buffer[0].to_bits(), 17.0f32.to_bits());
            assert_eq!(buffer[1].to_bits(), (-0.0f32).to_bits());
            assert_eq!(
                buffer[2..].iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
    }
}
