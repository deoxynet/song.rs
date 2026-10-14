//! FFT wrapper around `rustfft` for computing power spectrum.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::cell::RefCell;
use std::f64::consts::PI;
use std::sync::Arc;

struct SpectrumWorkspace {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buffer: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl SpectrumWorkspace {
    fn new(n: usize) -> Self {
        let fft = FftPlanner::new().plan_fft_forward(n);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        let window = (0..n)
            .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos())) as f32)
            .collect();
        Self {
            fft,
            window,
            buffer: vec![Complex::new(0.0, 0.0); n],
            scratch,
        }
    }
}

thread_local! {
    // One workspace per thread, replaced when the frame size changes.
    // Unlike a cache of arbitrary FFT sizes, this cannot accumulate an
    // unbounded number of plans. Threads never share mutable FFT buffers.
    static WORKSPACE: RefCell<Option<SpectrumWorkspace>> = const { RefCell::new(None) };
}

fn compute_spectrum(samples: &[f32], spectrum: &mut [f32], windowed: bool) {
    let n = samples.len();
    assert!(spectrum.len() >= n);
    WORKSPACE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache
            .as_ref()
            .map_or(true, |workspace| workspace.buffer.len() != n)
        {
            *cache = Some(SpectrumWorkspace::new(n));
        }
        let workspace = cache.as_mut().unwrap();
        for (i, (&sample, target)) in samples.iter().zip(&mut workspace.buffer).enumerate() {
            let sample = if windowed {
                sample * workspace.window[i]
            } else {
                sample
            };
            *target = Complex::new(sample, 0.0);
        }
        workspace
            .fft
            .process_with_scratch(&mut workspace.buffer, &mut workspace.scratch);
        for (power, value) in spectrum.iter_mut().zip(&workspace.buffer) {
            *power = value.norm_sqr();
        }
        for i in 1..n / 2 {
            spectrum[i] += spectrum[n - i];
        }
    });
}

/// Compute the Hann-windowed power spectrum of a real-valued audio frame.
///
/// Input: `samples` of arbitrary length N.
/// Output: `spectrum` of length >= N containing power (magnitude squared) at each bin.
///
/// Applies a Hann window before FFT to reduce spectral leakage,
/// then folds negative frequencies: spectrum[i] += spectrum[N - i] for i in 1..N/2.
pub fn power_spectrum(samples: &[f32], spectrum: &mut [f32]) {
    compute_spectrum(samples, spectrum, true);
}

/// Compute power spectrum without windowing (rectangular window).
///
/// More robust for preamble detection with misaligned windows,
/// since all samples contribute equally regardless of position.
pub fn power_spectrum_raw(samples: &[f32], spectrum: &mut [f32]) {
    compute_spectrum(samples, spectrum, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent copy of the original calculation: fresh Hann values,
    // planner, buffers, and process() scratch allocation on every call.
    fn reference(samples: &[f32], windowed: bool) -> Vec<f32> {
        let n = samples.len();
        let mut buffer: Vec<_> = samples
            .iter()
            .enumerate()
            .map(|(i, &sample)| {
                let value = if windowed {
                    let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos());
                    sample * w as f32
                } else {
                    sample
                };
                Complex::new(value, 0.0)
            })
            .collect();
        FftPlanner::new().plan_fft_forward(n).process(&mut buffer);
        let mut power: Vec<_> = buffer.iter().map(Complex::norm_sqr).collect();
        for i in 1..n / 2 {
            power[i] += power[n - i];
        }
        power
    }

    #[test]
    fn cached_spectra_are_bit_identical_to_the_original_calculation() {
        for n in [1, 2, 3, 7, 64, 400, 480, 2400, 1024, 400, 2400] {
            let mut rng = 3571u32;
            for kind in 0..4 {
                let samples: Vec<f32> = (0..n)
                    .map(|i| match kind {
                        0 => 0.0,
                        1 => 0.03,
                        2 => {
                            if i == n / 2 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        _ => {
                            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                            (rng as f32 / u32::MAX as f32 - 0.5) * 0.2
                        }
                    })
                    .collect();
                for windowed in [true, false, true] {
                    let mut actual = vec![-17.0; n + 3];
                    if windowed {
                        power_spectrum(&samples, &mut actual);
                    } else {
                        power_spectrum_raw(&samples, &mut actual);
                    }
                    let expected = reference(&samples, windowed);
                    assert_eq!(
                        actual[..n].iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|p| p.to_bits()).collect::<Vec<_>>(),
                        "length {n}, signal {kind}, windowed {windowed}"
                    );
                    assert_eq!(&actual[n..], &[-17.0; 3]);
                }
            }
        }
    }

    #[test]
    fn workspace_replaces_old_lengths_and_stays_thread_local() {
        let handles: Vec<_> = [400, 2400, 17]
            .into_iter()
            .map(|n| {
                std::thread::spawn(move || {
                    for len in [64, n, n] {
                        let samples = vec![0.01; len];
                        let mut power = vec![0.0; len];
                        power_spectrum(&samples, &mut power);
                        assert_eq!(power, reference(&samples, true));
                        WORKSPACE.with(|cache| {
                            let cache = cache.borrow();
                            let workspace = cache.as_ref().unwrap();
                            assert_eq!(workspace.fft.len(), len);
                            assert_eq!(workspace.buffer.len(), len);
                            assert_eq!(workspace.window.len(), len);
                            assert_eq!(
                                workspace.scratch.len(),
                                workspace.fft.get_inplace_scratch_len()
                            );
                        });
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
