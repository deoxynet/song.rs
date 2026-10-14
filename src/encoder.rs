//! Encoder: payload → f32 audio samples (vowel modem with melodic pitch).

use crate::formant::{self, PREAMBLE_END, PREAMBLE_START};
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Encode a payload into f32 audio samples.
///
/// `volume` is 0..=100 (typically 10-50).
/// Returns a Vec<f32> containing the complete waveform at 48 kHz.
///
/// The payload is sung to the default melody — the Giant Steps head
/// (`protocol::GIANT_STEPS`); pitch carries no data. The intro and outro
/// sing The Lick while sync, header, checksum, and parity ride the vowels. Use
/// [`encode_armored`] when the audio must survive worst-case channels
/// (heavy spectral tilt plus an aggressive denoiser) with maximum margin.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    let track = default_byte_pitch(payload.len());
    encode_with_pitch(payload, volume, Some(&track))
}

/// Encode with the armored payload melody: the payload follows the
/// margin-verified arpeggio (`protocol::MELODY`) instead of the default
/// Giant Steps head. Maximum channel margin; used by the transfer profile.
pub fn encode_armored(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    encode_with_pitch(payload, volume, None)
}

/// Encode with an expressive per-byte pitch track.
///
/// `byte_pitch[i]` is the F0 (Hz) for the 4 symbols carrying payload byte `i`
/// (must have `payload.len()` entries). Framing symbols sing The Lick.
/// Pitch carries no data — the decoder is
/// F0-agnostic — so callers may drive it semantically (e.g. token surprisal),
/// within the B2–F#4 singing register (approximately 123.5–370 Hz).
/// The chromatic register is tested through echo, noisy phone channels,
/// spectral tilt, and noise suppression; pitch still affects channel margin.
/// Nonfinite or nonpositive pitches return [`crate::Error::InvalidPitch`]
/// before synthesis. Valid pitches are used unchanged, without tuning or clamping;
/// positive frequencies outside the tested register have no channel guarantee.
pub fn encode_with_pitch(
    payload: &[u8],
    volume: u8,
    byte_pitch: Option<&[f64]>,
) -> Result<Vec<f32>, crate::Error> {
    encode_with_profile(payload, volume, byte_pitch, EccProfile::Standard)
}

/// Encode with an explicit ECC profile (receiver must use the same profile).
/// `Standard` includes CRC-32; use `Legacy` only for older checksum-free peers.
/// Pitch-track length and finite, positive frequencies are checked for every profile.
pub fn encode_with_profile(
    payload: &[u8],
    volume: u8,
    byte_pitch: Option<&[f64]>,
    profile: EccProfile,
) -> Result<Vec<f32>, crate::Error> {
    let data_length = payload.len();
    if data_length == 0 {
        return Err(crate::Error::EmptyPayload);
    }
    if data_length > MAX_LENGTH_VARIABLE {
        return Err(crate::Error::PayloadTooLarge {
            size: data_length,
            max: MAX_LENGTH_VARIABLE,
        });
    }
    if volume > 100 {
        return Err(crate::Error::InvalidVolume(volume));
    }
    if let Some(bp) = byte_pitch {
        if bp.len() != data_length {
            return Err(crate::Error::PitchTrackMismatch {
                got: bp.len(),
                expected: data_length,
            });
        }
        for (index, &hz) in bp.iter().enumerate() {
            if !hz.is_finite() || hz <= 0.0 {
                return Err(crate::Error::InvalidPitch { index, hz });
            }
        }
    }

    let send_volume = volume as f64 / 100.0;

    // The header describes caller-visible bytes. The checksum is protected
    // by the same RS codeword as the payload and has no melody override.
    let protected = profile.protect_payload(payload);

    // --- Step 2: Reed-Solomon encode ---
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let encoded_length = rs_length.encode(&[data_length as u8]);

    let n_ecc = profile.ecc_bytes(data_length);
    let rs_data = ReedSolomon::new(protected.len(), n_ecc);
    let encoded_data = rs_data.encode(&protected);

    let mut data_encoded = Vec::with_capacity(profile.encoded_bytes(data_length));
    data_encoded.extend_from_slice(&encoded_length);
    data_encoded.extend_from_slice(&encoded_data);

    // --- Step 3: Convert bytes to 2-bit symbol sequence ---
    let data_symbols = formant::bytes_to_symbols(&data_encoded);

    // --- Step 4: Build symbol sequence: preamble + pilot + data + preamble ---
    let total_symbols = PREAMBLE_LEN + PILOT_LEN + data_symbols.len() + PREAMBLE_LEN;
    let total_samples = total_symbols * SYMBOL_TOTAL_SAMPLES;

    let mut output = Vec::with_capacity(total_samples);

    // Synthesize and append a full symbol slot (voiced window + release tail).
    // Accent follows the melody position; pitch follows the melody unless the
    // caller supplied a per-payload-byte track.
    let mut position = 0usize;
    let payload_start = PREAMBLE_LEN + PILOT_LEN + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
    let payload_end = payload_start + formant::symbols_for_bytes(data_length);
    let mut emit_symbol = |sym_idx: usize, f0_override: Option<f64>| {
        let volume = send_volume * melody_accent(position);
        let f0 = f0_override.unwrap_or_else(|| {
            if profile == EccProfile::Legacy {
                melody_f0(position)
            } else if position < payload_start {
                framing_lick_f0(position, payload_start)
            } else if position >= payload_end {
                outro_f0(position - payload_end, total_symbols - payload_end)
            } else {
                melody_f0(position)
            }
        });
        let slot = formant::synthesize_symbol(sym_idx, f0, volume, position);
        position += 1;
        for &s in &slot {
            output.push(s as f32);
        }
    };

    // Start preamble
    for &sym in &PREAMBLE_START {
        emit_symbol(sym, None);
    }

    // Calibration pilot: known symbols the receiver uses to measure each
    // vowel's centroid through the live channel (and to let AGC settle).
    for &sym in &PILOT {
        emit_symbol(sym, None);
    }

    // Data symbols: byte b of data_encoded spans symbols 4b..4b+4.
    // Bytes 0..3 are the length header, then payload, checksum, and RS parity —
    // only payload bytes get the expressive pitch track.
    for (j, &sym) in data_symbols.iter().enumerate() {
        let byte_idx = j / (8 / BITS_PER_SYMBOL);
        let f0 = byte_pitch.and_then(|bp| {
            byte_idx
                .checked_sub(ENCODED_DATA_OFFSET)
                .and_then(|i| bp.get(i))
                .copied()
        });
        emit_symbol(sym, f0);
    }

    // End preamble
    for &sym in &PREAMBLE_END {
        emit_symbol(sym, None);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_pitch_values_are_rejected_before_synthesis() {
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for hz in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, -220.0] {
                for index in 0..3 {
                    let mut track = default_byte_pitch(3);
                    track[index] = hz;
                    for volume in [0, 50, 100] {
                        let result = encode_with_profile(b"abc", volume, Some(&track), profile);
                        assert!(matches!(result,
                            Err(crate::Error::InvalidPitch { index: got_index, hz: got_hz })
                                if got_index == index && got_hz.to_bits() == hz.to_bits()),
                            "{profile:?}, volume {volume}, pitch {hz} at byte {index}");
                    }
                }
            }
            let track = [220.0, f64::NAN, 0.0];
            assert!(matches!(encode_with_profile(b"a", 50, Some(&track), profile),
                Err(crate::Error::PitchTrackMismatch { got: 3, expected: 1 })));
            assert!(matches!(encode_with_profile(b"abc", 50, Some(&track), profile),
                Err(crate::Error::InvalidPitch { index: 1, hz }) if hz.is_nan()));
        }
    }

    #[test]
    fn valid_pitch_tracks_preserve_every_requested_note_and_sample() {
        let payload: Vec<_> = (0..60).map(|i| (i * 73 + 19) as u8).collect();
        // Every chromatic note in the singing register at -30/0/+30 cents,
        // in a wide-leap contour rather than the default tune.
        let track: Vec<_> = (0..payload.len()).map(|i| {
            let midi = 47.0 + (i * 7 % 20) as f64 + [-0.3, 0.0, 0.3][i % 3];
            440.0 * 2.0f64.powf((midi - 69.0) / 12.0)
        }).collect();
        let original: Vec<_> = track.iter().map(|hz| hz.to_bits()).collect();
        let symbols = formant::bytes_to_symbols(&payload);
        let start = PREAMBLE_LEN + PILOT_LEN + formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for volume in [0, 50, 100] {
                let audio = encode_with_profile(&payload, volume, Some(&track), profile).unwrap();
                assert!(audio.iter().all(|sample| sample.is_finite()));
                for (j, &symbol) in symbols.iter().enumerate() {
                    let pos = start + j;
                    let expected = formant::synthesize_symbol(symbol, track[j / 4],
                        volume as f64 / 100.0 * melody_accent(pos), pos);
                    let actual = &audio[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES];
                    assert!(actual.iter().zip(expected).all(|(&a, e)| a.to_bits() == (e as f32).to_bits()),
                        "{profile:?}, volume {volume}, payload slot {j} was retuned or resynthesized differently");
                }
                if volume == 50 {
                    let mut rx = crate::Decoder::with_profile(profile);
                    assert_eq!(rx.decode(&audio).unwrap(), Some(payload.clone()), "{profile:?}");
                }
            }
        }
        assert_eq!(track.iter().map(|hz| hz.to_bits()).collect::<Vec<_>>(), original);
    }

    #[test]
    fn current_frames_sing_the_lick_intro_and_solo_outro_in_the_actual_audio() {
        for profile in [EccProfile::Standard, EccProfile::Paranoid] {
            for len in [1, 64, MAX_LENGTH_VARIABLE] {
                let payload: Vec<u8> = (0..len).map(|i| (i * 37) as u8).collect();
                let pitches = default_byte_pitch(len);
                let audio = encode_with_profile(&payload, 50, Some(&pitches), profile).unwrap();
                let protected = profile.protect_payload(&payload);
                let header = ReedSolomon::new(1, 2).encode(&[len as u8]);
                let body = ReedSolomon::new(protected.len(), profile.ecc_bytes(len)).encode(&protected);
                let symbols: Vec<usize> = PREAMBLE_START.iter().copied()
                    .chain(PILOT)
                    .chain(formant::bytes_to_symbols(&header))
                    .chain(formant::bytes_to_symbols(&body))
                    .chain(PREAMBLE_END)
                    .collect();
                let payload_start = FRAMING_LICK.len();
                let payload_end = payload_start + formant::symbols_for_bytes(len);
                assert_eq!(audio.len(), symbols.len() * SYMBOL_TOTAL_SAMPLES);
                for (pos, &symbol) in symbols.iter().enumerate() {
                    let hz = if pos < payload_start {
                        FRAMING_LICK[pos]
                    } else if pos < payload_end {
                        pitches[(pos - payload_start) / 4]
                    } else {
                        outro_f0(pos - payload_end, symbols.len() - payload_end)
                    };
                    let expected: Vec<f32> = formant::synthesize_symbol(
                        symbol, hz, 0.5 * melody_accent(pos), pos,
                    ).into_iter().map(|s| s as f32).collect();
                    assert_eq!(
                        &audio[pos * SYMBOL_TOTAL_SAMPLES..(pos + 1) * SYMBOL_TOTAL_SAMPLES],
                        expected, "{profile:?}, {len} bytes, slot {pos}",
                    );
                }
            }
        }
    }
}
