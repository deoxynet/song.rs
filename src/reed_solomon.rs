//! Reed-Solomon codec over GF(2^8) with primitive polynomial 0x11d.
//!
//! Ported from ggwave's `src/reed-solomon/` for bit-compatibility.

use std::sync::OnceLock;

/// GF(2^8) exponential table (512 entries, wraps around at 255).
#[rustfmt::skip]
const GF_EXP: [u8; 512] = [
    0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26, 0x4c,
    0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9, 0x8f, 0x3, 0x6, 0xc, 0x18, 0x30, 0x60, 0xc0, 0x9d,
    0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35, 0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23, 0x46,
    0x8c, 0x5, 0xa, 0x14, 0x28, 0x50, 0xa0, 0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1, 0x5f,
    0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc, 0x65, 0xca, 0x89, 0xf, 0x1e, 0x3c, 0x78, 0xf0, 0xfd,
    0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f, 0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2, 0xd9,
    0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88, 0xd, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce, 0x81,
    0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93, 0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc, 0x85,
    0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9, 0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54, 0xa8,
    0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa, 0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73, 0xe6,
    0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e, 0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff, 0xe3,
    0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4, 0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41, 0x82,
    0x19, 0x32, 0x64, 0xc8, 0x8d, 0x7, 0xe, 0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6, 0x51,
    0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef, 0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x9, 0x12,
    0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5, 0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0xb, 0x16, 0x2c,
    0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83, 0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x1, 0x2,
    // second half (repeat)
    0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26, 0x4c, 0x98,
    0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9, 0x8f, 0x3, 0x6, 0xc, 0x18, 0x30, 0x60, 0xc0, 0x9d, 0x27,
    0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35, 0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23, 0x46, 0x8c,
    0x5, 0xa, 0x14, 0x28, 0x50, 0xa0, 0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1, 0x5f, 0xbe,
    0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc, 0x65, 0xca, 0x89, 0xf, 0x1e, 0x3c, 0x78, 0xf0, 0xfd, 0xe7,
    0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f, 0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2, 0xd9, 0xaf,
    0x43, 0x86, 0x11, 0x22, 0x44, 0x88, 0xd, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce, 0x81, 0x1f,
    0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93, 0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc, 0x85, 0x17,
    0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9, 0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54, 0xa8, 0x4d,
    0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa, 0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73, 0xe6, 0xd1,
    0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e, 0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff, 0xe3, 0xdb,
    0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4, 0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41, 0x82, 0x19,
    0x32, 0x64, 0xc8, 0x8d, 0x7, 0xe, 0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6, 0x51, 0xa2,
    0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef, 0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x9, 0x12, 0x24,
    0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5, 0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0xb, 0x16, 0x2c, 0x58,
    0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83, 0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x1, 0x2,
];

/// GF(2^8) logarithm table (256 entries). log[0] = 0 by convention.
#[rustfmt::skip]
const GF_LOG: [u8; 256] = [
    0x0, 0x0, 0x1, 0x19, 0x2, 0x32, 0x1a, 0xc6, 0x3, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b, 0x4,
    0x64, 0xe0, 0xe, 0x34, 0x8d, 0xef, 0x81, 0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x8, 0x4c, 0x71, 0x5,
    0x8a, 0x65, 0x2f, 0xe1, 0x24, 0xf, 0x21, 0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45, 0x1d,
    0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9, 0xc9, 0x9a, 0x9, 0x78, 0x4d, 0xe4, 0x72, 0xa6, 0x6,
    0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd, 0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88, 0x36,
    0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd, 0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40, 0x1e,
    0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e, 0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d, 0xca,
    0x5e, 0x9b, 0x9f, 0xa, 0x15, 0x79, 0x2b, 0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57, 0x7,
    0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0xd, 0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18, 0xe3,
    0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c, 0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e, 0x37,
    0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd, 0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61, 0xf2,
    0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e, 0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2, 0x1f,
    0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76, 0xc4, 0x17, 0x49, 0xec, 0x7f, 0xc, 0x6f, 0xf6, 0x6c,
    0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa, 0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a, 0xcb,
    0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51, 0xb, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7, 0x4f,
    0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8, 0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf,
];

// --- GF(2^8) arithmetic ---

#[inline]
fn gf_mul(x: u8, y: u8) -> u8 {
    // Polynomial loops normally keep the second operand fixed. Select its
    // row so scaling, evaluation, and Forney updates reuse 256 adjacent bytes.
    GF_PRODUCTS[y as usize][x as usize]
}

#[inline]
fn gf_div(x: u8, y: u8) -> u8 {
    debug_assert!(y != 0);
    if x == 0 {
        return 0;
    }
    GF_EXP[(GF_LOG[x as usize] as usize + 255 - GF_LOG[y as usize] as usize) % 255]
}

#[inline]
fn gf_pow(x: u8, power: isize) -> u8 {
    let mut i = GF_LOG[x as usize] as isize * power;
    i %= 255;
    if i < 0 {
        i += 255;
    }
    GF_EXP[i as usize]
}

#[inline]
fn gf_inverse(x: u8) -> u8 {
    GF_EXP[255 - GF_LOG[x as usize] as usize]
}

// --- Polynomial operations (dynamic Vec-based) ---

#[cfg(test)]
fn poly_scale(p: &[u8], x: u8) -> Vec<u8> {
    p.iter().map(|&c| gf_mul(c, x)).collect()
}

fn poly_add(p: &[u8], q: &[u8]) -> Vec<u8> {
    let len = p.len().max(q.len());
    let mut result = vec![0u8; len];
    for (i, &v) in p.iter().enumerate() {
        result[i + len - p.len()] = v;
    }
    for (i, &v) in q.iter().enumerate() {
        result[i + len - q.len()] ^= v;
    }
    result
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut result = vec![0u8; p.len() + q.len() - 1];
    for (j, &qv) in q.iter().enumerate() {
        for (i, &pv) in p.iter().enumerate() {
            result[i + j] ^= gf_mul(pv, qv);
        }
    }
    result
}

fn poly_div(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut result = p.to_vec();
    for i in 0..(p.len() - (q.len() - 1)) {
        let coef = result[i];
        if coef != 0 {
            for j in 1..q.len() {
                if q[j] != 0 {
                    result[i + j] ^= gf_mul(q[j], coef);
                }
            }
        }
    }
    let sep = p.len() - (q.len() - 1);
    result[sep..].to_vec()
}

fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut y = p[0];
    for &c in &p[1..] {
        y = gf_mul(y, x) ^ c;
    }
    y
}

// --- Generator polynomial ---

fn generator_poly(ecc_length: usize) -> Vec<u8> {
    let mut gen = vec![1u8];
    for i in 0..ecc_length {
        let mulp = [1u8, gf_pow(2, i as isize)];
        gen = poly_mul(&gen, &mulp);
    }
    gen
}

// --- Syndromes ---

// One fixed 64 KiB table is shared by field multiplication and syndrome
// evaluation. It replaces logarithm/exponent lookups and zero branches,
// with no per-message allocation, cache growth, or initialization work.
static GF_PRODUCTS: [[u8; 256]; 256] = {
    let mut products = [[0u8; 256]; 256];
    let mut x = 1;
    while x < 256 {
        let mut y = 1;
        while y < 256 {
            products[x][y] = GF_EXP[GF_LOG[x] as usize + GF_LOG[y] as usize];
            y += 1;
        }
        x += 1;
    }
    products
};

fn calc_syndromes(msg: &[u8], ecc_length: usize) -> Vec<u8> {
    let mut synd = vec![0u8; ecc_length + 1];
    for i in 1..=ecc_length {
        let products = &GF_PRODUCTS[gf_pow(2, (i - 1) as isize) as usize];
        let mut value = msg[0];
        for &coefficient in &msg[1..] {
            value = products[value as usize] ^ coefficient;
        }
        synd[i] = value;
    }
    synd
}

// --- Error locator (Berlekamp-Massey) ---

fn find_error_locator(synd: &[u8], ecc_length: usize, erase_count: usize) -> Option<Vec<u8>> {
    // A locator grows by at most one coefficient per iteration. Reserve
    // both work buffers once; scaling and polynomial addition stay in place.
    let mut err_loc = Vec::with_capacity(ecc_length + 1);
    let mut old_loc = Vec::with_capacity(ecc_length + 1);
    err_loc.push(1u8);
    old_loc.push(1u8);

    let synd_shift = if synd.len() > ecc_length {
        synd.len() - ecc_length
    } else {
        0
    };

    for i in 0..(ecc_length - erase_count) {
        let k = i + synd_shift;

        let mut delta = synd[k];
        for j in 1..err_loc.len() {
            let index = err_loc.len() - j - 1;
            delta ^= gf_mul(err_loc[index], synd[k - j]);
        }

        old_loc.push(0);

        if delta != 0 {
            if old_loc.len() > err_loc.len() {
                std::mem::swap(&mut old_loc, &mut err_loc);
                for coefficient in &mut err_loc {
                    *coefficient = gf_mul(*coefficient, delta);
                }
                let inverse = gf_inverse(delta);
                for coefficient in &mut old_loc {
                    *coefficient = gf_mul(*coefficient, inverse);
                }
            }
            // After the possible swap, err_loc is at least as long as
            // old_loc. This is exactly poly_add(err_loc, scale(old_loc,
            // delta)), right-aligned as in the original implementation.
            let offset = err_loc.len() - old_loc.len();
            for (i, &coefficient) in old_loc.iter().enumerate() {
                err_loc[offset + i] ^= gf_mul(coefficient, delta);
            }
        }
    }

    // strip leading zeros
    let mut shift = 0;
    while shift < err_loc.len() && err_loc[shift] == 0 {
        shift += 1;
    }
    let err_loc = err_loc[shift..].to_vec();

    let errs = err_loc.len() - 1;
    // Budget: each unknown error costs 2, each erasure costs 1.
    // When using Forney syndromes, errs already excludes erasures,
    // so the check is simply errs*2 + erase_count.
    if errs * 2 + erase_count > ecc_length {
        return None;
    }

    Some(err_loc)
}

// --- Find error positions (Chien search) ---

fn find_errors(error_loc: &[u8], msg_size: usize) -> Option<Vec<usize>> {
    let errs = error_loc.len() - 1;
    let mut err_pos = Vec::new();

    for i in 0..msg_size {
        if poly_eval(error_loc, gf_pow(2, i as isize)) == 0 {
            err_pos.push(msg_size - 1 - i);
        }
    }

    if err_pos.len() != errs {
        return None;
    }

    Some(err_pos)
}

// --- Errata locator ---

fn find_errata_locator(epos: &[usize], msg_len: usize) -> Vec<u8> {
    let c_pos: Vec<usize> = epos.iter().map(|&p| msg_len - 1 - p).collect();
    let mut errata_loc = vec![1u8];
    for &p in &c_pos {
        let addp = [gf_pow(2, p as isize), 0];
        let mulp = [1u8];
        let apol = poly_add(&mulp, &addp);
        errata_loc = poly_mul(&errata_loc, &apol);
    }
    errata_loc
}

// --- Error evaluator ---

fn find_error_evaluator(synd: &[u8], errata_loc: &[u8], ecc_len: usize) -> Vec<u8> {
    let mulp = poly_mul(synd, errata_loc);
    let mut divisor = vec![0u8; ecc_len + 2];
    divisor[0] = 1;
    poly_div(&mulp, &divisor)
}

// --- Correct errata ---

fn correct_errata(msg_in: &[u8], synd: &[u8], err_pos: &[usize]) -> Vec<u8> {
    let msg_len = msg_in.len();

    let c_pos: Vec<usize> = err_pos.iter().map(|&p| msg_len - 1 - p).collect();

    let errata_loc = find_errata_locator(err_pos, msg_len);

    // reverse syndromes
    let rsynd: Vec<u8> = synd.iter().rev().copied().collect();

    // error evaluator
    let re_eval = find_error_evaluator(&rsynd, &errata_loc, errata_loc.len() - 1);

    // X: error positions as GF elements
    let x_vals: Vec<u8> = c_pos
        .iter()
        .map(|&p| {
            let l = 255isize - p as isize;
            gf_pow(2, -l)
        })
        .collect();

    // Magnitude polynomial
    let mut e = vec![0u8; msg_len];

    for i in 0..x_vals.len() {
        let xi_inv = gf_inverse(x_vals[i]);

        // err_loc_prime
        let mut err_loc_prime = 1u8;
        for j in 0..x_vals.len() {
            if j != i {
                let val = 1 ^ gf_mul(xi_inv, x_vals[j]);
                err_loc_prime = gf_mul(err_loc_prime, val);
            }
        }

        let y = gf_mul(gf_pow(x_vals[i], 1), poly_eval(&re_eval, xi_inv));

        e[err_pos[i]] = gf_div(y, err_loc_prime);
    }

    poly_add(msg_in, &e)
}

/// Forney syndromes for error detection after erasure removal.
fn calc_forney_syndromes(synd: &[u8], erasures_pos: &[usize], msg_in_size: usize) -> Vec<u8> {
    let erase_pos_reversed: Vec<usize> =
        erasures_pos.iter().map(|&p| msg_in_size - 1 - p).collect();

    let mut forney_synd = synd[1..].to_vec();

    for &p in &erase_pos_reversed {
        let x = gf_pow(2, p as isize);
        for j in 0..forney_synd.len().saturating_sub(1) {
            forney_synd[j] = gf_mul(forney_synd[j], x) ^ forney_synd[j + 1];
        }
    }

    forney_synd
}

/// Reed-Solomon encoder/decoder.
pub struct ReedSolomon {
    msg_length: usize,
    ecc_length: usize,
    // Decoding does not use the generator polynomial. Build it only when
    // encoding, rather than for every length/alignment/Chase hypothesis.
    generator: OnceLock<Vec<u8>>,
}

/// A received codeword and its syndromes, shared by hard and erasure retries.
/// Keeping the bytes borrowed alongside their syndromes prevents a retry
/// from accidentally using checks calculated for a different codeword.
pub(crate) struct PreparedBlock<'a> {
    msg_length: usize,
    ecc_length: usize,
    encoded: &'a [u8],
    syndromes: Vec<u8>,
}

/// Reuse syndrome evaluation when candidate codewords are prefixes of one
/// immutable received body. The first request uses the ordinary calculation;
/// only a second actual request allocates prefix rows. Failure-cache hits
/// therefore do not trigger speculative work, and clean decoding stays cheap.
pub(crate) struct PrefixDecoder<'a> {
    encoded: &'a [u8],
    max_ecc: usize,
    first_request: bool,
    // One row per generator root, indexed by prefix length minus one.
    // At most 255 rows, each covering at most 255 received bytes.
    prefixes: Vec<[u8; 255]>,
    #[cfg(test)]
    prefix_steps: usize,
}

/// Exact syndrome reuse across sparse changes to one immutable received body.
/// Chase retries change at most a few bytes. Evaluate the original polynomial
/// prefixes once, then XOR each changed coefficient's contribution at the
/// requested codeword length. Corrected codewords still undergo the ordinary
/// independent full syndrome verification in `PreparedBlock`.
pub(crate) struct RetrySyndromes {
    original: Vec<u8>,
    prefixes: Vec<[u8; 255]>,
    #[cfg(test)]
    prefix_steps: usize,
    #[cfg(test)]
    delta_steps: usize,
}

impl RetrySyndromes {
    pub(crate) fn new(original: &[u8]) -> Self {
        assert!(original.len() <= 255);
        Self {
            original: original.to_vec(),
            prefixes: Vec::new(),
            #[cfg(test)]
            prefix_steps: 0,
            #[cfg(test)]
            delta_steps: 0,
        }
    }

    pub(crate) fn prepare_decode<'a>(
        &mut self, encoded: &'a [u8], msg_length: usize, ecc_length: usize,
    ) -> PreparedBlock<'a> {
        assert!(msg_length <= self.original.len()
            && ecc_length <= self.original.len() - msg_length);
        let length = msg_length + ecc_length;
        assert_eq!(encoded.len(), length);
        if self.prefixes.len() < ecc_length {
            self.prefixes.reserve_exact(ecc_length - self.prefixes.len());
        }
        for root in self.prefixes.len()..ecc_length {
            let products = &GF_PRODUCTS[gf_pow(2, root as isize) as usize];
            let mut row = [0u8; 255];
            let mut value = 0;
            for (prefix, &coefficient) in row.iter_mut().zip(&self.original) {
                value = products[value as usize] ^ coefficient;
                *prefix = value;
                #[cfg(test)]
                { self.prefix_steps += 1; }
            }
            self.prefixes.push(row);
        }
        let mut syndromes = vec![0; ecc_length + 1];
        for (root, row) in self.prefixes.iter().take(ecc_length).enumerate() {
            syndromes[root + 1] = row[length - 1];
        }
        for (position, (&before, &after)) in self.original.iter().zip(encoded).enumerate() {
            let delta = before ^ after;
            if delta == 0 { continue; }
            let power = length - 1 - position;
            for root in 0..ecc_length {
                syndromes[root + 1] ^= gf_mul(delta, gf_pow(2, (root * power) as isize));
                #[cfg(test)]
                { self.delta_steps += 1; }
            }
        }
        PreparedBlock { msg_length, ecc_length, encoded, syndromes }
    }
}

impl<'a> PrefixDecoder<'a> {
    pub(crate) fn new(encoded: &'a [u8], max_ecc: usize) -> Self {
        assert!(encoded.len() <= 255 && max_ecc <= encoded.len());
        Self {
            encoded,
            max_ecc,
            first_request: true,
            prefixes: Vec::new(),
            #[cfg(test)]
            prefix_steps: 0,
        }
    }

    pub(crate) fn prepare_decode(&mut self, msg_length: usize, ecc_length: usize) -> PreparedBlock<'a> {
        assert!(msg_length <= self.encoded.len() && ecc_length <= self.max_ecc);
        let length = msg_length + ecc_length;
        assert!(length <= self.encoded.len());
        let encoded = &self.encoded[..length];
        if self.first_request {
            self.first_request = false;
            return ReedSolomon::new(msg_length, ecc_length).prepare_decode(encoded);
        }
        if ecc_length > 0 && self.prefixes.capacity() == 0 {
            self.prefixes.reserve_exact(self.max_ecc);
        }
        for root in self.prefixes.len()..ecc_length {
            let products = &GF_PRODUCTS[gf_pow(2, root as isize) as usize];
            let mut row = [0u8; 255];
            let mut value = 0;
            for (prefix, &coefficient) in row.iter_mut().zip(self.encoded) {
                value = products[value as usize] ^ coefficient;
                *prefix = value;
                #[cfg(test)]
                { self.prefix_steps += 1; }
            }
            self.prefixes.push(row);
        }
        let mut syndromes = vec![0; ecc_length + 1];
        for (i, row) in self.prefixes.iter().take(ecc_length).enumerate() {
            syndromes[i + 1] = row[length - 1];
        }
        PreparedBlock { msg_length, ecc_length, encoded, syndromes }
    }
}

impl ReedSolomon {
    /// Create a new RS codec with the given message and ECC lengths.
    pub fn new(msg_length: usize, ecc_length: usize) -> Self {
        Self {
            msg_length,
            ecc_length,
            generator: OnceLock::new(),
        }
    }

    /// Encode: appends ECC bytes to the message.
    /// Returns a vector of `msg_length + ecc_length` bytes.
    pub fn encode(&self, msg: &[u8]) -> Vec<u8> {
        assert_eq!(msg.len(), self.msg_length);
        assert!(self.msg_length + self.ecc_length < 256);

        let generator = self
            .generator
            .get_or_init(|| generator_poly(self.ecc_length));
        let mut msg_out = vec![0u8; self.msg_length + self.ecc_length];
        msg_out[..self.msg_length].copy_from_slice(msg);

        for i in 0..self.msg_length {
            let coef = msg_out[i];
            if coef != 0 {
                for j in 1..generator.len() {
                    msg_out[i + j] ^= gf_mul(generator[j], coef);
                }
            }
        }

        // The first msg_length bytes get overwritten by the division;
        // restore the original message.
        msg_out[..self.msg_length].copy_from_slice(msg);

        msg_out
    }

    /// Encode block: returns only the ECC bytes (not the message).
    pub fn encode_block(&self, msg: &[u8]) -> Vec<u8> {
        let full = self.encode(msg);
        full[self.msg_length..].to_vec()
    }

    /// Decode: takes `msg_length + ecc_length` bytes, returns decoded message.
    /// Returns `None` if correction fails.
    pub fn decode(&self, encoded: &[u8]) -> Option<Vec<u8>> {
        self.decode_with_erasures(encoded, &[])
    }

    /// Decode with known erasure positions (byte indices into the encoded block).
    ///
    /// Erasures are positions where the symbol confidence was too low to trust.
    /// Each erasure costs 1 ECC symbol (vs 2 for an unknown error), so providing
    /// erasure positions lets RS correct more total damage.
    /// Returns `None` if correction fails.
    pub fn decode_with_erasures(&self, encoded: &[u8], erase_pos: &[usize]) -> Option<Vec<u8>> {
        self.prepare_decode(encoded).decode_with_erasures(erase_pos)
    }

    pub(crate) fn prepare_decode<'a>(&self, encoded: &'a [u8]) -> PreparedBlock<'a> {
        assert_eq!(encoded.len(), self.msg_length + self.ecc_length);
        PreparedBlock {
            msg_length: self.msg_length,
            ecc_length: self.ecc_length,
            encoded,
            syndromes: calc_syndromes(encoded, self.ecc_length),
        }
    }

    /// Decode block: takes separate message and ecc slices.
    /// Returns decoded message or None if correction fails.
    pub fn decode_block(&self, msg: &[u8], ecc: &[u8]) -> Option<Vec<u8>> {
        let mut encoded = Vec::with_capacity(self.msg_length + self.ecc_length);
        encoded.extend_from_slice(msg);
        encoded.extend_from_slice(ecc);
        self.decode(&encoded)
    }
}

impl PreparedBlock<'_> {
    pub(crate) fn decode_with_erasures(&self, erase_pos: &[usize]) -> Option<Vec<u8>> {
        let encoded = self.encoded;
        let src_len = self.msg_length + self.ecc_length;
        let synd = &self.syndromes;

        // Check for errors
        let has_errors = synd.iter().any(|&s| s != 0);
        if !has_errors {
            return Some(encoded[..self.msg_length].to_vec());
        }

        // Budget check: erasures alone must not exceed ECC capacity
        if erase_pos.len() > self.ecc_length {
            return None;
        }

        // Use Forney syndromes to find additional errors beyond known erasures
        let forney = calc_forney_syndromes(synd, erase_pos, src_len);

        let error_loc = find_error_locator(&forney, self.ecc_length, erase_pos.len())?;

        // reverse error locator
        let reloc: Vec<u8> = error_loc.iter().rev().copied().collect();

        let err = find_errors(&reloc, src_len)?;

        // Combine erasure positions with found errors
        let mut all_pos: Vec<usize> = erase_pos.to_vec();
        all_pos.extend_from_slice(&err);

        if all_pos.is_empty() {
            return None;
        }

        let corrected = correct_errata(encoded, synd, &all_pos);

        // Verify correction
        let verify_synd = calc_syndromes(&corrected, self.ecc_length);
        if verify_synd.iter().any(|&s| s != 0) {
            return None;
        }

        Some(corrected[..self.msg_length].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_syndromes_match_every_prefix_and_parity_width() {
        for seed in [0u8, 73, 255] {
            let original: Vec<_> = (0..255).map(|i| if seed == 73 { (i * 73 + 41) as u8 } else { seed }).collect();
            let mut retry = RetrySyndromes::new(&original);
            for length in 0..=255 {
                let mut changed = original[..length].to_vec();
                if length > 0 {
                    for (position, delta) in [(0, 0x13), (length / 2, 0x29), (length - 1, 0x40)] {
                        changed[position] ^= delta;
                    }
                }
                let expected = calc_syndromes(&changed, length);
                for ecc in 0..=length {
                    let prepared = retry.prepare_decode(&changed, length - ecc, ecc);
                    assert_eq!(prepared.syndromes, expected[..=ecc], "seed {seed}, length {length}, ecc {ecc}");
                    assert_eq!(prepared.encoded.as_ptr(), changed.as_ptr());
                }
            }
            assert_eq!(retry.prefix_steps, 255 * 255, "each root/prefix is evaluated once");
            assert_eq!(retry.prefixes.len(), 255);
            assert_eq!(retry.prefixes.capacity(), 255);
            assert_eq!(retry.original, original);
        }
    }

    #[test]
    fn retry_syndromes_are_lazy_bounded_and_independent_of_previous_trials() {
        let original: Vec<_> = (0..255).map(|i| (i * 73 + 41) as u8).collect();
        let mut retry = RetrySyndromes::new(&original);
        assert_eq!(retry.prefixes.capacity(), 0);
        assert_eq!(retry.prefix_steps, 0);
        for _ in 0..2 {
            assert_eq!(retry.prepare_decode(&[], 0, 0).syndromes, [0]);
        }
        assert_eq!(retry.prefixes.capacity(), 0);
        let mut changed = original.clone();
        for mask in 0..64 {
            changed.copy_from_slice(&original);
            for position in 0..6 {
                if mask & (1 << position) != 0 { changed[position * 31] ^= 0x5a; }
            }
            for (length, ecc) in [(255, 127), (128, 64), (17, 2)] {
                let prepared = retry.prepare_decode(&changed[..length], length - ecc, ecc);
                assert_eq!(prepared.syndromes, calc_syndromes(&changed[..length], ecc));
            }
        }
        assert_eq!(retry.prefix_steps, 255 * 127);
        assert!(retry.delta_steps <= 64 * 6 * (127 + 64 + 2));
        assert_eq!(retry.prefixes.capacity(), 127);
        assert_eq!(retry.original, original);
        assert_eq!(retry.prepare_decode(&original, 128, 127).syndromes, calc_syndromes(&original, 127),
            "previous hypotheses never accumulate into the next one");
    }

    #[test]
    fn retry_syndromes_preserve_profile_correction_and_erasures() {
        use crate::protocol::{EccProfile, MAX_LENGTH_VARIABLE};
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                let payload: Vec<_> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
                let protected = profile.protect_payload(&payload);
                let ecc = profile.ecc_bytes(len);
                let rs = ReedSolomon::new(protected.len(), ecc);
                let original = rs.encode(&protected);
                let mut reference = original.clone();
                reference.resize(255, 0xa5);
                let mut retry = RetrySyndromes::new(&reference);
                for damage in 0..4 {
                    let mut bytes = original.clone();
                    let count = match damage { 0 => 0, 1 => ecc / 2, _ => ecc / 2 + 1 };
                    for i in 0..count { bytes[i] ^= 0x5a; }
                    let erased: Vec<_> = if damage < 2 { vec![] }
                        else { (0..count - (damage == 3) as usize).rev().collect() };
                    let prepared = retry.prepare_decode(&bytes, protected.len(), ecc);
                    assert_eq!(prepared.decode_with_erasures(&[]), rs.decode(&bytes),
                        "{profile:?}, length {len}, damage {damage}");
                    assert_eq!(prepared.decode_with_erasures(&erased), rs.decode_with_erasures(&bytes, &erased),
                        "{profile:?}, length {len}, damage {damage}");
                }
            }
        }
    }

    #[test]
    fn prefix_syndromes_match_every_length_and_parity_width() {
        let mut state = 0x12ab34cdu32;
        let random: Vec<u8> = (0..255).map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }).collect();
        for bytes in [vec![0; 255], vec![255; 255], (0..255).collect(), random] {
            let mut prefixes = PrefixDecoder::new(&bytes, 255);
            for length in 0..=255 {
                let expected = if length == 0 { vec![0; 256] }
                    else { calc_syndromes(&bytes[..length], 255) };
                for ecc in 0..=length {
                    let prepared = prefixes.prepare_decode(length - ecc, ecc);
                    assert_eq!(prepared.syndromes, expected[..ecc + 1], "length {length}, ecc {ecc}");
                    assert_eq!(prepared.encoded, &bytes[..length]);
                }
            }
            // Revisit earlier prefixes after the table has grown fully.
            for length in (1..=255).rev() {
                let ecc = length / 2;
                assert_eq!(prefixes.prepare_decode(length - ecc, ecc).syndromes,
                    calc_syndromes(&bytes[..length], ecc));
            }
            assert_eq!(prefixes.prefix_steps, 255 * 255);
        }
    }

    #[test]
    fn prefix_decoder_is_lazy_bounded_and_reuses_existing_rows() {
        let bytes: Vec<u8> = (0..255).collect();
        let mut prefixes = PrefixDecoder::new(&bytes, 127);
        assert_eq!(prefixes.prefixes.capacity(), 0);
        let first = prefixes.prepare_decode(64, 64);
        assert_eq!(first.syndromes, calc_syndromes(&bytes[..128], 64));
        assert_eq!(prefixes.prefixes.capacity(), 0);
        assert_eq!(prefixes.prefix_steps, 0);
        prefixes.prepare_decode(64, 63);
        assert_eq!(prefixes.prefixes.len(), 63);
        assert!(prefixes.prefixes.capacity() >= 127);
        assert_eq!(prefixes.prefix_steps, 63 * bytes.len());
        let storage = prefixes.prefixes.as_ptr();
        for length in (64..=255).rev() {
            prefixes.prepare_decode(length - 63, 63);
        }
        assert_eq!(prefixes.prefix_steps, 63 * bytes.len());
        prefixes.prepare_decode(128, 127);
        assert_eq!(prefixes.prefix_steps, 127 * bytes.len());
        assert_eq!(prefixes.prefixes.as_ptr(), storage);
        // Prepared results own their syndromes and keep the original byte
        // slice, independently of subsequent table growth and queries.
        assert_eq!(first.syndromes, calc_syndromes(first.encoded, 64));
        let mut empty = PrefixDecoder::new(&[], 0);
        for _ in 0..2 {
            assert_eq!(empty.prepare_decode(0, 0).decode_with_erasures(&[]), Some(vec![]));
        }
        assert_eq!(empty.prefixes.capacity(), 0);
    }

    #[test]
    fn prefix_decoding_preserves_profile_error_and_erasure_results() {
        use crate::protocol::{EccProfile, MAX_LENGTH_VARIABLE};
        for profile in [EccProfile::Standard, EccProfile::Legacy, EccProfile::Paranoid] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                let payload: Vec<u8> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
                let protected = profile.protect_payload(&payload);
                let ecc = profile.ecc_bytes(len);
                let rs = ReedSolomon::new(protected.len(), ecc);
                let original = rs.encode(&protected);
                for damage in 0..4 {
                    let mut bytes = original.clone();
                    let count = match damage { 0 => 0, 1 => ecc / 2, _ => ecc / 2 + 1 };
                    for i in 0..count { bytes[i] ^= 0x5a; }
                    let erased: Vec<usize> = if damage < 2 { vec![] }
                        else { (0..count - (damage == 3) as usize).rev().collect() };
                    let expected_hard = rs.decode(&bytes);
                    let expected_soft = rs.decode_with_erasures(&bytes, &erased);
                    if damage <= 2 {
                        assert_eq!(expected_soft.as_deref(), Some(protected.as_slice()));
                    }
                    // Unrelated trailing observations must not affect this
                    // prefix. Warm a different length to exercise cached rows.
                    bytes.resize(255, 0xa5);
                    let mut prefixes = PrefixDecoder::new(&bytes, ecc);
                    prefixes.prepare_decode(1, 0);
                    let prepared = prefixes.prepare_decode(protected.len(), ecc);
                    assert_eq!(prepared.decode_with_erasures(&[]), expected_hard,
                        "{profile:?}, length {len}, damage {damage}");
                    assert_eq!(prepared.decode_with_erasures(&erased), expected_soft,
                        "{profile:?}, length {len}, damage {damage}");
                }
            }
        }
    }

    #[test]
    fn table_products_match_original_and_polynomial_multiplication_exhaustively() {
        for x in 0..=255u8 {
            for y in 0..=255u8 {
                // Original logarithm/exponent calculation, including zero.
                let original = if x == 0 || y == 0 {
                    0
                } else {
                    GF_EXP[GF_LOG[x as usize] as usize + GF_LOG[y as usize] as usize]
                };
                // Independent carry-less polynomial multiplication modulo
                // x^8 + x^4 + x^3 + x^2 + 1, without any lookup tables.
                let mut multiplicand = x as u16;
                let mut multiplier = y;
                let mut product = 0u16;
                for _ in 0..8 {
                    if multiplier & 1 != 0 { product ^= multiplicand; }
                    multiplier >>= 1;
                    multiplicand <<= 1;
                    if multiplicand & 0x100 != 0 { multiplicand ^= 0x11d; }
                }
                assert_eq!(GF_PRODUCTS[x as usize][y as usize], original);
                assert_eq!(gf_mul(x, y), original);
                assert_eq!(gf_mul(x, y), product as u8);
            }
        }
    }

    #[test]
    fn table_syndromes_match_polynomial_evaluation() {
        assert_eq!(calc_syndromes(&[], 0), [0]);
        for len in 1..=255 {
            for pattern in 0..4 {
                let mut state = len as u32;
                let message: Vec<u8> = (0..len)
                    .map(|i| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        match pattern {
                            0 => 0,
                            1 => 255,
                            2 => (state >> 24) as u8,
                            _ => {
                                if i == len / 2 {
                                    173
                                } else {
                                    0
                                }
                            }
                        }
                    })
                    .collect();
                // Include exponent wrapping as well as all profiles' range.
                for ecc in [0, 1, 2, 4, 8, 16, 56, 124, 255, 256, 257] {
                    let expected: Vec<u8> = std::iter::once(0)
                        .chain((0..ecc).map(|i| poly_eval(&message, gf_pow(2, i as isize))))
                        .collect();
                    assert_eq!(
                        calc_syndromes(&message, ecc),
                        expected,
                        "length {len}, pattern {pattern}, parity {ecc}"
                    );
                }
            }
        }
    }

    #[test]
    fn lazy_generator_preserves_encoded_bytes_for_every_parity_width() {
        use crate::protocol::{EccProfile, MAX_LENGTH_VARIABLE};
        let mut geometries: Vec<(usize, usize)> = (0..=255).map(|ecc| (255 - ecc, ecc)).collect();
        for profile in [
            EccProfile::Standard,
            EccProfile::Legacy,
            EccProfile::Paranoid,
        ] {
            for len in 1..=MAX_LENGTH_VARIABLE {
                geometries.push((profile.protected_bytes(len), profile.ecc_bytes(len)));
            }
        }
        for (len, ecc) in geometries {
            let message: Vec<u8> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
            // Original eager-generator division, independent of encode().
            let generator = generator_poly(ecc);
            let mut expected = message.clone();
            expected.resize(len + ecc, 0);
            for i in 0..len {
                let coefficient = expected[i];
                if coefficient != 0 {
                    for j in 1..generator.len() {
                        expected[i + j] ^= gf_mul(generator[j], coefficient);
                    }
                }
            }
            expected[..len].copy_from_slice(&message);

            let rs = ReedSolomon::new(len, ecc);
            assert!(rs.generator.get().is_none());
            assert_eq!(rs.encode(&message), expected);
            let cached = rs.generator.get().unwrap().as_ptr();
            assert_eq!(rs.encode(&message), expected);
            assert_eq!(rs.generator.get().unwrap().as_ptr(), cached);
            let receiver = ReedSolomon::new(len, ecc);
            assert_eq!(receiver.decode(&expected), Some(message));
            assert!(receiver.generator.get().is_none());
        }
    }

    #[test]
    fn correcting_errors_and_erasures_never_builds_an_encoder_generator() {
        let message: Vec<u8> = (0..120).map(|i| (i * 73 + 41) as u8).collect();
        let mut encoded = ReedSolomon::new(120, 120).encode(&message);
        let receiver = ReedSolomon::new(120, 120);
        for i in 0..60 {
            encoded[i * 3] ^= 173;
        }
        assert_eq!(receiver.decode(&encoded), Some(message.clone()));
        let erasures: Vec<usize> = (0..120).map(|i| i * 2).collect();
        encoded = ReedSolomon::new(120, 120).encode(&message);
        for &pos in &erasures {
            encoded[pos] ^= 71;
        }
        assert_eq!(
            receiver.decode_with_erasures(&encoded, &erasures),
            Some(message)
        );
        assert!(receiver.generator.get().is_none());
    }

    // Original allocation-heavy Berlekamp–Massey loop. Retain independent
    // buffer operations so the in-place swap/scaling/addition can be checked.
    fn reference_error_locator(synd: &[u8], ecc_length: usize, erase_count: usize) -> Option<Vec<u8>> {
        let mut err_loc = vec![1u8];
        let mut old_loc = vec![1u8];
        let synd_shift = synd.len().saturating_sub(ecc_length);
        for i in 0..ecc_length - erase_count {
            let k = i + synd_shift;
            let mut delta = synd[k];
            for j in 1..err_loc.len() {
                delta ^= gf_mul(err_loc[err_loc.len() - j - 1], synd[k - j]);
            }
            old_loc.push(0);
            if delta != 0 {
                if old_loc.len() > err_loc.len() {
                    let temp = poly_scale(&old_loc, delta);
                    old_loc = poly_scale(&err_loc, gf_inverse(delta));
                    err_loc = temp;
                }
                err_loc = poly_add(&err_loc, &poly_scale(&old_loc, delta));
            }
        }
        let shift = err_loc.iter().take_while(|&&coefficient| coefficient == 0).count();
        let err_loc = err_loc[shift..].to_vec();
        ((err_loc.len() - 1) * 2 + erase_count <= ecc_length).then_some(err_loc)
    }

    #[test]
    fn in_place_locators_match_original_for_every_parity_and_erasure_budget() {
        for ecc in 0..=255 {
            for erased in 0..=ecc {
                for pattern in 0..3 {
                    let mut random = (ecc * 257 + erased * 73 + 1) as u32;
                    let synd: Vec<_> = (0..ecc).map(|_| {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        match pattern {
                            0 => 0,
                            1 => 173,
                            _ => (random >> 24) as u8,
                        }
                    }).collect();
                    assert_eq!(find_error_locator(&synd, ecc, erased),
                        reference_error_locator(&synd, ecc, erased),
                        "parity {ecc}, erasures {erased}, pattern {pattern}");
                }
            }
            // Preserve both the raw syndrome's leading sentinel and the
            // original handling of longer prefixes (synd_shift).
            for prefix in [1, 3] {
                for erased in [0, ecc / 2, ecc] {
                    for position in 0..ecc + prefix {
                        let mut synd = vec![0; ecc + prefix];
                        synd[position] = 173;
                        assert_eq!(find_error_locator(&synd, ecc, erased),
                            reference_error_locator(&synd, ecc, erased));
                    }
                }
            }
        }
    }

    // Original decoder, using polynomial evaluation for its initial and
    // verification syndromes, to check the prepared-block refactor end to end.
    fn reference_decode(
        encoded: &[u8],
        len: usize,
        ecc: usize,
        erasures: &[usize],
    ) -> Option<Vec<u8>> {
        let syndromes = |bytes: &[u8]| -> Vec<u8> {
            std::iter::once(0)
                .chain((0..ecc).map(|i| poly_eval(bytes, gf_pow(2, i as isize))))
                .collect()
        };
        let synd = syndromes(encoded);
        if synd.iter().all(|&s| s == 0) {
            return Some(encoded[..len].to_vec());
        }
        if erasures.len() > ecc {
            return None;
        }
        let forney = calc_forney_syndromes(&synd, erasures, encoded.len());
        let locator = reference_error_locator(&forney, ecc, erasures.len())?;
        let reversed: Vec<u8> = locator.iter().rev().copied().collect();
        let errors = find_errors(&reversed, encoded.len())?;
        let mut positions = erasures.to_vec();
        positions.extend_from_slice(&errors);
        if positions.is_empty() {
            return None;
        }
        let corrected = correct_errata(encoded, &synd, &positions);
        syndromes(&corrected)
            .iter()
            .all(|&s| s == 0)
            .then(|| corrected[..len].to_vec())
    }

    #[test]
    fn prepared_retries_preserve_reference_results_at_all_profile_lengths() {
        use crate::protocol::{EccProfile, MAX_LENGTH_VARIABLE};
        for profile in [
            EccProfile::Standard,
            EccProfile::Legacy,
            EccProfile::Paranoid,
        ] {
            for payload_len in 1..=MAX_LENGTH_VARIABLE {
                let len = profile.protected_bytes(payload_len);
                let ecc = profile.ecc_bytes(payload_len);
                let message: Vec<u8> = (0..len).map(|i| (i * 73 + len * 17) as u8).collect();
                let rs = ReedSolomon::new(len, ecc);
                let codeword = rs.encode(&message);
                for damage in [0, ecc / 2, ecc, ecc + 1] {
                    let positions: Vec<usize> = (0..damage)
                        .map(|i| (i + len / 2) % codeword.len())
                        .collect();
                    let mut received = codeword.clone();
                    for &pos in &positions {
                        received[pos] ^= 173;
                    }
                    let prepared = rs.prepare_decode(&received);
                    let original_syndromes = prepared.syndromes.clone();
                    let hard = reference_decode(&received, len, ecc, &[]);
                    assert_eq!(prepared.decode_with_erasures(&[]), hard);
                    let expected = reference_decode(&received, len, ecc, &positions);
                    assert_eq!(prepared.decode_with_erasures(&positions), expected);
                    if damage <= ecc {
                        assert_eq!(expected, Some(message.clone()));
                    }
                    if damage <= ecc / 2 {
                        assert_eq!(hard, Some(message.clone()));
                    }
                    // An erasure retry cannot mutate the hard-decision input.
                    assert_eq!(prepared.decode_with_erasures(&[]), hard);
                    assert_eq!(prepared.syndromes, original_syndromes);
                }
            }
        }
    }

    #[test]
    fn test_gf_mul_basic() {
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(5, 0), 0);
        assert_eq!(gf_mul(1, 1), 1);
        assert_eq!(gf_mul(2, 2), 4);
    }

    #[test]
    fn test_gf_div_basic() {
        for x in 1..=255u8 {
            assert_eq!(gf_div(x, x), 1);
        }
    }

    #[test]
    fn test_gf_pow_basic() {
        assert_eq!(gf_pow(2, 0), 1);
        assert_eq!(gf_pow(2, 1), 2);
        assert_eq!(gf_pow(2, 8), 0x1d); // 2^8 mod p(x) = 0x1d
    }

    #[test]
    fn test_rs_encode_decode_roundtrip() {
        let rs = ReedSolomon::new(3, 2);
        let msg = [0x48, 0x65, 0x6c]; // "Hel"
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 5);

        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_correct_single_error() {
        let rs = ReedSolomon::new(3, 4);
        let msg = [0x01, 0x02, 0x03];
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[1] ^= 0xFF;

        let decoded = rs.decode(&corrupted).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_length_encoding() {
        // This matches ggwave's RS(1, 2) for the length byte
        let rs = ReedSolomon::new(1, 2);
        let msg = [5u8]; // payload length = 5
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 3);

        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_larger_message() {
        let msg_len = 20;
        let ecc_len = 8; // 2*(20/5) = 8
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // corrupt two bytes
        let mut corrupted = encoded;
        corrupted[3] ^= 0xAB;
        corrupted[10] ^= 0xCD;

        let decoded = rs.decode(&corrupted).unwrap();
        assert_eq!(&decoded, &msg[..]);
    }

    #[test]
    fn test_rs_erasure_known_positions() {
        let msg_len = 10;
        let ecc_len = 6;
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // Corrupt 3 known positions (erasures cost 1 each, budget = 6)
        let mut corrupted = encoded.clone();
        corrupted[0] ^= 0xFF;
        corrupted[4] ^= 0xAA;
        corrupted[8] ^= 0x55;

        // Without erasure info, 3 errors cost 6 ECC symbols — at the limit
        // With erasure info, 3 erasures cost only 3 — plenty of room
        let decoded = rs.decode_with_erasures(&corrupted, &[0, 4, 8]).unwrap();
        assert_eq!(&decoded, &msg[..]);
    }

    #[test]
    fn test_rs_erasure_plus_error() {
        let msg_len = 10;
        let ecc_len = 6;
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // 2 known erasures + 1 unknown error = 2 + 2 = 4 ECC symbols (within budget of 6)
        let mut corrupted = encoded.clone();
        corrupted[1] ^= 0xFF; // erasure
        corrupted[5] ^= 0xAA; // erasure
        corrupted[9] ^= 0x33; // unknown error

        let decoded = rs.decode_with_erasures(&corrupted, &[1, 5]).unwrap();
        assert_eq!(&decoded, &msg[..]);
    }

    #[test]
    fn test_rs_erasure_budget_exceeded() {
        let msg_len = 10;
        let ecc_len = 4;
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // 5 erasures exceeds budget of 4
        let mut corrupted = encoded.clone();
        for i in 0..5 {
            corrupted[i] ^= 0xFF;
        }

        let result = rs.decode_with_erasures(&corrupted, &[0, 1, 2, 3, 4]);
        assert!(
            result.is_none(),
            "should fail when erasures exceed ECC budget"
        );
    }

    #[test]
    fn test_rs_erasure_more_correctable_than_errors() {
        // Demonstrate erasures can fix what hard-decision cannot
        let msg_len = 10;
        let ecc_len = 4; // can fix 2 errors OR 4 erasures
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // 3 corrupted bytes — too many for hard-decision (needs 6 ECC, only have 4)
        let mut corrupted = encoded.clone();
        corrupted[2] ^= 0xDE;
        corrupted[5] ^= 0xAD;
        corrupted[7] ^= 0xBE;

        // Hard-decision fails
        assert!(
            rs.decode(&corrupted).is_none(),
            "hard-decision should fail with 3 errors and ecc=4"
        );

        // Erasure-aided succeeds (3 erasures cost 3, within budget of 4)
        let decoded = rs.decode_with_erasures(&corrupted, &[2, 5, 7]).unwrap();
        assert_eq!(&decoded, &msg[..]);
    }

    #[test]
    fn test_rs_erasure_empty_is_same_as_decode() {
        let rs = ReedSolomon::new(3, 4);
        let msg = [0x01, 0x02, 0x03];
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[1] ^= 0xFF;

        let d1 = rs.decode(&corrupted).unwrap();
        let d2 = rs.decode_with_erasures(&corrupted, &[]).unwrap();
        assert_eq!(d1, d2);
    }
}
