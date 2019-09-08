//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use crate::mixed_radix;
use num_complex::Complex;
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use num_traits::NumAssign;

pub fn convert_rad2_inplace<T: Float + NumAssign>(
    source: &mut [Complex<T>],
    level: usize,
    ids: &Vec<usize>,
    omega: &Vec<Complex<T>>,
    is_back: bool,
    pow2len_inv: T,
) {
    // 入力の並び替え
    for (i, &s) in ids.iter().enumerate() {
        if i != s {
            source.swap(i, s);
        }
    }

    if is_back {
        for i in 0..source.len() {
            source[i] = source[i].scale(pow2len_inv);
        }
    }

    // FFT
    let mut po2 = 1;
    let len = source.len();
    let mut rad = len;
    if (level & 1) == 1 {
        rad >>= 1;
        po2 = 2;
        for j in 0..rad {
            let pos_a = j << 1;
            let pos_b = pos_a + 1;
            let wfb = source[pos_b];
            source[pos_b] = source[pos_a] - wfb;
            source[pos_a] += wfb;
        }
    }

    let im_one = if is_back { -Complex::i() } else { Complex::i() };

    mixed_radix::mixed_kernel_radix4(source, level >> 1, &mut po2, &mut rad, omega, len, &im_one);
}

pub fn convert_chirpz<T: Float + NumAssign>(
    source: &[Complex<T>],
    srclen: usize,
    level: usize,
    ids: &Vec<usize>,
    omega: &Vec<Complex<T>>,
    omega_back: &Vec<Complex<T>>,
    src_omega: &Vec<Complex<T>>,
    rot_conj: &Vec<Complex<T>>,
    rot_ft: &Vec<Complex<T>>,
    is_back: bool,
    pow2len_inv: T,
    scaler: T,
) -> Vec<Complex<T>> {
    let len = 1 << level;
    let dlen = srclen << 1;

    let mut a = Vec::with_capacity(len);

    for i in 0..source.len() {
        let w = src_omega[(i * i) % dlen];
        a.push(source[i] * w);
    }
    for _ in srclen..len {
        a.push(zero());
    }

    convert_rad2_inplace(&mut a, level, ids, omega, false, pow2len_inv);
    for i in 0..a.len() {
        a[i] = a[i] * rot_ft[i];
    }
    convert_rad2_inplace(&mut a, level, ids, omega_back, true, pow2len_inv);

    // Multiply phase factor
    (0..srclen)
        .map(move |i| {
            if i == 0 {
                0
            } else if is_back {
                srclen - i
            } else {
                i
            }
        })
        .map(move |i| {
            if scaler != one() {
                a[i] * rot_conj[i].scale(scaler)
            } else {
                a[i] * rot_conj[i]
            }
        })
        .collect::<Vec<_>>()
}

pub fn convert_chirpz_inplace<T: Float + NumAssign>(
    source: &mut [Complex<T>],
    srclen: usize,
    level: usize,
    ids: &Vec<usize>,
    omega: &Vec<Complex<T>>,
    omega_back: &Vec<Complex<T>>,
    src_omega: &Vec<Complex<T>>,
    rot_conj: &Vec<Complex<T>>,
    rot_ft: &Vec<Complex<T>>,
    is_back: bool,
    pow2len_inv: T,
    scaler: T,
) {
    let len = 1 << level;
    let dlen = srclen << 1;

    let mut a = Vec::with_capacity(len);

    for i in 0..source.len() {
        let w = src_omega[(i * i) % dlen];
        a.push(source[i] * w);
    }
    for _ in srclen..len {
        a.push(zero());
    }

    convert_rad2_inplace(&mut a, level, ids, omega, false, pow2len_inv);
    for i in 0..a.len() {
        a[i] = a[i] * rot_ft[i];
    }
    convert_rad2_inplace(&mut a, level, ids, omega_back, true, pow2len_inv);

    // Multiply phase factor
    for i in 0..srclen {
        let j = if i == 0 {
            0
        } else if is_back {
            srclen - i
        } else {
            i
        };
        source[i] = if scaler != one() {
            a[j] * rot_conj[j].scale(scaler)
        } else {
            a[j] * rot_conj[j]
        };
    }
}
