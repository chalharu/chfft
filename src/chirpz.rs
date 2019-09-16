//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use crate::mixed_radix;
use crate::QuarterRotation;
use num_complex::Complex;
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use num_traits::NumAssign;

pub struct ChirpzData<T> {
    pub level: usize,
    pub ids: Vec<usize>,
    pub omega: Vec<Complex<T>>,
    pub omega_back: Vec<Complex<T>>,
    pub src_omega: Vec<Complex<T>>,
    pub rot_conj: Vec<Complex<T>>,
    pub rot_ft: Vec<Complex<T>>,
    pub pow2len_inv: T,
}

pub fn convert_rad2_inplace<T: Float + NumAssign>(
    source: &mut [Complex<T>],
    level: usize,
    ids: &[usize],
    omega: &[Complex<T>],
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
        for data in source.iter_mut() {
            *data = data.scale(pow2len_inv);
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

    if is_back {
        mixed_radix::mixed_kernel_radix4(
            source,
            level >> 1,
            &mut po2,
            &mut rad,
            omega,
            len,
            &QuarterRotation::three_quarter_turn,
        );
    } else {
        mixed_radix::mixed_kernel_radix4(
            source,
            level >> 1,
            &mut po2,
            &mut rad,
            omega,
            len,
            &QuarterRotation::quarter_turn,
        );
    }
}

pub fn convert_chirpz<T: Float + NumAssign>(
    source: &[Complex<T>],
    srclen: usize,
    is_back: bool,
    scaler: T,
    data: &ChirpzData<T>,
) -> Vec<Complex<T>> {
    let len = 1 << data.level;
    let dlen = srclen << 1;

    let mut a = Vec::with_capacity(len);

    for (i, s) in source.iter().enumerate() {
        let w = data.src_omega[(i * i) % dlen];
        a.push(s * w);
    }
    for _ in srclen..len {
        a.push(zero());
    }

    convert_rad2_inplace(
        &mut a,
        data.level,
        &data.ids,
        &data.omega,
        false,
        data.pow2len_inv,
    );
    for (i, d) in a.iter_mut().enumerate() {
        *d *= data.rot_ft[i];
    }
    convert_rad2_inplace(
        &mut a,
        data.level,
        &data.ids,
        &data.omega_back,
        true,
        data.pow2len_inv,
    );

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
                a[i] * data.rot_conj[i].scale(scaler)
            } else {
                a[i] * data.rot_conj[i]
            }
        })
        .collect::<Vec<_>>()
}

pub fn convert_chirpz_inplace<T: Float + NumAssign>(
    source: &mut [Complex<T>],
    srclen: usize,
    is_back: bool,
    scaler: T,
    data: &ChirpzData<T>,
) {
    let len = 1 << data.level;
    let dlen = srclen << 1;

    let mut a = Vec::with_capacity(len);

    for (i, s) in source.iter().enumerate() {
        let w = data.src_omega[(i * i) % dlen];
        a.push(s * w);
    }
    for _ in srclen..len {
        a.push(zero());
    }

    convert_rad2_inplace(
        &mut a,
        data.level,
        &data.ids,
        &data.omega,
        false,
        data.pow2len_inv,
    );
    for (i, d) in a.iter_mut().enumerate() {
        *d *= data.rot_ft[i];
    }
    convert_rad2_inplace(
        &mut a,
        data.level,
        &data.ids,
        &data.omega_back,
        true,
        data.pow2len_inv,
    );

    // Multiply phase factor
    for (i, si) in source.iter_mut().take(srclen).enumerate() {
        let j = if i == 0 {
            0
        } else if is_back {
            srclen - i
        } else {
            i
        };
        *si = if scaler != one() {
            a[j] * data.rot_conj[j].scale(scaler)
        } else {
            a[j] * data.rot_conj[j]
        };
    }
}
