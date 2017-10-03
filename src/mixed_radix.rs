//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

use num_complex::Complex;
use num_traits::{cast, NumAssign};
use num_traits::float::{Float, FloatConst};
use num_traits::identities::one;
use prime_factorization::Factor;

pub fn convert_mixed<T: Float + NumAssign + FloatConst>(
    source: &[Complex<T>],
    len: usize,
    ids: &Vec<usize>,
    omega: &Vec<Complex<T>>,
    omega_back: &Vec<Complex<T>>,
    factors: &Vec<Factor>,
    is_back: bool,
    scaler: T,
) -> Vec<Complex<T>> {
    // 入力の並び替え
    let mut ret = ids.iter()
        .map(|&i| if scaler != one() {
            source[i].scale(scaler) // このタイミングで割り戻しておく
        } else {
            source[i]
        })
        .collect::<Vec<_>>();

    let (omega, im_one) = if is_back {
        (omega_back, -Complex::i())
    } else {
        (omega, Complex::i())
    };

    fft_kernel(&mut ret, len, omega, factors, &im_one);
    ret
}

pub fn convert_mixed_inplace<T: Float + NumAssign + FloatConst>(
    source: &mut [Complex<T>],
    len: usize,
    ids: &Vec<usize>,
    omega: &Vec<Complex<T>>,
    omega_back: &Vec<Complex<T>>,
    factors: &Vec<Factor>,
    is_back: bool,
    scaler: T,
) {
    // 入力の並び替え
    for (i, &s) in ids.iter().enumerate() {
        if i != s {
            source.swap(i, s);
        }
    }

    let (omega, im_one) = if is_back {
        (omega_back, -Complex::i())
    } else {
        (omega, Complex::i())
    };

    fft_kernel(source, len, omega, factors, &im_one);

    if scaler != one() {
        for i in 0..source.len() {
            source[i] = source[i].scale(scaler);
        }
    }
}

pub fn fft_kernel<T: Float + NumAssign + FloatConst>(
    source: &mut [Complex<T>],
    len: usize,
    omega: &Vec<Complex<T>>,
    factors: &Vec<Factor>,
    im_one: &Complex<T>,
) {
    // FFT
    let mut po2 = 1;
    let mut rad = len;

    for factor in factors {
        match factor.value {
            2 => {
                mixed_kernel_radix2(source, factor.count, &mut po2, &mut rad, omega, len);
            }
            3 => {
                mixed_kernel_radix3(source, factor.count, &mut po2, &mut rad, omega, len, im_one);
            }
            4 => {
                mixed_kernel_radix4(source, factor.count, &mut po2, &mut rad, omega, len, im_one);
            }
            5 => {
                mixed_kernel_radix5(source, factor.count, &mut po2, &mut rad, omega, len, im_one);
            }
            _ => {
                mixed_kernel_other(
                    source,
                    factor.value,
                    factor.count,
                    &mut po2,
                    &mut rad,
                    omega,
                    len,
                );
            }
        }
    }
}

#[inline(always)]
fn mixed_kernel_radix2<T: Float + NumAssign>(
    ret: &mut [Complex<T>],
    count: usize,
    po2: &mut usize,
    rad: &mut usize,
    omega: &Vec<Complex<T>>,
    len: usize,
) {
    for _ in 0..count {
        let po2m = *po2;
        *po2 <<= 1;
        *rad >>= 1;
        for mut j in 0..po2m {
            let w1 = omega[*rad * j];
            while j < len {
                let pos1 = j + po2m;
                let z1 = ret[pos1] * w1;
                ret[pos1] = ret[j] - z1;
                ret[j] += z1;
                j += *po2;
            }
        }
    }
}

#[inline]
fn mixed_kernel_radix3<T: Float + FloatConst + NumAssign>(
    ret: &mut [Complex<T>],
    count: usize,
    po2: &mut usize,
    rad: &mut usize,
    omega: &Vec<Complex<T>>,
    len: usize,
    im_one: &Complex<T>,
) {
    let t3scaler = im_one.scale(
        (cast::<_, T>(-2.0).unwrap() * T::PI() / cast(3.0).unwrap()).sin(),
    );
    for _ in 0..count {
        let po2m = *po2;
        *po2 *= 3;
        *rad /= 3;
        for mut j in 0..po2m {
            let wpos = *rad * j;
            let (w1, w2) = (omega[wpos], omega[wpos << 1]);

            while j < len {
                let pos1 = j + po2m;
                let pos2 = pos1 + po2m;
                let z1 = ret[pos1] * w1;
                let z2 = ret[pos2] * w2;
                let t1 = z1 + z2;
                let t2 = ret[j] - t1.scale(cast(0.5).unwrap());
                let t3 = (z1 - z2) * t3scaler;
                ret[j] += t1;
                ret[pos1] = t2 + t3;
                ret[pos2] = t2 - t3;
                j += *po2;
            }
        }
    }
}

#[inline(always)]
pub fn mixed_kernel_radix4<T: Float>(
    ret: &mut [Complex<T>],
    count: usize,
    po2: &mut usize,
    rad: &mut usize,
    omega: &Vec<Complex<T>>,
    len: usize,
    im_one: &Complex<T>,
) {
    for _ in 0..count {
        let po2m = *po2;
        *po2 <<= 2;
        *rad >>= 2;
        for mut j in 0..po2m {
            let wpos = *rad * j;
            let (w1, w2, w3) = (omega[wpos], omega[wpos << 1], omega[wpos * 3]);
            while j < len {
                let pos1 = j + po2m;
                let pos2 = pos1 + po2m;
                let pos3 = pos2 + po2m;
                let wfb = ret[pos2] * w2;
                let wfab = ret[j] + wfb;
                let wfamb = ret[j] - wfb;
                let wfc = ret[pos1] * w1;
                let wfd = ret[pos3] * w3;
                let wfcd = wfc + wfd;
                let wfcimdi = (wfc - wfd) * im_one;

                ret[j] = wfab + wfcd;
                ret[pos1] = wfamb - wfcimdi;
                ret[pos2] = wfab - wfcd;
                ret[pos3] = wfamb + wfcimdi;
                j += *po2;
            }
        }
    }
}

#[inline]
fn mixed_kernel_radix5<T: Float + FloatConst>(
    ret: &mut [Complex<T>],
    count: usize,
    po2: &mut usize,
    rad: &mut usize,
    omega: &Vec<Complex<T>>,
    len: usize,
    im_one: &Complex<T>,
) {
    for _ in 0..count {
        let po2m = *po2;
        *po2 *= 5;
        *rad /= 5;
        for mut j in 0..po2m {
            let wpos = *rad * j;
            let (w1, w2, w3, w4) = (
                omega[wpos],
                omega[wpos << 1],
                omega[wpos * 3],
                omega[wpos << 2],
            );
            while j < len {
                let pos2 = j + po2m;
                let pos3 = pos2 + po2m;
                let pos4 = pos3 + po2m;
                let pos5 = pos4 + po2m;

                let z0 = ret[j];
                let z1 = ret[pos2] * w1;
                let z2 = ret[pos3] * w2;
                let z3 = ret[pos4] * w3;
                let z4 = ret[pos5] * w4;

                let t1 = z1 + z4;
                let t2 = z2 + z3;
                let t3 = z1 - z4;
                let t4 = z2 - z3;
                let t5 = t1 + t2;
                let t6 = (t1 - t2).scale(
                    cast::<_, T>(0.25).unwrap() * (cast::<_, T>(5.0).unwrap()).sqrt(),
                );
                let t7 = z0 - t5.scale(cast(0.25).unwrap());
                let t8 = t6 + t7;
                let t9 = t7 - t6;
                let t10 = (t3.scale((cast::<_, T>(-0.4).unwrap() * T::PI()).sin()) +
                    t4.scale((cast::<_, T>(-0.2).unwrap() * T::PI()).sin())) *
                    im_one;
                let t11 = (t3.scale((cast::<_, T>(-0.2).unwrap() * T::PI()).sin()) -
                    t4.scale((cast::<_, T>(-0.4).unwrap() * T::PI()).sin())) *
                    im_one;

                ret[j] = z0 + t5;
                ret[pos2] = t8 + t10;
                ret[pos3] = t9 + t11;
                ret[pos4] = t9 - t11;
                ret[pos5] = t8 - t10;
                j += *po2;
            }
        }
    }
}

#[inline]
fn mixed_kernel_other<T: Float>(
    ret: &mut [Complex<T>],
    value: usize,
    count: usize,
    po2: &mut usize,
    rad: &mut usize,
    omega: &Vec<Complex<T>>,
    len: usize,
) {
    let rot_width = len / value;
    let rot = (0..value).map(|i| omega[rot_width * i]).collect::<Vec<_>>();

    for _ in 0..count {
        let po2m = *po2;
        *po2 *= value;
        *rad /= value;
        for mut j in 0..po2m {
            let wpos = *rad * j;
            while j < len {
                // 定義式DFT
                let pos = (0..value).map(|i| j + po2m * i).collect::<Vec<_>>();
                let z = (0..value)
                    .map(|i| ret[pos[i]] * omega[wpos * i])
                    .collect::<Vec<_>>();
                for i in 0..value {
                    ret[pos[i]] = (1..value).fold(z[0], |x, l| x + z[l] * rot[(i * l) % value]);
                }
                j += *po2;
            }
        }
    }
}
