//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use crate::prime_factorization::Factor;
use num_complex::Complex;
use num_traits::cast;
use num_traits::float::{Float, FloatConst};
use num_traits::identities::one;
use std::cmp;

#[inline]
pub fn calc_omega_item<T: Float + FloatConst>(len: usize, position: usize) -> Complex<T> {
    Complex::from_polar(
        &one(),
        &(cast::<_, T>(-2.0).unwrap() * T::PI() / cast(len).unwrap() * cast(position).unwrap()),
    )
}

// ωの事前計算
pub fn calc_omega<T: Float + FloatConst>(len: usize) -> Vec<Complex<T>> {
    let mut omega = Vec::with_capacity(len + 1);
    omega.push(one());
    if len.trailing_zeros() >= 2 {
        // 4で割り切れる(下位2ビットが0)ならば
        let q = len >> 2;
        let h = len >> 1;
        // 1/4周分を計算
        for i in 1..q {
            omega.push(calc_omega_item(len, i));
        }

        // 1/4～1/2周分を計算
        for i in q..h {
            let tmp: Complex<T> = omega[i - q];
            omega.push(Complex::new(tmp.im, -tmp.re));
        }

        // 1/2周目から計算
        for i in h..len {
            let tmp = omega[i - h];
            omega.push(Complex::new(-tmp.re, -tmp.im));
        }
    } else if len & 1 == 0 {
        // 2で割り切れる(下位1ビットが0)ならば
        let h = cmp::max(len >> 1, 1);
        // 1/2周分を計算
        for i in 1..h {
            omega.push(calc_omega_item(len, i));
        }

        // 1/2周目から計算
        for i in h..len {
            let tmp = omega[i - h];
            omega.push(Complex::new(-tmp.re, -tmp.im));
        }
    } else {
        for i in 1..len {
            omega.push(calc_omega_item(len, i));
        }
    }
    // 1周ちょうど
    omega.push(one());
    omega
}

#[inline]
pub fn calc_bitreverse(len: usize, factors: &[Factor]) -> Vec<usize> {
    // ビットリバースの計算
    let mut ids = Vec::<usize>::with_capacity(len);
    let mut llen = 1_usize;
    ids.push(0);
    for f in factors {
        for _ in 0..f.count {
            for id in ids.iter_mut().take(llen) {
                *id *= f.value;
            }
            for i in 1..f.value {
                for j in 0..llen {
                    let id = ids[j] + i;
                    ids.push(id);
                }
            }
            llen *= f.value;
        }
    }
    ids
}

#[inline]
pub fn calc_bitreverse2inplace(source: Vec<usize>) -> Vec<usize> {
    let mut nums = (0..source.len()).collect::<Vec<_>>();

    (0..source.len())
        .map(|i| {
            let r = (i..nums.len()).find(|&j| nums[j] == source[i]).unwrap();
            nums.swap(i, r);
            r
        })
        .collect()
}
