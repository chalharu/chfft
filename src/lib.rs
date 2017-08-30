#![crate_type="lib"]

//! Chalharu's Fastest Fourier Transform.
//!
//! At the moment, this only provides one-dimensional
//! complex-to-complex transforms.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

extern crate num_traits;
extern crate num;

use std::cmp;
use num::{Complex, Float, cast, one, zero};
use num_traits::NumAssign;
use num_traits::float::FloatConst;

enum WorkData<T> {
    MixedRadix {
        ids: Vec<usize>,
        omega: Vec<Complex<T>>,
        omega_back: Vec<Complex<T>>,
        factors: Vec<Factor>,
    },
    ChirpZ {
        level: usize,
        ids: Vec<usize>,
        omega: Vec<Complex<T>>,
        omega_back: Vec<Complex<T>>,
        src_omega: Vec<Complex<T>>,
        rot_conj: Vec<Complex<T>>,
        rot_ft: Vec<Complex<T>>,
        pow2len_inv: T,
    },
    None,
}

#[derive(Copy, Clone)]
struct Factor {
    value: usize,
    count: usize,
}

struct FactorIterator {
    value: usize,
    prime: usize,
}

impl Iterator for FactorIterator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            match self.prime {
                4 => {
                    if self.value & 3 == 0 {
                        self.value >>= 2;
                        return Some(4);
                    } else {
                        self.prime = 2;
                    }
                }
                2 => {
                    if self.value & 1 == 0 {
                        self.value >>= 1;
                        return Some(2);
                    } else {
                        self.prime = 3;
                    }
                }
                _ => {
                    if self.value >= self.prime * self.prime {
                        if self.value % self.prime == 0 {
                            self.value /= self.prime;
                            return Some(self.prime);
                        } else {
                            self.prime += 2;
                        }
                    } else if self.value > 1 {
                        let v = self.value;
                        self.value = 0;
                        return Some(v);
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

fn prime_factorization_iter(value: usize) -> FactorIterator {
    FactorIterator {
        value: value,
        prime: 4,
    }
}

/// Perform a complex-to-complex one-dimensional Fourier transform
///
/// # Example
///
/// ```rust
/// extern crate chfft;
/// extern crate num;
/// use num::Complex;
/// use chfft::CFft1D;
///
/// fn main() {
///     let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
///                  Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
///
///     let mut fft = CFft1D::<f64>::with_len(input.len());
///
///     let output = fft.forward(&input);
///
///     println!("the transform of {:?} is {:?}", input, output);
/// }
/// ```
pub struct CFft1D<T> {
    len: usize,
    scaler_n: T,
    scaler_u: T,
    work: WorkData<T>,
}

impl<T: Float + FloatConst + NumAssign> CFft1D<T> {
    /// Returns a instances to execute FFT
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// let mut fft = CFft1D::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            len: 0,
            scaler_n: zero(),
            scaler_u: zero(),
            work: WorkData::None,
        }
    }

    /// Returns a instances to execute length initialized FFT
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// let mut fft = CFft1D::<f64>::with_len(1024);
    /// ```
    pub fn with_len(len: usize) -> Self {
        let mut ret = Self {
            len: 0,
            scaler_n: zero(),
            scaler_u: zero(),
            work: WorkData::None,
        };
        ret.setup(len);
        ret
    }

    /// Reinitialize length
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// let mut fft = CFft1D::<f64>::with_len(1024);
    ///
    /// // reinitialize
    /// fft.setup(2048);
    /// ```
    pub fn setup(&mut self, len: usize) {
        const MAX_PRIME: usize = 7;
        self.len = len;

        self.scaler_n = T::one() / cast(len).unwrap();
        self.scaler_u = self.scaler_n.sqrt();

        // 素因数分解
        let factors = Self::prime_factorization(len, MAX_PRIME);
        if factors.len() == 0 {
            // Chrip-Z
            let pow2len = len.next_power_of_two() << 1;
            let lv = pow2len.trailing_zeros() as usize;

            let dlen = len << 1;
            let src_omega = Self::calc_omega(dlen);

            let mut rot = Vec::with_capacity(pow2len);
            let mut rot_conj = Vec::with_capacity(pow2len);

            for i in 0..len {
                let w = src_omega[(i * i) % dlen];
                let w_back = src_omega[dlen - ((i * i) % dlen)];
                rot_conj.push(w);
                rot.push(w_back);
            }

            let hlen = (pow2len >> 1) + 1;
            for _ in len..hlen {
                rot_conj.push(zero());
                rot.push(zero());
            }
            for i in hlen..pow2len {
                let t = rot_conj[pow2len - i];
                rot_conj.push(t);
                let t = rot[pow2len - i];
                rot.push(t);
            }

            match &mut self.work {
                &mut WorkData::ChirpZ {
                    level,
                    ref ids,
                    ref omega,
                    omega_back: _,
                    src_omega: ref mut org_src_omega,
                    rot_conj: ref mut org_rot_conj,
                    rot_ft: ref mut org_rot_ft,
                    ref pow2len_inv,
                } => {
                    if level == lv {
                        *org_src_omega = src_omega;
                        *org_rot_conj = rot_conj;
                        *org_rot_ft = Self::convert_rad2(&rot, lv, ids, omega, false, *pow2len_inv);
                        return;
                    }
                }
                _ => {}
            }
            // ビットリバースの計算
            let ids = Self::calc_bitreverse(
                len,
                &[
                    Factor {
                        value: 2,
                        count: lv & 1,
                    },
                    Factor {
                        value: 4,
                        count: lv >> 1,
                    },
                ],
            );
            let omega = Self::calc_omega(pow2len);
            let omega_back = omega.iter().rev().map(|x| *x).collect::<Vec<_>>();
            let pow2len_inv = T::one() / cast(pow2len).unwrap();
            let rot_ft = Self::convert_rad2(&rot, lv, &ids, &omega, false, pow2len_inv);

            self.work = WorkData::ChirpZ {
                level: lv,
                ids: ids,
                omega: omega,
                omega_back: omega_back,
                src_omega: src_omega,
                rot_conj: rot_conj,
                rot_ft: rot_ft,
                pow2len_inv: pow2len_inv,
            };
        } else {
            // Mixed-Radix

            // ωの事前計算
            let omega = Self::calc_omega(len);
            let omega_back = omega.iter().rev().map(|x| *x).collect::<Vec<_>>();

            self.work = WorkData::MixedRadix {
                ids: Self::calc_bitreverse(len, &factors),
                omega: omega,
                omega_back: omega_back,
                factors: factors,
            }
        }
    }

    #[inline]
    fn calc_bitreverse(len: usize, factors: &[Factor]) -> Vec<usize> {
        // ビットリバースの計算
        let mut ids = Vec::<usize>::with_capacity(len);
        let mut llen = 1_usize;
        ids.push(0);
        for &f in factors {
            for _ in 0..f.count {
                for i in 0..llen {
                    ids[i] *= f.value;
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
    fn calc_omega_item(len: usize, position: usize) -> Complex<T> {
        Complex::from_polar(
            &one(),
            &(cast::<_, T>(-2.0).unwrap() * T::PI() / cast(len).unwrap() * cast(position).unwrap()),
        )
    }

    // ωの事前計算
    fn calc_omega(len: usize) -> Vec<Complex<T>> {
        let mut omega = Vec::with_capacity(len + 1);
        omega.push(one());
        if len & 3 == 0 {
            // 4で割り切れる(下位2ビットが0)ならば
            let q = len >> 2;
            let h = len >> 1;
            // 1/4周分を計算
            for i in 1..q {
                omega.push(Self::calc_omega_item(len, i));
            }

            // 1/4～1/2周分を計算
            for i in q..h {
                let tmp = omega[i - q];
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
                omega.push(Self::calc_omega_item(len, i));
            }

            // 1/2周目から計算
            for i in h..len {
                let tmp = omega[i - h];
                omega.push(Complex::new(-tmp.re, -tmp.im));
            }
        } else {
            for i in 1..len {
                omega.push(Self::calc_omega_item(len, i));
            }
        }
        // 1周ちょうど
        omega.push(one());
        omega
    }

    // 素因数分解
    fn prime_factorization(value: usize, max: usize) -> Vec<Factor> {
        let mut factors = Vec::<Factor>::new();
        let mut count = 0;
        let mut prime = 0;
        let mut fac4count = 0;
        for p in prime_factorization_iter(value) {
            if p > max {
                return Vec::<Factor>::new();
            }
            if prime != p {
                if count > 0 {
                    if prime == 4 {
                        fac4count = count;
                    } else {
                        if prime > 4 && fac4count != 0 {
                            factors.push(Factor {
                                value: 4,
                                count: fac4count,
                            });
                            fac4count = 0;
                        }
                        factors.push(Factor {
                            value: prime,
                            count: count,
                        });
                    }
                }
                count = 1;
                prime = p;
            } else {
                count += 1;
            }
        }
        if count > 0 {
            factors.push(Factor {
                value: prime,
                count: count,
            });
        }
        if fac4count != 0 {
            factors.push(Factor {
                value: 4,
                count: fac4count,
            });
        }
        factors
    }

    fn convert_rad2(
        source: &[Complex<T>],
        level: usize,
        ids: &Vec<usize>,
        omega: &Vec<Complex<T>>,
        is_back: bool,
        pow2len_inv: T,
    ) -> Vec<Complex<T>> {
        // 入力の並び替え
        let mut ret = ids.iter().map(|&i| if is_back {
            source[i].scale(pow2len_inv) // 逆変換の場合はこのタイミングで割り戻しておく
        } else {
            source[i]
        }).collect::<Vec<_>>();

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
                let wfb = ret[pos_b];
                ret[pos_b] = ret[pos_a] - wfb;
                ret[pos_a] += wfb;
            }
        }

        let im_one = if is_back { -Complex::i() } else { Complex::i() };

        Self::mixed_kernel_radix4(
            &mut ret,
            level >> 1,
            &mut po2,
            &mut rad,
            omega,
            len,
            &im_one,
        );

        ret
    }

    fn convert_mixed(
        source: &[Complex<T>],
        len: usize,
        ids: &Vec<usize>,
        omega: &Vec<Complex<T>>,
        omega_back: &Vec<Complex<T>>,
        factors: &Vec<Factor>,
        is_back: bool,
        scaler: T,
    ) -> Vec<Complex<T>> {
        let omega = if is_back { omega_back } else { omega };

        // 入力の並び替え
        let mut ret = ids.iter()
            .map(|&i| if scaler != one() {
                source[i].scale(scaler) // このタイミングで割り戻しておく
            } else {
                source[i]
            })
            .collect::<Vec<_>>();

        // FFT
        let mut po2 = 1;
        let mut rad = len;

        let im_one = if is_back { -Complex::i() } else { Complex::i() };

        for factor in factors {
            match factor.value {
                2 => {
                    Self::mixed_kernel_radix2(
                        &mut ret,
                        factor.count,
                        &mut po2,
                        &mut rad,
                        omega,
                        len,
                    );
                }
                3 => {
                    Self::mixed_kernel_radix3(
                        &mut ret,
                        factor.count,
                        &mut po2,
                        &mut rad,
                        omega,
                        len,
                        &im_one,
                    );
                }
                4 => {
                    Self::mixed_kernel_radix4(
                        &mut ret,
                        factor.count,
                        &mut po2,
                        &mut rad,
                        omega,
                        len,
                        &im_one,
                    );
                }
                5 => {
                    Self::mixed_kernel_radix5(
                        &mut ret,
                        factor.count,
                        &mut po2,
                        &mut rad,
                        omega,
                        len,
                        &im_one,
                    );
                }
                _ => {
                    Self::mixed_kernel_other(
                        &mut ret,
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
        ret
    }

    #[inline(always)]
    fn mixed_kernel_radix2(
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
    fn mixed_kernel_radix3(
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
    fn mixed_kernel_radix4(
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
    fn mixed_kernel_radix5(
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
                        cast::<_, T>(0.25).unwrap() *
                            (cast::<_, T>(5.0).unwrap()).sqrt(),
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
    fn mixed_kernel_other(
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

    fn convert_chirpz(
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

        let aa = Self::convert_rad2(&a, level, ids, omega, false, pow2len_inv);

        let gg = aa.iter()
            .zip(rot_ft)
            .map(|(&x, &y)| x * y)
            .collect::<Vec<_>>();
        let g = Self::convert_rad2(&gg, level, ids, omega_back, true, pow2len_inv);

        // Multiply phase factor
        (0..srclen)
            .map(move |i| if i == 0 {
                0
            } else if is_back {
                srclen - i
            } else {
                i
            })
            .map(move |i| if scaler != one() {
                g[i] * rot_conj[i].scale(scaler)
            } else {
                g[i] * rot_conj[i]
            })
            .collect::<Vec<_>>()
    }

    #[inline]
    fn convert(&mut self, source: &[Complex<T>], is_back: bool, scaler: T) -> Vec<Complex<T>> {
        let len = source.len();

        // １要素以下ならば入力値をそのまま返す
        if len <= 1 {
            source.to_vec()
        } else {
            if len != self.len {
                self.setup(len);
            }

            match &self.work {
                &WorkData::MixedRadix {
                    ref ids,
                    ref omega,
                    ref omega_back,
                    ref factors,
                } => {
                    Self::convert_mixed(
                        source,
                        len,
                        ids,
                        omega,
                        omega_back,
                        factors,
                        is_back,
                        scaler,
                    )
                }
                &WorkData::ChirpZ {
                    level,
                    ref ids,
                    ref omega,
                    ref omega_back,
                    ref src_omega,
                    ref rot_conj,
                    ref rot_ft,
                    ref pow2len_inv,
                } => {
                    Self::convert_chirpz(
                        source,
                        len,
                        level,
                        ids,
                        omega,
                        omega_back,
                        src_omega,
                        rot_conj,
                        rot_ft,
                        is_back,
                        *pow2len_inv,
                        scaler,
                    )
                }
                &WorkData::None => source.to_vec(),
            }
        }
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forward(&input);
    /// ```
    pub fn forward(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, false, one())
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forward0(&input);
    /// ```
    pub fn forward0(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, false, one())
    }

    /// The $\frac 1 {\sqrt n}$ scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forwardu(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_u;
        self.convert(source, false, scaler)
    }

    /// The $\frac 1 n$ scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forwardn(&input);
    /// ```
    pub fn forwardn(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, false, scaler)
    }

    /// The $\frac 1 n$ scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backward(&input);
    /// ```
    pub fn backward(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, true, scaler)
    }

    /// The 1 scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backward0(&input);
    /// ```
    pub fn backward0(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, true, one())
    }

    /// The $\frac 1 {\sqrt n}$ scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backwardu(&input);
    /// ```
    pub fn backwardu(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_u;
        self.convert(source, true, scaler)
    }

    /// The $\frac 1 n$ scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num;
    ///
    /// let input = [num::Complex::new(2.0, 0.0), num::Complex::new(1.0, 1.0),
    ///              num::Complex::new(0.0, 3.0), num::Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = chfft::CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backwardn(&input);
    /// ```
    pub fn backwardn(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, true, scaler)
    }
}
