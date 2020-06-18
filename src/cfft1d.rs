//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use crate::chirpz;
use crate::mixed_radix;
use crate::precompute_utils;
use crate::prime_factorization;
use crate::prime_factorization::Factor;
use num_complex::Complex;
use num_traits::float::{Float, FloatConst};
use num_traits::identities::{one, zero};
use num_traits::{cast, NumAssign};

#[derive(Debug)]
enum WorkData<T> {
    MixedRadix(mixed_radix::MixedRadixData<T>),
    ChirpZ(chirpz::ChirpzData<T>),
    None,
}

/// Perform a complex-to-complex one-dimensional Fourier transform
///
/// <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML"></script>
/// When X is input array and Y is output array,
/// the forward discrete Fourier transform of the one-dimensional array is
///
/// \\[ \Large Y_k = \sum_{j=0}\^{n-1} X_j e\^{- \frac {2 \pi i j k}{n}} \\]
///
/// also, the backward discrete Fourier transform of the one-dimensional array is
///
/// \\[ \Large Y_k = \sum_{j=0}\^{n-1} X_j e\^{\frac {2 \pi i j k}{n}} \\]
///
/// # Example
///
/// ```rust
/// use num_complex::Complex;
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
#[derive(Debug)]
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
        let factors = prime_factorization::prime_factorization(len, MAX_PRIME);
        if factors.is_empty() {
            // Chrip-Z
            let pow2len = len.next_power_of_two() << 1;
            let lv = pow2len.trailing_zeros() as usize;

            let dlen = len << 1;
            let src_omega = precompute_utils::calc_omega(dlen);

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

            if let WorkData::ChirpZ(chirpz::ChirpzData {
                level,
                ref ids,
                ref omega,
                src_omega: ref mut org_src_omega,
                rot_conj: ref mut org_rot_conj,
                rot_ft: ref mut org_rot_ft,
                ref pow2len_inv,
                ..
            }) = self.work
            {
                if level == lv {
                    *org_src_omega = src_omega;
                    *org_rot_conj = rot_conj;
                    chirpz::convert_rad2_inplace(&mut rot, lv, ids, omega, false, *pow2len_inv);
                    *org_rot_ft = rot;
                    return;
                }
            }

            // ビットリバースの計算
            let ids = precompute_utils::calc_bitreverse2inplace(precompute_utils::calc_bitreverse(
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
            ));
            let omega = precompute_utils::calc_omega(pow2len);
            let omega_back = omega.iter().rev().copied().collect::<Vec<_>>();
            let pow2len_inv = T::one() / cast(pow2len).unwrap();
            chirpz::convert_rad2_inplace(&mut rot, lv, &ids, &omega, false, pow2len_inv);

            self.work = WorkData::ChirpZ(chirpz::ChirpzData {
                level: lv,
                ids,
                omega,
                omega_back,
                src_omega,
                rot_conj,
                rot_ft: rot,
                pow2len_inv,
            });
        } else {
            // Mixed-Radix

            // ωの事前計算
            let omega = precompute_utils::calc_omega(len);
            let omega_back = omega.iter().rev().copied().collect::<Vec<_>>();

            self.work = WorkData::MixedRadix(mixed_radix::MixedRadixData {
                ids: precompute_utils::calc_bitreverse(len, &factors),
                omega,
                omega_back,
                factors,
                ids_inplace: None,
            });
        }
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
                WorkData::MixedRadix(ref data) => {
                    mixed_radix::convert_mixed(source, len, is_back, scaler, data)
                }
                WorkData::ChirpZ(ref data) => {
                    chirpz::convert_chirpz(source, len, is_back, scaler, data)
                }
                WorkData::None => source.to_vec(),
            }
        }
    }

    #[inline]
    fn convert_inplace(&mut self, source: &mut [Complex<T>], is_back: bool, scaler: T) {
        let len = source.len();

        // １要素以下ならば入力値をそのまま返す
        if len > 1 {
            if len != self.len {
                self.setup(len);
            }

            match self.work {
                WorkData::MixedRadix(ref mut data) => {
                    if data.ids_inplace.is_none() {
                        data.ids_inplace =
                            Some(precompute_utils::calc_bitreverse2inplace(data.ids.to_vec()))
                    }
                    mixed_radix::convert_mixed_inplace(source, len, is_back, scaler, data);
                }
                WorkData::ChirpZ(ref data) => {
                    chirpz::convert_chirpz_inplace(source, len, is_back, scaler, data)
                }
                _ => {}
            };
        }
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forward(&input);
    /// ```
    pub fn forward(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, false, one())
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forward0(&input);
    /// ```
    pub fn forward0(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, false, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forwardu(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_u;
        self.convert(source, false, scaler)
    }

    /// The \\(\frac 1 {n}\\) scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.forwardn(&input);
    /// ```
    pub fn forwardn(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, false, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backward(&input);
    /// ```
    pub fn backward(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, true, scaler)
    }

    /// The 1 scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backward0(&input);
    /// ```
    pub fn backward0(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        self.convert(source, true, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backwardu(&input);
    /// ```
    pub fn backwardu(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_u;
        self.convert(source, true, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// let output = fft.backwardn(&input);
    /// ```
    pub fn backwardn(&mut self, source: &[Complex<T>]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, true, scaler)
    }

    /// The 1 scaling factor and in-place forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let mut input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// fft.forward0i(&mut input);
    /// ```
    pub fn forward0i(&mut self, source: &mut [Complex<T>]) {
        self.convert_inplace(source, false, one());
    }

    /// The 1 scaling factor and in-place backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let mut input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// fft.backward0i(&mut input);
    /// ```
    pub fn backward0i(&mut self, source: &mut [Complex<T>]) {
        self.convert_inplace(source, true, one());
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor and in-place forward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let mut input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// fft.forwardui(&mut input);
    /// ```
    pub fn forwardui(&mut self, source: &mut [Complex<T>]) {
        let scaler = self.scaler_u;
        self.convert_inplace(source, false, scaler);
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor and in-place backward transform
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// use num_complex::Complex;
    ///
    /// let mut input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
    ///              Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    ///
    /// let mut fft = CFft1D::<f64>::with_len(input.len());
    /// fft.backwardui(&mut input);
    /// ```
    pub fn backwardui(&mut self, source: &mut [Complex<T>]) {
        let scaler = self.scaler_u;
        self.convert_inplace(source, true, scaler);
    }
}

impl<T: Float + FloatConst + NumAssign> Default for CFft1D<T> {
    /// Returns a instances to execute FFT
    ///
    /// ```rust
    /// use chfft::CFft1D;
    /// let mut fft = CFft1D::<f64>::default();
    /// ```
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_appro_eq;
    use crate::FloatEps;
    use appro_eq::AbsError;
    use rand::distributions::{Distribution, Standard};
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;
    use std::fmt::Debug;

    fn convert<T: Float + FloatConst>(source: &[Complex<T>], scalar: T) -> Vec<Complex<T>> {
        (0..source.len())
            .map(|i| {
                (1..source.len()).fold(source[0], |x, j| {
                    x + source[j]
                        * Complex::<T>::from_polar(
                            one(),
                            -cast::<_, T>(2 * i * j).unwrap() * T::PI()
                                / cast(source.len()).unwrap(),
                        )
                }) * scalar
            })
            .collect::<Vec<_>>()
    }

    fn test_with_source<T: Float + FloatConst + NumAssign + AbsError + Debug + FloatEps>(
        fft: &mut CFft1D<T>,
        source: &[Complex<T>],
    ) {
        let expected = convert(source, one());
        let actual = fft.forward(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = fft.backward(&actual);
        assert_appro_eq(source, &actual_source);

        let actual = fft.forward0(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = fft.backwardn(&actual);
        assert_appro_eq(source, &actual_source);

        let expected = convert(
            source,
            T::one() / cast::<_, T>(source.len()).unwrap().sqrt(),
        );
        let actual = fft.forwardu(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = fft.backwardu(&actual);
        assert_appro_eq(source, &actual_source);

        let expected = convert(source, T::one() / cast(source.len()).unwrap());
        let actual = fft.forwardn(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = fft.backward0(&actual);
        assert_appro_eq(source, &actual_source);

        let expected = fft.forward0(source);
        let mut actual = source.to_vec();
        fft.forward0i(&mut actual);
        assert_appro_eq(&expected, &actual);

        let mut actual = fft.forwardn(source);
        fft.backward0i(&mut actual);
        assert_appro_eq(source, &actual);

        let expected = fft.forwardu(source);
        let mut actual = source.to_vec();
        fft.forwardui(&mut actual);
        assert_appro_eq(&expected, &actual);

        let mut actual = fft.forwardu(source);
        fft.backwardui(&mut actual);
        assert_appro_eq(source, &actual);
    }

    fn test_with_len<T: Float + FloatConst + NumAssign + AbsError + Debug + FloatEps>(
        fft: &mut CFft1D<T>,
        len: usize,
    ) where
        Standard: Distribution<T>,
    {
        let mut rng = XorShiftRng::from_seed([
            0xDA, 0xE1, 0x4B, 0x0B, 0xFF, 0xC2, 0xFE, 0x64, 0x23, 0xFE, 0x3F, 0x51, 0x6D, 0x3E,
            0xA2, 0xF3,
        ]);

        // 10パターンのテスト
        for _ in 0..10 {
            let arr = (0..len)
                .map(|_| Complex::new(rng.gen::<T>(), rng.gen::<T>()))
                .collect::<Vec<Complex<T>>>();

            test_with_source(fft, &arr);
        }
    }

    #[test]
    fn f64_new() {
        for i in 1..100 {
            test_with_len(&mut CFft1D::<f64>::new(), i);
        }
    }

    #[test]
    fn f32_new() {
        for i in 1..100 {
            test_with_len(&mut CFft1D::<f32>::new(), i);
        }
    }

    #[test]
    fn f64_with_len() {
        for i in 1..100 {
            test_with_len(&mut CFft1D::<f64>::with_len(i), i);
        }
    }

    #[test]
    fn f32_with_len() {
        for i in 1..100 {
            test_with_len(&mut CFft1D::<f32>::with_len(i), i);
        }
    }

    #[test]
    fn f64_primes() {
        let mut dft = CFft1D::<f64>::new();
        for &i in &[
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89, 97,
        ] {
            test_with_len(&mut dft, i);
        }
    }
}
