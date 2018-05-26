//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use CFft1D;
use num_complex::Complex;
use num_traits::float::{Float, FloatConst};
use num_traits::identities::{one, zero};
use num_traits::{cast, NumAssign};
use precompute_utils;

/// Perform a real-to-complex one-dimensional Fourier transform
///
/// <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML"></script>
///
/// # Example
///
/// ```rust
/// extern crate chfft;
///
/// use chfft::RFft1D;
///
/// fn main() {
///     let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
///
///     let mut fft = RFft1D::<f64>::new(input.len());
///
///     let output = fft.forward(&input);
///
///     println!("the transform of {:?} is {:?}", input, output);
/// }
/// ```
pub struct RFft1D<T> {
    fft: CFft1D<T>,
    len: usize,
    coef: Vec<Complex<T>>,
    bcoef: Vec<Complex<T>>,
    work: Vec<Complex<T>>,
    scaler_n: T,
    scaler_u: T,
}

impl<T: Float + FloatConst + NumAssign> RFft1D<T> {
    fn calc_rfft_coef(len: usize) -> (Vec<Complex<T>>, Vec<Complex<T>>) {
        let omega = precompute_utils::calc_omega(len);
        (
            omega
                .iter()
                .take(len >> 1)
                .map(|w| {
                    (Complex::<T>::new(one(), zero()) + Complex::<T>::i() * w)
                        .scale(cast(0.5).unwrap())
                })
                .collect(),
            omega
                .iter()
                .rev()
                .take(len >> 1)
                .map(|w| {
                    (Complex::<T>::new(T::one(), zero()) - Complex::<T>::i() * w)
                        .scale(cast(0.5).unwrap())
                })
                .collect(),
        )
    }

    /// Returns a instances to execute FFT
    ///
    /// ```rust
    /// use chfft::RFft1D;
    /// let mut rfft = RFft1D::<f64>::new(1024);
    /// ```
    pub fn new(len: usize) -> Self {
        if len & 1 != 0 {
            panic!("invalid length")
        }

        let (coef, bcoef) = Self::calc_rfft_coef(len);
        let scaler_n = T::one() / cast(len).unwrap();

        Self {
            fft: CFft1D::with_len(len >> 1),
            len: len,
            scaler_n: scaler_n,
            scaler_u: scaler_n.sqrt(),
            coef: coef,
            bcoef: bcoef,
            work: vec![zero(); len >> 1],
        }
    }

    /// Reinitialize length
    ///
    /// ```rust
    /// use chfft::RFft1D;
    /// let mut rfft = RFft1D::<f64>::new(1024);
    ///
    /// // reinitialize
    /// rfft.setup(2048);
    /// ```
    pub fn setup(&mut self, len: usize) {
        if len & 1 != 0 {
            panic!("invalid length")
        }
        self.len = len;
        self.fft = CFft1D::with_len(len >> 1);
        self.scaler_n = T::one() / cast(len).unwrap();
        self.scaler_u = self.scaler_n.sqrt();
        let coef = Self::calc_rfft_coef(len);
        self.coef = coef.0;
        self.bcoef = coef.1;
        self.work = vec![zero(); len >> 1];
    }

    fn convert(&mut self, source: &[T], scaler: T) -> Vec<Complex<T>> {
        if source.len() != self.len {
            panic!(
                "invalid length (soure: {}, rdft.len: {})",
                source.len(),
                self.len
            )
        }

        if source.len() == 1 {
            return vec![Complex::new(source[0] * scaler, zero()); 1];
        }

        let t = self.len >> 1;
        for i in 0..t {
            self.work[i] = Complex::new(source[i << 1], source[(i << 1) + 1]).scale(scaler);
        }

        self.fft.forward0i(&mut self.work);
        let hlen = (self.len + 1) >> 1;
        let qlen = (self.len + 3) >> 2;

        let mut ret = vec![zero(); t + 1];
        for i in 1..qlen {
            let x = self.coef[i] * (self.work[i] - self.work[hlen - i].conj());
            ret[i] = self.work[i] - x;
            ret[hlen - i] = self.work[hlen - i] + x.conj();
        }

        ret[0] = Complex::new(self.work[0].re + self.work[0].im, zero());
        if self.len & 3 == 0 {
            ret[qlen] = self.work[qlen].conj();
        }
        ret[hlen] = Complex::new(self.work[0].re - self.work[0].im, zero());

        ret
    }

    fn convert_back(&mut self, source: &[Complex<T>], scaler: T) -> Vec<T> {
        if source.len() != ((self.len >> 1) + 1) {
            panic!(
                "invalid length (soure: {}, rdft.len: {})",
                source.len(),
                self.len
            )
        }
        let scaler = scaler * cast(2.0).unwrap();
        let hlen = (self.len + 1) >> 1;
        let qlen = (self.len + 3) >> 2;
        self.work[0] = Complex::new(
            source[0].re + source[hlen].re,
            source[0].re - source[hlen].re,
        ).scale(cast(0.5).unwrap());
        if self.len & 3 == 0 {
            self.work[qlen] = source[qlen].conj();
        }

        for i in 1..qlen {
            let x = self.bcoef[i] * (source[i] - source[hlen - i].conj());
            self.work[i] = source[i] - x;
            self.work[hlen - i] = source[hlen - i] + x.conj();
        }

        self.fft.backward0i(&mut self.work);

        let mut ret = Vec::with_capacity(self.len);
        for i in 0..hlen {
            ret.push(self.work[i].re * scaler);
            ret.push(self.work[i].im * scaler);
        }
        ret
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(input.len());
    /// let output = fft.forward0(&input);
    /// ```
    pub fn forward(&mut self, source: &[T]) -> Vec<Complex<T>> {
        self.convert(source, one())
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(input.len());
    /// let output = fft.forward0(&input);
    /// ```
    pub fn forward0(&mut self, source: &[T]) -> Vec<Complex<T>> {
        self.convert(source, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(input.len());
    /// let output = fft.forwardu(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[T]) -> Vec<Complex<T>> {
        let scaler = self.scaler_u;
        self.convert(source, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(input.len());
    /// let output = fft.forwardn(&input);
    /// ```
    pub fn forwardn(&mut self, source: &[T]) -> Vec<Complex<T>> {
        let scaler = self.scaler_n;
        self.convert(source, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [num_complex::Complex::new(2.0, 0.0), num_complex::Complex::new(1.0, 1.0),
    ///              num_complex::Complex::new(4.0, 3.0), num_complex::Complex::new(2.0, 0.0)];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(6);
    /// let output = fft.backward(&input);
    /// ```
    pub fn backward(&mut self, source: &[Complex<T>]) -> Vec<T> {
        let scaler = self.scaler_n;
        self.convert_back(source, scaler)
    }

    /// The 1 scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [num_complex::Complex::new(2.0, 0.0), num_complex::Complex::new(1.0, 1.0),
    ///              num_complex::Complex::new(4.0, 3.0), num_complex::Complex::new(2.0, 0.0)];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(6);
    /// let output = fft.backward0(&input);
    /// ```
    pub fn backward0(&mut self, source: &[Complex<T>]) -> Vec<T> {
        self.convert_back(source, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [num_complex::Complex::new(2.0, 0.0), num_complex::Complex::new(1.0, 1.0),
    ///              num_complex::Complex::new(4.0, 3.0), num_complex::Complex::new(2.0, 0.0)];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(6);
    /// let output = fft.backwardu(&input);
    /// ```
    pub fn backwardu(&mut self, source: &[Complex<T>]) -> Vec<T> {
        let scaler = self.scaler_u;
        self.convert_back(source, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [num_complex::Complex::new(2.0, 0.0), num_complex::Complex::new(1.0, 1.0),
    ///              num_complex::Complex::new(4.0, 3.0), num_complex::Complex::new(2.0, 0.0)];
    ///
    /// let mut fft = chfft::RFft1D::<f64>::new(6);
    /// let output = fft.backwardn(&input);
    /// ```
    pub fn backwardn(&mut self, source: &[Complex<T>]) -> Vec<T> {
        let scaler = self.scaler_n;
        self.convert_back(source, scaler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use FloatEps;
    use appro_eq::AbsError;
    use assert_appro_eq;
    use rand::distributions::{Distribution, Standard};
    use rand::{Rng, SeedableRng, XorShiftRng};
    use std::fmt::Debug;

    fn convert<T: Float + FloatConst>(source: &[T], scalar: T) -> Vec<Complex<T>> {
        (0..((source.len() >> 1) + 1))
            .map(|i| {
                (1..source.len()).fold(Complex::new(source[0], zero()), |x, j| {
                    x + Complex::new(source[j], zero())
                        * Complex::<T>::from_polar(
                            &one(),
                            &(-cast::<_, T>(2 * i * j).unwrap() * T::PI()
                                / cast(source.len()).unwrap()),
                        )
                }) * scalar
            })
            .collect::<Vec<_>>()
    }

    fn test_with_source<T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps>(
        fft: &mut RFft1D<T>,
        source: &[T],
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
    }

    fn test_with_len<T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps>(
        dct: &mut RFft1D<T>,
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
            let arr = (0..len).map(|_| rng.gen::<T>()).collect::<Vec<T>>();
            test_with_source(dct, &arr);
        }
    }

    #[test]
    fn f64_new() {
        for i in 1..100 {
            test_with_len(&mut RFft1D::<f64>::new(i << 1), i << 1);
        }
    }

    #[test]
    fn f32_new() {
        for i in 1..100 {
            test_with_len(&mut RFft1D::<f32>::new(i << 1), i << 1);
        }
    }
}
