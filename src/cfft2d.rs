//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

use num_complex::Complex;
use num_traits::{cast, NumAssign};
use num_traits::float::{Float, FloatConst};
use num_traits::identities::{one, zero};
use CFft1D;

/// Perform a complex-to-complex two-dimensional Fourier transform
///
/// <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML"></script>
///
/// # Example
///
/// ```rust
/// extern crate chfft;
/// extern crate num_complex;
/// use num_complex::Complex;
/// use chfft::CFft2D;
///
/// fn main() {
///     let input = [
///         vec![
///             Complex::new(2.0, 0.0),
///             Complex::new(1.0, 1.0),
///             Complex::new(0.0, 3.0),
///             Complex::new(2.0, 4.0),
///         ],
///         vec![
///             Complex::new(5.0, 0.0),
///             Complex::new(3.0, 1.0),
///             Complex::new(2.0, 3.0),
///             Complex::new(2.0, 8.0),
///         ],
///         vec![
///             Complex::new(2.0, 5.0),
///             Complex::new(2.0, 3.0),
///             Complex::new(3.0, 7.0),
///             Complex::new(2.0, 1.0),
///         ],
///         vec![
///             Complex::new(5.0, 4.0),
///             Complex::new(1.0, 2.0),
///             Complex::new(4.0, 3.0),
///             Complex::new(2.0, 1.0),
///         ],
///     ];
///
///     let mut fft = CFft2D::<f64>::with_len(input.len(), input[0].len());
///
///     let output = fft.forward(&input);
///
///     println!("the transform of {:?} is {:?}", input, output);
/// }
/// ```
pub struct CFft2D<T> {
    len_m: usize,
    len_n: usize,
    scaler_n: T,
    scaler_u: T,
    fft_m: CFft1D<T>,
    fft_n: CFft1D<T>,
    work: Vec<Vec<Complex<T>>>,
}

impl<T: Float + FloatConst + NumAssign> CFft2D<T> {
    /// Returns a instances to execute FFT
    ///
    /// ```rust
    /// use chfft::CFft2D;
    /// let mut fft = CFft2D::<f64>::new();
    /// ```
    pub fn new() -> Self {
        Self {
            len_m: 0,
            len_n: 0,
            scaler_n: zero(),
            scaler_u: zero(),
            fft_m: CFft1D::new(),
            fft_n: CFft1D::new(),
            work: Vec::new(),
        }
    }

    /// Returns a instances to execute length initialized FFT
    ///
    /// ```rust
    /// use chfft::CFft2D;
    /// let mut fft = CFft2D::<f64>::with_len(1024, 1024);
    /// ```
    pub fn with_len(len_m: usize, len_n: usize) -> Self {
        Self {
            len_m: len_m,
            len_n: len_n,
            scaler_n: T::one() / cast(len_m * len_n).unwrap(),
            scaler_u: T::one() / cast::<_, T>(len_m * len_n).unwrap().sqrt(),
            fft_m: CFft1D::with_len(len_m),
            fft_n: CFft1D::with_len(len_n),
            work: vec![vec![zero(); len_m]; len_n],
        }
    }

    /// Reinitialize length
    ///
    /// ```rust
    /// use chfft::CFft2D;
    /// let mut fft = CFft2D::<f64>::with_len(1024, 1024);
    ///
    /// // reinitialize
    /// fft.setup(2048, 2048);
    /// ```
    pub fn setup(&mut self, len_m: usize, len_n: usize) {
        self.len_m = len_m;
        self.len_n = len_n;
        self.scaler_n = T::one() / cast(len_m * len_n).unwrap();
        self.scaler_u = self.scaler_n.sqrt();
        self.fft_m.setup(len_m);
        self.fft_n.setup(len_n);
        if self.work.len() != len_n || (self.work.len() > 0 && self.work[0].len() != len_m) {
            self.work = vec![vec![zero(); len_m]; len_n];
        }
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.forward(&input);
    /// ```
    pub fn forward(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        self.convert(source, false, one())
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.forward0(&input);
    /// ```
    pub fn forward0(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        self.convert(source, false, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.forwardu(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        let scaler = self.scaler_u;
        self.convert(source, false, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor forward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.forwardn(&input);
    /// ```
    pub fn forwardn(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        let scaler = self.scaler_n;
        self.convert(source, false, scaler)
    }

    /// The \\(\frac 1 n\\) scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.backward(&input);
    /// ```
    pub fn backward(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        let scaler = self.scaler_n;
        self.convert(source, true, scaler)
    }

    /// The 1 scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.backward0(&input);
    /// ```
    pub fn backward0(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        self.convert(source, true, one())
    }

    /// The \\(\frac 1 {\sqrt n}\\) scaling factor backward transform
    ///
    /// ```rust
    /// extern crate chfft;
    /// extern crate num_complex;
    ///
    /// let input = [
    ///     vec![
    ///         num_complex::Complex::new(2.0, 0.0),
    ///         num_complex::Complex::new(1.0, 1.0),
    ///         num_complex::Complex::new(0.0, 3.0),
    ///         num_complex::Complex::new(2.0, 4.0),
    ///     ],
    ///     vec![
    ///         num_complex::Complex::new(5.0, 0.0),
    ///         num_complex::Complex::new(3.0, 1.0),
    ///         num_complex::Complex::new(2.0, 3.0),
    ///         num_complex::Complex::new(2.0, 8.0),
    ///     ],
    /// ];
    ///
    /// let mut fft = chfft::CFft2D::<f64>::with_len(input.len(), input[0].len());
    /// let output = fft.backwardu(&input);
    /// ```
    pub fn backwardu(&mut self, source: &[Vec<Complex<T>>]) -> Vec<Vec<Complex<T>>> {
        let scaler = self.scaler_u;
        self.convert(source, true, scaler)
    }

    #[inline]
    fn convert(
        &mut self,
        source: &[Vec<Complex<T>>],
        is_back: bool,
        scaler: T,
    ) -> Vec<Vec<Complex<T>>> {
        if source.len() == 0 {
            return Vec::new();
        }
        if source.len() != self.len_m || source[0].len() != self.len_n {
            self.setup(source.len(), source[0].len());
        }

        for i in 0..source.len() {
            let work = if is_back {
                self.fft_m.backward0(&source[i])
            } else {
                self.fft_m.forward0(&source[i])
            };
            for j in 0..work.len() {
                self.work[j][i] = work[j];
            }
        }

        let mut ret = vec![Vec::with_capacity(self.len_n); self.len_m];
        for i in 0..self.work.len() {
            let work = if is_back {
                self.fft_n.backward0(&self.work[i])
            } else {
                self.fft_n.forward0(&self.work[i])
            };
            for j in 0..work.len() {
                ret[j].push(work[j] * scaler);
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nearly_eq::NearlyEq;
    use rand::{Rand, Rng, SeedableRng, XorShiftRng};
    use std::fmt::Debug;

    fn convert<T: Float + FloatConst>(source: &[Vec<Complex<T>>], scalar: T) -> Vec<Vec<Complex<T>>> {
        (0..source.len())
            .map(|i| {
                (0..source[0].len())
                    .map(|k| {
                        (0..source.len()).fold(zero(), |x: Complex<T>, j| {
                            x + (0..source[0].len()).fold(zero(), |y: Complex<T>, l| {
                                y
                                    + source[j][l]
                                        * Complex::<T>::from_polar(
                                            &one(),
                                            &(-cast::<_, T>(2).unwrap() * T::PI()
                                                * ((cast::<_, T>(i * j).unwrap()
                                                    / cast(source.len()).unwrap())
                                                    + cast::<_, T>(k * l).unwrap()
                                                        / cast(source[0].len()).unwrap())),
                                        )
                            })
                        }) * scalar
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn test_with_source<T: Float + FloatConst + NumAssign + Debug + NearlyEq>(
        fft: &mut CFft2D<T>,
        source: &[Vec<Complex<T>>],
    ) {
        let expected = convert(source, one());
        let actual = fft.forward(source);
        assert_nearly_eq!(&expected, &actual);
        let actual_source = fft.backward(&actual);
        assert_nearly_eq!(&source, &actual_source);
    }

    fn test_with_len<T: Float + Rand + FloatConst + NumAssign + Debug + NearlyEq>(
        fft: &mut CFft2D<T>,
        len_m: usize,
        len_n: usize,
    ) {
        let mut rng = XorShiftRng::from_seed([189522394, 1694417663, 1363148323, 4087496301]);

        // 10パターンのテスト
        for _ in 0..10 {
            let arr = (0..len_m)
                .map(|_| {
                    (0..len_n)
                        .map(|_| Complex::new(rng.gen::<T>(), rng.gen::<T>()))
                        .collect::<Vec<Complex<T>>>()
                })
                .collect::<Vec<Vec<Complex<T>>>>();

            test_with_source(fft, &arr);
        }
    }

    #[test]
    fn f64_new() {
        for i in 1..10 {
            for j in 1..10 {
                test_with_len(&mut CFft2D::<f64>::new(), i, j);
            }
        }
    }

    #[test]
    fn f32_new() {
        for i in 1..10 {
            for j in 1..10 {
                test_with_len(&mut CFft2D::<f32>::new(), i, j);
            }
        }
    }

    #[test]
    fn f64_with_len() {
        for i in 1..10 {
            for j in 1..10 {
                test_with_len(&mut CFft2D::<f64>::with_len(i, j), i, j);
            }
        }
    }

    #[test]
    fn f32_with_len() {
        for i in 1..10 {
            for j in 1..10 {
                test_with_len(&mut CFft2D::<f32>::with_len(i, j), i, j);
            }
        }
    }
}