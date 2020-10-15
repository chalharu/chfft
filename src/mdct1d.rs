//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

use crate::precompute_utils;
use crate::CFft1D;
use num_complex::Complex;
use num_traits::float::{Float, FloatConst};
use num_traits::identities::zero;
use num_traits::{cast, one, NumAssign};

/// Perform a Modified discrete cosine transform
///
/// # Example
///
/// ```rust
/// use chfft::Mdct1D;
///
/// fn main() {
///     let input = [2.0 as f64, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
///
///     let mut mdct = Mdct1D::with_sine(input.len());
///     let output = mdct.forward(&input);
///     println!("the transform of {:?} is {:?}", input, output);
/// }
/// ```
#[derive(Debug)]
pub struct Mdct1D<T, F: Fn(usize, usize) -> T> {
    fft: CFft1D<T>,
    len: usize,
    twiddle: Vec<Complex<T>>,
    work: Vec<Complex<T>>,
    scaler_u: T,
    scaler_ui: T,
    window_func: F,
    window_scaler: Vec<T>,
}

fn sine_window<T: Float + FloatConst>(l: usize, p: usize) -> T {
    (T::PI() * (cast::<_, T>(p).unwrap() + cast(0.5).unwrap()) / cast(l).unwrap()).sin()
}

fn vorbis_window<T: Float + FloatConst>(l: usize, p: usize) -> T {
    ((T::PI() * (cast::<_, T>(p).unwrap() + cast(0.5).unwrap()) / cast(l).unwrap())
        .sin()
        .powi(2)
        * T::PI()
        * cast(0.5).unwrap())
    .sin()
}

impl<T: Float + FloatConst + NumAssign> Mdct1D<T, fn(usize, usize) -> T> {
    /// Returns a instances to execute DCT with sine window
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    /// let mut mdct = Mdct1D::<f64, _>::with_sine(1024);
    /// ```
    pub fn with_sine(len: usize) -> Self {
        Self::new(sine_window, len)
    }

    /// Returns a instances to execute DCT with vorbis window
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    /// let mut mdct = Mdct1D::<f64, _>::with_vorbis(1024);
    /// ```
    pub fn with_vorbis(len: usize) -> Self {
        Self::new(vorbis_window, len)
    }
}

impl<T: Float + FloatConst + NumAssign, F: Fn(usize, usize) -> T> Mdct1D<T, F> {
    fn calc_twiddle(len: usize) -> Vec<Complex<T>> {
        let scaler = (cast::<_, T>(2.0).unwrap() / cast(len).unwrap())
            .sqrt()
            .sqrt();
        (0..(len >> 2))
            .map(|i| {
                precompute_utils::calc_omega_item(len << 3, (i << 3) + 1)
                    // .conj()
                    .scale(scaler)
            })
            .collect()
    }

    fn calc_window(window_func: &F, len: usize) -> Vec<T> {
        (0..len).map(|i| window_func(len, i)).collect()
    }

    /// Returns a instances to execute DCT
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    /// let mut mdct = Mdct1D::new(|l, p| (std::f64::consts::PI * (p as f64 + 0.5) / l as f64).sin(), 1024);
    /// ```
    pub fn new(window_func: F, len: usize) -> Self {
        if len & 3 != 0 {
            panic!("invalid length")
        }
        Self {
            fft: CFft1D::with_len(len >> 2),
            len,
            scaler_u: T::one() / cast::<_, T>(len >> 1).unwrap().sqrt(),
            scaler_ui: cast::<_, T>(len >> 1).unwrap().sqrt(),
            twiddle: Self::calc_twiddle(len),
            work: vec![zero(); len >> 2],
            window_scaler: Self::calc_window(&window_func, len),
            window_func,
        }
    }

    /// Reinitialize length
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    /// let mut mdct = Mdct1D::<f64, _>::with_sine(1024);
    ///
    /// // reinitialize
    /// mdct.setup(2048);
    /// ```
    pub fn setup(&mut self, len: usize) {
        if len & 3 != 0 {
            panic!("invalid length")
        }
        self.len = len;
        self.fft = CFft1D::with_len(len >> 2);
        self.scaler_u = T::one() / cast::<_, T>(len >> 1).unwrap().sqrt();
        self.scaler_ui = cast::<_, T>(len >> 1).unwrap().sqrt();
        self.twiddle = Self::calc_twiddle(len);
        self.work = vec![zero(); len >> 2];
        self.window_scaler = Self::calc_window(&self.window_func, len);
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    ///
    /// let input = [2.0 as f64, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut mdct = Mdct1D::with_sine(input.len());
    /// let output = mdct.forward(&input);
    /// ```
    pub fn forward(&mut self, source: &[T]) -> Vec<T> {
        let scaler = self.scaler_ui;
        self.convert(source, scaler)
    }

    /// The 1 scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    ///
    /// let input = [2.0 as f64, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut mdct = Mdct1D::with_sine(input.len() << 1);
    /// let output = mdct.backward(&input);
    /// ```
    pub fn backward(&mut self, source: &[T]) -> Vec<T> {
        let scaler = self.scaler_u;
        self.convert_back(source, scaler)
    }

    /// The \\(\sqrt{\frac 2 n}\\) scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    ///
    /// let input = [2.0 as f64, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut mdct = Mdct1D::with_sine(input.len());
    /// let output = mdct.forward(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[T]) -> Vec<T> {
        self.convert(source, one())
    }

    /// The \\(\sqrt{\frac 2 n}\\) scaling factor backward transform
    ///
    /// ```rust
    /// use chfft::Mdct1D;
    ///
    /// let input = [2.0 as f64, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut mdct = Mdct1D::with_sine(input.len() << 1);
    /// let output = mdct.backward(&input);
    /// ```
    pub fn backwardu(&mut self, source: &[T]) -> Vec<T> {
        self.convert_back(source, one())
    }

    fn convert(&mut self, source: &[T], scaler: T) -> Vec<T> {
        if source.len() != self.len {
            panic!(
                "invalid length (soure: {}, mdct.len: {})",
                source.len(),
                self.len
            )
        }

        let lenh = self.len >> 1;
        let lenq = lenh >> 1;
        let leno = lenq - (lenq >> 1);
        let len3q = lenh + lenq;
        let len5q = lenh + len3q;

        for i in 0..leno {
            let n = i << 1;
            self.work[i] = Complex::new(
                source[len3q - 1 - n] * self.window_scaler[len3q - 1 - n]
                    + source[len3q + n] * self.window_scaler[len3q + n],
                source[lenq + n] * self.window_scaler[lenq + n]
                    - source[lenq - 1 - n] * self.window_scaler[lenq - 1 - n],
            ) * self.twiddle[i];
        }

        for i in leno..lenq {
            let n = i << 1;
            self.work[i] = Complex::new(
                source[len3q - 1 - n] * self.window_scaler[len3q - 1 - n]
                    - source[n - lenq] * self.window_scaler[n - lenq],
                source[lenq + n] * self.window_scaler[lenq + n]
                    + source[len5q - 1 - n] * self.window_scaler[len5q - 1 - n],
            ) * self.twiddle[i];
        }

        self.fft.forward0i(&mut self.work);

        let mut ret = vec![zero(); lenh];
        for i in 0..lenq {
            let n = i << 1;
            let t = self.work[i] * self.twiddle[i].scale(scaler);
            ret[n] = -t.re;
            ret[lenh - 1 - n] = t.im;
        }
        ret
    }

    fn convert_back(&mut self, source: &[T], scaler: T) -> Vec<T> {
        if source.len() << 1 != self.len {
            panic!(
                "invalid length (soure: {}, mdct.len: {})",
                source.len(),
                self.len
            )
        }

        let lenh = self.len >> 1;
        let lenq = lenh >> 1;
        let leno = lenq - (lenq >> 1);
        let len3q = lenh + lenq;
        let len5q = lenh + len3q;

        /* pre-twiddle */

        for i in 0..lenq {
            let n = i << 1;
            self.work[i] = Complex::new(source[n], source[lenh - 1 - n])
                * self.twiddle[i].scale(cast(-2.0).unwrap());
        }
        self.fft.forward0i(&mut self.work);

        let mut ret = vec![zero(); self.len];

        for i in 0..leno {
            let n = i << 1;
            let t = self.work[i] * self.twiddle[i].scale(scaler);

            ret[len3q - 1 - n] = t.re * self.window_scaler[len3q - 1 - n];
            ret[len3q + n] = t.re * self.window_scaler[len3q + n];
            ret[lenq + n] = -t.im * self.window_scaler[lenq + n];
            ret[lenq - 1 - n] = t.im * self.window_scaler[lenq - 1 - n];
        }

        for i in leno..lenq {
            let n = i << 1;
            let t = self.work[i] * self.twiddle[i].scale(scaler);

            ret[len3q - 1 - n] = t.re * self.window_scaler[len3q - 1 - n];
            ret[n - lenq] = -t.re * self.window_scaler[n - lenq];
            ret[lenq + n] = -t.im * self.window_scaler[lenq + n];
            ret[len5q - 1 - n] = -t.im * self.window_scaler[len5q - 1 - n];
        }

        ret
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

    fn convert<T: Float + FloatConst, F>(window_func: F, source: &[T]) -> Vec<T>
    where
        F: Fn(usize, usize) -> T,
    {
        let n = source.len();
        (0..(n >> 1))
            .map(|m| {
                (0..source.len()).fold(zero(), |x: T, k| {
                    x + window_func(n, k)
                        * source[k]
                        * (T::PI() / cast(n << 1).unwrap()
                            * cast::<_, T>((1 + (n >> 1) + (k << 1)) * (1 + (m << 1))).unwrap())
                        .cos()
                })
            })
            .collect::<Vec<_>>()
    }

    fn convert_back<T: Float + FloatConst, F>(window_func: F, source: &[T]) -> Vec<T>
    where
        F: Fn(usize, usize) -> T,
    {
        let n = source.len() << 1;
        (0..n)
            .map(|k| {
                cast::<_, T>(4.0).unwrap() * window_func(n, k) / cast(n).unwrap()
                    * (0..(n >> 1)).fold(zero(), |x: T, m| {
                        x + source[m]
                            * (T::PI() / cast(n << 1).unwrap()
                                * cast::<_, T>((1 + (n >> 1) + (k << 1)) * (1 + (m << 1))).unwrap())
                            .cos()
                    })
            })
            .collect::<Vec<_>>()
    }

    fn test_with_source<
        T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps,
        F: Fn(usize, usize) -> T,
        G: Fn(usize, usize) -> T,
    >(
        mdct: &mut Mdct1D<T, F>,
        source: &[T],
        window_func: &G,
    ) {
        let expected = convert(window_func, source);
        let actual = mdct.forward(source);
        assert_appro_eq(&expected, &actual);
        let expected = convert_back(window_func, &actual);
        let actual_source = mdct.backward(&actual);
        assert_appro_eq(&expected, &actual_source);
        let actual = mdct.forwardu(source);
        let actual_source = mdct.backwardu(&actual);
        assert_appro_eq(&expected, &actual_source);
    }

    fn test_with_len<
        T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps,
        F: Fn(usize, usize) -> T,
        G: Fn(usize, usize) -> T,
    >(
        mdct: &mut Mdct1D<T, F>,
        len: usize,
        window_func: &G,
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
            test_with_source(mdct, &arr, window_func);
        }
    }

    #[test]
    fn f32_with_sine() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f32, _>::with_sine(i << 2),
                i << 2,
                &sine_window,
            );
        }
    }

    #[test]
    fn f64_with_sine() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f64, _>::with_sine(i << 2),
                i << 2,
                &sine_window,
            );
        }
    }

    #[test]
    fn f32_with_vorbis() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f32, _>::with_vorbis(i << 2),
                i << 2,
                &vorbis_window,
            );
        }
    }

    #[test]
    fn f64_with_vorbis() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f64, _>::with_vorbis(i << 2),
                i << 2,
                &vorbis_window,
            );
        }
    }

    #[test]
    fn f32_with_new() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f32, _>::new(sine_window, i << 2),
                i << 2,
                &sine_window,
            );
        }
    }

    #[test]
    fn f64_with_new() {
        for i in 1..100 {
            test_with_len(
                &mut Mdct1D::<f64, _>::new(sine_window, i << 2),
                i << 2,
                &sine_window,
            );
        }
    }

    #[test]
    fn f32_with_setup() {
        for i in 1..100 {
            let mut mdct = Mdct1D::<f32, _>::new(sine_window, i << 4);
            mdct.setup(i << 2);
            test_with_len(&mut mdct, i << 2, &sine_window);
        }
    }

    #[test]
    fn f64_with_setup() {
        for i in 1..100 {
            let mut mdct = Mdct1D::<f64, _>::new(sine_window, i << 4);
            mdct.setup(i << 2);
            test_with_len(&mut mdct, i << 2, &sine_window);
        }
    }

    #[test]
    #[should_panic(expected = "invalid length")]
    fn invalid_length() {
        let _ = Mdct1D::<f64, _>::new(sine_window, 10);
    }

    #[test]
    #[should_panic(expected = "invalid length")]
    fn invalid_length_convert() {
        let mut fft = Mdct1D::<f64, _>::new(sine_window, 8);
        let _ = fft.forward(&(0..).take(10).flat_map(cast::<_, _>).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "invalid length")]
    fn invalid_length_convert_back() {
        let mut fft = Mdct1D::<f64, _>::new(sine_window, 8);
        let _ = fft.backward(&(0..).take(8).flat_map(cast::<_, _>).collect::<Vec<_>>());
    }
}
