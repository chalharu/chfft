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
use num_traits::identities::{one, zero};
use num_traits::{cast, NumAssign};

/// Perform a discrete cosine transform
///
/// # Example
///
/// ```rust
/// use chfft::{Dct1D, DctType};
///
/// fn main() {
///     let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
///
///     let mut dct = Dct1D::<f64>::new(DctType::Two, input.len(), false);
///     let output = dct.forward(&input);
///     println!("the transform of {:?} is {:?}", input, output);
/// }
/// ```
pub struct Dct1D<T> {
    worker: Dct1DWorkers<T>,
}

#[derive(PartialEq)]
pub enum DctType {
    // One,
    Two,
    Three,
    // Four
}

impl<T: Float + FloatConst + NumAssign> Dct1D<T> {
    /// Returns a instances to execute DCT
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, 1024, false);
    /// ```
    pub fn new(dct_type: DctType, len: usize, is_ortho: bool) -> Self {
        if len & 1 != 0 {
            panic!("invalid length")
        }
        Self {
            worker: match dct_type {
                DctType::Two => Dct1DWorkers::Dct2(Dct2Worker1D::new(len, is_ortho)),
                DctType::Three => Dct1DWorkers::Dct3(Dct3Worker1D::new(len, is_ortho)),
            },
        }
    }

    /// Reinitialize length
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, 1024, false);
    ///
    /// // reinitialize
    /// dct.setup(DctType::Two, 2048, false);
    /// ```
    pub fn setup(&mut self, dct_type: DctType, len: usize, is_ortho: bool) {
        if len & 1 != 0 {
            panic!("invalid length")
        }
        if self.worker.unwrap().dct_type() != dct_type {
            self.worker = match dct_type {
                DctType::Two => Dct1DWorkers::Dct2(Dct2Worker1D::new(len, is_ortho)),
                DctType::Three => Dct1DWorkers::Dct3(Dct3Worker1D::new(len, is_ortho)),
            }
        } else {
            self.worker.unwrap().setup(len, is_ortho);
        }
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, input.len(), false);
    /// let output = dct.forward0(&input);
    /// ```
    pub fn forward(&mut self, source: &[T]) -> Vec<T> {
        self.worker.unwrap().convert(source, one())
    }

    /// The 1 scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, input.len(), false);
    /// let output = dct.forward0(&input);
    /// ```
    pub fn forward0(&mut self, source: &[T]) -> Vec<T> {
        self.worker.unwrap().convert(source, one())
    }

    /// The unitary transform scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, input.len(), false);
    /// let output = dct.forwardu(&input);
    /// ```
    pub fn forwardu(&mut self, source: &[T]) -> Vec<T> {
        let scaler = self.worker.unwrap().unitary_scaler();
        self.worker.unwrap().convert(source, scaler)
    }

    /// The inverse scaling factor forward transform
    ///
    /// ```rust
    /// use chfft::{Dct1D, DctType};
    ///
    /// let input = [2.0, 0.0, 1.0, 1.0, 0.0, 3.0, 2.0, 4.0];
    ///
    /// let mut dct = Dct1D::<f64>::new(DctType::Two, input.len(), false);
    /// let output = dct.forwardu(&input);
    /// ```
    pub fn forwardn(&mut self, source: &[T]) -> Vec<T> {
        let scaler = self.worker.unwrap().unitary_scaler();
        self.worker.unwrap().convert(source, scaler * scaler)
    }
}

enum Dct1DWorkers<T> {
    Dct2(Dct2Worker1D<T>),
    Dct3(Dct3Worker1D<T>),
}

impl<T: Float + FloatConst + NumAssign> Dct1DWorkers<T> {
    // 本当はDerefMutを定義したいが、error[E0495]となる
    fn unwrap(&mut self) -> &mut dyn DctWorker1D<T> {
        match *self {
            Dct1DWorkers::Dct2(ref mut worker) => worker,
            Dct1DWorkers::Dct3(ref mut worker) => worker,
        }
    }
}

trait DctWorker1D<T> {
    fn setup(&mut self, len: usize, is_ortho: bool);
    fn dct_type(&self) -> DctType;
    fn unitary_scaler(&self) -> T;
    fn convert(&mut self, source: &[T], scaler: T) -> Vec<T>;
}

struct Dct2Worker1D<T> {
    cfft: Option<CFft1D<T>>,
    omega: Vec<Complex<T>>,
    coef: Vec<Complex<T>>,
    work: Vec<Complex<T>>,
    len: usize,
    ortho_scaler: T,
    unitary_scaler: T,
    is_ortho: bool,
}

impl<T: Float + FloatConst + NumAssign> Dct2Worker1D<T> {
    pub fn new(len: usize, is_ortho: bool) -> Self {
        let mut ret = Self {
            cfft: None,
            omega: Vec::new(),
            coef: Vec::new(),
            work: Vec::new(),
            len: 0,
            ortho_scaler: one(),
            unitary_scaler: one(),
            is_ortho: false,
        };
        ret.setup(len, is_ortho);
        ret
    }
}

impl<T: Float + FloatConst + NumAssign> DctWorker1D<T> for Dct2Worker1D<T> {
    fn setup(&mut self, len: usize, is_ortho: bool) {
        if len != self.len {
            self.cfft = Some(CFft1D::with_len(len >> 1));
            self.omega = precompute_utils::calc_omega(len << 2)
                .into_iter()
                .rev()
                .take((len >> 1) + 1)
                .collect();
            self.coef = precompute_utils::calc_omega(len)
                .iter()
                .take(len >> 1)
                .map(|w| {
                    (Complex::<T>::new(one(), zero()) + Complex::<T>::i() * w)
                        .scale(cast(0.5).unwrap())
                })
                .collect();
            self.work = vec![zero(); len >> 1];
            self.len = len;
            self.unitary_scaler = (cast::<_, T>(2.0).unwrap() / cast(len).unwrap()).sqrt();
        }
        self.ortho_scaler = if is_ortho {
            cast::<_, T>(0.5).unwrap().sqrt()
        } else {
            one()
        };
        self.is_ortho = is_ortho;
    }

    fn dct_type(&self) -> DctType {
        DctType::Two
    }

    fn unitary_scaler(&self) -> T {
        self.unitary_scaler
    }

    fn convert(&mut self, source: &[T], scaler: T) -> Vec<T> {
        if source.len() != self.len {
            panic!(
                "invalid length (soure: {}, dct.len: {})",
                source.len(),
                self.len
            )
        }

        let hlen = self.len >> 1;
        let qlen = hlen >> 1;

        if qlen == 0 {
            self.work[0] = Complex::new(source[0] * self.ortho_scaler, source[1]).scale(scaler);
        } else {
            self.work[0] = Complex::new(source[0] * self.ortho_scaler, source[2]).scale(scaler);
            for i in 1..qlen {
                self.work[i] = Complex::new(source[i << 2], source[(i << 2) + 2]).scale(scaler);
            }
            for i in 0..qlen {
                self.work[hlen - i - 1] =
                    Complex::new(source[(i << 2) + 3], source[(i << 2) + 1]).scale(scaler);
            }
            if hlen & 1 == 1 {
                self.work[qlen] =
                    Complex::new(source[self.len - 2], source[self.len - 1]).scale(scaler);
            }
        }
        self.cfft.as_mut().unwrap().forward0i(&mut self.work);

        let mut ret = vec![zero(); self.len];

        let qlen = (self.len + 3) >> 2;

        for i in 1..qlen {
            let wconj = self.work[hlen - i].conj();
            let x = self.coef[i] * (self.work[i] - wconj);
            let j = (self.work[i] - x).conj() * self.omega[i];
            ret[i] = j.re;
            ret[self.len - i] = j.im;
            let k = (wconj + x) * self.omega[hlen - i];
            ret[hlen - i] = k.re;
            ret[hlen + i] = k.im;
        }
        if self.len.trailing_zeros() >= 2 {
            let x = self.work[qlen] * self.omega[qlen];
            ret[qlen] = x.re;
            ret[self.len - qlen] = x.im;
        }

        ret[0] = self.work[0].re + self.work[0].im;
        ret[hlen] = (self.work[0].re - self.work[0].im) * self.omega[hlen].re;

        ret
    }
}

struct Dct3Worker1D<T> {
    cfft: Option<CFft1D<T>>,
    omega: Vec<Complex<T>>,
    coef: Vec<Complex<T>>,
    work: Vec<Complex<T>>,
    len: usize,
    ortho_scaler: T,
    unitary_scaler: T,
    is_ortho: bool,
}

impl<T: Float + FloatConst + NumAssign> Dct3Worker1D<T> {
    pub fn new(len: usize, is_ortho: bool) -> Self {
        let mut ret = Self {
            cfft: None,
            omega: Vec::new(),
            coef: Vec::new(),
            work: Vec::new(),
            len: 0,
            ortho_scaler: one(),
            unitary_scaler: one(),
            is_ortho: false,
        };
        ret.setup(len, is_ortho);
        ret
    }
}

impl<T: Float + FloatConst + NumAssign> DctWorker1D<T> for Dct3Worker1D<T> {
    fn setup(&mut self, len: usize, is_ortho: bool) {
        if len != self.len {
            self.cfft = Some(CFft1D::with_len(len >> 1));
            self.omega = precompute_utils::calc_omega(len << 2)
                .into_iter()
                .take((len >> 1) + 1)
                .collect();
            self.coef = precompute_utils::calc_omega(len)
                .iter()
                .rev()
                .take(len >> 1)
                .map(|w| {
                    (Complex::<T>::new(one(), zero()) - Complex::<T>::i() * w)
                        .scale(cast(0.5).unwrap())
                })
                .collect();
            self.work = vec![zero(); len >> 1];
            self.len = len;
            self.unitary_scaler = (cast::<_, T>(2.0).unwrap() / cast(len).unwrap()).sqrt();
        }
        self.ortho_scaler = if is_ortho {
            cast::<_, T>(0.5).unwrap().sqrt()
        } else {
            cast(0.5).unwrap()
        };
        self.is_ortho = is_ortho;
    }

    fn dct_type(&self) -> DctType {
        DctType::Three
    }

    fn unitary_scaler(&self) -> T {
        self.unitary_scaler
    }

    fn convert(&mut self, source: &[T], scaler: T) -> Vec<T> {
        if source.len() != self.len {
            panic!(
                "invalid length (soure: {}, dct.len: {})",
                source.len(),
                self.len
            )
        }
        let hlen = self.len >> 1;

        let zval = source[0] * self.ortho_scaler;
        let hval = source[hlen] * self.omega[hlen].re;

        self.work[0] = Complex::new(zval + hval, zval - hval).scale(scaler);

        let qlen = (self.len + 3) >> 2;

        if self.len.trailing_zeros() >= 2 {
            self.work[qlen] = Complex::new(source[qlen], source[self.len - qlen])
                * self.omega[qlen].scale(scaler);
        }

        for i in 1..qlen {
            let j = (Complex::new(source[i], source[self.len - i]) * self.omega[i])
                .scale(scaler)
                .conj();
            let k = (Complex::new(source[hlen - i], source[hlen + i]) * self.omega[hlen - i])
                .scale(scaler);
            let x = self.coef[i] * (j - k);

            self.work[i] = j - x;
            self.work[hlen - i] = k.conj() + x.conj();
        }

        self.cfft.as_mut().unwrap().backward0i(&mut self.work);

        let mut ret = Vec::with_capacity(self.len);

        let qlen = hlen >> 1;
        for i in 0..qlen {
            ret.push(self.work[i].re);
            ret.push(self.work[hlen - i - 1].im);
            ret.push(self.work[i].im);
            ret.push(self.work[hlen - i - 1].re);
        }
        if hlen & 1 == 1 {
            ret.push(self.work[qlen].re);
            ret.push(self.work[qlen].im);
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

    fn convert<T: Float + FloatConst>(source: &[T], scalar: T) -> Vec<T> {
        (0..source.len())
            .map(|i| {
                (0..source.len()).fold(zero(), |x: T, j| {
                    x + source[j]
                        * (T::PI() / cast(source.len() * 2).unwrap()
                            * cast::<_, T>((j * 2 + 1) * i).unwrap())
                        .cos()
                }) * scalar
            })
            .collect::<Vec<_>>()
    }

    fn test_with_source<T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps>(
        dct2: &mut Dct1D<T>,
        dct3: &mut Dct1D<T>,
        source: &[T],
    ) {
        let expected = convert(
            source,
            (cast::<_, T>(2).unwrap() / cast(source.len()).unwrap()).sqrt(),
        );
        let actual = dct2.forwardu(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = dct3.forwardu(&actual);
        assert_appro_eq(source, &actual_source);
        let expected = convert(
            source,
            cast::<_, T>(2).unwrap() / cast(source.len()).unwrap(),
        );
        let actual = dct2.forwardn(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = dct3.forward0(&actual);
        assert_appro_eq(source, &actual_source);
        let expected = convert(source, one());
        let actual = dct2.forward0(source);
        assert_appro_eq(&expected, &actual);
        let actual_source = dct3.forwardn(&actual);
        assert_appro_eq(source, &actual_source);
    }

    fn test_with_len<T: Float + FloatConst + NumAssign + Debug + AbsError + FloatEps>(
        dct2: &mut Dct1D<T>,
        dct3: &mut Dct1D<T>,
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
            test_with_source(dct2, dct3, &arr);
        }
    }

    #[test]
    fn f64_new() {
        for i in 1..100 {
            test_with_len(
                &mut Dct1D::<f64>::new(DctType::Two, i << 1, false),
                &mut Dct1D::<f64>::new(DctType::Three, i << 1, false),
                i << 1,
            );
        }
    }

    #[test]
    fn f32_new() {
        for i in 1..100 {
            test_with_len(
                &mut Dct1D::<f32>::new(DctType::Two, i << 1, false),
                &mut Dct1D::<f32>::new(DctType::Three, i << 1, false),
                i << 1,
            );
        }
    }

    #[test]
    fn f64_with_setup() {
        for i in 1..100 {
            let mut dct2 = Dct1D::<f64>::new(DctType::Two, i << 2, true);
            let mut dct3 = Dct1D::<f64>::new(DctType::Two, i << 2, true);
            dct2.setup(DctType::Two, i << 1, false);
            dct3.setup(DctType::Three, i << 1, false);
            test_with_len(&mut dct2, &mut dct3, i << 1);
        }
    }

    #[test]
    fn f32_with_setup() {
        for i in 1..100 {
            let mut dct2 = Dct1D::<f32>::new(DctType::Three, i << 2, true);
            let mut dct3 = Dct1D::<f32>::new(DctType::Three, i << 2, true);
            dct2.setup(DctType::Two, i << 1, false);
            dct3.setup(DctType::Three, i << 1, false);
            test_with_len(&mut dct2, &mut dct3, i << 1);
        }
    }

    #[test]
    #[should_panic(expected = "invalid length")]
    fn invalid_length() {
        Dct1D::<f64>::new(DctType::Two, 11, false);
    }

    #[test]
    #[should_panic(expected = "invalid length")]
    fn invalid_length_convert() {
        let mut fft = Dct1D::<f64>::new(DctType::Two, 8, false);
        fft.forward(&(0..).take(10).flat_map(cast::<_, _>).collect::<Vec<_>>());
    }
}
