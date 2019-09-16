#![crate_type = "lib"]

//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

mod chirpz;
mod mixed_radix;
mod precompute_utils;
mod prime_factorization;

mod cfft1d;
mod cfft2d;
mod dct1d;
mod mdct1d;
mod rfft1d;

pub use cfft1d::CFft1D;
pub use cfft2d::CFft2D;
pub use dct1d::{Dct1D, DctType};
pub use mdct1d::Mdct1D;
pub use rfft1d::RFft1D;

pub(crate) trait QuarterRotation {
    fn quarter_turn(self) -> Self;
    fn three_quarter_turn(self) -> Self;
}

impl<T: num_traits::Float> QuarterRotation for num_complex::Complex<T> {
    fn quarter_turn(self) -> Self {
        num_complex::Complex::new(-self.im, self.re)
    }
    fn three_quarter_turn(self) -> Self {
        num_complex::Complex::new(self.im, -self.re)
    }
}

#[cfg(test)]
trait FloatEps {
    fn eps() -> Self;
}

#[cfg(test)]
mod tests {
    impl crate::FloatEps for f32 {
        fn eps() -> Self {
            1e-2
        }
    }

    impl crate::FloatEps for f64 {
        fn eps() -> Self {
            1e-10
        }
    }
}

#[cfg(test)]
fn assert_appro_eq<
    A: FloatEps + std::fmt::Debug + PartialOrd,
    B: std::fmt::Debug + ?Sized,
    C: appro_eq::AbsError<B, A> + std::fmt::Debug + ?Sized,
>(
    expected: &C,
    actual: &B,
) {
    appro_eq::assert_appro_eq!(&expected, &actual, A::eps());
}
