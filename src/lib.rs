#![crate_type = "lib"]

//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

extern crate num_complex;
extern crate num_traits;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate appro_eq;

mod precompute_utils;
mod prime_factorization;
mod chirpz;
mod mixed_radix;

mod cfft1d;
mod rfft1d;
mod dct1d;
mod mdct1d;
mod cfft2d;

pub use cfft1d::CFft1D;
pub use rfft1d::RFft1D;
pub use dct1d::Dct1D;
pub use mdct1d::Mdct1D;
pub use cfft2d::CFft2D;

#[cfg(test)]
trait FloatEps {
    fn eps() -> Self;
}

#[cfg(test)]
mod tests {
    impl ::FloatEps for f32 {
        fn eps() -> Self {
            1e-2
        }
    }

    impl ::FloatEps for f64 {
        fn eps() -> Self {
            1e-10
        }
    }
}

#[cfg(test)]
#[inline(always)]
fn assert_appro_eq<
    A: FloatEps + std::fmt::Debug + PartialOrd,
    B: std::fmt::Debug + ?Sized,
    C: appro_eq::AbsError<B, A> + std::fmt::Debug + ?Sized,
>(
    expected: &C,
    actual: &B,
) {
    assert_appro_eq!(&expected, &actual, A::eps());
}
