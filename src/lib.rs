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
mod nearly_eq;

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
