//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

use num_complex::Complex;
use nearly_eq::NearlyEq;

impl<A, B, C: NearlyEq<A, B>> NearlyEq<Complex<A>, B> for Complex<C> {
    fn eps() -> B {
        C::eps()
    }

    fn eq(&self, other: &Complex<A>, eps: &B) -> bool {
        self.re.eq(&other.re, eps) && self.im.eq(&other.im, eps)
    }
}
