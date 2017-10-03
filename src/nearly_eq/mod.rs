//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

mod complex;

#[macro_use]
mod assert;

use std::{f32, f64};

pub trait NearlyEq<Rhs: ?Sized = Self, Diff: ?Sized = Self> {
    fn eps() -> Diff;

    fn eq(&self, other: &Rhs, eps: &Diff) -> bool;

    #[inline]
    fn ne(&self, other: &Rhs, eps: &Diff) -> bool {
        !self.eq(other, eps)
    }
}

impl NearlyEq for f32 {
    fn eps() -> f32 {
        1e-2
    }

    fn eq(&self, other: &f32, eps: &f32) -> bool {
        let diff = (*self - *other).abs();

        if *self == *other {
            true
        } else {
            diff < *eps
        }
    }
}

impl NearlyEq for f64 {
    fn eps() -> f64 {
        1e-11
    }

    fn eq(&self, other: &f64, eps: &f64) -> bool {
        let diff = (*self - *other).abs();

        if *self == *other {
            true
        } else {
            diff < *eps
        }
    }
}

impl<A, B, C: NearlyEq<A, B>> NearlyEq<[A], B> for [C] {
    fn eps() -> B {
        C::eps()
    }

    fn eq(&self, other: &[A], eps: &B) -> bool {
        if self.len() != other.len() {
            false
        } else {
            for i in 0..self.len() {
                if self[i].ne(&other[i], eps) {
                    return false;
                }
            }
            true
        }
    }
}

impl<A, B, C: NearlyEq<A, B>> NearlyEq<Vec<A>, B> for Vec<C> {
    fn eps() -> B {
        C::eps()
    }

    fn eq(&self, other: &Vec<A>, eps: &B) -> bool {
        if self.len() != other.len() {
            false
        } else {
            for i in 0..self.len() {
                if self[i].ne(&other[i], eps) {
                    return false;
                }
            }
            true
        }
    }
}

impl<'a, A: ?Sized, B, C: NearlyEq<A, B> + ?Sized> NearlyEq<A, B> for &'a C {
    fn eps() -> B {
        C::eps()
    }

    fn eq(&self, other: &A, eps: &B) -> bool {
        (*self).eq(&other, eps)
    }
}
