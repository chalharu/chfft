//! # Licensing
//!　This Source Code is subject to the terms of the Mozilla Public License
//!　version 2.0 (the "License"). You can obtain a copy of the License at
//!　http://mozilla.org/MPL/2.0/.

extern crate chfft;
extern crate rand;
extern crate num;
extern crate num_traits;

use chfft::*;
use num::{Complex, Float, cast, one};
use num_traits::float::FloatConst;
use num_traits::NumAssign;
use std::fmt::Debug;
use rand::{Rand, XorShiftRng, SeedableRng, Rng};

fn epsilon<T: Float + Debug>() -> T {
    let mut epsold: T = one();
    let mut eps = T::one() / (T::one() + T::one());
    let mut delta: T = T::one() + eps;
    while delta != one() {
        delta = T::one() + eps;
        epsold = eps;
        eps = eps / (T::one() + T::one());
    }
    return epsold;
}

fn assert_complexes_eq<T: Float + Debug>(actual: &[Complex<T>], expected: &[Complex<T>]) {
    assert!(
        actual.len() == expected.len(),
        "assertion failed: `(left == right)` (left: `{:?}` , right: `{:?}`)",
        actual,
        expected
    );
    for i in 0..actual.len() {
        let (mantissa, exponent, _) = (actual[i] - expected[i]).norm().integer_decode();
        let diff: T = cast::<_, T>(mantissa).unwrap() *
            cast::<_, T>(2.0).unwrap().powi(exponent as i32);
        assert!(
            diff <= epsilon::<T>().sqrt().sqrt(),
            "assertion failed: `(left == right)` (left: `{:?}` , right: `{:?}`), diff: {:?}, eps: {:?}",
            actual,
            expected,
            diff,
            epsilon::<T>()
        );
    }
}

fn convert<T: Float + FloatConst>(source: &[Complex<T>], scalar: T) -> Vec<Complex<T>> {
    (0..source.len())
        .map(|i| {
            (1..source.len()).fold(source[0], |x, j| {
                x +
                    source[j] *
                        Complex::<T>::from_polar(
                            &one(),
                            &(-cast::<_, T>(2 * i * j).unwrap() * T::PI() /
                                    cast(source.len()).unwrap()),
                        )
            }) * scalar
        })
        .collect::<Vec<_>>()
}

fn test_with_source<T: Float + FloatConst + NumAssign + Debug>(
    fft: &mut CFft1D<T>,
    source: &[Complex<T>],
) {
    let expected = convert(source, one());
    let actual = fft.forward(source);
    assert_complexes_eq(&expected, &actual);
    let actual_source = fft.backward(&actual);
    assert_complexes_eq(&source, &actual_source);

    let expected = convert(source, one());
    let actual = fft.forward0(source);
    assert_complexes_eq(&expected, &actual);
    let actual_source = fft.backwardn(&actual);
    assert_complexes_eq(&source, &actual_source);

    let expected = convert(
        source,
        T::one() / cast::<_, T>(source.len()).unwrap().sqrt(),
    );
    let actual = fft.forwardu(source);
    assert_complexes_eq(&expected, &actual);
    let actual_source = fft.backwardu(&actual);
    assert_complexes_eq(&source, &actual_source);

    let expected = convert(source, T::one() / cast(source.len()).unwrap());
    let actual = fft.forwardn(source);
    assert_complexes_eq(&expected, &actual);
    let actual_source = fft.backward0(&actual);
    assert_complexes_eq(&source, &actual_source);
}

fn test_with_len<T: Float + Rand + FloatConst + NumAssign + Debug>(
    fft: &mut CFft1D<T>,
    len: usize,
) {
    let mut rng = XorShiftRng::from_seed([189522394, 1694417663, 1363148323, 4087496301]);

    // 10パターンのテスト
    for _ in 0..10 {
        let arr = (0..len)
            .map(|_| Complex::new(rng.gen::<T>(), rng.gen::<T>()))
            .collect::<Vec<Complex<T>>>();

        test_with_source(fft, &arr);
    }
}

#[test]
fn test() {
    for i in 1..100 {
        test_with_len(&mut CFft1D::<f64>::new(), i);
        test_with_len(&mut CFft1D::<f32>::new(), i);
        test_with_len(&mut CFft1D::<f64>::with_len(i), i);
        test_with_len(&mut CFft1D::<f32>::with_len(i), i);
    }
}
