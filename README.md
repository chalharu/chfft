# chfft

[![crates.io badge](https://img.shields.io/crates/v/chfft.svg)](https://crates.io/crates/chfft)
[![Build Status](https://travis-ci.org/chalharu/chfft.svg?branch=master)](https://travis-ci.org/chalharu/chfft)
[![docs.rs](https://docs.rs/chfft/badge.svg)](https://docs.rs/chfft)
[![Coverage Status](https://coveralls.io/repos/github/chalharu/chfft/badge.svg?branch=master)](https://coveralls.io/github/chalharu/chfft?branch=master)

Fastest Fourier Transform library implemented with pure Rust.

## How-to Use

See the [crate documentation](https://docs.rs/chfft/) for more details.

## Features

- **`CFft1D`** - Perform a complex-to-complex one-dimensional Fourier transform.

- **`CFft2D`** - Perform a complex-to-complex two-dimensional Fourier transform.

- **`Dct1D`** - Perform a discrete cosine transform.

- **`RFft1D`** - Perform a real-to-complex one-dimensional Fourier transform.

- **`Mdct1D`** - Perform a Modified discrete cosine transform.

### Examples

```rust
use num_complex::Complex;
use chfft::CFft1D;

fn main() {
    let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
                 Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
    let mut fft = CFft1D::<f64>::with_len(input.len());
    let output = fft.forward(&input);
    println!("the transform of {:?} is {:?}", input, output);
}
```
