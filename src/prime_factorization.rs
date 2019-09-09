//! Chalharu's Fastest Fourier Transform.
//!
//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/ .

pub struct Factor {
    pub value: usize,
    pub count: usize,
}

struct FactorIterator {
    value: usize,
    prime: usize,
}

impl Iterator for FactorIterator {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            match self.prime {
                4 => {
                    if self.value.trailing_zeros() >= 2 {
                        self.value >>= 2;
                        return Some(4);
                    } else {
                        self.prime = 2;
                    }
                }
                2 => {
                    if self.value & 1 == 0 {
                        self.value >>= 1;
                        return Some(2);
                    } else {
                        self.prime = 3;
                    }
                }
                _ => {
                    if self.value >= self.prime * self.prime {
                        if self.value % self.prime == 0 {
                            self.value /= self.prime;
                            return Some(self.prime);
                        } else {
                            self.prime += 2;
                        }
                    } else if self.value > 1 {
                        let v = self.value;
                        self.value = 0;
                        return Some(v);
                    } else {
                        return None;
                    }
                }
            }
        }
    }
}

fn prime_factorization_iter(value: usize) -> FactorIterator {
    FactorIterator { value, prime: 4 }
}

// 素因数分解
pub fn prime_factorization(value: usize, max: usize) -> Vec<Factor> {
    let mut factors = Vec::<Factor>::new();
    let mut count = 0;
    let mut prime = 0;
    let mut fac4count = 0;
    for p in prime_factorization_iter(value) {
        if p > max {
            return Vec::<Factor>::new();
        }
        if prime != p {
            if count > 0 {
                if prime == 4 {
                    fac4count = count;
                } else {
                    if prime > 4 && fac4count != 0 {
                        factors.push(Factor {
                            value: 4,
                            count: fac4count,
                        });
                        fac4count = 0;
                    }
                    factors.push(Factor {
                        value: prime,
                        count,
                    });
                }
            }
            count = 1;
            prime = p;
        } else {
            count += 1;
        }
    }
    if count > 0 {
        factors.push(Factor {
            value: prime,
            count,
        });
    }
    if fac4count != 0 {
        factors.push(Factor {
            value: 4,
            count: fac4count,
        });
    }
    factors
}
