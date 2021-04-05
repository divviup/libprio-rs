// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::benchmarked::{benchmarked_iterative_fft, benchmarked_recursive_fft};
use prio::finite_field::{Field, FieldElement};

pub fn fft(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        let mut rng = rand::thread_rng();
        let mut inp = vec![Field::zero(); *size];
        let mut outp = vec![Field::zero(); *size];
        for i in 0..*size {
            inp[i] = Field::rand(&mut rng);
        }

        c.bench_function(&format!("iterative/{}", *size), |b| {
            b.iter(|| {
                benchmarked_iterative_fft(&mut outp, &inp);
            })
        });

        c.bench_function(&format!("recursive/{}", *size), |b| {
            b.iter(|| {
                benchmarked_recursive_fft(&mut outp, &inp);
            })
        });
    }
}

criterion_group!(benches, fft);
criterion_main!(benches);
