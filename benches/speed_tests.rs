// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::benchmarked::*;
use prio::client::Client;
use prio::encrypt::PublicKey;
use prio::field::{Field126 as F, FieldElement};
use prio::pcp::gadgets::Mul;
use prio::pcp::types::{ParallelPolyCheckedVector, PolyCheckedVector};
use prio::pcp::{prove, query, Gadget, Value};
use prio::server::{generate_verification_message, ValidationMemory};

/// This benchmark compares the performance of recursive and iterative FFT.
pub fn fft(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        let mut inp = vec![F::zero(); *size];
        let mut outp = vec![F::zero(); *size];
        for i in 0..*size {
            inp[i] = F::rand();
        }

        c.bench_function(&format!("iterative FFT, size={}", *size), |b| {
            b.iter(|| {
                benchmarked_iterative_fft(&mut outp, &inp);
            })
        });

        c.bench_function(&format!("recursive FFT, size={}", *size), |b| {
            b.iter(|| {
                benchmarked_recursive_fft(&mut outp, &inp);
            })
        });
    }
}

/// The asymptotic cost of polynomial multiplication is `O(n log n)` using FFT and `O(n^2)` using
/// the naive method. This benchmark demonstrates that the latter has better concrete performance
/// for small polynomials. The result is used to pick the `FFT_THRESHOLD` constant in
/// `src/pcp/gadgets.rs`.
pub fn poly_mul(c: &mut Criterion) {
    let test_sizes = [1_usize, 30, 60, 90, 120, 150];
    for size in test_sizes.iter() {
        let m = size.next_power_of_two();
        let mut g: Mul<F> = Mul::new(m);
        let mut outp = vec![F::zero(); 2 * m];
        let mut inp = vec![vec![F::zero(); m]; 2];
        for i in 0..2 {
            for j in 0..*size {
                inp[i][j] = F::rand();
            }
        }

        c.bench_function(&format!("poly mul FFT, size={}", *size), |b| {
            b.iter(|| {
                benchmarked_gadget_mul_call_poly_fft(&mut g, &mut outp, &inp).unwrap();
            })
        });

        c.bench_function(&format!("poly mul direct, size={}", *size), |b| {
            b.iter(|| {
                benchmarked_gadget_mul_call_poly_direct(&mut g, &mut outp, &inp).unwrap();
            })
        });
    }
}

// Public keys used to instantiate the v2 client.
const PUBKEY1: &str =
    "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVQ=";
const PUBKEY2: &str =
    "BNNOqoU54GPo+1gTPv+hCgA9U2ZCKd76yOMrWa1xTWgeb4LhFLMQIQoRwDVaW64g/WTdcxT4rDULoycUNFB60LE=";

/// Benchmark generation and verification of boolean vectors.
pub fn bool_vec(c: &mut Criterion) {
    let test_sizes = [1, 10, 100, 1_000, 10_000, 100_000];
    for size in test_sizes.iter() {
        let data = vec![F::zero(); *size];

        // v2
        let pk1 = PublicKey::from_base64(PUBKEY1).unwrap();
        let pk2 = PublicKey::from_base64(PUBKEY2).unwrap();
        let mut client: Client<F> = Client::new(data.len(), pk1.clone(), pk2.clone()).unwrap();

        c.bench_function(&format!("bool vec v2 prove, size={}", *size), |b| {
            b.iter(|| {
                benchmarked_v2_prove(&data, &mut client);
            })
        });
        println!(
            "bool vec v2 proof size={}\n",
            benchmarked_v2_prove(&data, &mut client).len()
        );

        let data_and_proof = benchmarked_v2_prove(&data, &mut client);
        let mut validator: ValidationMemory<F> = ValidationMemory::new(data.len());
        let eval_at = F::rand();

        c.bench_function(&format!("bool vec v2 query, size={}", *size), |b| {
            b.iter(|| {
                generate_verification_message(
                    data.len(),
                    eval_at,
                    &data_and_proof,
                    true,
                    &mut validator,
                )
                .unwrap();
            })
        });

        // v3
        let x: PolyCheckedVector<F> = PolyCheckedVector::new_range_checked(data.clone(), 0, 2);
        let (query_rand, joint_rand) = gen_rand(&x);

        c.bench_function(&format!("bool vec v3 prove, size={}", *size), |b| {
            b.iter(|| {
                prove(&x, &joint_rand).unwrap();
            })
        });

        let pf = prove(&x, &joint_rand).unwrap();
        println!("bool vec v3 proof size={}\n", pf.as_slice().len());

        c.bench_function(&format!("bool vec v3 query, size={}", *size), |b| {
            b.iter(|| {
                query(&x, &pf, &query_rand, &joint_rand).unwrap();
            })
        });

        // v3 (parallel)
        let x: ParallelPolyCheckedVector<F> =
            ParallelPolyCheckedVector::new_range_checked(data.clone(), 0, 2);
        let (query_rand, joint_rand) = gen_rand(&x);

        c.bench_function(
            &format!("bool vec v3 parallel prove, size={}", *size),
            |b| {
                b.iter(|| {
                    prove(&x, &joint_rand).unwrap();
                })
            },
        );

        let pf = prove(&x, &joint_rand).unwrap();
        println!("bool vec v3 parallel proof size={}\n", pf.as_slice().len());

        c.bench_function(
            &format!("bool vec v3 parallel query, size={}", *size),
            |b| {
                b.iter(|| {
                    query(&x, &pf, &query_rand, &joint_rand).unwrap();
                })
            },
        );
    }
}

fn gen_rand<F, G, V>(x: &V) -> (Vec<F>, Vec<F>)
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
{
    let query_rand = vec![F::rand()];
    let rand_len = x.valid_rand_len();
    let mut joint_rand: Vec<F> = Vec::with_capacity(rand_len);
    for _ in 0..rand_len {
        joint_rand.push(F::rand());
    }
    (query_rand, joint_rand)
}

criterion_group!(benches, bool_vec, poly_mul, fft);
criterion_main!(benches);
