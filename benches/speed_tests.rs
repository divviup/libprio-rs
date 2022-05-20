// SPDX-License-Identifier: MPL-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use prio::benchmarked::*;
use prio::client::Client as Prio2Client;
use prio::codec::Encode;
use prio::encrypt::PublicKey;
use prio::field::{random_vector, Field128 as F, FieldElement};
use prio::flp::gadgets::Mul;
use prio::server::{generate_verification_message, ValidationMemory};
#[cfg(feature = "multithreaded")]
use prio::vdaf::prio3::Prio3Aes128CountVecMultithreaded;
use prio::vdaf::{
    prio3::{
        Prio3Aes128Count, Prio3Aes128CountVec, Prio3Aes128Histogram, Prio3Aes128Sum,
        Prio3InputShare,
    },
    Client as Prio3Client,
};

/// This benchmark compares the performance of recursive and iterative FFT.
pub fn fft(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        let inp = random_vector(*size).unwrap();
        let mut outp = vec![F::zero(); *size];

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

/// Speed test for generating a seed and deriving a pseudorandom sequence of field elements.
pub fn prng(c: &mut Criterion) {
    let test_sizes = [16, 256, 1024, 4096];
    for size in test_sizes.iter() {
        c.bench_function(&format!("rand, size={}", *size), |b| {
            b.iter(|| random_vector::<F>(*size))
        });
    }
}

/// The asymptotic cost of polynomial multiplication is `O(n log n)` using FFT and `O(n^2)` using
/// the naive method. This benchmark demonstrates that the latter has better concrete performance
/// for small polynomials. The result is used to pick the `FFT_THRESHOLD` constant in
/// `src/flp/gadgets.rs`.
pub fn poly_mul(c: &mut Criterion) {
    let test_sizes = [1_usize, 30, 60, 90, 120, 150];
    for size in test_sizes.iter() {
        let m = (*size + 1).next_power_of_two();
        let mut g: Mul<F> = Mul::new(*size);
        let mut outp = vec![F::zero(); 2 * m];
        let inp = vec![random_vector(m).unwrap(); 2];

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
    let test_sizes = [1, 10, 100, 1_000, 10_000];
    for size in test_sizes.iter() {
        let data = vec![F::zero(); *size];

        // v2
        let pk1 = PublicKey::from_base64(PUBKEY1).unwrap();
        let pk2 = PublicKey::from_base64(PUBKEY2).unwrap();
        let mut client: Prio2Client<F> =
            Prio2Client::new(data.len(), pk1.clone(), pk2.clone()).unwrap();

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
        let eval_at = random_vector(1).unwrap()[0];

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

        // TODO(cjpatton) Add benchmark for comparable FLP functionality.
    }
}

/// Benchmark prio3 client performance.
pub fn prio3_client(c: &mut Criterion) {
    let num_shares = 2;

    let prio3 = Prio3Aes128Count::new(num_shares).unwrap();
    let measurement = 1;
    println!(
        "prio3 count size = {}",
        prio3_input_share_size(&prio3.shard(&measurement).unwrap())
    );
    c.bench_function("prio3 count", |b| {
        b.iter(|| {
            prio3.shard(&1).unwrap();
        })
    });

    let buckets: Vec<u64> = (1..10).collect();
    let prio3 = Prio3Aes128Histogram::new(num_shares, &buckets).unwrap();
    let measurement = 17;
    println!(
        "prio3 histogram ({} buckets) size = {}",
        buckets.len() + 1,
        prio3_input_share_size(&prio3.shard(&measurement).unwrap())
    );
    c.bench_function(
        &format!("prio3 histogram ({} buckets)", buckets.len() + 1),
        |b| {
            b.iter(|| {
                prio3.shard(&measurement).unwrap();
            })
        },
    );

    let bits = 32;
    let prio3 = Prio3Aes128Sum::new(num_shares, bits).unwrap();
    let measurement = 1337;
    println!(
        "prio3 sum ({} bits) size = {}",
        bits,
        prio3_input_share_size(&prio3.shard(&measurement).unwrap())
    );
    c.bench_function(&format!("prio3 sum ({} bits)", bits), |b| {
        b.iter(|| {
            prio3.shard(&measurement).unwrap();
        })
    });

    let len = 1000;
    let prio3 = Prio3Aes128CountVec::new(num_shares, len).unwrap();
    let measurement = vec![0; len];
    println!(
        "prio3 countvec ({} len) size = {}",
        len,
        prio3_input_share_size(&prio3.shard(&measurement).unwrap())
    );
    c.bench_function(&format!("prio3 countvec ({} len)", len), |b| {
        b.iter(|| {
            prio3.shard(&measurement).unwrap();
        })
    });

    #[cfg(feature = "multithreaded")]
    {
        let prio3 = Prio3Aes128CountVecMultithreaded::new(num_shares, len).unwrap();
        let measurement = vec![0; len];
        println!(
            "prio3 countvec multithreaded ({} len) size = {}",
            len,
            prio3_input_share_size(&prio3.shard(&measurement).unwrap())
        );
        c.bench_function(&format!("prio3 parallel countvec ({} len)", len), |b| {
            b.iter(|| {
                prio3.shard(&measurement).unwrap();
            })
        });
    }
}

fn prio3_input_share_size<F: FieldElement, const L: usize>(
    input_shares: &[Prio3InputShare<F, L>],
) -> usize {
    let mut size = 0;
    for input_share in input_shares {
        size += input_share.get_encoded().len();
    }

    size
}

criterion_group!(benches, prio3_client, bool_vec, poly_mul, prng, fft);
criterion_main!(benches);
