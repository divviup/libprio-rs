use fixed::{
    types::{extra::U15, I1F15},
    FixedI16,
};

use prio::{
    codec::Encode,
    vdaf::{
        prio2::Prio2,
        prio3::{
            Prio3, Prio3Count, Prio3FixedPointBoundedL2VecSum, Prio3Histogram, Prio3Sum,
            Prio3SumVec,
        },
        Client, Vdaf,
    },
};

const PRIO2_CTX_STR: &[u8] = b"prio2 ctx";
const PRIO3_CTX_STR: &[u8] = b"prio3 ctx";

fn main() {
    let num_shares = 2;
    let nonce = [0; 16];

    let prio3 = Prio3::new_count(num_shares).unwrap();
    let measurement = true;
    println!(
        "prio3 count share size = {}",
        vdaf_input_share_size::<Prio3Count, 16>(
            prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
        )
    );

    let length = 10;
    let prio3 = Prio3::new_histogram(num_shares, length, 3).unwrap();
    let measurement = 9;
    println!(
        "prio3 histogram ({} buckets) share size = {}",
        length,
        vdaf_input_share_size::<Prio3Histogram, 16>(
            prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
        )
    );

    let max_measurement = 0xffff_ffff;
    let prio3 = Prio3::new_sum(num_shares, max_measurement).unwrap();
    let measurement = 1337;
    println!(
        "prio3 sum ({} bits) share size = {}",
        max_measurement.ilog2() + 1,
        vdaf_input_share_size::<Prio3Sum, 16>(
            prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
        )
    );

    let len = 1000;
    let prio3 = Prio3::new_sum_vec(num_shares, 1, len, 31).unwrap();
    let measurement = vec![0; len];
    println!(
        "prio3 sumvec ({} len) share size = {}",
        len,
        vdaf_input_share_size::<Prio3SumVec, 16>(
            prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
        )
    );

    let len = 1000;
    let prio3 = Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, len).unwrap();
    const FP_NUM: I1F15 = I1F15::lit("0.0001");
    let measurement = vec![FP_NUM; len];
    println!(
        "prio3 fixedpoint16 boundedl2 vec ({} entries) size = {}",
        len,
        vdaf_input_share_size::<Prio3FixedPointBoundedL2VecSum<FixedI16<U15>>, 16>(
            prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
        )
    );

    println!();

    for (size, chunk_length) in [(10, 3), (100, 10), (1_000, 31)] {
        // Prio2
        let measurement = vec![0u32; size];
        let prio2 = Prio2::new(size).unwrap();
        println!(
            "prio2 ({} entries) size = {}",
            size,
            vdaf_input_share_size::<Prio2, 16>(
                prio2.shard(PRIO2_CTX_STR, &measurement, &nonce).unwrap()
            )
        );

        // Prio3
        let measurement = vec![0u128; size];
        let prio3 = Prio3::new_sum_vec(2, 1, size, chunk_length).unwrap();
        println!(
            "prio3 sumvec ({} entries) size = {}",
            size,
            vdaf_input_share_size::<Prio3SumVec, 16>(
                prio3.shard(PRIO3_CTX_STR, &measurement, &nonce).unwrap()
            )
        );
    }
}

fn vdaf_input_share_size<V: Vdaf, const SEED_SIZE: usize>(
    shares: (V::PublicShare, Vec<V::InputShare>),
) -> usize {
    let mut size = shares.0.get_encoded().unwrap().len();
    for input_share in shares.1 {
        size += input_share.get_encoded().unwrap().len();
    }

    size
}
