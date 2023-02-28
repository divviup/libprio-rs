use fixed_macro::fixed;
use prio::{
    benchmarked::benchmarked_v2_prove,
    client::Client as Prio2Client,
    codec::Encode,
    encrypt::PublicKey,
    field::{random_vector, FftFriendlyFieldElement, Field128, FieldElement},
    flp::{
        gadgets::{BlindPolyEval, ParallelSum},
        types::SumVec,
        Type,
    },
    vdaf::{
        prio3::{Prio3, Prio3InputShare},
        Client,
    },
};

fn main() {
    let num_shares = 2;
    let nonce = [0; 16];

    let prio3 = Prio3::new_count(num_shares).unwrap();
    let measurement = 1;
    println!(
        "prio3 count share size = {}",
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let buckets: Vec<u64> = (1..10).collect();
    let prio3 = Prio3::new_histogram(num_shares, &buckets).unwrap();
    let measurement = 17;
    println!(
        "prio3 histogram ({} buckets) share size = {}",
        buckets.len() + 1,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let bits = 32;
    let prio3 = Prio3::new_sum(num_shares, bits).unwrap();
    let measurement = 1337;
    println!(
        "prio3 sum ({} bits) share size = {}",
        bits,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let len = 1000;
    let prio3 = Prio3::new_sum_vec(num_shares, 1, len).unwrap();
    let measurement = vec![0; len];
    println!(
        "prio3 countvec ({} len) share size = {}",
        len,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let prio3 = Prio3::new_sum_vec_multithreaded(num_shares, 1, len).unwrap();
    println!(
        "prio3 countvec multithreaded ({} len) share size = {}",
        len,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let len = 1000;
    let prio3 = Prio3::new_fixedpoint_boundedl2_vec_sum(num_shares, len).unwrap();
    let fp_num = fixed!(0.0001: I1F15);
    let measurement = vec![fp_num; len];
    println!(
        "prio3 fixedpoint16 boundedl2 vec ({} entries) size = {}",
        len,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    let prio3 = Prio3::new_fixedpoint_boundedl2_vec_sum_multithreaded(num_shares, len).unwrap();
    println!(
        "prio3 fixedpoint16 boundedl2 vec multithreaded ({} entries) size = {}",
        len,
        prio3_input_share_size(&prio3.shard(&measurement, &nonce).unwrap().1)
    );

    println!();

    for size in [10, 100, 1_000] {
        // Prio2
        // Public keys used to instantiate the v2 client.
        const PUBKEY1: &str =
        "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVQ=";
        const PUBKEY2: &str =
        "BNNOqoU54GPo+1gTPv+hCgA9U2ZCKd76yOMrWa1xTWgeb4LhFLMQIQoRwDVaW64g/WTdcxT4rDULoycUNFB60LE=";
        let pk1 = PublicKey::from_base64(PUBKEY1).unwrap();
        let pk2 = PublicKey::from_base64(PUBKEY2).unwrap();
        let input = vec![Field128::zero(); size];
        let mut client: Prio2Client<Field128> =
            Prio2Client::new(input.len(), pk1.clone(), pk2.clone()).unwrap();
        println!(
            "prio2 ({} len) proof size={}",
            size,
            benchmarked_v2_prove(&input, &mut client).len()
        );

        // Prio3
        let count_vec: SumVec<Field128, ParallelSum<Field128, BlindPolyEval<Field128>>> =
            SumVec::new(1, size).unwrap();
        let joint_rand = random_vector(count_vec.joint_rand_len()).unwrap();
        let prove_rand = random_vector(count_vec.prove_rand_len()).unwrap();
        let proof = count_vec.prove(&input, &prove_rand, &joint_rand).unwrap();
        println!("prio3 countvec ({} len) proof size={}", size, proof.len());
    }
}

fn prio3_input_share_size<F: FftFriendlyFieldElement, const L: usize>(
    input_shares: &[Prio3InputShare<F, L>],
) -> usize {
    let mut size = 0;
    for input_share in input_shares {
        size += input_share.get_encoded().len();
    }

    size
}
