use std::time::Instant;

use prio::{
    codec::Encode, idpf::test_utils::generate_zipf_distributed_batch,
    vdaf::poplar1::Poplar1AggregationParam,
};
use rand::rng;

fn main() {
    let bits = 256;
    let measurement_count = 10_000;
    let threshold = ((measurement_count as f64) * 0.01) as usize; // 1%
    let zipf_support = 128;
    let zipf_exponent = 1.03;

    println!("Generating inputs and computing the prefix tree. This may take some time...");
    let start = Instant::now();
    let (_measurements, prefix_tree) = generate_zipf_distributed_batch(
        &mut rng(),
        bits,
        threshold,
        measurement_count,
        zipf_support,
        zipf_exponent,
    );
    let elapsed = start.elapsed();
    println!("Finished in {elapsed:?}");

    let mut max_agg_param_len = 0;
    let mut max_agg_param_level = 0;
    for (level, prefixes) in prefix_tree.into_iter().enumerate() {
        let num_prefixes = prefixes.len();
        let agg_param = Poplar1AggregationParam::try_from_prefixes(prefixes)
            .expect("failed to encode prefixes at level {level}");
        let agg_param_len = agg_param
            .get_encoded()
            .expect("failed to encode the aggregation parameter at level {level}")
            .len();
        if agg_param_len > max_agg_param_len {
            max_agg_param_len = agg_param_len;
            max_agg_param_level = level;
        }
        println!("{level}: {agg_param_len} {num_prefixes}");
    }
    println!("max: {max_agg_param_level}: {max_agg_param_len}");
}
