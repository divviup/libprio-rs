use serde::Serialize;
use std::fs::File;
use std::io::Write;

use prio::vdaf::prg::Prg;
use prio::vdaf::prg::PrgAes128;
use prio::vdaf::prg::SeedStream;

#[derive(Serialize, Debug)]
struct TestCase {
    seed: [u8; 16],
    info_string: Vec<u8>,
    buffer1_out: Vec<u8>,
    buffer2_out: Vec<u8>,
}

impl TestCase {
    fn new(seed: [u8; 16], info_string: &[u8], buffer1_size: usize, buffer2_size: usize) -> Self {
        TestCase {
            seed,
            info_string: info_string.to_vec(),
            buffer1_out: vec![0; buffer1_size],
            buffer2_out: vec![0; buffer2_size],
        }
    }
}

fn main() {
    let mut testcases = vec![
        TestCase::new([0; 16], &[0; 10], 20, 500),
        TestCase::new(
            [1; 16],
            &[
                0x69, 0x6e, 0x66, 0x6f, 0x20, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67,
            ],
            0,
            16,
        ),
        TestCase::new(
            [5; 16],
            &[
                0x69, 0x6e, 0x66, 0x6f, 0x20, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67,
            ],
            0,
            16,
        ),
        TestCase::new(
            [1; 16],
            &[
                0x6e, 0x6e, 0x66, 0x6f, 0x20, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67,
            ],
            0,
            16,
        ),
        TestCase::new([3; 16], &[0; 0], 2, 5),
        TestCase::new([3; 16], &[0x6e; 5], 2, 5),
    ];

    for testcase in &mut testcases {
        let mut p = PrgAes128::init(&testcase.seed);
        p.update(&testcase.info_string);
        let mut s = p.into_seed_stream();
        s.fill(&mut testcase.buffer1_out);
        s.fill(&mut testcase.buffer2_out);
    }

    let mut file = File::create("PrgAes128_tests.json").unwrap();
    let json = serde_json::to_string(&testcases).unwrap();
    println!("{}", json);
    write!(file, "{}", json).expect("Failed to write output to file.");
}
