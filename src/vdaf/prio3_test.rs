// SPDX-License-Identifier: MPL-2.0

use crate::{
    codec::{Encode, ParameterizedDecode},
    flp::Type,
    vdaf::{
        prg::Prg,
        prio3::{
            Prio3, Prio3Aes128Count, Prio3Aes128Histogram, Prio3Aes128Sum, Prio3InputShare,
            Prio3PrepareShare,
        },
        Aggregator, PrepareTransition,
    },
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
struct TEncoded(#[serde(with = "hex")] Vec<u8>);

impl AsRef<[u8]> for TEncoded {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Deserialize, Serialize)]
struct TPrio3Prep<M> {
    measurement: M,
    #[serde(with = "hex")]
    nonce: Vec<u8>,
    input_shares: Vec<TEncoded>,
    prep_shares: Vec<Vec<TEncoded>>,
    out_shares: Vec<Vec<M>>,
}

#[derive(Deserialize, Serialize)]
struct TPrio3<M> {
    prep: Vec<TPrio3Prep<M>>,
}

macro_rules! err {
    (
        $test_num:ident,
        $error:expr,
        $msg:expr
    ) => {
        panic!("test #{} failed: {} err: {}", $test_num, $msg, $error)
    };
}

// TODO Generalize this method to work with any VDAF. To do so we would need to add
// `test_vec_setup()` and `test_vec_shard()` to traits. (There may be a less invasive alternative.)
fn check_prep_test_vec<M, T, A, P, const L: usize>(
    prio3: &Prio3<T, A, P, L>,
    test_num: usize,
    t: &TPrio3Prep<M>,
) where
    T: Type<Measurement = M>,
    A: Clone + Debug + Sync + Send,
    P: Prg<L>,
    M: From<<T as Type>::Field> + Debug + PartialEq,
{
    let input_shares = prio3
        .test_vec_shard(&t.measurement)
        .expect("failed to generate input shares");

    let agg_key = [1; L];

    assert_eq!(2, t.input_shares.len(), "#{}", test_num);
    for (agg_id, want) in t.input_shares.iter().enumerate() {
        assert_eq!(
            input_shares[agg_id],
            Prio3InputShare::get_decoded_with_param(&(prio3, agg_id), want.as_ref())
                .unwrap_or_else(|e| err!(test_num, e, "decode test vector (input share)")),
            "#{}",
            test_num
        );
        assert_eq!(
            input_shares[agg_id].get_encoded(),
            want.as_ref(),
            "#{}",
            test_num
        )
    }

    let mut states = Vec::new();
    let mut prep_shares = Vec::new();
    for (agg_id, input_share) in input_shares.iter().enumerate() {
        let (state, prep_share) = prio3
            .prepare_init(&agg_key, agg_id, &(), &t.nonce, input_share)
            .unwrap_or_else(|e| err!(test_num, e, "prep state init"));
        states.push(state);
        prep_shares.push(prep_share);
    }

    assert_eq!(1, t.prep_shares.len(), "#{}", test_num);
    for (i, want) in t.prep_shares[0].iter().enumerate() {
        assert_eq!(
            prep_shares[i],
            Prio3PrepareShare::get_decoded_with_param(&states[i], want.as_ref())
                .unwrap_or_else(|e| err!(test_num, e, "decode test vector (prep share)")),
            "#{}",
            test_num
        );
        assert_eq!(prep_shares[i].get_encoded(), want.as_ref(), "#{}", test_num);
    }

    let inbound = prio3
        .prepare_preprocess(prep_shares)
        .unwrap_or_else(|e| err!(test_num, e, "prep preprocess"));

    let mut out_shares = Vec::new();
    for state in states.iter_mut() {
        match prio3.prepare_step(state.clone(), inbound.clone()).unwrap() {
            PrepareTransition::Finish(out_share) => {
                out_shares.push(out_share);
            }
            _ => panic!("unexpected transition"),
        }
    }

    for (got, want) in out_shares.iter().zip(t.out_shares.iter()) {
        let got: Vec<M> = got.as_ref().iter().map(|x| M::from(*x)).collect();
        assert_eq!(&got, want);
    }
}

#[test]
fn test_vec_prio3_count() {
    let t: TPrio3<u64> =
        serde_json::from_str(include_str!("testdata/vdaf_00_prio3_count.json")).unwrap();
    let prio3 = Prio3Aes128Count::new(2).unwrap();

    for (test_num, p) in t.prep.iter().enumerate() {
        check_prep_test_vec(&prio3, test_num, p);
    }
}

#[test]
fn test_vec_prio3_sum() {
    let t: TPrio3<u128> =
        serde_json::from_str(include_str!("testdata/vdaf_00_prio3_sum.json")).unwrap();
    let prio3 = Prio3Aes128Sum::new(2, 8).unwrap();

    for (test_num, p) in t.prep.iter().enumerate() {
        check_prep_test_vec(&prio3, test_num, p);
    }
}

#[test]
fn test_vec_prio3_histogram() {
    let t: TPrio3<u128> =
        serde_json::from_str(include_str!("testdata/vdaf_00_prio3_histogram.json")).unwrap();
    let prio3 = Prio3Aes128Histogram::new(2, &[1, 10, 100]).unwrap();

    for (test_num, p) in t.prep.iter().enumerate() {
        check_prep_test_vec(&prio3, test_num, p);
    }
}
