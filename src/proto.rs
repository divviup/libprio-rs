// SPDX-License-Identifier: MPL-2.0

//! Implementation of the Prio protocol.
//!
//! TODO: Decide how to authenticate the leader in leader<->helper communication.

use crate::field::FieldElement;
use crate::pcp::gadgets::MeanVarUnsigned;
use crate::pcp::types::MeanVarUnsignedVector;
use crate::pcp::{Gadget, Proof, Value, Verifier};

use rand::Rng;
use std::marker::PhantomData;

/// An HPKE public key.
pub struct HpkePublicKey {
    // XXX
}

/// An HPKE secret key.
pub struct HpkeSecretKey {
    // XXX
}

/// Code points for pseudorandom generators used for deriving pseudorandom field elements.
#[derive(Clone, Copy)]
pub enum Prg {
    /// AES-128 in CTR mode.
    ///
    /// XXX Figure out how `prio::prng` sets the IV.
    Aes128Ctr,
}

/// Code points for finite fields.
pub enum Field {
    /// The 64-bit field implemented by `prio::field::Field64`.
    Field64,
}

/// Parameters for the Prio protocol.
pub struct PrioParam {
    /// The type of data being collected.
    pub data_type: PrioDataType,

    /// The field used to encode client inputs.
    pub field: Field,

    /// The PRG used to generate pseudorandom field elements.
    pub prg: Prg,
}

/// Data types.
pub enum PrioDataType {
    /// Corresponds to `prio;:pcp::types::MeanVarUnsignedVector`.
    MeanVarUnsignedVector {
        /// Length in bits of each integer in the vector.
        bits: usize,
        /// Length of the vector.
        length: usize,
    },
}

/// Errors emitted by the message processors.
#[derive(Debug)]
pub enum PrioError {
    // XXX
}

/// Message processor for a Prio client.
pub trait PrioClient<'a, F, G, V>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
    Self: Sized,
{
    /// Returns an instance of a client message processor. `pk` is the helper's public key.
    fn new(param: &PrioParam, pk: &'a HpkePublicKey) -> Result<Self, PrioError>;

    /// Generates the aggregator shares for input `inp`.
    fn upload(
        &self,
        leader_share: &mut Vec<u8>,
        helper_share: &mut Vec<u8>,
        joint_rand_seed: &[u8],
        inp: &V,
    ) -> Result<(), PrioError> {
        panic!("TODO");
    }
}

/// Message processor for a Prio leader.
pub trait PrioLeader<F, G, V>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
    Self: Sized,
{
    /// Returns an instance of a leader message processor.
    fn new(param: &PrioParam, leader_share: &[u8]) -> Result<Self, PrioError>;

    /// Produces the verify request. This message contains the following fields:
    /// * A seed for deriving joint_rand (`joint_rand_seed`)
    /// * A seed for deriving query_rand
    /// * The leader's verifier share
    fn verify_start(&mut self, req: &mut Vec<u8>, joint_rand_seed: &[u8]) -> Result<(), PrioError> {
        panic!("TODO");
    }

    /// Consumes the verify response sent by the helper. This message contains the helper's
    /// verifier share. If the input is valid, then this call returns the leader's share of the
    /// input.
    fn verify_finish(&self, resp: &[u8]) -> Result<V, PrioError> {
        panic!("TODO");
    }
}

/// Message processor for a Prio helper.
pub trait PrioHelper<'a, F, G, V>
where
    F: FieldElement,
    G: Gadget<F>,
    V: Value<F, G>,
    Self: Sized,
{
    /// Returns an instance of a helper message processor. `sk` is the helper's secret key.
    fn new(
        param: &PrioParam,
        sk: &'a HpkeSecretKey,
        helper_share: &[u8],
    ) -> Result<Self, PrioError>;

    /// Consumes the verify request sent by the leader and produces the response.
    fn verify(&self, resp: &mut Vec<u8>, req: &[u8]) -> Result<V, PrioError> {
        panic!("TODO");
    }
}

/// The client message processor for type `prio::pcp::types::MeanVarUnsignedVector`
pub struct MeanVarUnsignedVectorClient<'a, F: FieldElement> {
    phantom: PhantomData<F>,
    pk: &'a HpkePublicKey,
    prg: Prg,
}

impl<'a, F: FieldElement> PrioClient<'a, F, MeanVarUnsigned<F>, MeanVarUnsignedVector<F>>
    for MeanVarUnsignedVectorClient<'a, F>
{
    fn new(param: &PrioParam, pk: &'a HpkePublicKey) -> Result<Self, PrioError> {
        match param.data_type {
            PrioDataType::MeanVarUnsignedVector { bits, length } => Ok(Self {
                phantom: PhantomData::<F>,
                pk,
                prg: param.prg,
            }),
        }
    }
}

pub struct MeanVarUnsignedVectorLeader<F: FieldElement> {
    phantom: PhantomData<F>,
    bits: usize,
    length: usize,
}

impl<F: FieldElement> PrioLeader<F, MeanVarUnsigned<F>, MeanVarUnsignedVector<F>>
    for MeanVarUnsignedVectorLeader<F>
{
    fn new(param: &PrioParam, leader_share: &[u8]) -> Result<Self, PrioError> {
        match param.data_type {
            PrioDataType::MeanVarUnsignedVector { bits, length } => Ok(Self {
                phantom: PhantomData::<F>,
                bits,
                length,
            }),
        }
    }
}

pub struct MeanVarUnsignedVectorHelper<'a, F: FieldElement> {
    phantom: PhantomData<F>,
    sk: &'a HpkeSecretKey,
    prg: Prg,
    bits: usize,
    length: usize,
}

impl<'a, F: FieldElement> PrioHelper<'a, F, MeanVarUnsigned<F>, MeanVarUnsignedVector<F>>
    for MeanVarUnsignedVectorHelper<'a, F>
{
    fn new(
        param: &PrioParam,
        sk: &'a HpkeSecretKey,
        helper_share: &[u8],
    ) -> Result<Self, PrioError> {
        match param.data_type {
            PrioDataType::MeanVarUnsignedVector { bits, length } => Ok(Self {
                phantom: PhantomData::<F>,
                sk,
                prg: param.prg,
                bits,
                length,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::Field64 as F;

    #[test]
    fn test_upload_verify() {
        let buf_len = 1024;

        let pk = HpkePublicKey {}; // XXX
        let sk = HpkeSecretKey {}; // XXX

        let bits = 12;
        let measurement = [23, 42, 99, 0, 1, 2, 6, 1337];

        let param = PrioParam {
            data_type: PrioDataType::MeanVarUnsignedVector {
                bits,
                length: measurement.len(),
            },
            field: Field::Field64,
            prg: Prg::Aes128Ctr,
        };

        let mut joint_rand_seed = Vec::<u8>::with_capacity(buf_len);
        let mut leader_share = Vec::<u8>::with_capacity(buf_len);
        let mut helper_share = Vec::<u8>::with_capacity(buf_len);
        let mut verify_req = Vec::<u8>::with_capacity(buf_len);
        let mut verify_resp = Vec::<u8>::with_capacity(buf_len);

        // Upload Start: The leader responds to an upload start request by sending the client a
        // string called `joint_rand_seed`. The aggregators will use this string to verify the
        // proof.
        let joint_rand_seed_len = match param.prg {
            Prg::Aes128Ctr => 16,
        };
        let mut rng = rand::thread_rng();
        let vals: Vec<u8> = (0..joint_rand_seed_len)
            .map(|_| rand::random::<u8>())
            .collect();

        // Upload Finish: The client generates `(leader_share, helper_share)` and uploads them to
        // the leader. (This computation uses `joint_rand_seed`.)
        let inp = MeanVarUnsignedVector::<F>::new(bits, &measurement).unwrap();
        let client = MeanVarUnsignedVectorClient::<F>::new(&param, &pk).unwrap();
        client
            .upload(&mut leader_share, &mut helper_share, &joint_rand_seed, &inp)
            .unwrap();

        // Verify Finish (we don't need to do Verify Start): The leader sends its verifier share to
        // the helper. The helper responds with its verifier share and decides if the input is
        // valid. Finally, the leader decides if the input is valid.
        //
        // leader -> helper
        let mut leader = MeanVarUnsignedVectorLeader::<F>::new(&param, &leader_share).unwrap();
        leader
            .verify_start(&mut verify_req, &joint_rand_seed)
            .unwrap();

        // helper -> leader
        let helper = MeanVarUnsignedVectorHelper::<F>::new(&param, &sk, &helper_share).unwrap();
        let valid_helper_share = helper.verify(&mut verify_resp, &verify_req).unwrap();

        // leader
        let valid_leader_share = leader.verify_finish(&verify_resp).unwrap();
    }
}
