// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

use prio::{
    client::Client,
    encrypt::{decrypt_share, encrypt_share, PrivateKey, PublicKey},
    field::{Field32, FieldElement},
    server::Server,
    util::{deserialize, serialize, unpack_proof_mut},
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tweak {
    None,
    WrongInput,
    DataPartOfShare,
    ZeroTermF,
    ZeroTermG,
    ZeroTermH,
    PointsH,
    VerificationF,
    VerificationG,
    VerificationH,
}

fn tweaks(tweak: Tweak) {
    let dim = 123;

    let priv_key1 = PrivateKey::from_base64(
        "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9HUC2E0w+9Rq\
         Zx3XMkdEHBHfNuCSMpOwofVSq3TfyKwn0NrftKisKKVSaTOt5seJ67P5QL4hxgPWvxw==",
    )
    .unwrap();

    let priv_key2 = PrivateKey::from_base64(
        "BNNOqoU54GPo+1gTPv+hCgA9U2ZCKd76yOMrWa1xTWgeb4LhFLMQIQoRwDVaW64g\
         /WTdcxT4rDULoycUNFB60LER6hPEHg/ObBnRPV1rwS3nj9Bj0tbjVPPyL9p8QW8B+w==",
    )
    .unwrap();

    let pub_key1 = PublicKey::from(&priv_key1);
    let pub_key2 = PublicKey::from(&priv_key2);

    let priv_key1_clone = priv_key1.clone();
    let pub_key1_clone = pub_key1.clone();

    let mut server1: Server<Field32> = Server::new(dim, true, priv_key1).unwrap();
    let mut server2: Server<Field32> = Server::new(dim, false, priv_key2).unwrap();

    let mut client_mem = Client::new(dim, pub_key1, pub_key2).unwrap();

    // all zero data
    let mut data = vec![Field32::zero(); dim];

    if let Tweak::WrongInput = tweak {
        data[0] = Field32::from(2);
    }

    let (share1_original, share2) = client_mem.encode_simple(&data).unwrap();

    let decrypted_share1 = decrypt_share(&share1_original, &priv_key1_clone).unwrap();
    let mut share1_field: Vec<Field32> = deserialize(&decrypted_share1).unwrap();
    let unpacked_share1 = unpack_proof_mut(&mut share1_field, dim).unwrap();

    let one = Field32::from(1);

    match tweak {
        Tweak::DataPartOfShare => unpacked_share1.data[0] += one,
        Tweak::ZeroTermF => *unpacked_share1.f0 += one,
        Tweak::ZeroTermG => *unpacked_share1.g0 += one,
        Tweak::ZeroTermH => *unpacked_share1.h0 += one,
        Tweak::PointsH => unpacked_share1.points_h_packed[0] += one,
        _ => (),
    };

    // reserialize altered share1
    let share1_modified = encrypt_share(&serialize(&share1_field), &pub_key1_clone).unwrap();

    let eval_at = server1.choose_eval_at();

    let mut v1 = server1
        .generate_verification_message(eval_at, &share1_modified)
        .unwrap();
    let v2 = server2
        .generate_verification_message(eval_at, &share2)
        .unwrap();

    match tweak {
        Tweak::VerificationF => v1.f_r += one,
        Tweak::VerificationG => v1.g_r += one,
        Tweak::VerificationH => v1.h_r += one,
        _ => (),
    }

    let should_be_valid = matches!(tweak, Tweak::None);
    assert_eq!(
        server1.aggregate(&share1_modified, &v1, &v2).unwrap(),
        should_be_valid
    );
    assert_eq!(
        server2.aggregate(&share2, &v1, &v2).unwrap(),
        should_be_valid
    );
}

#[test]
fn tweak_none() {
    tweaks(Tweak::None);
}

#[test]
fn tweak_input() {
    tweaks(Tweak::WrongInput);
}

#[test]
fn tweak_data() {
    tweaks(Tweak::DataPartOfShare);
}

#[test]
fn tweak_f_zero() {
    tweaks(Tweak::ZeroTermF);
}

#[test]
fn tweak_g_zero() {
    tweaks(Tweak::ZeroTermG);
}

#[test]
fn tweak_h_zero() {
    tweaks(Tweak::ZeroTermH);
}

#[test]
fn tweak_h_points() {
    tweaks(Tweak::PointsH);
}

#[test]
fn tweak_f_verif() {
    tweaks(Tweak::VerificationF);
}

#[test]
fn tweak_g_verif() {
    tweaks(Tweak::VerificationG);
}

#[test]
fn tweak_h_verif() {
    tweaks(Tweak::VerificationH);
}
