// SPDX-License-Identifier: MPL-2.0

use prio::client::*;
use prio::encrypt::*;
use prio::finite_field::*;
use prio::server::*;

fn main() {
    let pub_key = PublicKey::from_base64(
        "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9\
        HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVQ=",
    )
    .unwrap();
    let priv_key = PrivateKey::from_base64(
        "BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgN\
        t9HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVSq3TfyKwn0NrftKisKKVSaTOt5seJ67P5QL4hxgPWvxw==",
    )
    .unwrap();

    let dim = 8;

    let mut client1 = Client::new(dim, pub_key.clone(), pub_key.clone()).unwrap();
    let mut client2 = Client::new(dim, pub_key.clone(), pub_key.clone()).unwrap();

    let data1_u32 = [0, 0, 1, 0, 0, 0, 0, 0];
    println!("Client 1 Input: {:?}", data1_u32);

    let data1 = data1_u32
        .iter()
        .map(|x| Field::from(*x))
        .collect::<Vec<Field>>();

    let data2_u32 = [0, 0, 1, 0, 0, 0, 0, 0];
    println!("Client 2 Input: {:?}", data2_u32);

    let data2 = data2_u32
        .iter()
        .map(|x| Field::from(*x))
        .collect::<Vec<Field>>();

    let (share1_1, share1_2) = client1.encode_simple(&data1).unwrap();
    let (share2_1, share2_2) = client2.encode_simple(&data2).unwrap();
    let eval_at = Field::from(12313);

    let mut server1 = Server::new(dim, true, priv_key.clone());
    let mut server2 = Server::new(dim, false, priv_key.clone());

    let v1_1 = server1
        .generate_verification_message(eval_at, &share1_1)
        .unwrap();
    let v1_2 = server2
        .generate_verification_message(eval_at, &share1_2)
        .unwrap();

    let v2_1 = server1
        .generate_verification_message(eval_at, &share2_1)
        .unwrap();
    let v2_2 = server2
        .generate_verification_message(eval_at, &share2_2)
        .unwrap();

    let _ = server1.aggregate(&share1_1, &v1_1, &v1_2).unwrap();
    let _ = server2.aggregate(&share1_2, &v1_1, &v1_2).unwrap();

    let _ = server1.aggregate(&share2_1, &v2_1, &v2_2).unwrap();
    let _ = server2.aggregate(&share2_2, &v2_1, &v2_2).unwrap();

    server1.merge_total_shares(server2.total_shares());
    println!("Final Publication: {:?}", server1.total_shares());
}
