use libprio_rs::client::*;
use libprio_rs::finite_field::Field;
use libprio_rs::server::*;
use libprio_rs::util::*;

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

    let mut server1 = Server::new(dim, true);
    let mut server2 = Server::new(dim, false);

    let mut client_mem = ClientMemory::new(dim).unwrap();

    // all zero data
    let mut data = vector_with_length(dim);

    if let Tweak::WrongInput = tweak {
        data[0] = Field::from(2);
    }

    let (mut share1, share2) = client_mem.encode_simple(&data);

    let unpacked_share1 = unpack_share_mut(&mut share1, dim).unwrap();

    let one = Field::from(1);

    match tweak {
        Tweak::DataPartOfShare => unpacked_share1.data[0] += one,
        Tweak::ZeroTermF => *unpacked_share1.f0 += one,
        Tweak::ZeroTermG => *unpacked_share1.g0 += one,
        Tweak::ZeroTermH => *unpacked_share1.h0 += one,
        Tweak::PointsH => unpacked_share1.points_h_packed[0] += one,
        _ => (),
    };

    let eval_at = server1.choose_eval_at();

    let mut v1 = server1
        .generate_verification_message(eval_at, &share1)
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

    let should_be_valid = match tweak {
        Tweak::None => true,
        _ => false,
    };
    assert_eq!(server1.aggregate(&share1, &v1, &v2), should_be_valid);
    assert_eq!(server2.aggregate(&share2, &v1, &v2), should_be_valid);
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
