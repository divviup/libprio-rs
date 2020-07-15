use crate::finite_field::Field;

pub struct UnpackedShare<'a> {
    pub data: &'a [Field],
    pub f0: &'a Field,
    pub g0: &'a Field,
    pub h0: &'a Field,
    pub points_h_packed: &'a [Field],
}

pub struct UnpackedShareMut<'a> {
    pub data: &'a mut [Field],
    pub f0: &'a mut Field,
    pub g0: &'a mut Field,
    pub h0: &'a mut Field,
    pub points_h_packed: &'a mut [Field],
}

pub fn share_length(dimension: usize) -> usize {
    dimension + 3 + (dimension + 1).next_power_of_two()
}

pub fn unpack_share<'a>(share: &'a [Field], dimension: usize) -> Option<UnpackedShare<'a>> {
    // check the share length
    if share.len() != share_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = share.split_at(dimension);
    let (zero_terms, points_h_packed) = rest.split_at(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedShare {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

pub fn unpack_share_mut<'a>(
    share: &'a mut [Field],
    dimension: usize,
) -> Option<UnpackedShareMut<'a>> {
    // check the share length
    if share.len() != share_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = share.split_at_mut(dimension);
    let (zero_terms, points_h_packed) = rest.split_at_mut(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedShareMut {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

#[test]
fn test_unpack_share() {
    let dim = 15;
    let len = share_length(dim);

    let mut share = vec![Field::from(0); len];
    let unpacked = unpack_share_mut(&mut share, dim).unwrap();
    *unpacked.f0 = Field::from(12);
    assert_eq!(share[dim], 12.into());
}
