use crate::finite_field::Field;

pub fn vector_with_length(len: usize) -> Vec<Field> {
    vec![Field::from(0); len]
}

pub fn share_length(dimension: usize) -> usize {
    dimension + 3 + (dimension + 1).next_power_of_two()
}

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

pub fn unpack_share(share: &[Field], dimension: usize) -> Option<UnpackedShare> {
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

pub fn unpack_share_mut(share: &mut [Field], dimension: usize) -> Option<UnpackedShareMut> {
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

pub fn secret_share(share: &mut [Field]) -> Vec<Field> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut random = vec![0u32; share.len()];
    let mut share2 = vector_with_length(share.len());

    rng.fill(&mut random[..]);

    for (r, f) in random.iter().zip(share2.iter_mut()) {
        *f = Field::from(*r);
    }

    for (f1, f2) in share.iter_mut().zip(share2.iter()) {
        *f1 -= *f2;
    }

    share2
}

pub fn reconstruct_shares(share1: &[Field], share2: &[Field]) -> Option<Vec<Field>> {
    if share1.len() != share2.len() {
        return None;
    }

    let mut reconstructed = vector_with_length(share1.len());

    for (r, (s1, s2)) in reconstructed
        .iter_mut()
        .zip(share1.iter().zip(share2.iter()))
    {
        *r = *s1 + *s2;
    }

    Some(reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_unpack_share() {
        let dim = 15;
        let len = share_length(dim);

        let mut share = vec![Field::from(0); len];
        let unpacked = unpack_share_mut(&mut share, dim).unwrap();
        *unpacked.f0 = Field::from(12);
        assert_eq!(share[dim], 12);
    }

    #[test]
    fn secret_sharing() {
        let mut share1 = vector_with_length(10);
        share1[3] = 21.into();
        share1[8] = 123.into();

        let original_data = share1.clone();

        let share2 = secret_share(&mut share1);

        let reconstructed = reconstruct_shares(&share1, &share2).unwrap();
        assert_eq!(reconstructed, original_data);
    }
}
