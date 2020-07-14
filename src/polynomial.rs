use crate::finite_field::*;

pub struct PolyFFTTempMemory {
    fft_tmp: Vec<Field>,
    fft_y_sub: Vec<Field>,
    fft_roots_sub: Vec<Field>,
}

impl PolyFFTTempMemory {
    fn new(point_count: usize) -> Self {
        PolyFFTTempMemory {
            fft_tmp: vector_with_length(point_count),
            fft_y_sub: vector_with_length(point_count),
            fft_roots_sub: vector_with_length(point_count),
        }
    }
}

pub struct PolyTempMemory {
    pub roots: Vec<Field>,
    pub roots_inverted: Vec<Field>,
    pub roots_half: Vec<Field>,
    pub roots_half_inverted: Vec<Field>,
    pub coeffs: Vec<Field>,
    pub fft_memory: PolyFFTTempMemory,
}

impl PolyTempMemory {
    pub fn new(point_count: usize) -> Self {
        PolyTempMemory {
            roots: fft_get_roots(point_count, false),
            roots_inverted: fft_get_roots(point_count, true),
            roots_half: fft_get_roots(point_count / 2, false),
            roots_half_inverted: fft_get_roots(point_count / 2, true),
            coeffs: vector_with_length(point_count),
            fft_memory: PolyFFTTempMemory::new(point_count),
        }
    }
}

pub fn vector_with_length(len: usize) -> Vec<Field> {
    vec![Field::from(0); len]
}

fn fft_recurse(
    out: &mut [Field],
    n: usize,
    roots: &[Field],
    ys: &[Field],
    tmp: &mut [Field],
    y_sub: &mut [Field],
    roots_sub: &mut [Field],
) {
    if n == 1 {
        out[0] = ys[0];
        return;
    }

    let half_n = n / 2;

    let (mut tmp_first, mut tmp_second) = tmp.split_at_mut(half_n);
    let (mut y_sub_first, mut y_sub_second) = y_sub.split_at_mut(half_n);
    let (mut roots_sub_first, mut roots_sub_second) = roots_sub.split_at_mut(half_n);

    // Recurse on the first half
    for i in 0..half_n {
        y_sub_first[i] = ys[i] + ys[i + half_n];
        roots_sub_first[i] = roots[2 * i];
    }
    fft_recurse(
        &mut tmp_first,
        half_n,
        roots_sub_first,
        y_sub_first,
        &mut tmp_second,
        &mut y_sub_second,
        &mut roots_sub_second,
    );
    for i in 0..half_n {
        out[2 * i] = tmp_first[i];
    }

    // Recurse on the second half
    for i in 0..half_n {
        y_sub_first[i] = ys[i] - ys[i + half_n];
        y_sub_first[i] = y_sub_first[i] * roots[i];
    }
    fft_recurse(
        &mut tmp_first,
        half_n,
        roots_sub_first,
        y_sub_first,
        &mut tmp_second,
        &mut y_sub_second,
        &mut roots_sub_second,
    );
    for i in 0..half_n {
        out[2 * i + 1] = tmp[i];
    }
}

fn fft_get_roots(count: usize, invert: bool) -> Vec<Field> {
    let mut roots = vec![Field::from(0); count];
    let mut gen: Field = GENERATOR.into();
    if invert {
        gen = gen.inv();
    }

    roots[0] = 1.into();
    let step_size: u32 = N_ROOTS / (count as u32);
    // generator for subgroup of order count
    gen = gen.pow(step_size.into());

    roots[1] = gen;

    for i in 2..count {
        roots[i] = gen * roots[i - 1];
    }

    roots
}

fn fft_interpolate_raw(
    out: &mut [Field],
    ys: &[Field],
    n_points: usize,
    roots: &[Field],
    invert: bool,
    mem: &mut PolyFFTTempMemory,
) {
    fft_recurse(
        out,
        n_points,
        roots,
        ys,
        &mut mem.fft_tmp,
        &mut mem.fft_y_sub,
        &mut mem.fft_roots_sub,
    );
    if invert {
        let n_inverse = Field::from(n_points as u32).inv();
        for i in 0..n_points {
            out[i] = out[i] * n_inverse;
        }
    }
}

pub fn poly_fft(
    points_out: &mut [Field],
    points_in: &[Field],
    scaled_roots: &[Field],
    n_points: usize,
    invert: bool,
    mem: &mut PolyFFTTempMemory,
) {
    fft_interpolate_raw(points_out, points_in, n_points, scaled_roots, invert, mem)
}

pub fn poly_horner_eval(poly: &[Field], eval_at: Field) -> Field {
    let mut result = poly[poly.len() - 1];

    for i in (0..(poly.len() - 1)).rev() {
        result = result * eval_at;
        result = result + poly[i];
    }

    result
}

pub fn poly_interpret_eval(
    points: &[Field],
    roots: &[Field],
    eval_at: Field,
    mut mem: PolyTempMemory,
) {
    poly_fft(
        &mut mem.coeffs,
        points,
        roots,
        points.len(),
        true,
        &mut mem.fft_memory,
    );
}

#[test]
fn test_roots() {
    let count = 128;
    let roots = fft_get_roots(count, false);
    let roots_inv = fft_get_roots(count, true);

    for i in 0..count {
        assert_eq!(roots[i] * roots_inv[i], 1.into());
        assert_eq!(roots[i].pow(Field::from(count as u32)), 1.into());
        assert_eq!(roots_inv[i].pow(Field::from(count as u32)), 1.into());
    }
}

#[test]
fn test_horner_eval() {
    let mut poly = vec![Field::from(0); 4];
    poly[0] = 2.into();
    poly[1] = 1.into();
    poly[2] = 5.into();
    // 5*3^2 + 3 + 2 = 50
    assert_eq!(poly_horner_eval(&poly, 3.into()), 50.into());
    poly[3] = 4.into();
    // 4*3^3 + 5*3^2 + 3 + 2 = 158
    assert_eq!(poly_horner_eval(&poly, 3.into()), 158.into());
}

#[test]
fn test_fft() {
    let count = 128;
    let mut mem = PolyTempMemory::new(count);

    use rand::prelude::*;

    let mut poly = vec![Field::from(0); count];
    let mut points2 = vec![Field::from(0); count];

    let points = (0..count)
        .into_iter()
        .map(|_| Field::from(random::<u32>()))
        .collect::<Vec<Field>>();

    // From points to coeffs and back
    poly_fft(
        &mut poly,
        &points,
        &mem.roots,
        count,
        false,
        &mut mem.fft_memory,
    );
    poly_fft(
        &mut points2,
        &poly,
        &mem.roots_inverted,
        count,
        true,
        &mut mem.fft_memory,
    );

    assert_eq!(points, points2);

    // simple
    poly_fft(
        &mut poly,
        &points,
        &mem.roots,
        count,
        false,
        &mut mem.fft_memory,
    );
    for i in 0..count {
        let mut should_be = Field::from(0);
        for j in 0..count {
            should_be = mem.roots[i].pow(Field::from(j as u32)) * points[j] + should_be;
        }
        assert_eq!(should_be, poly[i]);
    }
}
