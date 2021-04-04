// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Functions for polynomial interpolation and evaluation

use crate::field::FieldElement;

use std::convert::TryFrom;

/// Temporary memory used for FFT
#[derive(Debug)]
pub struct PolyFFTTempMemory<F: FieldElement> {
    fft_tmp: Vec<F>,
    fft_y_sub: Vec<F>,
    fft_roots_sub: Vec<F>,
}

impl<F: FieldElement> PolyFFTTempMemory<F> {
    fn new(length: usize) -> Self {
        PolyFFTTempMemory {
            fft_tmp: vec![F::zero(); length],
            fft_y_sub: vec![F::zero(); length],
            fft_roots_sub: vec![F::zero(); length],
        }
    }
}

/// Auxiliary memory for polynomial interpolation and evaluation
#[derive(Debug)]
pub struct PolyAuxMemory<F: FieldElement> {
    pub roots_2n: Vec<F>,
    pub roots_2n_inverted: Vec<F>,
    pub roots_n: Vec<F>,
    pub roots_n_inverted: Vec<F>,
    pub coeffs: Vec<F>,
    pub fft_memory: PolyFFTTempMemory<F>,
}

impl<F: FieldElement> PolyAuxMemory<F> {
    pub fn new(n: usize) -> Self {
        PolyAuxMemory {
            roots_2n: fft_get_roots(2 * n, false),
            roots_2n_inverted: fft_get_roots(2 * n, true),
            roots_n: fft_get_roots(n, false),
            roots_n_inverted: fft_get_roots(n, true),
            coeffs: vec![F::zero(); 2 * n],
            fft_memory: PolyFFTTempMemory::new(2 * n),
        }
    }
}

fn fft_recurse<F: FieldElement>(
    out: &mut [F],
    n: usize,
    roots: &[F],
    ys: &[F],
    tmp: &mut [F],
    y_sub: &mut [F],
    roots_sub: &mut [F],
) {
    if n == 1 {
        out[0] = ys[0];
        return;
    }

    let half_n = n / 2;

    let (mut tmp_first, mut tmp_second) = tmp.split_at_mut(half_n);
    let (y_sub_first, mut y_sub_second) = y_sub.split_at_mut(half_n);
    let (roots_sub_first, mut roots_sub_second) = roots_sub.split_at_mut(half_n);

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
        y_sub_first[i] *= roots[i];
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

/// Calculate `count` number of roots of unity of order `count`
fn fft_get_roots<F: FieldElement>(count: usize, invert: bool) -> Vec<F> {
    let mut roots = vec![F::zero(); count];
    let mut gen = F::generator();
    if invert {
        gen = gen.inv();
    }

    roots[0] = F::one();
    let step_size = F::generator_order() / F::Integer::try_from(count).unwrap();
    // generator for subgroup of order count
    gen = gen.pow(step_size);

    roots[1] = gen;

    for i in 2..count {
        roots[i] = gen * roots[i - 1];
    }

    roots
}

fn fft_interpolate_raw<F: FieldElement>(
    out: &mut [F],
    ys: &[F],
    n_points: usize,
    roots: &[F],
    invert: bool,
    mem: &mut PolyFFTTempMemory<F>,
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
        let n_inverse = F::from(F::Integer::try_from(n_points).unwrap()).inv();
        for i in 0..n_points {
            out[i] *= n_inverse;
        }
    }
}

pub fn poly_fft<F: FieldElement>(
    points_out: &mut [F],
    points_in: &[F],
    scaled_roots: &[F],
    n_points: usize,
    invert: bool,
    mem: &mut PolyFFTTempMemory<F>,
) {
    fft_interpolate_raw(points_out, points_in, n_points, scaled_roots, invert, mem)
}

pub fn poly_horner_eval<F: FieldElement>(poly: &[F], eval_at: F, len: usize) -> F {
    let mut result = poly[len - 1];

    for i in (0..(len - 1)).rev() {
        result *= eval_at;
        result += poly[i];
    }

    result
}

pub fn poly_interpret_eval<F: FieldElement>(
    points: &[F],
    roots: &[F],
    eval_at: F,
    tmp_coeffs: &mut [F],
    fft_memory: &mut PolyFFTTempMemory<F>,
) -> F {
    poly_fft(tmp_coeffs, points, roots, points.len(), true, fft_memory);
    poly_horner_eval(&tmp_coeffs, eval_at, points.len())
}

#[test]
fn test_roots() {
    use crate::field::Field32;

    let count = 128;
    let roots = fft_get_roots::<Field32>(count, false);
    let roots_inv = fft_get_roots::<Field32>(count, true);

    for i in 0..count {
        assert_eq!(roots[i] * roots_inv[i], 1);
        assert_eq!(roots[i].pow(u32::try_from(count).unwrap()), 1);
        assert_eq!(roots_inv[i].pow(u32::try_from(count).unwrap()), 1);
    }
}

#[test]
fn test_horner_eval() {
    use crate::field::Field32;

    let mut poly = vec![Field32::from(0); 4];
    poly[0] = 2.into();
    poly[1] = 1.into();
    poly[2] = 5.into();
    // 5*3^2 + 3 + 2 = 50
    assert_eq!(poly_horner_eval(&poly, 3.into(), 3), 50);
    poly[3] = 4.into();
    // 4*3^3 + 5*3^2 + 3 + 2 = 158
    assert_eq!(poly_horner_eval(&poly, 3.into(), 4), 158);
}

#[test]
fn test_fft() {
    use crate::field::Field32;

    use rand::prelude::*;
    use std::convert::TryFrom;

    let count = 128;
    let mut mem = PolyAuxMemory::new(count / 2);

    let mut poly = vec![Field32::from(0); count];
    let mut points2 = vec![Field32::from(0); count];

    let points = (0..count)
        .into_iter()
        .map(|_| Field32::from(random::<u32>()))
        .collect::<Vec<Field32>>();

    // From points to coeffs and back
    poly_fft(
        &mut poly,
        &points,
        &mem.roots_2n,
        count,
        false,
        &mut mem.fft_memory,
    );
    poly_fft(
        &mut points2,
        &poly,
        &mem.roots_2n_inverted,
        count,
        true,
        &mut mem.fft_memory,
    );

    assert_eq!(points, points2);

    // interpolation
    poly_fft(
        &mut poly,
        &points,
        &mem.roots_2n,
        count,
        false,
        &mut mem.fft_memory,
    );
    for i in 0..count {
        let mut should_be = Field32::from(0);
        for j in 0..count {
            should_be = mem.roots_2n[i].pow(u32::try_from(j).unwrap()) * points[j] + should_be;
        }
        assert_eq!(should_be, poly[i]);
    }
}
