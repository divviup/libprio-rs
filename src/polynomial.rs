// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Functions for polynomial interpolation and evaluation

use crate::finite_field::*;
use crate::util::*;

/// Temporary memory used for FFT
#[derive(Debug)]
pub struct PolyFFTTempMemory {
    fft_tmp: Vec<Field>,
    fft_y_sub: Vec<Field>,
    fft_roots_sub: Vec<Field>,
}

impl PolyFFTTempMemory {
    fn new(length: usize) -> Self {
        PolyFFTTempMemory {
            fft_tmp: vector_with_length(length),
            fft_y_sub: vector_with_length(length),
            fft_roots_sub: vector_with_length(length),
        }
    }
}

/// Auxiliary memory for polynomial interpolation and evaluation
#[derive(Debug)]
pub struct PolyAuxMemory {
    pub roots_2n: Vec<Field>,
    pub roots_2n_inverted: Vec<Field>,
    pub roots_n: Vec<Field>,
    pub roots_n_inverted: Vec<Field>,
    pub coeffs: Vec<Field>,
    pub fft_memory: PolyFFTTempMemory,
}

impl PolyAuxMemory {
    pub fn new(n: usize) -> Self {
        PolyAuxMemory {
            roots_2n: fft_get_roots(2 * n, false),
            roots_2n_inverted: fft_get_roots(2 * n, true),
            roots_n: fft_get_roots(n, false),
            roots_n_inverted: fft_get_roots(n, true),
            coeffs: vector_with_length(2 * n),
            fft_memory: PolyFFTTempMemory::new(2 * n),
        }
    }
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
fn fft_get_roots(count: usize, invert: bool) -> Vec<Field> {
    let mut roots = vec![Field::from(0); count];
    let mut gen = Field::generator();
    if invert {
        gen = gen.inv();
    }

    roots[0] = 1.into();
    let step_size: u32 = (Field::generator_order() as u32) / (count as u32);
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
            out[i] *= n_inverse;
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

pub fn poly_horner_eval(poly: &[Field], eval_at: Field, len: usize) -> Field {
    let mut result = poly[len - 1];

    for i in (0..(len - 1)).rev() {
        result *= eval_at;
        result += poly[i];
    }

    result
}

pub fn poly_interpret_eval(
    points: &[Field],
    roots: &[Field],
    eval_at: Field,
    tmp_coeffs: &mut [Field],
    fft_memory: &mut PolyFFTTempMemory,
) -> Field {
    poly_fft(tmp_coeffs, points, roots, points.len(), true, fft_memory);
    poly_horner_eval(&tmp_coeffs, eval_at, points.len())
}

#[test]
fn test_roots() {
    let count = 128;
    let roots = fft_get_roots(count, false);
    let roots_inv = fft_get_roots(count, true);

    for i in 0..count {
        assert_eq!(roots[i] * roots_inv[i], 1);
        assert_eq!(roots[i].pow(Field::from(count as u32)), 1);
        assert_eq!(roots_inv[i].pow(Field::from(count as u32)), 1);
    }
}

#[test]
fn test_horner_eval() {
    let mut poly = vec![Field::from(0); 4];
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
    let count = 128;
    let mut mem = PolyAuxMemory::new(count / 2);

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
        let mut should_be = Field::from(0);
        for j in 0..count {
            should_be = mem.roots_2n[i].pow(Field::from(j as u32)) * points[j] + should_be;
        }
        assert_eq!(should_be, poly[i]);
    }
}
