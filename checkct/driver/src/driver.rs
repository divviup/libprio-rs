use core::{
    alloc::{GlobalAlloc, Layout},
    cell::UnsafeCell,
    hint::black_box,
    ptr::null_mut,
    sync::atomic::{AtomicUsize, Ordering},
};

use rand::RngExt;

use prio::field::{Field64, Field128, Field255, FieldElementWithInteger};

use crate::rng::{PrivateRng, PublicRng};
use checkct_macros::checkct;

static SIZE: usize = 1024 * 1024;

#[repr(C, align(4096))]
struct StubAllocator {
    memory: UnsafeCell<[u8; SIZE]>,
    next: AtomicUsize,
}

#[global_allocator]
static ALLOCATOR: StubAllocator = StubAllocator {
    memory: UnsafeCell::new([0; SIZE]),
    next: AtomicUsize::new(0),
};

unsafe impl Sync for StubAllocator {}

unsafe impl GlobalAlloc for StubAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        if align > 4096 {
            return null_mut();
        }

        let Ok(previous) =
            self.next
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |mut next| {
                    if next % align != 0 {
                        let remainder = align - next % align;
                        next = next.checked_add(remainder)?;
                    }

                    next = next.checked_add(size)?;

                    if next > SIZE {
                        return None;
                    }

                    Some(next)
                })
        else {
            return null_mut();
        };

        unsafe { self.memory.get().cast::<u8>().add(previous) }
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[checkct]
pub fn field64_multiply() {
    let a: Field64 = PrivateRng.random();
    let b: Field64 = PrivateRng.random();
    black_box(a * b);
}

#[checkct]
pub fn field128_multiply() {
    let a: Field128 = PrivateRng.random();
    let b: Field128 = PrivateRng.random();
    black_box(a * b);
}

#[checkct]
pub fn field255_multiply() {
    let a: Field255 = PrivateRng.random();
    let b: Field255 = PrivateRng.random();
    black_box(a * b);
}

#[checkct]
pub fn field64_add() {
    let a: Field64 = PrivateRng.random();
    let b: Field64 = PrivateRng.random();
    black_box(a + b);
}

#[checkct]
pub fn field128_add() {
    let a: Field128 = PrivateRng.random();
    let b: Field128 = PrivateRng.random();
    black_box(a + b);
}

#[checkct]
pub fn field255_add() {
    let a: Field255 = PrivateRng.random();
    let b: Field255 = PrivateRng.random();
    black_box(a + b);
}

#[checkct]
pub fn field64_sub() {
    let a: Field64 = PrivateRng.random();
    let b: Field64 = PrivateRng.random();
    black_box(a - b);
}

#[checkct]
pub fn field128_sub() {
    let a: Field128 = PrivateRng.random();
    let b: Field128 = PrivateRng.random();
    black_box(a - b);
}

#[checkct]
pub fn field255_sub() {
    let a: Field255 = PrivateRng.random();
    let b: Field255 = PrivateRng.random();
    black_box(a - b);
}

#[checkct]
pub fn field64_pow() {
    let a: Field64 = PrivateRng.random();
    let b = PublicRng.random();
    black_box(a.pow(b));
}

#[checkct]
pub fn field128_pow() {
    let a: Field128 = PrivateRng.random();
    let b = PublicRng.random();
    black_box(a.pow(b));
}
