// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Slice C FFI

extern crate libc;
use libc::size_t;
use std::slice;

///Wrapper for a u8 slice that can be accessed from C FFI
pub struct Cu8Vec {
    ptr: *mut [u8],
}

impl Cu8Vec {
    //Create a u8 slice for C
    fn new(data: *mut [u8]) -> Cu8Vec {
         Cu8Vec {
            ptr: data
        }
    }
}

///Send a C accessible slice
pub fn send_c_u8_vec_new(data: Vec<u8>) -> *mut Cu8Vec {
    let boxed_array: Box<[u8]> = data.into_boxed_slice();
    let boxed_data = Box::into_raw(boxed_array);
    Box::into_raw(Box::new(Cu8Vec::new(boxed_data)))
}

///Free a C accessible slice
#[no_mangle]
pub extern "C" fn c_u8_vec_free(ptr: *mut Cu8Vec) {
    if ptr.is_null() {
        return;
    }
    unsafe {
    	Box::from_raw((*ptr).ptr);
        Box::from_raw(ptr);
    }
}

///Fill from a C accesible slice
#[no_mangle]
pub extern "C" fn c_u8_vec_fill_buffer(ptr: *const Cu8Vec, buffer: *mut u8) {
    unsafe  {
    	let c_slice = slice::from_raw_parts_mut(buffer, (*ptr).ptr.as_ref().unwrap().len());
	c_slice.copy_from_slice((*ptr).ptr.as_ref().unwrap());
     }
}

///Get the length of a C accesible slice
#[no_mangle]
pub extern "C" fn c_u8_vec_length(ptr: *const Cu8Vec)-> size_t {
    unsafe {
        (*ptr).ptr.as_ref().unwrap().len()
    }
}

///Wrapper for a u32 slice that can be accessed from C FFI
pub struct Cu32Vec {
    ptr: *mut [u32],
}

impl Cu32Vec {
    //Create a u32 slice for C
    fn new(data: *mut [u32]) -> Cu32Vec {
         Cu32Vec {
            ptr: data
        }
    }
}

///Send a C accessible slice
pub fn send_c_u32_vec_new(data: Vec<u32>) -> *mut Cu32Vec {
    let boxed_array: Box<[u32]> = data.into_boxed_slice();
    let boxed_data = Box::into_raw(boxed_array);
    Box::into_raw(Box::new(Cu32Vec::new(boxed_data)))
}

///Free a C accessible slice
#[no_mangle]
pub extern "C" fn c_u32_vec_free(ptr: *mut Cu32Vec) {
    if ptr.is_null() {
        return;
    }
    unsafe {
    	Box::from_raw((*ptr).ptr);
        Box::from_raw(ptr);
    }
}

///Fill from a C accesible slice
#[no_mangle]
pub extern "C" fn c_u32_vec_fill_buffer(ptr: *const Cu32Vec, buffer: *mut u32) {
    unsafe  {
    	let c_slice = slice::from_raw_parts_mut(buffer, (*ptr).ptr.as_ref().unwrap().len());
	c_slice.copy_from_slice((*ptr).ptr.as_ref().unwrap());
     }
}

///Get the length of a C accesible slice
#[no_mangle]
pub extern "C" fn c_u32_vec_length(ptr: *const Cu32Vec)-> size_t {
    unsafe {
        (*ptr).ptr.as_ref().unwrap().len()
    }
}

///Wrapper for two u8 slices that can be accessed from C FFI
pub struct Cu8VecPair {
    ptr1: *mut [u8],
    ptr2: *mut [u8]
}

impl Cu8VecPair {
    //Create a u8 slice pair for C
    fn new(data1: *mut [u8], data2: *mut [u8]) -> Cu8VecPair {
         Cu8VecPair {
            ptr1: data1,
	    ptr2: data2
        }
    }
}

///Send a C accessible slice pair
pub fn send_c_u8_vec_pair_new(data1: Vec<u8>, data2: Vec<u8>) -> *mut Cu8VecPair {
    let boxed_array1: Box<[u8]> = data1.into_boxed_slice();
    let boxed_data1 = Box::into_raw(boxed_array1);
    let boxed_array2: Box<[u8]> = data2.into_boxed_slice();
    let boxed_data2 = Box::into_raw(boxed_array2);
    Box::into_raw(Box::new(Cu8VecPair::new(boxed_data1,boxed_data2)))
}

///Free a C accessible slice pair
#[no_mangle]
pub extern "C" fn c_u8_vec_pair_free(ptr: *mut Cu8VecPair) {
    if ptr.is_null() {
        return;
    }
    unsafe {
    	Box::from_raw((*ptr).ptr1);
	Box::from_raw((*ptr).ptr2);
        Box::from_raw(ptr);
    }
}

///Fill from a C accesible slice pair (entry 1)
#[no_mangle]
pub extern "C" fn c_u8_vec_pair_fill_buffer1(ptr: *const Cu8VecPair, buffer: *mut u8) {
    unsafe  {
    	let c_slice = slice::from_raw_parts_mut(buffer, (*ptr).ptr1.as_ref().unwrap().len());
	c_slice.copy_from_slice((*ptr).ptr1.as_ref().unwrap());
     }
}

///Get the length of a C accesible slice pair (entry 1)
#[no_mangle]
pub extern "C" fn c_u8_vec_pair_length1(ptr: *const Cu8VecPair)-> size_t {
    unsafe {
        (*ptr).ptr1.as_ref().unwrap().len()
    }
}

///Fill from a C accesible slice pair (entry 2)
#[no_mangle]
pub extern "C" fn c_u8_vec_pair_fill_buffer2(ptr: *const Cu8VecPair, buffer: *mut u8) {
    unsafe  {
    	let c_slice = slice::from_raw_parts_mut(buffer, (*ptr).ptr2.as_ref().unwrap().len());
	c_slice.copy_from_slice((*ptr).ptr2.as_ref().unwrap());
     }
}

///Get the length of a C accesible slice pair (entry 2)
#[no_mangle]
pub extern "C" fn c_u8_vec_pair_length2(ptr: *const Cu8VecPair)-> size_t {
    unsafe {
        (*ptr).ptr2.as_ref().unwrap().len()
    }
}

