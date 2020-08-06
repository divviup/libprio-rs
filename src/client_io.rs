// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Prio client C FFI

extern crate libc;
use libc::c_char;
use libc::size_t;

use std::slice;
use std::ffi::CStr;
use std::str;

use crate::c_vec::*;
use crate::encrypt::*;
use crate::client::*;
use crate::finite_field::Field;


///C binding for the client
pub struct ClientIO {
   client: Client,
   dimension: usize
}

impl ClientIO {
    ///Construct a new client to communicate through an FFI interface
    pub fn new(dimension: usize, public_key1_str:&str, public_key2_str: &str) -> ClientIO {
    	let public_key1 = PublicKey::from_base64(public_key1_str).unwrap();
	let public_key2 = PublicKey::from_base64(public_key2_str).unwrap();
        ClientIO{
	    client: Client::new(dimension, public_key1, public_key2).unwrap(),
	    dimension: dimension
	}
    }

    ///Ingest data and return Prio shares
    pub fn encode(&mut self, data: &[u32]) -> (Vec<u8>, Vec<u8>) {
    	let mut data_field:Vec<Field> = Vec::with_capacity(data.len());
	for i in 0..data.len() {
    	    data_field.push(Field::from(data[i]));
	}
    	self.client.encode_simple(data_field.as_slice()).ok().unwrap()
    }

    ///Return the dimension of the client
    pub fn get_dimension(&self) -> usize {
    	self.dimension
    }
}

///Return a C version of the client
#[no_mangle]
pub extern "C" fn client_io_new(dimension: size_t, public_key1: *const c_char, public_key2: *const c_char ) -> *mut ClientIO {
    let c_public_key1_str = unsafe {
        assert!(!public_key1.is_null());
        CStr::from_ptr(public_key1)
    };
    let r_public_key1_str = c_public_key1_str.to_str().unwrap();
    
    let c_public_key2_str = unsafe {
        assert!(!public_key2.is_null());
        CStr::from_ptr(public_key2)
    };
    let r_public_key2_str = c_public_key2_str.to_str().unwrap();
    
    Box::into_raw(Box::new(ClientIO::new(dimension, r_public_key1_str, r_public_key2_str)))
}

///Free memory of C version of the client
#[no_mangle]
pub extern "C" fn client_io_free(ptr: *mut ClientIO) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

///Get shares from input data
#[no_mangle]
pub extern "C" fn client_io_get_shares(ptr: *mut ClientIO, data: *const u32)-> *mut Cu8VecPair {
    let client = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let data = unsafe {
        assert!(!data.is_null());
        slice::from_raw_parts(data, client.get_dimension())
    };
    let shares = client.encode(data);
    send_c_u8_vec_pair_new(shares.0,shares.1)
}
