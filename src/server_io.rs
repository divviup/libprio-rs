// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Prio server C FFI

extern crate libc;
use libc::c_char;
use libc::size_t;

use std::slice;
use std::ffi::CStr;
use std::str;

use crate::c_vec::*;
use crate::encrypt::*;
use crate::server::*;
use crate::finite_field::Field;
use crate::util::reconstruct_shares;

///A server object to communicate between C and Rust
pub struct ServerIO {
    server: Server
}

impl ServerIO {
    ///Create a server object
    pub fn new(dimension: usize, is_first_server: bool, private_key_str: &str) -> ServerIO {
    	let private_key = PrivateKey::from_base64(private_key_str).unwrap();
        ServerIO{
	    server: Server::new(dimension, is_first_server, private_key)
	}
    }

    ///Choose a random point in group
    pub fn choose_eval_at(&self) -> u32 {
    	self.server.choose_eval_at().into()
    }

    ///Generate a verification message
    pub fn generate_verification_message(&mut self, eval_at: u32, share: &[u8]) -> VerificationTuple {
    	let v = self.server.generate_verification_message(Field::from(eval_at),share).unwrap();
	VerificationTuple{f_r: v.f_r.into(), g_r: v.g_r.into(), h_r: v.h_r.into()}
    }

    ///Perform an aggregation
    pub fn aggregate (&mut self, share: &[u8],
		      v1: VerificationMessage,
		      v2: VerificationMessage) -> bool {
	self.server.aggregate(share,&v1,&v2).unwrap()
    }

    ///Return total of shares aggregated
    pub fn total_shares(&self) -> Vec<u32> {
	let totals = self.server.total_shares();
	let mut totals_u32:Vec<u32> = Vec::with_capacity(totals.len());
	for i in 0..totals.len() {
    	    totals_u32.push(totals[i].into());
	}
	totals_u32
    }

}

///Send a server object to C interface
#[no_mangle]
pub extern "C" fn server_io_new(dimension: size_t, is_first_server: bool, private_key: *const c_char ) -> *mut ServerIO {
    let c_private_key_str = unsafe {
        assert!(!private_key.is_null());
        CStr::from_ptr(private_key)
    };
    let r_private_key_str = c_private_key_str.to_str().unwrap();
    
    Box::into_raw(Box::new(ServerIO::new(dimension, is_first_server, r_private_key_str)))
}

///Free memory of an object passed to C
#[no_mangle]
pub extern "C" fn server_io_free(ptr: *mut ServerIO) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

///Get random element from group
#[no_mangle]
pub extern "C" fn server_io_choose_eval_at(ptr: *mut ServerIO) -> u32 {
    let server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    server.choose_eval_at()
}

///Generate a verification message
#[no_mangle]
pub extern "C" fn server_io_generate_verification_message(ptr: *mut ServerIO, eval_at: u32, share: *const u8, len: size_t,) -> VerificationTuple {
    let server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let share = unsafe {
        assert!(!share.is_null());
        slice::from_raw_parts(share, len as usize)
    };
    server.generate_verification_message(eval_at,share)
}

///Create a verification object to pass to C
#[repr(C)]
pub struct VerificationTuple {
    f_r: u32,
    g_r: u32,
    h_r: u32
}

/// Convert between VerificationMessage and VerificationTuple
impl From<VerificationMessage> for VerificationTuple {
    fn from(vm: VerificationMessage) -> VerificationTuple {
        VerificationTuple { f_r: vm.f_r.into(), g_r: vm.g_r.into(), h_r: vm.h_r.into() }
    }
}

/// Convert between VerificationTuple and VerificationMessage
impl From<VerificationTuple> for VerificationMessage {
    fn from(vt: VerificationTuple) -> VerificationMessage {
        VerificationMessage{f_r: Field::from(vt.f_r), g_r: Field::from(vt.g_r), h_r: Field::from(vt.h_r)}
    }
}

///Add messages to server
#[no_mangle]
pub extern "C" fn server_io_aggregate(ptr: *mut ServerIO, share: *const u8, len: size_t, vt1: VerificationTuple, vt2: VerificationTuple) -> bool {
    let server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    let share = unsafe {
        assert!(!share.is_null());
        slice::from_raw_parts(share, len as usize)
    };
    let out = server.aggregate(share,vt1.into(),vt2.into());
    out
}

///Get total shares
#[no_mangle]
pub extern "C" fn server_io_total_shares(ptr: *mut ServerIO) -> *mut Cu32Vec {
    let server = unsafe {
        assert!(!ptr.is_null());
        &mut *ptr
    };
    send_c_u32_vec_new(server.total_shares())
}

///Sum two pairs of shares
#[no_mangle]
pub extern "C" fn server_io_reconstruct_shares(share1: *const u32, share2: *const u32, len: size_t) -> *mut Cu32Vec {
    let share1 = unsafe {
        assert!(!share1.is_null());
        slice::from_raw_parts(share1, len as usize)
    };
    let mut share1_field:Vec<Field> = Vec::with_capacity(len);

    let share2 = unsafe {
        assert!(!share2.is_null());
        slice::from_raw_parts(share2, len as usize)
    };
    let mut share2_field:Vec<Field> = Vec::with_capacity(len);
    
    for i in 0..len {
    	share1_field.push(Field::from(share1[i]));
	share2_field.push(Field::from(share2[i]));
    }
    
    let reconstructed = reconstruct_shares(share1_field.as_slice(),share2_field.as_slice()).unwrap();
    let mut reconstructed_u32:Vec<u32> = Vec::with_capacity(len);
    for i in 0..len {
    	reconstructed_u32.push(reconstructed[i].into());
    }
    send_c_u32_vec_new(reconstructed_u32)
}
