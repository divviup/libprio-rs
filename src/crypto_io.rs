// Copyright (c) 2020 Apple Inc.
// SPDX-License-Identifier: MPL-2.0

//! Prio cryptograhic function C FFI

extern crate libc;
use libc::c_char;
use libc::size_t;

use std::slice;
use std::ffi::CStr;
use std::str;

use crate::encrypt::*;

use crate::c_vec::*;

///Making cryptographic utilities available to C
pub struct CryptoIO {
   public_key: PublicKey,
   private_key:PrivateKey
}

impl CryptoIO {
    ///Create a new cryptographic encryption and decryption object
    pub fn new(public_key_str:&str, private_key_str: &str) -> CryptoIO {
    	let public_key = PublicKey::from_base64(public_key_str).unwrap();
	let private_key = PrivateKey::from_base64(private_key_str).unwrap();
        CryptoIO {
	   public_key : public_key,
           private_key : private_key
        }
    }

    ///Encrypt a u8 slice
    pub fn encrypt(&self, share: &[u8]) -> Option<Vec<u8>> {
    	encrypt_share(share, &self.public_key).ok()
    }

    ///Decrypt a u8 slice
    pub fn decrypt(&self, share: &[u8]) -> Option<Vec<u8>> {
    	decrypt_share(share, &self.private_key).ok()
    }
    
}

///Create and pass an object to C
#[no_mangle]
pub extern "C" fn crypto_io_new(public_key: *const c_char, private_key: *const c_char ) -> *mut CryptoIO {
    let c_public_key_str = unsafe {
        assert!(!public_key.is_null());
        CStr::from_ptr(public_key)
    };
    let r_public_key_str = c_public_key_str.to_str().unwrap();
    
    let c_private_key_str = unsafe {
        assert!(!private_key.is_null());
        CStr::from_ptr(private_key)
    };
    let r_private_key_str = c_private_key_str.to_str().unwrap();
    
    Box::into_raw(Box::new(CryptoIO::new(r_public_key_str, r_private_key_str)))
}

///Free an object passed to C
#[no_mangle]
pub extern "C" fn crypto_io_free(ptr: *mut CryptoIO) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

///Encrypt a buffer from C
#[no_mangle]
pub extern "C" fn crypto_io_encrypt(ptr: *const CryptoIO, data: *const u8, len: size_t)-> *mut Cu8Vec {
    let crypto = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };
    let plain_text = unsafe {
        assert!(!data.is_null());
        slice::from_raw_parts(data, len as usize)
    };
    let crypt_text = crypto.encrypt(plain_text).unwrap();
    send_c_u8_vec_new(crypt_text)
}

///Decrypt a buffer from C
#[no_mangle]
pub extern "C" fn crypto_io_decrypt(ptr: *const CryptoIO, data: *const u8, len: size_t)-> *mut Cu8Vec {
    let crypto = unsafe {
        assert!(!ptr.is_null());
        &*ptr
    };
    let crypt_text = unsafe {
        assert!(!data.is_null());
        slice::from_raw_parts(data, len as usize)
    };
    let plain_text = crypto.decrypt(crypt_text).unwrap();
    send_c_u8_vec_new(plain_text)
}




