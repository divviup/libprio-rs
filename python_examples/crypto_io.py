# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import ctypes
from ctypes import Structure, c_char_p, Structure, POINTER, c_uint8, c_size_t
from cu_vec import *

class CryptoIOS(Structure):
    pass

def setup_crypto_io(lib):
    lib.crypto_io_new.argtypes = (c_char_p, c_char_p, )
    lib.crypto_io_new.restype = POINTER(CryptoIOS)
    lib.crypto_io_free.argtypes = (POINTER(CryptoIOS), )
    lib.crypto_io_encrypt.argtypes = (POINTER(CryptoIOS), POINTER(c_uint8), c_size_t, )
    lib.crypto_io_encrypt.restype = POINTER(Cu8VecS)
    lib.crypto_io_decrypt.argtypes = (POINTER(CryptoIOS), POINTER(c_uint8), c_size_t, )
    lib.crypto_io_decrypt.restype = POINTER(Cu8VecS)

class CryptoIO:
    def __init__(self, lib, public_key, private_key):
        self.lib = lib
        self.obj = self.lib.crypto_io_new(public_key.encode('utf-8'),private_key.encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.crypto_io_free(self.obj)

    def encrypt(self, clear_text):
        input_buffer_type = c_uint8 * len(clear_text)
        input_buffer = input_buffer_type(*clear_text)
        ptr = self.lib.crypto_io_encrypt(self.obj, input_buffer, len(clear_text))
        length = self.lib.c_u8_vec_length(ptr)
        output_buffer_type = c_uint8 * length
        output_buffer = output_buffer_type()
        self.lib.c_u8_vec_fill_buffer(ptr, output_buffer)
        try:
            return bytearray(output_buffer)
        finally:
            self.lib.c_u8_vec_free(ptr)
            
    def decrypt(self, crypt_text):
        input_buffer_type = c_uint8 * len(crypt_text)
        input_buffer = input_buffer_type(*crypt_text)
        ptr = self.lib.crypto_io_decrypt(self.obj, input_buffer, len(crypt_text))
        length = self.lib.c_u8_vec_length(ptr)
        output_buffer_type = c_uint8 * length
        output_buffer = output_buffer_type()
        self.lib.c_u8_vec_fill_buffer(ptr, output_buffer)
        try:
            return bytearray(output_buffer)
        finally:
            self.lib.c_u8_vec_free(ptr)
