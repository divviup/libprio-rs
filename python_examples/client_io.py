# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import ctypes
from ctypes import Structure, c_char_p, Structure, POINTER, c_uint8, c_uint32, c_size_t
from random import Random
from cu_vec import *

class ClientIOS(Structure):
    pass

def setup_client_io(lib):
    lib.client_io_new.argtypes = (c_size_t, c_char_p, c_char_p, )
    lib.client_io_new.restype = POINTER(ClientIOS)
    lib.client_io_free.argtypes = (POINTER(ClientIOS), )
    lib.client_io_get_shares.argtypes = (POINTER(ClientIOS), POINTER(c_uint32), )
    lib.client_io_get_shares.restype = POINTER(Cu8VecPairS)

class ClientIO:
    def __init__(self, lib, dimension, public_key1, public_key2, seed=0):
        self.lib = lib
        self.rng = Random(seed)
        self.dimension = int(dimension)
        self.obj = self.lib.client_io_new(self.dimension, public_key1.encode('utf-8'),public_key2.encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.client_io_free(self.obj)

    def get_shares(self, signal):
        assert(len(signal)==self.dimension)
        input_buffer_type = c_uint32 * len(signal)
        input_buffer = input_buffer_type(*signal)
        ptr = self.lib.client_io_get_shares(self.obj, input_buffer)
        length1 = self.lib.c_u8_vec_pair_length1(ptr)
        length2 = self.lib.c_u8_vec_pair_length2(ptr)
        output_buffer1_type = c_uint8 * length1
        output_buffer1 = output_buffer1_type()
        self.lib.c_u8_vec_pair_fill_buffer1(ptr, output_buffer1)
        output_buffer2_type = c_uint8 * length2
        output_buffer2 = output_buffer2_type()
        self.lib.c_u8_vec_pair_fill_buffer2(ptr, output_buffer2)
        try:
            return (output_buffer1,output_buffer2)
        finally:
            self.lib.c_u8_vec_pair_free(ptr)
            
    def generate_random_signal(self,):
        r = self.rng.getrandbits(self.dimension) 
        return [r >> i & 1 for i in range(self.dimension)]
