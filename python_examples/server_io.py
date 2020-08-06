# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import ctypes
from ctypes import Structure, c_char_p, Structure, POINTER, c_uint8, c_uint32, c_size_t, c_bool
from cu_vec import *

class ServerIOS(Structure):
    pass

class VerificationTuple(Structure):
    _fields_ = [("f_r", c_uint32),
                ("g_r", c_uint32),
                ("h_r", c_uint32),]

    def __str__(self):
        return "[{},{},{}]".format(self.f_r, self.g_r,self.h_r)
    def from_dict(input_dict):
        return VerificationTuple(int(input_dict["f_r"]),int(input_dict["g_r"]),int(input_dict["h_r"]))
    def to_dict(self):
        return {"f_r": self.f_r,
                "g_r": self.g_r,
                "h_r": self.h_r}

def setup_server_io(lib):
    lib.server_io_new.argtypes = (c_size_t, c_bool, c_char_p, )
    lib.server_io_new.restype = POINTER(ServerIOS)
    lib.server_io_choose_eval_at.argtypes = (POINTER(ServerIOS), )
    lib.server_io_choose_eval_at.restype = c_uint32
    lib.server_io_generate_verification_message.argtypes = (POINTER(ServerIOS), c_uint32, POINTER(c_uint8), c_size_t)
    lib.server_io_generate_verification_message.restype = VerificationTuple
    lib.server_io_aggregate.argtypes = (POINTER(ServerIOS), POINTER(c_uint8), c_size_t, VerificationTuple, VerificationTuple)
    lib.server_io_aggregate.restype = c_bool
    lib.server_io_total_shares.argtypes = (POINTER(ServerIOS),)
    lib.server_io_total_shares.restype = POINTER(Cu32VecS)
    lib.server_io_reconstruct_shares.argtypes = (POINTER(c_uint32),POINTER(c_uint32), c_size_t, )
    lib.server_io_reconstruct_shares.restype = POINTER(Cu32VecS)

class ServerIO:
    def __init__(self, lib, dimension, is_first_server, private_key):
        self.lib = lib
        self.dimension = int(dimension)
        self.is_first_server = bool(is_first_server)
        self.obj = self.lib.server_io_new(self.dimension, self.is_first_server,private_key.encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lib.server_io_free(self.obj)

    def choose_eval_at(self):
        return self.lib.server_io_choose_eval_at(self.obj)

    def generate_verification_message(self,eval_at,share):
        input_buffer_type = c_uint8 * len(share)
        input_buffer = input_buffer_type(*share)
        verification = self.lib.server_io_generate_verification_message(self.obj, int(eval_at), input_buffer, len(share))
        return verification.to_dict()
    
    def aggregate(self,share,v1,v2):
        input_buffer_type = c_uint8 * len(share)
        input_buffer = input_buffer_type(*share)
        return self.lib.server_io_aggregate(self.obj, input_buffer, len(share), VerificationTuple.from_dict(v1), VerificationTuple.from_dict(v2))

    def total_shares(self):
        ptr = self.lib.server_io_total_shares(self.obj)
        length = self.lib.c_u32_vec_length(ptr)
        output_buffer_type = c_uint32 * length
        output_buffer = output_buffer_type()
        self.lib.c_u32_vec_fill_buffer(ptr, output_buffer)
        try:
            return output_buffer
        finally:
            self.lib.c_u32_vec_free(ptr)
            
    def reconstruct_shares(lib,share1,share2):
        assert(len(share1) == len(share2))
        buffer_type = c_uint32 * len(share1)
        input_buffer1 = buffer_type(*share1)
        input_buffer2 = buffer_type(*share2)
        ptr = lib.server_io_reconstruct_shares(input_buffer1, input_buffer2, len(share1))
        output_buffer = buffer_type()
        lib.c_u32_vec_fill_buffer(ptr, output_buffer)
        try:
            return output_buffer
        finally:
            lib.c_u32_vec_free(ptr)
        
