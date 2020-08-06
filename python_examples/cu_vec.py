# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import ctypes
from ctypes import Structure, POINTER, c_uint8, c_uint32, c_size_t

class Cu8VecS(Structure):
    pass

class Cu32VecS(Structure):
    pass

class Cu8VecPairS(Structure):
    pass

def setup_cu_vectors(lib):
    lib.c_u8_vec_fill_buffer.argtypes = (POINTER(Cu8VecS), POINTER(c_uint8), )
    lib.c_u8_vec_length.argtypes = (POINTER(Cu8VecS), )
    lib.c_u8_vec_length.argtype = c_size_t
    lib.c_u8_vec_free.argtypes = (POINTER(Cu8VecS), )

    lib.c_u32_vec_fill_buffer.argtypes = (POINTER(Cu32VecS), POINTER(c_uint32), )
    lib.c_u32_vec_length.argtypes = (POINTER(Cu32VecS), )
    lib.c_u32_vec_length.argtype = c_size_t
    lib.c_u32_vec_free.argtypes = (POINTER(Cu32VecS), )

    lib.c_u8_vec_pair_fill_buffer1.argtypes = (POINTER(Cu8VecPairS), POINTER(c_uint8))
    lib.c_u8_vec_pair_length1.argtypes = (POINTER(Cu8VecPairS), )
    lib.c_u8_vec_pair_length1.argtype = c_size_t
    lib.c_u8_vec_pair_fill_buffer2.argtypes = (POINTER(Cu8VecPairS), POINTER(c_uint8))
    lib.c_u8_vec_pair_length2.argtypes = (POINTER(Cu8VecPairS), )
    lib.c_u8_vec_pair_length2.argtype = c_size_t
    lib.c_u8_vec_pair_free.argtypes = (POINTER(Cu8VecPairS), )
