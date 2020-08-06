# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import base64
import json

from client_io import *

class ClientSimulator:
    def __init__(self, lib, aggregate_name, dimension, public_key1, public_key2, seed=0, ENCODING = "utf-8"):
        self.ENCODING = ENCODING
        self.aggregate_name = aggregate_name
        self.true_aggregate = [0]*int(dimension)
        self.n_clients = 0
        self.output_list = list()
        self.client = ClientIO(lib = lib,
                               dimension = int(dimension),
                               public_key1 = public_key1,
                               public_key2 = public_key2,
                               seed = seed)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.client.__exit__(exc_type, exc_value, traceback)

    def simulate_1(self):
        signal = self.client.generate_random_signal()
        for i in range(self.client.dimension):
            self.true_aggregate[i] += signal[i]
        (share1, share2) = self.client.get_shares(signal)
        share1_str = base64.encodebytes(bytearray(share1)).decode(self.ENCODING)
        share2_str = base64.encodebytes(bytearray(share2)).decode(self.ENCODING)
        result = {"aggregate_name": self.aggregate_name,
                  "aggregate_dimension": self.client.dimension,
                  "share1":share1_str,
                  "share2":share2_str
                 }
        self.output_list.append(json.dumps(result))
        
    def simulate_n(self, n):
        for i in range(n):
            self.simulate_1()
            
    def get_simulation(self):
        return (self.output_list, self.true_aggregate)
