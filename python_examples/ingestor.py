# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import uuid
import base64
import json
import zlib
from server_io import *

class Ingestor:
    def __init__(self, lib, dimension, s3_session1=None, s3_session2=None, seed = 0, ENCODING = "utf-8"):
        self.ENCODING = ENCODING
        if s3_session1 is not None: self.s3_client1 = s3_session1.client('s3')
        else: self.s3_client1 = None
        if s3_session2 is not None: self.s3_client2 = s3_session2.client('s3')
        else: self.s3_client2 = None
        self.server1_output_list = []
        self.server2_output_list = []
        self.rng_server = ServerIO(lib,
                                   dimension = int(dimension),
                                   is_first_server = True,
                                   private_key = "N/A")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.rng_server.__exit__(exc_type, exc_value, traceback)

    def get_roots_2n(self):
        return list(self.rng_server.get_roots_2n())
        
    def read_share(self,input_str):
        data = json.loads(input_str)
        uuid_str = str(uuid.uuid4())
        eval_at = self.rng_server.choose_eval_at()
        server1_data = {"aggregate_name": data["aggregate_name"],
                        "aggregate_dimension": data["aggregate_dimension"],
                        "uuid":uuid_str,
                        "eval_at":eval_at,
                        "share":data["share1"]
                       }
        self.server1_output_list.append(server1_data)
        server2_data = {"aggregate_name": data["aggregate_name"],
                        "aggregate_dimension": data["aggregate_dimension"],
                        "uuid":uuid_str,
                        "eval_at":eval_at,
                        "share":data["share2"]
                       }
        self.server2_output_list.append(server2_data)
        
    def read_shares(self, input_list):
        for share in input_list:
            self.read_share(share)
            
    def get_shares(self):
        return (json.dumps(self.server1_output_list), json.dumps(self.server2_output_list))

    def s3_put_shares(self,bucket1,bucket2,prefix):
        if self.s3_client1 is None or self.s3_client2 is None or len(self.server1_output_list) == 0: return
        key = prefix+"shares"
        server1_compressed_data = zlib.compress(json.dumps(self.server1_output_list).encode(self.ENCODING))
        server2_compressed_data = zlib.compress(json.dumps(self.server2_output_list).encode(self.ENCODING))        
        self.s3_client1.put_object(Body=server1_compressed_data,
                                   Bucket=bucket1,
                                   Key=key)
        self.s3_client2.put_object(Body=server2_compressed_data,
                                   Bucket=bucket2,
                                   Key=key)
    
