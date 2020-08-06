# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

import base64
import json
import zlib
from server_io import *
from crypto_io import *

class Server:
    def __init__(self, lib, dimension, is_first_server, private_key, public_key1, public_key2, s3_session1=None, s3_session2=None, encrypt_data = True, ENCODING = "utf-8"):
        self.encrypt_data = encrypt_data
        if s3_session1 is not None: self.s3_client1 = s3_session1.client('s3')
        else: self.s3_client1 = None
        if s3_session2 is not None: self.s3_client2 = s3_session2.client('s3')
        else: self.s3_client2 = None
        self.lib = lib
        self.ENCODING = ENCODING
        self.is_first_server = is_first_server
        self.verification_share_output = list()
        self.valid_share_output = list()
        self.server = ServerIO(lib = lib,
                               dimension = int(dimension),
                               is_first_server = is_first_server,
                               private_key = private_key)
        self.self_crypto = CryptoIO(lib = lib,
                                    public_key = public_key1 if is_first_server else public_key2,
                                    private_key =private_key)
        self.other_crypto = CryptoIO(lib = lib,
                                     public_key = public_key2 if is_first_server else public_key1,
                                     private_key = "")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.server.__exit__(exc_type, exc_value, traceback)
        self.self_crypto.__exit__(exc_type, exc_value, traceback)
        self.other_crypto.__exit__(exc_type, exc_value, traceback)

    def get_share(self,input_dict):
        return base64.decodebytes(input_dict["share"].encode(self.ENCODING))

    def get_r_pit(self,input_dict):
        return int(input_dict["eval_at"])

    def get_uuid(self,input_dict):
        return input_dict["uuid"]
        
    def generate_verification_message(self,input_dict):
        share = self.get_share(input_dict)
        eval_at = self.get_r_pit(input_dict)
        output = self.server.generate_verification_message(eval_at,share)
        output["uuid"] = self.get_uuid(input_dict)
        self.verification_share_output.append(output)
    
    def generate_verification_messages(self,input_list):
        for d in input_list:
            self.generate_verification_message(d)
            
    def get_verification_messages(self):
        raw_out = zlib.compress(json.dumps(self.verification_share_output).encode(self.ENCODING))
        if self.encrypt_data:
            self_verification_messages = self.self_crypto.encrypt(raw_out)
            other_verification_messages = self.other_crypto.encrypt(raw_out)
        else:
            self_verification_messages = raw_out
            other_verification_messages = raw_out
        return (self_verification_messages,other_verification_messages) if self.is_first_server else (other_verification_messages,self_verification_messages)
    
    def read_verification_messages(self, verification_input):
        if self.encrypt_data: verification_input = self.self_crypto.decrypt(verification_input)
        verification_messages = json.loads(zlib.decompress(verification_input).decode(self.ENCODING))
        output = {}
        for verification_message in verification_messages:
            output[verification_message["uuid"]] = verification_message
        return output
    
    def aggregate_1(self,uuid,share,v1,v2):
        valid = self.server.aggregate(share,v1,v2)
        self.valid_share_output.append({uuid:valid})
    
    def aggregate(self, input_list, v1_dict, v2_dict):
        for d in input_list:
            uuid = self.get_uuid(d)
            share = self.get_share(d)
            v1 = v1_dict[uuid]
            v2 = v2_dict[uuid]
            self.aggregate_1(uuid,share,v1,v2)

    def get_valid_shares(self):
        raw_out = zlib.compress(json.dumps(self.valid_share_output).encode(self.ENCODING))
        if self.encrypt_data:
            self_valid_shares = self.self_crypto.encrypt(raw_out)
            other_valid_shares = self.other_crypto.encrypt(raw_out)
        else:
            self_valid_shares = raw_out
            other_valid_shares = raw_out
        return (self_valid_shares,other_valid_shares) if self.is_first_server else (other_valid_shares,self_valid_shares)
      
    def get_total_shares(self,):
        x = self.server.total_shares()
        raw_out = zlib.compress(json.dumps(list(self.server.total_shares())).encode(self.ENCODING))
        if self.encrypt_data:
            self_total_shares = self.self_crypto.encrypt(raw_out)
            other_total_shares = self.other_crypto.encrypt(raw_out)
        else:
            self_total_shares = raw_out
            other_total_shares = raw_out
        return (self_total_shares,other_total_shares) if self.is_first_server else (other_total_shares,self_total_shares)
    
    def reconstruct_shares(self,share1,share2):
        if self.encrypt_data:
            share1 = self.self_crypto.decrypt(share1)
            share2 = self.self_crypto.decrypt(share2)
        share1 = json.loads(zlib.decompress(share1).decode(self.ENCODING))
        share2 = json.loads(zlib.decompress(share2).decode(self.ENCODING))
        return ServerIO.reconstruct_shares(self.lib,share1,share2)

    def s3_get_key(self,bucket,key):
        if self.s3_client1 is None or self.s3_client2 is None: return
        client = self.s3_client1 if self.is_first_server else self.s3_client2
        return client.get_object(Bucket=bucket, Key=key)["Body"].read()
    
    def s3_get_shares(self,bucket,prefix):
        key = prefix+"shares"
        return json.loads(zlib.decompress(self.s3_get_key(bucket,key)).decode(self.ENCODING))

    def s3_get_verification_messages(self,bucket,prefix):
        key1 = prefix+"verifications1"
        v1 = self.s3_get_key(bucket,key1)
        key2 = prefix+"verifications2"
        v2 = self.s3_get_key(bucket,key2)
        return (v1,v2)

    def s3_get_total_shares(self,bucket,prefix):
        key1 = prefix+"total_shares1"
        ts1 = self.s3_get_key(bucket,key1)
        key2 = prefix+"total_shares2"
        ts2 = self.s3_get_key(bucket,key2)
        return (ts1,ts2)
       
    def s3_put_verification_messages(self,bucket1,bucket2,prefix):
        if self.s3_client1 is None or self.s3_client2 is None: return
        key = prefix+("verifications1" if self.is_first_server else "verifications2")
        (b1,b2) = self.get_verification_messages()
        self.s3_client1.put_object(Body=b1,
                                   Bucket=bucket1,
                                   Key=key)
        self.s3_client2.put_object(Body=b2,
                                   Bucket=bucket2,
                                   Key=key)

    def s3_put_total_shares(self,bucket1,bucket2,prefix):
        if self.s3_client1 is None or self.s3_client2 is None: return
        key = prefix+("total_shares1" if self.is_first_server else "total_shares2")
        (b1,b2) = self.get_total_shares()
        self.s3_client1.put_object(Body=b1,
                                   Bucket=bucket1,
                                   Key=key)

    def s3_put_valid_shares(self,bucket1,bucket2,prefix):
        if self.s3_client1 is None or self.s3_client2 is None: return
        key = prefix+("valid_shares1" if self.is_first_server else "valid_shares2")
        (b1,b2) = self.get_valid_shares()
        self.s3_client1.put_object(Body=b1,
                                   Bucket=bucket1,
                                   Key=key)

    def s3_cleanup(self,bucket,prefix):
        client = self.s3_client1 if self.is_first_server else self.s3_client2
        cleanup_list = ["shares","verifications1","verifications2"]
        if self.is_first_server: cleanup_list += ["total_shares1","total_shares2","total_shares1","total_shares2"]
        for key in cleanup_list:
            client.delete_object(Bucket=bucket, Key=prefix+key)
