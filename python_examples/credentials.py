# Copyright (c) 2020 Apple Inc.
# SPDX-License-Identifier: MPL-2.0

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
import base64

def get_key_pairs_x_9_63(pk):
    # For an elliptic curve public key, the format follows the ANSI X9.63 standard using a byte string of 04 || X || Y
    # For an elliptic curve private key, the output is formatted as the public key concatenated with the big endian encoding of the secret scalar, or 04 || X || Y || K.
    four = int(4).to_bytes(1,'big')
    private_key_size = pk.key_size // 8
    k = pk.private_numbers().private_value.to_bytes(private_key_size,'big')
    public_key_size = pk.public_key().key_size // 8
    x = pk.public_key().public_numbers().x.to_bytes(public_key_size,'big')
    y = pk.public_key().public_numbers().y.to_bytes(public_key_size,'big')
    public_key_x_9_63 = four + x + y
    private_key_x_9_63 = public_key_x_9_63 + k
    return(private_key_x_963, public_key_x_9_63)

def get_random_key_pairs(ENCODING="utf-8"):
    pk1 = ec.generate_private_key(ec.SECP256R1(), default_backend())
    pk2 = ec.generate_private_key(ec.SECP256R1(), default_backend())
    private_key1,public_key1 = get_key_pairs_x_9_63(pk1)
    private_key2,public_key2 = get_key_pairs_x_9_63(pk2)
    
    return {"private_key1":base64.b64encode(private_key1).decode(ENCODING),
            "public_key1":base64.b64encode(public_key1).decode(ENCODING),
            "private_key2":base64.b64encode(private_key2).decode(ENCODING),
            "public_key2":base64.b64encode(public_key2).decode(ENCODING)}

server1_bucket = ""
server1_region = ""
server1_access_key_id = ""
server1_secret_access_key = ""

server2_bucket = ""
server2_region = ""
server2_access_key_id = ""
server2_secret_access_key = ""
