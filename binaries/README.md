# Prio binary targets

This sub-crate contains binary targets that use crate `prio`.

## `generate-test-vector`: test vector generation for backward compatibility

Crate `prio` supports Prio v2, used in [Exposure Notification Private Analytics][enpa-whitepaper],
as implemented in [`prio-server`][prio-server], as well as the Prio v3 Verifiable Distributed
Aggregation Function used in [Privacy Preserving Measurements][ppm-spec].

To ensure backward compatibility with older versions of crate `prio` and [other implementations of
Prio v2][libprio-cc], we continuously test `prio`'s Prio v2 implementation against a test vector
recorded with `prio` 0.5.0. That test is implemented in `tests/backward_compatibility.rs`. The test
vector was generated using `generate-test-vector`. See that tool's usage text for more information,
and module `prio::test_vector` for utilities for working with test vectors.

## `crypt`: encrypt and decrypt Prio v2 inputs

`crypt` transforms inputs (a vector of `FieldPriov2`) into encrypted input shares as well as
transforming encrypted input shares back into inputs. To decrypt input shares:

    cargo run --bin crypt -- \
        -d 10 \
        --server-1-private-key /path/to/base64/server-1/private/key \
        --server-2-private-key /path/to/base64/server-2/private/key \
        decrypt \
        --output /tmp/decrypted \
        --server-1-encrypted-share "BGuETGFrn8w+CYPBd4f2G/Ju0RNc50rmAGPtptV2kO4KLiMDeALBtCk04zgQ3pwG/2yCh1ouLAPSeS8WgDtCtMuQNz6YUG+/6UYcBm8LtrzDTHBXoWaOuuafmj4I9d2UdjrTkWMZoAlq/MLKoLDP3MyRNX7MCqkGRgdIyk50b5yvwzt4eMQPJDPnp4TuhV5I2W2ddLXBVGFk9NeJyahbJkbOvYYEnc1NAKMejdiWOcfVfS0vEk+s1Br01UtqEEOYFtyW+CA=" \
        --server-2-encrypted-share "BKqwkUPnrpvlwdGKc92cD0iy3FlE4Bcv+2pLUm+0JrcC9r9ufnD7pMEgAVYmNP7z0PNmuKuDD2PZRdpp5/h330BeWX31n6quI/DvLPQUcTT54M1gM8f3HXfegMaPCCWihqdAh6V7FRw8CdsQI3tr86c="

Encrypt the inputs again:

    cargo run --bin crypt -- \
        -d 10 \
        --server-1-private-key /path/to/base64/server-1/private/key \
        --server-2-private-key /path/to/base64/server-2/private/key \
        encrypt \
        --input "AQAAAAEAAAABAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==" \
        --server-1-encrypted-share /tmp/server-1 \
        --server-2-encrypted-share /tmp/server-2

Where the contents of `/path/to/base64/server-1/private/key` and
`/path/to/base64/server-2/private/key` are base64 encoded private keys like:

    server 1: BIl6j+J6dYttxALdjISDv6ZI4/VWVEhUzaS05LgrsfswmbLOgNt9HUC2E0w+9RqZx3XMkdEHBHfNuCSMpOwofVSq3TfyKwn0NrftKisKKVSaTOt5seJ67P5QL4hxgPWvxw==
    server 2: BNNOqoU54GPo+1gTPv+hCgA9U2ZCKd76yOMrWa1xTWgeb4LhFLMQIQoRwDVaW64g/WTdcxT4rDULoycUNFB60LER6hPEHg/ObBnRPV1rwS3nj9Bj0tbjVPPyL9p8QW8B+w==

[enpa-whitepaper]: https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ENPA_White_Paper.pdf
[ppm-spec]: https://github.com/abetterinternet/ppm-specification
[prio-server]: https://github.com/abetterinternet/prio-server
[libprio-cc]: https://github.com/google/libprio-cc
