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

[enpa-whitepaper]: https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ENPA_White_Paper.pdf
[ppm-spec]: https://github.com/abetterinternet/ppm-specification
[prio-server]: https://github.com/abetterinternet/prio-server
[libprio-cc]: https://github.com/google/libprio-cc
