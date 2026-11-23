# Test harness for `cargo-checkct`

This workspace provides a test harness for checking routines from libprio-rs
with [cargo-checkct](https://github.com/Ledger-Donjon/cargo-checkct). This
tooling performs symbolic execution, and confirms whether all instructions and
memory accesses run in constant time with respect to the values of secret
inputs.

## Usage

First, install BINSEC and other OCaml dependencies per the instructions in the
README of `cargo-checkct`. Then, install `cargo-checkct` itself.

Compile and analyze binaries with the following command.

```bash
cargo checkct run --checks control-flow,memory-access,dividend,divisor --all-leaks -j 1 --timeout 600
```

For each combination of target triple and entrypoint, a verdict will be output,
either "secure", "insecure", or "unknown". If "unknown" is output, that
indicates that the analysis timed out. The `--timeout` flag can be used to
extend this limit, but analysis of some complicated programs may be intractable
with these tools.

We expect to see false positives from most targets due to the rejection
sampling loop that is used when generating random field elements, in
`try_from_random()`.  The `--all-leaks` flag continues analysis after finding
the first instruction that causes a leak, and lists all potential leaks.
Disassemble the binary at `checkct/target/<TARGET-TRIPLE>/release/driver`, and
inspect the instructions at the listed addresses to confirm whether they are
part of this rejection sampling loop. If the `[checkct:result]` log line is
emitted soon after the timeout has elapsed, this indicates that analysis did
not finish after finding the first leak, and the output is inconclusive about
whether there are any non-false positive leaks later on.
