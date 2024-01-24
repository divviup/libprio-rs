// SPDX-License-Identifier: MPL-2.0

//! Implementations of VIDPF specified in [[draft-mouris-cfrg-mastic]].
//!
//! [draft-mouris-cfrg-mastic]: https://datatracker.ietf.org/doc/draft-mouris-cfrg-mastic/01/

/// VERSION is a tag.
pub static VERSION: &str = "MyVIDPF";

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version() {
        assert_eq!(VERSION, "MyVIDPF")
    }
}
