pub mod client;
pub mod encrypt;
pub mod finite_field;
mod polynomial;
mod prng;
pub mod server;
pub mod util;

pub use encrypt::PrivateKey;
pub use encrypt::PublicKey;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
