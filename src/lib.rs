pub mod client;
pub mod encrypt;
pub mod finite_field;
mod polynomial;
mod prng;
pub mod server;
pub mod util;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
