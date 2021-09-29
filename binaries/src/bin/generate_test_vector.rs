// SPDX-License-Identifier: MPL-2.0

use color_eyre::eyre::{Result, WrapErr};
use prio::test_vector::Priov2TestVector;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(about = "Generate Priov2 test vector", rename_all = "kebab-case")]
enum Subcommand {
    Priov2 {
        /// Dimension (number of bins) of the inputs
        #[structopt(short, long, required = true)]
        dimension: usize,
    },
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "generate-test-vector",
    about = "Generate test vectors for Prio",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Options {
    /// Number of inputs to generate
    #[structopt(short, long, required = true)]
    number_of_inputs: usize,
    /// Subcommand determines what kind of vector to construct
    #[structopt(subcommand)]
    command: Subcommand,
}

fn generate_and_print_priov2_vector(number_of_inputs: usize, dimension: usize) -> Result<()> {
    let test_vector = Priov2TestVector::new(dimension, number_of_inputs)
        .wrap_err("failed to create test vector")?;
    let json =
        serde_json::to_string(&test_vector).wrap_err("failed to encode test vector to JSON")?;
    println!("{}", json);

    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let options = Options::from_args();

    match options.command {
        Subcommand::Priov2 { dimension } => {
            generate_and_print_priov2_vector(dimension, options.number_of_inputs)
        }
    }
}
