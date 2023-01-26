// SPDX-License-Identifier: MPL-2.0

use base64::{engine::Engine, prelude::BASE64_STANDARD};
use color_eyre::eyre::{eyre, Result, WrapErr};
use prio::{
    client::Client,
    encrypt::{PrivateKey, PublicKey},
    field::{FieldElement, FieldPrio2},
    server::Server,
    util::reconstruct_shares,
};
use std::{
    fs::{read_to_string, File},
    io::{stdin, stdout, Cursor, Read, Write},
    path::PathBuf,
    str::FromStr,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
enum Subcommand {
    /// Split the input into two shares and encrypt them to the public portions
    /// of the server private keys
    Encrypt {
        /// Plaintext input to be shared and encrypted
        #[structopt(short, long, default_value = "-")]
        input: Input,
        /// Where to write server 1's encrypted share. Pass "-" for stdout.
        #[structopt(long, default_value = "-")]
        server_1_encrypted_share: Output,
        /// Where to write server 2's encrypted share. Pass "-" for stdout.
        #[structopt(long, default_value = "-")]
        server_2_encrypted_share: Output,
    },
    /// Decrypt the provided input shares using the server private keys
    Decrypt {
        /// Server 1's encrypted share
        #[structopt(long, default_value = "-")]
        server_1_encrypted_share: Input,
        /// Server 2's encrypted share
        #[structopt(long, default_value = "-")]
        server_2_encrypted_share: Input,
        /// Where to write decrypted input. Pass "-" for stdout.
        #[structopt(short, long, default_value = "-")]
        output: Output,
        /// Pretty-print the decrypted input
        #[structopt(short, long)]
        pretty_print: bool,
    },
}

#[derive(Debug)]
enum Input {
    /// Input provided as base64 encoded bytes
    Parameter(Vec<u8>),
    /// Read input from the provided path
    Path(PathBuf),
    /// Read input from stdin
    Stdin,
}

impl FromStr for Input {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq("-") {
            return Ok(Self::Stdin);
        }

        if let Ok(decoded) = BASE64_STANDARD.decode(s) {
            return Ok(Self::Parameter(decoded));
        }

        Ok(Self::Path(PathBuf::from_str(s).map_err(|e| {
            format!("argument could not be parsed as base64 or path: {e:?}")
        })?))
    }
}

impl Input {
    fn contents(self) -> Result<Vec<u8>> {
        let mut reader = match self {
            Self::Parameter(bytes) => Box::new(Cursor::new(bytes)) as Box<dyn Read>,
            Self::Path(path_buf) => {
                Box::new(File::open(path_buf).wrap_err("failed to open input path")?)
                    as Box<dyn Read>
            }
            Self::Stdin => Box::new(stdin()) as Box<dyn Read>,
        };
        let mut contents = Vec::new();
        reader
            .read_to_end(&mut contents)
            .wrap_err("failed to read to end of input")?;

        Ok(contents)
    }
}

#[derive(Debug)]
enum Output {
    /// Write output to the provided path
    Path(PathBuf),
    /// Write output to process stdout
    Stdout,
}

impl FromStr for Output {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq("-") {
            return Ok(Self::Stdout);
        }

        Ok(Self::Path(PathBuf::from_str(s).map_err(|e| {
            format!("argument could not be parsed as base64 or path: {e:?}")
        })?))
    }
}

impl Output {
    fn into_writer(self) -> Result<Box<dyn Write>> {
        match self {
            Self::Path(path_buf) => Ok(Box::new(
                File::create(path_buf).wrap_err("failed to open output path")?,
            ) as Box<dyn Write>),
            Self::Stdout => Ok(Box::new(stdout()) as Box<dyn Write>),
        }
    }
}

#[derive(Debug, StructOpt)]
#[structopt(
    name = "crypt",
    about = "Encrypt and decrypt Priov2 inputs",
    rename_all = "kebab-case",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Options {
    /// Dimension of the inputs
    #[structopt(long, short, required = true)]
    dimension: usize,
    /// Path to a file containing the Base64 encoded private key for the first
    /// server
    #[structopt(long, required = true, value_name = "path")]
    server_1_private_key: String,
    /// Path to a file containing the Base64 encoded private key for the second
    /// server
    #[structopt(long, required = true, value_name = "path")]
    server_2_private_key: String,

    /// Subcommand determines whether to encrypt or decrypt inputs
    #[structopt(subcommand)]
    command: Subcommand,
}

fn encrypt(
    mut client: Client<FieldPrio2>,
    input: Input,
    server_1_encrypted_share: Output,
    server_2_encrypted_share: Output,
) -> Result<()> {
    let input_fields = FieldPrio2::byte_slice_into_vec(&input.contents()?)
        .wrap_err("could not decode bytes into field elements")?;

    let (share_1, share_2) = client
        .encode_simple(&input_fields)
        .wrap_err("could not share and encrypt input")?;

    server_1_encrypted_share
        .into_writer()?
        .write_all(&share_1)
        .wrap_err("could not write server 1 output")?;
    server_2_encrypted_share
        .into_writer()?
        .write_all(&share_2)
        .wrap_err("could not write server 2 output")?;
    Ok(())
}

fn decrypt(
    mut server_1: Server<FieldPrio2>,
    mut server_2: Server<FieldPrio2>,
    server_1_encrypted_share: Input,
    server_2_encrypted_share: Input,
    decrypted_input: Output,
    pretty_print: bool,
) -> Result<()> {
    let eval_at = server_1.choose_eval_at();
    let server_1_share = server_1_encrypted_share.contents()?;
    let server_2_share = server_2_encrypted_share.contents()?;

    let verification_1 = server_1
        .generate_verification_message(eval_at, &server_1_share)
        .wrap_err("failed to verify server 1 input")?;
    let verification_2 = server_2
        .generate_verification_message(eval_at, &server_2_share)
        .wrap_err("failed to verify server 2 input")?;

    if !server_1
        .aggregate(&server_1_share, &verification_1, &verification_2)
        .wrap_err("failed to verify input")?
    {
        return Err(eyre!("server 1 share proof validation failed"));
    }
    if !server_2
        .aggregate(&server_2_share, &verification_1, &verification_2)
        .wrap_err("failed to verify input")?
    {
        return Err(eyre!("server 2 share proof validation failed"));
    }

    let reconstructed = reconstruct_shares(server_1.total_shares(), server_2.total_shares())
        .ok_or_else(|| eyre!("failed to reconstruct input shares"))?;

    if pretty_print {
        writeln!(&mut decrypted_input.into_writer()?, "{reconstructed:?}")
            .wrap_err("failed to pretty-print reconstructed output")
    } else {
        decrypted_input
            .into_writer()?
            .write_all(&FieldPrio2::slice_into_byte_vec(&reconstructed))
            .wrap_err("failed to write reconstructed share to output")
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let options = Options::from_args();

    let server_1_private_key = PrivateKey::from_base64(
        read_to_string(&options.server_1_private_key)
            .wrap_err("could not read server 1 private key from file")?
            .trim(),
    )
    .wrap_err("could not decode base64 private key")?;

    let server_2_private_key = PrivateKey::from_base64(
        read_to_string(&options.server_2_private_key)
            .wrap_err("could not read server 2 private key from file")?
            .trim(),
    )
    .wrap_err("could not decode base64 private key")?;

    match options.command {
        Subcommand::Encrypt {
            input,
            server_1_encrypted_share,
            server_2_encrypted_share,
        } => {
            let client: Client<FieldPrio2> = Client::new(
                options.dimension,
                PublicKey::from(&server_1_private_key),
                PublicKey::from(&server_2_private_key),
            )
            .wrap_err("could not create client")?;
            encrypt(
                client,
                input,
                server_1_encrypted_share,
                server_2_encrypted_share,
            )
        }
        Subcommand::Decrypt {
            server_1_encrypted_share,
            server_2_encrypted_share,
            output,
            pretty_print,
        } => {
            let server_1: Server<FieldPrio2> =
                Server::new(options.dimension, true, server_1_private_key)
                    .wrap_err("could not create first server")?;
            let server_2: Server<FieldPrio2> =
                Server::new(options.dimension, false, server_2_private_key)
                    .wrap_err("could not create second server")?;
            decrypt(
                server_1,
                server_2,
                server_1_encrypted_share,
                server_2_encrypted_share,
                output,
                pretty_print,
            )
        }
    }
}
