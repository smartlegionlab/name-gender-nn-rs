mod check_data;
mod model;
mod predict;
mod train;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "name-gender-nn-rs")]
#[command(about = "Gender classifier for first names — Rust port", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model from a dataset
    Train {
        #[arg(long)]
        data: String,
        #[arg(long)]
        out: String,
    },
    /// Predict gender for a name
    Predict {
        #[arg(long)]
        weights: String,
        #[arg(trailing_var_arg = true)]
        name: Vec<String>,
    },
    /// Validate a dataset
    Check {
        #[arg(long)]
        data: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train { data, out } => train::run(&data, &out)?,
        Commands::Predict { weights, name } => predict::run(&weights, &name)?,
        Commands::Check { data } => check_data::run(&data)?,
    }

    Ok(())
}
