use crate::model::NameNet;
use std::io::{self, Write};

fn print_prediction(net: &NameNet, name: &str) {
    let (label, prob) = net.predict(name);
    let confidence = prob.max(1.0 - prob);
    println!(
        "{} -> {}  (confidence {:.1}%)",
        name,
        label,
        confidence * 100.0
    );
}

pub fn run(weights_path: &str, name_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let net = NameNet::load(weights_path)?;

    if !name_args.is_empty() {
        let name = name_args.join(" ");
        print_prediction(&net, &name);
        return Ok(());
    }

    println!("Enter a name (or 'exit' to quit):");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break;
        }

        let name = line.trim();
        if name.is_empty() {
            continue;
        }
        if matches!(name.to_lowercase().as_str(), "exit" | "quit" | "выход") {
            break;
        }
        print_prediction(&net, name);
    }

    Ok(())
}
