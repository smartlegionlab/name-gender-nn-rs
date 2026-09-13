use crate::model::NameNet;
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

const LR: f64 = 0.1;
const EPOCHS: usize = 8000;

#[derive(Deserialize)]
struct Dataset {
    alphabet: String,
    female: Vec<String>,
    male: Vec<String>,
}

fn load_data(path: &str) -> Result<(String, Vec<(String, f64)>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let raw: Dataset = serde_json::from_reader(reader)?;

    let mut data = Vec::with_capacity(raw.female.len() + raw.male.len());
    for name in raw.female {
        data.push((name, 1.0));
    }
    for name in raw.male {
        data.push((name, 0.0));
    }

    Ok((raw.alphabet, data))
}

pub fn run(data_path: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let (alphabet, data) = load_data(data_path)?;
    println!("Loaded {} names from {}", data.len(), data_path);
    println!("Alphabet size: {}", alphabet.chars().count());

    let mut net = NameNet::new(&alphabet, 42);

    let encoded: Vec<(Vec<(usize, f64)>, f64)> = data
        .iter()
        .map(|(name, target)| (net.encode_sparse(name), *target))
        .collect();

    let t0 = Instant::now();
    for epoch in 0..EPOCHS {
        let mut total_error = 0.0;
        for (x, target) in &encoded {
            total_error += net.train_step(x, *target, LR);
        }

        if epoch % 500 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "epoch {:5}  error={:.6}  t={:.1}s",
                epoch, total_error, elapsed
            );
        }
    }

    let correct: usize = data
        .iter()
        .filter(|(name, target)| {
            let (_, y) = net.predict(name);
            (y > 0.5) == (*target == 1.0)
        })
        .count();

    println!(
        "\nTrain accuracy: {}/{} = {:.1}%",
        correct,
        data.len(),
        100.0 * correct as f64 / data.len() as f64
    );

    net.save(out_path)?;
    println!("Weights saved to {}", out_path);
    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());

    Ok(())
}
