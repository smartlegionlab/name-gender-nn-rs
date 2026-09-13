use regex::Regex;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct Dataset {
    alphabet: String,
    female: Vec<String>,
    male: Vec<String>,
}

pub fn run(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let data: Dataset = serde_json::from_reader(reader)?;

    let chars: String = data.alphabet.chars().filter(|&c| c != '_').collect();
    let escaped = regex::escape(&chars);
    let pattern = Regex::new(&format!("^[{}]+$", escaped))?;

    println!("File: {}", path);
    println!("Alphabet size: {}", data.alphabet.chars().count());
    println!(
        "Loaded {} female and {} male names",
        data.female.len(),
        data.male.len()
    );
    println!();

    let mut failed = false;

    let mut bad: Vec<(&str, &str)> = Vec::new();
    for name in &data.female {
        if !pattern.is_match(name) {
            bad.push(("female", name));
        }
    }
    for name in &data.male {
        if !pattern.is_match(name) {
            bad.push(("male", name));
        }
    }
    if bad.is_empty() {
        println!("Alphabet check: OK");
    } else {
        failed = true;
        println!("Names with characters outside the alphabet:");
        for (gender, name) in bad {
            println!("  [{}] {:?}", gender, name);
        }
    }

    let mut dup_found = false;
    for (label, names) in [("female", &data.female), ("male", &data.male)] {
        let mut seen = HashSet::new();
        let mut repeated = Vec::new();
        for n in names {
            if !seen.insert(n.clone()) {
                if !repeated.contains(n) {
                    repeated.push(n.clone());
                }
            }
        }
        if !repeated.is_empty() {
            dup_found = true;
            failed = true;
            println!("Duplicates in {}:", label);
            for n in repeated {
                println!("  {}", n);
            }
        }
    }
    if !dup_found {
        println!("Duplicate check: OK");
    }

    let female_set: HashSet<&String> = data.female.iter().collect();
    let male_set: HashSet<&String> = data.male.iter().collect();
    let cross: Vec<&&String> = female_set.intersection(&male_set).collect();
    if cross.is_empty() {
        println!("Cross-gender check: OK");
    } else {
        failed = true;
        println!("Names present in both female and male:");
        for name in cross {
            println!("  {}", name);
        }
    }

    if failed {
        std::process::exit(1);
    }

    println!();
    println!("All checks passed.");
    Ok(())
}
