# name-gender-nn-rs <sup>v0.0.2</sup>

**name-gender-nn-rs** — Rust port of
[name-gender-nn-py](https://github.com/smartlegionlab/name-gender-nn-py).
Same algorithm, same datasets, ~40× faster training.

Gender classifier for first names. A small fully connected neural
network written from scratch, no external ML libraries.

## What it does

Determines whether a first name is male or female. Ships with two
datasets:

- Russian (`data/names_ru.json`) — Cyrillic alphabet, 108 names
- English (`data/names_en.json`) — Latin alphabet, 100 names

Each dataset defines its own alphabet. You train a separate model per
dataset, and each model works only with names from its own alphabet.

## Scope and limitations

The models are trained on **full name forms** (`александр`,
`екатерина`, `alexander`, `elizabeth`). Diminutive and hypocoristic
forms (`коля`, `соня`, `катя` in Russian; `kate`, `bob`, `liz` in
English) are **not** in the datasets and are outside the intended
use case.

On full forms not present in the training data, accuracy is high but
not perfect. See the sanity check section below for real examples.

## Requirements

- Rust 1.85+ (2024 edition)
- Cargo

## Build

```bash
git clone https://github.com/smartlegionlab/name-gender-nn-rs.git
cd name-gender-nn-rs
cargo build --release
```

Binary: `target/release/name-gender-nn-rs`.

## Usage

All commands are subcommands of a single binary.

### Validate a dataset

```bash
cargo run --release -- check --data data/names_ru.json
cargo run --release -- check --data data/names_en.json
```

Output for the Russian dataset:

```
File: data/names_ru.json
Alphabet size: 34
Loaded 49 female and 59 male names

Alphabet check: OK
Duplicate check: OK
Cross-gender check: OK

All checks passed.
```

This checks:
- all names use only characters from the dataset alphabet
- no duplicates inside each list
- no name appears in both `female` and `male`

### Train

```bash
cargo run --release -- train --data data/names_ru.json --out weights_ru.json
cargo run --release -- train --data data/names_en.json --out weights_en.json
```

Output for the Russian dataset:

```
Loaded 108 names from data/names_ru.json
Alphabet size: 34
epoch     0  error=13.671195  t=0.0s
epoch   500  error=0.019290  t=0.1s
...
epoch  7500  error=0.000797  t=0.9s

Train accuracy: 108/108 = 100.0%
Weights saved to weights_ru.json
Total time: 1.0s
```

### Predict

```bash
cargo run --release -- predict --weights weights_ru.json александр
cargo run --release -- predict --weights weights_en.json alexander
```

One-liner output:

```
александр -> male  (confidence 99.7%)
alexander -> male  (confidence 99.8%)
```

Interactive mode (no name argument):

```bash
cargo run --release -- predict --weights weights_ru.json
```

```
Enter a name (or 'exit' to quit):
> анна
анна -> female  (confidence 99.8%)
> exit
```

## Performance

Measured on the same machine as the Python reference implementation.

| Task                | Python | Rust (release) |
|---------------------|--------|----------------|
| Train Russian model | ~41 s  | ~1.0 s         |
| Train English model | ~38 s  | ~0.9 s         |
| Single prediction   | ~0.2 s | ~0.01 s        |

Same algorithm, same accuracy. The speedup comes from the release
build profile (`opt-level = 3`, `lto = true`, `codegen-units = 1`).

## Sanity check

Inputs from the datasets:

```
анна       -> female  (confidence 99.8%)
дмитрий    -> male    (confidence 100.0%)
ольга      -> female  (confidence 99.5%)
александр  -> male    (confidence 99.7%)
екатерина  -> female  (confidence 100.0%)
mary       -> female  (confidence 99.0%)
john       -> male    (confidence 100.0%)
elizabeth  -> female  (confidence 99.7%)
alexander  -> male    (confidence 99.8%)
catherine  -> female  (confidence 99.9%)
```

Full forms not in the datasets:

```
сара       -> female  (confidence 85.5%)
мара       -> female  (confidence 93.9%)
федот      -> male    (confidence 96.1%)
жанна      -> female  (confidence 99.9%)
ева        -> female  (confidence 62.1%)
karl       -> male    (confidence 95.0%)
victoria   -> female  (confidence 94.9%)
diana      -> female  (confidence 99.7%)
george     -> male    (confidence 100.0%)
peter      -> female  (confidence 99.7%)   wrong
```

19 out of 20 correct. The single failure (`peter`) is a known
limitation of the small training set: the model has never seen a
short male name ending in `-er`, and the closest patterns in the
training data are female (`elizabeth`, `catherine`, `mary`).

## Project layout

```
name-gender-nn-rs/
├── Cargo.toml
├── data/
│   ├── names_ru.json   # Russian dataset (Cyrillic)
│   └── names_en.json   # English dataset (Latin)
├── src/
│   ├── main.rs         # CLI entry point (clap)
│   ├── model.rs        # NameNet: forward, train_step, save, load
│   ├── train.rs        # training
│   ├── predict.rs      # inference
│   └── check_data.rs   # dataset validation
├── LICENSE             # BSD 3-Clause License
├── DISCLAIMER.md       # full legal disclaimer
└── README.md
```

`weights_*.json` files are generated by `train` and are not tracked
by git.

## How it works

Same as the Python reference:

- **Alphabet.** Each dataset defines its own alphabet as a string of
  characters plus a `_` pad character. The Russian dataset uses the
  33-letter Cyrillic alphabet; the English dataset uses the 26-letter
  Latin alphabet.
- **Features.** Each name is split into three blocks: letters from the
  start, letters from the end, and the 2-letter suffix. Every character
  is one-hot encoded over the dataset alphabet.
- **Model.** Fully connected: input -> hidden layer (40 neurons,
  sigmoid) -> output (1 neuron, sigmoid). Trained with SGD, `lr = 0.1`,
  8000 epochs.
- **Data.** Russian: 108 names (49 female, 59 male). English: 100 names
  (50 female, 50 male).

## Adding names

Open the dataset file and add a name to either `female` or `male`:

```json
{
  "alphabet": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя_",
  "female": ["анна", "...", "александра"],
  "male":   ["иван", "...", "александр"]
}
```

Then retrain:

```bash
cargo run --release -- train --data data/names_ru.json --out weights_ru.json
```

## Adding a new language

Create `data/names_XX.json` with the correct alphabet and name lists,
then train:

```bash
cargo run --release -- train --data data/names_XX.json --out weights_XX.json
cargo run --release -- predict --weights weights_XX.json
```

The model reads the alphabet from the dataset, so any language with a
single-alphabet script works out of the box.

## Related projects

- [name-gender-nn-py](https://github.com/smartlegionlab/name-gender-nn-py) —
  Python reference implementation. Same algorithm, same datasets,
  intended as an educational project.

## Author

Alexander Suvorov — [@smartlegionlab](https://github.com/smartlegionlab)

## License

BSD 3-Clause License. See [LICENSE](LICENSE).

## Disclaimer

**By using this software, you agree to the full disclaimer terms.**

Software provided "AS IS" without warranty. You assume all risks.

Full legal disclaimer: [DISCLAIMER.md](DISCLAIMER.md)

