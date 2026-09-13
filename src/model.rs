use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub const MAX_LEN: usize = 12;
pub const HIDDEN_SIZE: usize = 40;

#[derive(Serialize, Deserialize)]
pub struct NameNet {
    pub alphabet: String,
    pub max_len: usize,
    pub hidden_size: usize,
    pub input_size: usize,
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
}

fn sigmoid(z: f64) -> f64 {
    if z < -500.0 {
        0.0
    } else if z > 500.0 {
        1.0
    } else {
        1.0 / (1.0 + (-z).exp())
    }
}

impl NameNet {
    pub fn new(alphabet: &str, seed: u64) -> Self {
        let alpha_size = alphabet.chars().count();
        let input_size = 2 * MAX_LEN * alpha_size + alpha_size * alpha_size;

        // Simple LCG for reproducible pseudo-random initialization.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };

        let scale1 = 1.0 / (input_size as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..HIDDEN_SIZE)
            .map(|_| (0..input_size).map(|_| (next() * 2.0 - 1.0) * scale1).collect())
            .collect();
        let b1 = vec![0.0; HIDDEN_SIZE];

        let scale2 = 1.0 / (HIDDEN_SIZE as f64).sqrt();
        let w2: Vec<f64> = (0..HIDDEN_SIZE).map(|_| (next() * 2.0 - 1.0) * scale2).collect();
        let b2 = 0.0;

        NameNet {
            alphabet: alphabet.to_string(),
            max_len: MAX_LEN,
            hidden_size: HIDDEN_SIZE,
            input_size,
            w1,
            b1,
            w2,
            b2,
        }
    }

    fn char_to_idx(&self, ch: char) -> Option<usize> {
        self.alphabet.chars().position(|c| c == ch)
    }

    fn pad_idx(&self) -> usize {
        self.char_to_idx('_').unwrap()
    }

    fn suffix_idx(&self, name: &str) -> usize {
        let chars: Vec<char> = name.chars().collect();
        let n = chars.len();
        let alpha_size = self.alphabet.chars().count();

        let a = if n >= 2 {
            self.char_to_idx(chars[n - 2].to_ascii_lowercase()).unwrap_or(0)
        } else {
            0
        };
        let b = if n >= 1 {
            self.char_to_idx(chars[n - 1].to_ascii_lowercase()).unwrap_or(0)
        } else {
            0
        };
        a * alpha_size + b
    }

    pub fn encode_sparse(&self, name: &str) -> Vec<(usize, f64)> {
        let lowered = name.to_lowercase();
        let chars: Vec<char> = lowered.chars().collect();
        let n = chars.len();
        let alpha_size = self.alphabet.chars().count();
        let pad = self.pad_idx();

        let mut out = Vec::with_capacity(2 * MAX_LEN + 1);

        for (pos, ch) in chars.iter().take(MAX_LEN).enumerate() {
            let idx = self.char_to_idx(*ch).unwrap_or(pad);
            out.push((pos * alpha_size + idx, 1.0));
        }

        for k in 1..=MAX_LEN {
            let ch = if k <= n { chars[n - k] } else { '_' };
            let idx = self.char_to_idx(ch).unwrap_or(pad);
            let offset = MAX_LEN * alpha_size + (k - 1) * alpha_size;
            out.push((offset + idx, 1.0));
        }

        let suffix_offset = 2 * MAX_LEN * alpha_size;
        out.push((suffix_offset + self.suffix_idx(&lowered), 1.0));

        out
    }

    pub fn forward(&self, x_sparse: &[(usize, f64)]) -> (Vec<f64>, f64) {
        let mut h = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let wi = &self.w1[i];
            let mut z = self.b1[i];
            for &(j, v) in x_sparse {
                z += wi[j] * v;
            }
            h[i] = sigmoid(z);
        }

        let mut z_out = self.b2;
        for i in 0..self.hidden_size {
            z_out += self.w2[i] * h[i];
        }
        let y = sigmoid(z_out);

        (h, y)
    }

    pub fn train_step(&mut self, x_sparse: &[(usize, f64)], target: f64, lr: f64) -> f64 {
        let (h, y) = self.forward(x_sparse);
        let error = target - y;
        let d_y = error * y * (1.0 - y);

        for i in 0..self.hidden_size {
            let hi = h[i];
            let d_hi = d_y * self.w2[i] * hi * (1.0 - hi);
            for &(j, v) in x_sparse {
                self.w1[i][j] += lr * d_hi * v;
            }
            self.b1[i] += lr * d_hi;
            self.w2[i] += lr * d_y * hi;
        }
        self.b2 += lr * d_y;

        error * error
    }

    pub fn predict(&self, name: &str) -> (&'static str, f64) {
        let x = self.encode_sparse(name);
        let (_, y) = self.forward(&x);
        let label = if y > 0.5 { "female" } else { "male" };
        (label, y)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let net: NameNet = serde_json::from_reader(reader)?;
        Ok(net)
    }
}
