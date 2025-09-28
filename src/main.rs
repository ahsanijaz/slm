use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use std::collections::{HashMap, HashSet};
use std::fs;
fn main() {
    // Read the text data from a file
    let text = fs::read_to_string("input.txt").expect("Failed to read input.txt");
    println!("Loaded {} characters of text.", text.len());

    // Create the vocabulary (all unique characters)
    let mut chars: Vec<char> = text
        .chars()
        .collect::<HashSet<char>>()
        .into_iter()
        .collect();
    chars.sort();
    let vocab_size = chars.len();
    println!("Vocabulary size: {}", vocab_size);
    println!("Vocabulary: {:?}", chars.iter().collect::<String>());

    // Create character-to-integer (stoi) and integer-to-character (itos) mappings
    let stoi: HashMap<char, i32> = chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| (ch, i as i32))
        .collect();
    let itos: HashMap<i32, char> = chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| (i as i32, ch))
        .collect();

    // Simple encoder and decoder functions
    let encode = |s: &str| -> Vec<i32> { s.chars().map(|ch| stoi[&ch]).collect() };
    let decode = |v: &[i32]| -> String { v.iter().map(|&i| itos[&i]).collect() };

    let test_string = "hello world";
    let encoded = encode(test_string);
    let decoded = decode(&encoded);

    println!("\n--- Tokenization Example ---");
    println!("Original: {}", test_string);
    println!("Encoded (Tokens): {:?}", encoded);
    println!("Decoded: {}", decoded);

    println!("\n--- Linear Layer Test ---");

    // Create a layer that takes 10 input features and produces 20 output features.
    let linear_layer = Linear::new(10, 20);

    // Create a dummy input batch of 5 samples, each with 10 features.
    let input = Array::random((5, 10), StandardNormal);

    println!("Input shape: {:?}", input.shape());

    // Perform the forward pass.
    let output = linear_layer.forward(&input);

    println!("Output shape: {:?}", output.shape());
    // The output shape should be [5, 20], meaning 5 samples, each with 20 features.
}
struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
}

impl Linear {
    /// Creates a new Linear layer with randomly initialized weights and biases.
    fn new(in_features: usize, out_features: usize) -> Self {
        // Initialize weights with random values from a standard normal distribution.
        // This helps the model learn effectively.
        let weights = Array::random((in_features, out_features), StandardNormal);
        // Initialize biases with zeros.
        let bias = Array::zeros((1, out_features));

        Linear { weights, bias }
    }

    /// Performs the forward pass: output = input * weights + bias
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // `input.dot(&self.weights)` is the matrix multiplication.
        // `+ &self.bias` adds the bias vector to each row of the result.
        input.dot(&self.weights) + &self.bias
    }
}
