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
}
