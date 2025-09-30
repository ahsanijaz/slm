use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::Rng;
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
    println!("\n--- Embedding Layer Test ---");

    // Define some parameters
    let embedding_dim = 10; // Each character will be represented by a 10-element vector

    // Create the layer
    let embedding_layer = Embedding::new(vocab_size, embedding_dim);

    // Get the tokens for "hello" from your tokenizer
    let hello_tokens = encode("hello");
    println!("Input tokens for 'hello': {:?}", hello_tokens);

    // Perform the forward pass
    let hello_vectors = embedding_layer.forward(&hello_tokens);

    println!("Output shape: {:?}", hello_vectors.shape());
    // The output shape should be [5, 10], meaning 5 tokens,
    // each represented by a 10-dimensional vector.
    println!("\n--- Self-Attention Head Test ---");
    let batch_size = 5; // e.g., "hello"
    let embedding_dim = 10;
    let head_size = 16;

    // Create a dummy input (the output of an embedding layer)
    let attention_input = Array::random((batch_size, embedding_dim), StandardNormal);
    println!("Attention Input shape: {:?}", attention_input.shape());

    // Create the attention head
    let attention_head = SelfAttentionHead::new(embedding_dim, head_size);

    // Perform the forward pass
    let attention_output = attention_head.forward(&attention_input);

    println!("Attention Output shape: {:?}", attention_output.shape());
    // The output shape should be [5, 16], as it's a weighted sum of Value vectors.
    let full_data = encode(&text);

    // --- Hyperparameters ---
    let embedding_dim = 32;
    let num_blocks = 3;
    let head_size = embedding_dim;
    let block_size = 8; // The length of the sequences we'll train on
    let learning_rate = 1e-3;
    let training_iterations = 1000;

    // --- Model Initialization ---
    let mut model = LanguageModel::new(vocab_size, embedding_dim, num_blocks, head_size);
    println!("Model created. Starting training...");

    // --- The Training Loop ---
    for i in 0..training_iterations {
        // 1. Get a random batch of data
        let mut rng = rand::thread_rng();
        let start_index = rng.gen_range(0..full_data.len() - block_size);
        let end_index = start_index + block_size;

        let inputs = &full_data[start_index..end_index];
        let targets = &full_data[start_index + 1..end_index + 1];

        // 2. Forward pass: Get the model's predictions
        let logits = model.forward(inputs);

        // 3. Calculate the loss
        let loss = cross_entropy_loss(&logits, targets);
        if i % 100 == 0 {
            println!("Iteration {}: Loss = {}", i, loss);
        }

        // 4. Backward pass (calculates gradients)
        model.backward();

        // 5. Update weights (optimizer step)
        model.update_weights(learning_rate);
    }
    println!("Training finished.");
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
    fn backward(&mut self) { /* Placeholder for backpropagation */
    }
    fn update_weights(&mut self, learning_rate: f32) { /* Placeholder for optimizer step */
    }
}
// add
/// An embedding layer that turns tokens into vectors.
struct Embedding {
    // This matrix is our learnable lookup table.
    // Shape: (vocabulary_size, embedding_dimension)
    embedding_matrix: Array2<f32>,
}

impl Embedding {
    /// Creates a new Embedding layer with a randomly initialized matrix.
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let embedding_matrix = Array::random((vocab_size, embedding_dim), StandardNormal);
        Embedding { embedding_matrix }
    }

    /// Performs the forward pass: looks up the embedding vectors for input tokens.
    fn forward(&self, input_tokens: &[i32]) -> Array2<f32> {
        // `select` is a powerful `ndarray` method for indexing.
        // We are selecting rows (Axis(0)) from the embedding matrix
        // at the indices specified by `input_tokens`.
        self.embedding_matrix.select(
            ndarray::Axis(0),
            &input_tokens.iter().map(|&i| i as usize).collect::<Vec<_>>(),
        )
    }
    fn backward(&mut self) { /* Placeholder for backpropagation */
    }
    fn update_weights(&mut self, learning_rate: f32) { /* Placeholder for optimizer step */
    }
}

/// A single head of self-attention.
struct SelfAttentionHead {
    query: Linear,
    key: Linear,
    value: Linear,
}

impl SelfAttentionHead {
    /// Creates a new self-attention head.
    /// `embedding_dim`: The dimension of the input vectors.
    /// `head_size`: The dimension of the query, key, and value vectors.
    fn new(embedding_dim: usize, head_size: usize) -> Self {
        Self {
            query: Linear::new(embedding_dim, head_size),
            key: Linear::new(embedding_dim, head_size),
            value: Linear::new(embedding_dim, head_size),
        }
    }

    /// Performs the forward pass for self-attention network.
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (t, _c) = (input.shape()[0], input.shape()[1]);

        // 1. Get Query, Key, and Value vectors for all tokens in the sequence.
        let q = self.query.forward(input);
        let k = self.key.forward(input);
        let v = self.value.forward(input);

        // 2. Compute attention scores ("affinities").
        //    q @ k.T --> (T, C) @ (C, T) = (T, T)
        let mut weights = q.dot(&k.t()); // k.t() is the transpose of k

        // 3. Apply a causal mask to prevent "cheating".
        //    A token should only see itself and past tokens, not future ones.
        //    We set the scores for future tokens to negative infinity.
        for i in 0..t {
            for j in (i + 1)..t {
                weights[[i, j]] = f32::NEG_INFINITY;
            }
        }

        // 4. Apply softmax to normalize the scores into weights.
        weights = softmax(&weights, 1);

        // 5. Perform the weighted aggregation of the Value vectors.
        let output = weights.dot(&v);
        output
    }
    fn backward(&mut self) {
        self.query.backward();
        self.key.backward();
        self.value.backward();
    }
    fn update_weights(&mut self, learning_rate: f32) {
        self.query.update_weights(learning_rate);
        self.key.update_weights(learning_rate);
        self.value.update_weights(learning_rate);
    }
}

// A single block of the transformer architecture.
struct TransformerBlock {
    attention: SelfAttentionHead,
    ffn_linear1: Linear,
    ffn_linear2: Linear,
}
impl TransformerBlock {
    fn new(embedding_dim: usize, head_size: usize) -> Self {
        Self {
            attention: SelfAttentionHead::new(embedding_dim, head_size),
            // The feed-forward network. The inner layer is often larger.
            ffn_linear1: Linear::new(embedding_dim, 4 * embedding_dim),
            ffn_linear2: Linear::new(4 * embedding_dim, embedding_dim),
        }
    }

    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // Pass through self-attention
        let attention_output = self.attention.forward(input);
        // Pass through the feed-forward network
        let ffn_output = self.ffn_linear1.forward(&attention_output);
        let ffn_output = relu(&ffn_output); // Apply activation function
        self.ffn_linear2.forward(&ffn_output)
    }
    fn backward(&mut self) {
        self.attention.backward();
        self.ffn_linear1.backward();
        self.ffn_linear2.backward();
    }
    fn update_weights(&mut self, learning_rate: f32) {
        self.attention.update_weights(learning_rate);
        self.ffn_linear1.update_weights(learning_rate);
        self.ffn_linear2.update_weights(learning_rate);
    }
}

// --- Add the final LanguageModel struct and its impl block ---

/// The complete language model.
struct LanguageModel {
    token_embedding_table: Embedding,
    blocks: Vec<TransformerBlock>,
    lm_head: Linear,
}

impl LanguageModel {
    fn new(vocab_size: usize, embedding_dim: usize, num_blocks: usize, head_size: usize) -> Self {
        let token_embedding_table = Embedding::new(vocab_size, embedding_dim);

        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(TransformerBlock::new(embedding_dim, head_size));
        }

        // The final layer projects the output back to the vocabulary size
        let lm_head = Linear::new(embedding_dim, vocab_size);

        Self {
            token_embedding_table,
            blocks,
            lm_head,
        }
    }

    fn forward(&self, input_tokens: &[i32]) -> Array2<f32> {
        // 1. Get token embeddings
        let mut x = self.token_embedding_table.forward(input_tokens);

        // 2. Pass through all the transformer blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 3. Get the final output logits
        let logits = self.lm_head.forward(&x);
        logits
    }
    fn backward(&mut self) {
        self.lm_head.backward();
        for block in self.blocks.iter_mut().rev() {
            block.backward();
        }
        self.token_embedding_table.backward();
    }
    fn update_weights(&mut self, learning_rate: f32) {
        self.lm_head.update_weights(learning_rate);
        for block in &mut self.blocks {
            block.update_weights(learning_rate);
        }
        self.token_embedding_table.update_weights(learning_rate);
    }
}

// Helper function for the ReLU activation
fn relu(matrix: &Array2<f32>) -> Array2<f32> {
    matrix.mapv(|x| x.max(0.0))
}
fn cross_entropy_loss(logits: &Array2<f32>, targets: &[i32]) -> f32 {
    let mut loss = 0.0;
    for (i, &target_token) in targets.iter().enumerate() {
        let logit_row = logits.row(i);
        // Apply softmax to get probabilities
        let max_logit = logit_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Array1<f32> = logit_row.mapv(|logit| (logit - max_logit).exp());
        let sum_exp_logits = exp_logits.sum();
        let probs = exp_logits / sum_exp_logits;

        // Calculate the negative log likelihood for the target token
        loss -= probs[target_token as usize].ln();
    }
    loss / targets.len() as f32 // Average the loss
}
/// A simple softmax function applied along a specific axis.
fn softmax(matrix: &Array2<f32>, axis: usize) -> Array2<f32> {
    let mut exp_matrix = matrix.mapv(f32::exp);
    let sum_exp = exp_matrix.sum_axis(ndarray::Axis(axis));

    // Use broadcasting to divide each row/column by its sum.
    exp_matrix / &sum_exp.insert_axis(ndarray::Axis(axis))
}
