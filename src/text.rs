//! Text generation utilities for character-level language models.
//!
//! Provides vocabulary management, character embeddings, and sampling strategies.

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::optimizers::Optimizer;

/// Character vocabulary for text generation tasks.
///
/// Maps characters to indices and vice versa.
#[derive(Clone, Debug)]
pub struct TextVocabulary {
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    vocab_size: usize,
}

impl TextVocabulary {
    /// Create vocabulary from text, extracting unique characters.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        chars.sort();

        let vocab_size = chars.len();
        let char_to_idx: HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        let idx_to_char: HashMap<usize, char> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        Self { char_to_idx, idx_to_char, vocab_size }
    }

    /// Create vocabulary from explicit character list.
    pub fn from_chars(chars: &[char]) -> Self {
        let vocab_size = chars.len();
        let char_to_idx: HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        let idx_to_char: HashMap<usize, char> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        Self { char_to_idx, idx_to_char, vocab_size }
    }

    /// Get index for a character.
    pub fn char_to_index(&self, ch: char) -> Option<usize> {
        self.char_to_idx.get(&ch).copied()
    }

    /// Get character for an index.
    pub fn index_to_char(&self, idx: usize) -> Option<char> {
        self.idx_to_char.get(&idx).copied()
    }

    /// Get vocabulary size.
    pub fn size(&self) -> usize {
        self.vocab_size
    }

    /// Check if character is in vocabulary.
    pub fn contains(&self, ch: char) -> bool {
        self.char_to_idx.contains_key(&ch)
    }

    /// Get all characters in vocabulary order.
    pub fn chars(&self) -> Vec<char> {
        let mut chars: Vec<_> = self.idx_to_char.iter().collect();
        chars.sort_by_key(|(idx, _)| *idx);
        chars.into_iter().map(|(_, &ch)| ch).collect()
    }

    /// Encode string to indices.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|ch| self.char_to_index(ch))
            .collect()
    }

    /// Decode indices to string.
    pub fn decode(&self, indices: &[usize]) -> String {
        indices.iter()
            .filter_map(|&idx| self.index_to_char(idx))
            .collect()
    }
}

/// Gradients for character embedding layer.
#[derive(Clone, Debug)]
pub struct EmbeddingGradients {
    pub weight: Array2<f64>,
}

/// Trainable character embedding layer.
///
/// Maps character indices to dense vectors.
#[derive(Clone, Debug)]
pub struct CharacterEmbedding {
    pub weight: Array2<f64>, // (vocab_size, embed_dim)
    vocab_size: usize,
    embed_dim: usize,
    input_cache: Option<Vec<usize>>,
}

impl CharacterEmbedding {
    /// Create new embedding with random initialization.
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let scale = (1.0 / embed_dim as f64).sqrt();
        let weight = Array2::random((vocab_size, embed_dim), Uniform::new(-scale, scale));

        Self {
            weight,
            vocab_size,
            embed_dim,
            input_cache: None,
        }
    }

    /// Create embedding with zero initialization.
    pub fn new_zeros(vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            weight: Array2::zeros((vocab_size, embed_dim)),
            vocab_size,
            embed_dim,
            input_cache: None,
        }
    }

    /// Create embedding from existing weights.
    pub fn from_weights(weight: Array2<f64>) -> Self {
        let (vocab_size, embed_dim) = weight.dim();
        Self {
            weight,
            vocab_size,
            embed_dim,
            input_cache: None,
        }
    }

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Lookup single character embedding.
    pub fn lookup(&self, char_idx: usize) -> Array1<f64> {
        assert!(char_idx < self.vocab_size, "Index {} out of vocabulary size {}", char_idx, self.vocab_size);
        self.weight.row(char_idx).to_owned()
    }

    /// Forward pass for sequence of indices.
    /// Returns (seq_len, embed_dim) matrix.
    pub fn forward(&mut self, char_indices: &[usize]) -> Array2<f64> {
        self.input_cache = Some(char_indices.to_vec());

        let seq_len = char_indices.len();
        let mut output = Array2::zeros((seq_len, self.embed_dim));

        for (i, &idx) in char_indices.iter().enumerate() {
            assert!(idx < self.vocab_size, "Index {} out of vocabulary size {}", idx, self.vocab_size);
            output.row_mut(i).assign(&self.weight.row(idx));
        }

        output
    }

    /// Backward pass - compute gradients.
    /// grad_output shape: (seq_len, embed_dim)
    pub fn backward(&self, grad_output: &Array2<f64>) -> EmbeddingGradients {
        let indices = self.input_cache.as_ref().expect("No cached input for backward pass");

        let mut weight_grad = Array2::zeros((self.vocab_size, self.embed_dim));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..self.embed_dim {
                weight_grad[[idx, j]] += grad_output[[i, j]];
            }
        }

        EmbeddingGradients { weight: weight_grad }
    }

    /// Update parameters with optimizer.
    pub fn update_parameters<O: Optimizer>(&mut self, gradients: &EmbeddingGradients, optimizer: &mut O, prefix: &str) {
        optimizer.update(&format!("{}_weight", prefix), &mut self.weight, &gradients.weight);
    }

    /// Get number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.weight.len()
    }
}

/// Sample from logits with temperature scaling.
///
/// Higher temperature = more random, lower = more deterministic.
pub fn sample_with_temperature(logits: &Array1<f64>, temperature: f64) -> usize {
    assert!(temperature > 0.0, "Temperature must be positive");

    // Scale logits by temperature
    let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();

    // Softmax with numerical stability
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    let probs: Vec<f64> = exp_vals.iter().map(|&x| x / sum).collect();

    // Sample from distribution
    let mut rng_val = rand::random::<f64>();
    for (i, &prob) in probs.iter().enumerate() {
        rng_val -= prob;
        if rng_val <= 0.0 {
            return i;
        }
    }

    probs.len() - 1
}

/// Sample from top-k most likely tokens.
///
/// Filters to k highest probability tokens before sampling.
pub fn sample_top_k(logits: &Array1<f64>, k: usize, temperature: f64) -> usize {
    assert!(k > 0, "k must be positive");
    assert!(temperature > 0.0, "Temperature must be positive");

    let k = k.min(logits.len());

    // Get indices sorted by logit value (descending)
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Keep top k
    let top_k: Vec<(usize, f64)> = indexed.into_iter().take(k).collect();

    // Apply temperature and softmax to top-k only
    let scaled: Vec<f64> = top_k.iter().map(|(_, v)| v / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    let probs: Vec<f64> = exp_vals.iter().map(|&x| x / sum).collect();

    // Sample
    let mut rng_val = rand::random::<f64>();
    for (i, &prob) in probs.iter().enumerate() {
        rng_val -= prob;
        if rng_val <= 0.0 {
            return top_k[i].0;
        }
    }

    top_k[k - 1].0
}

/// Nucleus (top-p) sampling.
///
/// Samples from smallest set of tokens whose cumulative probability exceeds p.
pub fn sample_nucleus(logits: &Array1<f64>, p: f64, temperature: f64) -> usize {
    assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1]");
    assert!(temperature > 0.0, "Temperature must be positive");

    // Apply temperature and softmax
    let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();
    let probs: Vec<f64> = exp_vals.iter().map(|&x| x / sum).collect();

    // Sort by probability (descending)
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find nucleus (cumulative prob >= p)
    let mut cumulative = 0.0;
    let mut nucleus: Vec<(usize, f64)> = Vec::new();
    for (idx, prob) in indexed {
        cumulative += prob;
        nucleus.push((idx, prob));
        if cumulative >= p {
            break;
        }
    }

    // Renormalize nucleus probabilities
    let nucleus_sum: f64 = nucleus.iter().map(|(_, prob)| prob).sum();
    let nucleus_probs: Vec<f64> = nucleus.iter().map(|(_, prob)| prob / nucleus_sum).collect();

    // Sample from nucleus
    let mut rng_val = rand::random::<f64>();
    for (i, &prob) in nucleus_probs.iter().enumerate() {
        rng_val -= prob;
        if rng_val <= 0.0 {
            return nucleus[i].0;
        }
    }

    nucleus.last().map(|(idx, _)| *idx).unwrap_or(0)
}

/// Get argmax (greedy decoding).
pub fn argmax(logits: &Array1<f64>) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Apply softmax to logits.
pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_vals: Array1<f64> = logits.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_vocabulary_from_text() {
        let vocab = TextVocabulary::from_text("hello");
        assert_eq!(vocab.size(), 4); // h, e, l, o
        assert!(vocab.contains('h'));
        assert!(vocab.contains('l'));
        assert!(!vocab.contains('x'));
    }

    #[test]
    fn test_vocabulary_encode_decode() {
        let vocab = TextVocabulary::from_text("abc");
        let encoded = vocab.encode("cab");
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, "cab");
    }

    #[test]
    fn test_embedding_forward() {
        let mut emb = CharacterEmbedding::new(10, 8);
        let output = emb.forward(&[0, 3, 5]);
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = CharacterEmbedding::new(10, 8);
        let vec = emb.lookup(5);
        assert_eq!(vec.len(), 8);
    }

    #[test]
    fn test_sample_with_temperature() {
        let logits = arr1(&[1.0, 2.0, 3.0]);
        let idx = sample_with_temperature(&logits, 1.0);
        assert!(idx < 3);
    }

    #[test]
    fn test_sample_top_k() {
        let logits = arr1(&[1.0, 5.0, 2.0, 0.5]);
        let idx = sample_top_k(&logits, 2, 1.0);
        // Should only sample from indices 1 or 2 (top 2)
        assert!(idx == 1 || idx == 2);
    }

    #[test]
    fn test_sample_nucleus() {
        let logits = arr1(&[0.0, 10.0, 0.0]); // Very peaked distribution
        let idx = sample_nucleus(&logits, 0.9, 1.0);
        assert_eq!(idx, 1); // Should almost always be 1
    }

    #[test]
    fn test_argmax() {
        let logits = arr1(&[1.0, 5.0, 2.0]);
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_softmax() {
        let logits = arr1(&[1.0, 2.0, 3.0]);
        let probs = softmax(&logits);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
    }
}
