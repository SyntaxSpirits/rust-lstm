//! Example demonstrating text generation utilities.
//!
//! Shows TextVocabulary, CharacterEmbedding, and sampling functions.

use ndarray::Array2;
use rust_lstm::text::{
    TextVocabulary, CharacterEmbedding,
    sample_with_temperature, sample_top_k, sample_nucleus, argmax, softmax
};
use rust_lstm::layers::linear::LinearLayer;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    println!("Text Generation Utilities Demo");
    println!("==============================\n");

    // 1. Vocabulary
    let text = "Hello, World! This is a test.";
    let vocab = TextVocabulary::from_text(text);

    println!("1. TextVocabulary");
    println!("   Text: \"{}\"", text);
    println!("   Vocabulary size: {}", vocab.size());
    println!("   Characters: {:?}", vocab.chars());

    let encoded = vocab.encode("Hello");
    let decoded = vocab.decode(&encoded);
    println!("   Encode 'Hello': {:?}", encoded);
    println!("   Decode back: \"{}\"", decoded);

    // 2. Character Embedding
    println!("\n2. CharacterEmbedding");
    let embed_dim = 16;
    let mut embedding = CharacterEmbedding::new(vocab.size(), embed_dim);
    println!("   Embedding: {} chars -> {} dimensions", vocab.size(), embed_dim);
    println!("   Parameters: {}", embedding.num_parameters());

    // Lookup single character
    let h_idx = vocab.char_to_index('H').unwrap();
    let h_vec = embedding.lookup(h_idx);
    println!("   'H' embedding (first 4): [{:.3}, {:.3}, {:.3}, {:.3}, ...]",
             h_vec[0], h_vec[1], h_vec[2], h_vec[3]);

    // Forward pass for sequence
    let seq_indices = vocab.encode("Hi");
    let seq_embeddings = embedding.forward(&seq_indices);
    println!("   'Hi' embeddings shape: {:?}", seq_embeddings.shape());

    // 3. LSTM + Linear for text generation
    println!("\n3. LSTM + Linear Pipeline");
    let hidden_size = 32;
    let mut lstm = LSTMNetwork::new(embed_dim, hidden_size, 1);
    let mut output_layer = LinearLayer::new(hidden_size, vocab.size());

    // Process a character
    let char_idx = vocab.char_to_index('H').unwrap();
    let char_emb = embedding.lookup(char_idx);
    let input = Array2::from_shape_vec((embed_dim, 1), char_emb.to_vec()).unwrap();

    let h0 = Array2::zeros((hidden_size, 1));
    let c0 = Array2::zeros((hidden_size, 1));

    let (hidden, _cell) = lstm.forward(&input, &h0, &c0);
    let logits_2d = output_layer.forward(&hidden);
    let logits = logits_2d.column(0).to_owned();

    println!("   Input: 'H' -> embed({}) -> LSTM -> Linear -> logits({})",
             embed_dim, vocab.size());

    // 4. Sampling strategies
    println!("\n4. Sampling Strategies");
    println!("   Logits range: [{:.2}, {:.2}]",
             logits.iter().cloned().fold(f64::INFINITY, f64::min),
             logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Greedy
    let greedy_idx = argmax(&logits);
    let greedy_char = vocab.index_to_char(greedy_idx).unwrap_or('?');
    println!("   Greedy (argmax): '{}' (idx {})", greedy_char, greedy_idx);

    // Temperature sampling
    for temp in [0.5, 1.0, 1.5] {
        let idx = sample_with_temperature(&logits, temp);
        let ch = vocab.index_to_char(idx).unwrap_or('?');
        println!("   Temperature {:.1}: '{}' (idx {})", temp, ch, idx);
    }

    // Top-k sampling
    let k = 5;
    let idx = sample_top_k(&logits, k, 1.0);
    let ch = vocab.index_to_char(idx).unwrap_or('?');
    println!("   Top-{} sampling: '{}' (idx {})", k, ch, idx);

    // Nucleus sampling
    let p = 0.9;
    let idx = sample_nucleus(&logits, p, 1.0);
    let ch = vocab.index_to_char(idx).unwrap_or('?');
    println!("   Nucleus (p={:.1}): '{}' (idx {})", p, ch, idx);

    // 5. Softmax probabilities
    println!("\n5. Probability Distribution");
    let probs = softmax(&logits);
    let mut prob_chars: Vec<_> = probs.iter()
        .enumerate()
        .map(|(i, &p)| (vocab.index_to_char(i).unwrap_or('?'), p))
        .collect();
    prob_chars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("   Top 5 most likely next characters:");
    for (ch, prob) in prob_chars.iter().take(5) {
        let display = if *ch == ' ' { "' '" } else { &ch.to_string() };
        println!("   {:>3}: {:.1}%", display, prob * 100.0);
    }

    println!("\nDone!");
}
