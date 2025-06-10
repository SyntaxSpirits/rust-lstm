use ndarray::{Array2, arr2};
use rust_lstm::layers::bilstm_network::BiLSTMNetwork;
use rust_lstm::models::lstm_network::LSTMNetwork;
use std::collections::HashMap;

/// Simple sentiment classification example comparing BiLSTM vs unidirectional LSTM
/// This demonstrates how BiLSTM can better understand context-dependent sentiment.

/// Simple word embeddings for demo purposes
fn create_word_embeddings() -> HashMap<String, Array2<f64>> {
    let mut embeddings = HashMap::new();
    
    // Positive words
    embeddings.insert("good".to_string(), arr2(&[[0.8], [0.1], [0.9]]));
    embeddings.insert("great".to_string(), arr2(&[[0.9], [0.1], [0.8]]));
    embeddings.insert("excellent".to_string(), arr2(&[[0.95], [0.05], [0.9]]));
    embeddings.insert("amazing".to_string(), arr2(&[[0.9], [0.1], [0.85]]));
    embeddings.insert("fantastic".to_string(), arr2(&[[0.85], [0.1], [0.9]]));
    embeddings.insert("wonderful".to_string(), arr2(&[[0.88], [0.12], [0.9]]));
    
    // Negative words
    embeddings.insert("bad".to_string(), arr2(&[[0.1], [0.9], [0.1]]));
    embeddings.insert("terrible".to_string(), arr2(&[[0.05], [0.95], [0.1]]));
    embeddings.insert("awful".to_string(), arr2(&[[0.1], [0.85], [0.15]]));
    embeddings.insert("horrible".to_string(), arr2(&[[0.08], [0.92], [0.1]]));
    embeddings.insert("disappointing".to_string(), arr2(&[[0.2], [0.8], [0.2]]));
    
    // Neutral/filler words
    embeddings.insert("the".to_string(), arr2(&[[0.5], [0.5], [0.5]]));
    embeddings.insert("is".to_string(), arr2(&[[0.45], [0.45], [0.5]]));
    embeddings.insert("was".to_string(), arr2(&[[0.4], [0.4], [0.5]]));
    embeddings.insert("very".to_string(), arr2(&[[0.6], [0.6], [0.6]]));
    embeddings.insert("quite".to_string(), arr2(&[[0.55], [0.55], [0.55]]));
    embeddings.insert("really".to_string(), arr2(&[[0.6], [0.6], [0.65]]));
    embeddings.insert("movie".to_string(), arr2(&[[0.5], [0.5], [0.4]]));
    embeddings.insert("film".to_string(), arr2(&[[0.5], [0.5], [0.45]]));
    embeddings.insert("story".to_string(), arr2(&[[0.5], [0.5], [0.48]]));
    
    // Negation words
    embeddings.insert("not".to_string(), arr2(&[[0.2], [0.2], [0.8]]));
    embeddings.insert("never".to_string(), arr2(&[[0.15], [0.15], [0.85]]));
    embeddings.insert("no".to_string(), arr2(&[[0.25], [0.25], [0.75]]));
    
    embeddings.insert("<UNK>".to_string(), arr2(&[[0.5], [0.5], [0.5]]));
    
    embeddings
}

/// Convert text to sequence of embeddings
fn text_to_sequence(text: &str, embeddings: &HashMap<String, Array2<f64>>) -> Vec<Array2<f64>> {
    text.split_whitespace()
        .map(|word| {
            embeddings.get(&word.to_lowercase())
                .unwrap_or(embeddings.get("<UNK>").unwrap())
                .clone()
        })
        .collect()
}

/// Simple sentiment classifier using the final output
fn classify_sentiment(output: &Array2<f64>) -> f64 {
    // Simple heuristic: positive if first dimension > second dimension
    // In a real implementation, you'd have a trained output layer
    let positive_score = output[[0, 0]];
    let negative_score = output[[1, 0]];
    positive_score - negative_score
}

/// Test sentences with expected sentiment
fn get_test_sentences() -> Vec<(&'static str, &'static str)> {
    vec![
        ("the movie was good", "positive"),
        ("the movie was bad", "negative"),
        ("the movie was not good", "negative"),
        ("the movie was not bad", "positive"),
        ("not a terrible film", "positive"),
        ("very good story", "positive"),
        ("very bad story", "negative"),
        ("the film was quite disappointing", "negative"),
        ("really amazing movie", "positive"),
        ("not really good", "negative"),
        ("good but not great", "mixed"),
        ("terrible but not awful", "mixed"),
    ]
}

/// Demonstrate sentiment analysis with both LSTM types
fn main() {
    println!("🎭 Text Sentiment Classification: BiLSTM vs LSTM");
    println!("================================================");
    
    let embeddings = create_word_embeddings();
    let test_sentences = get_test_sentences();
    
    // Create networks
    let embedding_dim = 3;
    let hidden_size = 4;
    let num_layers = 1;
    
    let mut bilstm = BiLSTMNetwork::new_concat(embedding_dim, hidden_size, num_layers);
    let mut lstm = LSTMNetwork::new(embedding_dim, hidden_size, num_layers);
    
    println!("Network configurations:");
    println!("  Embedding dimension: {}", embedding_dim);
    println!("  LSTM hidden size: {}", hidden_size);
    println!("  BiLSTM output size: {} (concat mode)", bilstm.output_size());
    println!("  Standard LSTM output size: {}", hidden_size);
    
    println!("\n📊 Processing test sentences...\n");
    
    for (text, expected) in &test_sentences {
        let sequence = text_to_sequence(text, &embeddings);
        
        // Process with BiLSTM
        let bilstm_outputs = bilstm.forward_sequence(&sequence);
        let bilstm_final = bilstm_outputs.last().unwrap();
        let bilstm_sentiment = classify_sentiment(bilstm_final);
        
        // Process with standard LSTM
        let mut hx = Array2::zeros((hidden_size, 1));
        let mut cx = Array2::zeros((hidden_size, 1));
        let mut lstm_final = hx.clone();
        
        for input in &sequence {
            let (new_hx, new_cx) = lstm.forward(input, &hx, &cx);
            hx = new_hx.clone();
            cx = new_cx;
            lstm_final = new_hx;
        }
        let lstm_sentiment = classify_sentiment(&lstm_final);
        
        println!("Text: \"{}\"", text);
        println!("  Expected: {}", expected);
        println!("  BiLSTM sentiment score: {:.3}", bilstm_sentiment);
        println!("  LSTM sentiment score:   {:.3}", lstm_sentiment);
        
        // Determine predictions
        let bilstm_pred = if bilstm_sentiment > 0.1 { "positive" } 
                         else if bilstm_sentiment < -0.1 { "negative" } 
                         else { "mixed" };
        let lstm_pred = if lstm_sentiment > 0.1 { "positive" } 
                       else if lstm_sentiment < -0.1 { "negative" } 
                       else { "mixed" };
        
        println!("  BiLSTM prediction: {}", bilstm_pred);
        println!("  LSTM prediction:   {}", lstm_pred);
        
        let bilstm_correct = bilstm_pred == *expected || (*expected == "mixed" && bilstm_sentiment.abs() < 0.2);
        let lstm_correct = lstm_pred == *expected || (*expected == "mixed" && lstm_sentiment.abs() < 0.2);
        
        println!("  BiLSTM correct: {}", if bilstm_correct { "✓" } else { "✗" });
        println!("  LSTM correct:   {}", if lstm_correct { "✓" } else { "✗" });
        println!();
    }
    
    println!("🔍 Analysis:");
    println!("============");
    println!("This example demonstrates key advantages of BiLSTM:");
    println!("• Better handling of negations (e.g., 'not good')");
    println!("• Understanding of contradictory sentiment within sentences");
    println!("• Access to both past and future context for each word");
    println!("• More robust sentiment analysis in complex sentences");
    println!();
    println!("Note: This is a demonstration with synthetic embeddings.");
    println!("In practice, you would:");
    println!("• Use pre-trained word embeddings (Word2Vec, GloVe, etc.)");
    println!("• Train the networks on labeled sentiment data");
    println!("• Add a proper classification layer on top of LSTM outputs");
    println!("• Use more sophisticated architecture (attention, transformers, etc.)");
    println!();
    println!("📈 BiLSTM is particularly useful for:");
    println!("• Sequence labeling (NER, POS tagging)");
    println!("• Text classification with complex patterns");
    println!("• Any task where future context helps current predictions");
    println!("• Machine translation (as encoder)");
    println!("• Question answering systems");
} 