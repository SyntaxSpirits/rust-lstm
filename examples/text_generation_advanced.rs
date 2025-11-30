use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::Adam;
use std::collections::HashMap;

/// Advanced character-level language model using LSTM with embedded representations
struct CharacterLSTM {
    network: LSTMNetwork,
    trainer: Option<LSTMTrainer<MSELoss, Adam>>,
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    vocab_size: usize,
    sequence_length: usize,
    embedding_size: usize,
}

impl CharacterLSTM {
    fn new(text: &str, sequence_length: usize, hidden_size: usize, embedding_size: usize) -> Self {
        // Build vocabulary from text
        let unique_chars: std::collections::HashSet<char> = text.chars().collect();
        let mut chars: Vec<char> = unique_chars.into_iter().collect();
        chars.sort(); // Ensure consistent ordering
        
        let vocab_size = chars.len();
        
        // Create character mappings
        let char_to_idx: HashMap<char, usize> = chars.iter().enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        let idx_to_char: HashMap<usize, char> = chars.iter().enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        
        // Create network: embedding_size input -> hidden_size (single layer)
        let network = LSTMNetwork::new(embedding_size, hidden_size, 1);
        
        println!("Built vocabulary: {} unique characters", vocab_size);
        println!("Characters: {:?}", chars.iter().take(20).collect::<Vec<_>>());
        println!("Network: {} -> {} -> {}", embedding_size, hidden_size, embedding_size);
        
        Self {
            network,
            trainer: None,
            char_to_idx,
            idx_to_char,
            vocab_size,
            sequence_length,
            embedding_size,
        }
    }

    /// Convert character to embedded representation
    fn char_to_embedding(&self, ch: char) -> Array2<f64> {
        let idx = self.char_to_idx.get(&ch).copied().unwrap_or(0);
        let mut embedding = vec![0.0; self.embedding_size];
        
        // Simple embedding: use sine/cosine features based on character index
        for i in 0..self.embedding_size {
            let freq = (i + 1) as f64 / self.embedding_size as f64;
            if i % 2 == 0 {
                embedding[i] = ((idx as f64) * freq).sin();
            } else {
                embedding[i] = ((idx as f64) * freq).cos();
            }
        }
        
        Array2::from_shape_vec((self.embedding_size, 1), embedding).unwrap()
    }

    /// Convert embedding back to character using similarity
    fn embedding_to_char(&self, embedding: &Array2<f64>) -> char {
        let mut best_char = ' ';
        let mut best_similarity = f64::NEG_INFINITY;
        
        // Find character with most similar embedding
        for (&ch, &_idx) in &self.char_to_idx {
            let char_embedding = self.char_to_embedding(ch);
            
            // Compute cosine similarity
            let dot_product: f64 = (0..self.embedding_size)
                .map(|i| embedding[[i, 0]] * char_embedding[[i, 0]])
                .sum();
            
            let norm1: f64 = (0..self.embedding_size)
                .map(|i| embedding[[i, 0]] * embedding[[i, 0]])
                .sum::<f64>().sqrt();
            
            let norm2: f64 = (0..self.embedding_size)
                .map(|i| char_embedding[[i, 0]] * char_embedding[[i, 0]])
                .sum::<f64>().sqrt();
            
            let similarity = if norm1 > 0.0 && norm2 > 0.0 {
                dot_product / (norm1 * norm2)
            } else {
                0.0
            };
            
            if similarity > best_similarity {
                best_similarity = similarity;
                best_char = ch;
            }
        }
        
        best_char
    }

    /// Sample character with temperature control
    fn sample_char_with_temperature(&self, embedding: &Array2<f64>, temperature: f64) -> char {
        let mut scores = Vec::new();
        
        // Calculate similarity scores for all characters
        for (&ch, &_idx) in &self.char_to_idx {
            let char_embedding = self.char_to_embedding(ch);
            
            let dot_product: f64 = (0..self.embedding_size)
                .map(|i| embedding[[i, 0]] * char_embedding[[i, 0]])
                .sum();
            
            scores.push((ch, dot_product / temperature));
        }
        
        // Apply softmax and sample
        let max_score = scores.iter().map(|(_, score)| *score).fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<(char, f64)> = scores.iter()
            .map(|(ch, score)| (*ch, (score - max_score).exp()))
            .collect();
        
        let sum: f64 = exp_scores.iter().map(|(_, exp_score)| *exp_score).sum();
        let probabilities: Vec<(char, f64)> = exp_scores.iter()
            .map(|(ch, exp_score)| (*ch, exp_score / sum))
            .collect();
        
        // Sample from distribution
        let mut rng_val = rand::random::<f64>();
        for &(ch, prob) in &probabilities {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return ch;
            }
        }
        
        // Fallback to most probable character
        probabilities.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(ch, _)| *ch)
            .unwrap_or(' ')
    }

    /// Project embedding to hidden space for targets
    fn embedding_to_hidden(&self, embedding: &Array2<f64>) -> Array2<f64> {
        // Simple projection: repeat/pad embedding to hidden size
        let hidden_size = 32; // This should match the network hidden size
        let mut hidden = vec![0.0; hidden_size];
        
        for i in 0..hidden_size {
            if i < self.embedding_size {
                hidden[i] = embedding[[i, 0]];
            } else {
                // Pad with scaled values or zeros
                hidden[i] = 0.0;
            }
        }
        
        Array2::from_shape_vec((hidden_size, 1), hidden).unwrap()
    }

    /// Create training sequences from text
    fn create_sequences(&self, text: &str) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let chars: Vec<char> = text.chars().collect();
        let mut sequences = Vec::new();
        
        for i in 0..chars.len().saturating_sub(self.sequence_length) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();
            
            // Create input sequence and corresponding target sequence
            for j in i..i + self.sequence_length {
                inputs.push(self.char_to_embedding(chars[j]));
                
                // Target is the next character's embedding projected to hidden space
                if j + 1 < chars.len() {
                    let target_embedding = self.char_to_embedding(chars[j + 1]);
                    let hidden_target = self.embedding_to_hidden(&target_embedding);
                    targets.push(hidden_target);
                }
            }
            
            if inputs.len() == targets.len() && !inputs.is_empty() {
                sequences.push((inputs, targets));
            }
        }
        
        sequences
    }

    /// Train the character-level language model
    fn train(&mut self, text: &str, epochs: usize, validation_split: f64) {
        println!("Creating character sequences from text...");
        let sequences = self.create_sequences(text);
        
        if sequences.is_empty() {
            println!("No training sequences created!");
            return;
        }
        
        let split_idx = ((sequences.len() as f64) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = sequences.split_at(split_idx);
        
        println!("Training on {} sequences, validating on {} sequences",
                train_data.len(), val_data.len());
        
        // Create trainer with MSE loss for embedding regression
        let loss_function = MSELoss;
        let optimizer = Adam::new(0.002);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_function, optimizer);
        
        // Configure training
        let mut config = rust_lstm::training::TrainingConfig::default();
        config.epochs = epochs;
        config.print_every = epochs / 5; // Print 5 times during training
        config.clip_gradient = Some(5.0);
        
        trainer = trainer.with_config(config);
        
        // Train the model
        trainer.train(train_data, if val_data.is_empty() { None } else { Some(val_data) });
        
        self.trainer = Some(trainer);
        println!("Character LSTM training completed!");
    }

    /// Generate text starting with a seed string
    fn generate_text(&mut self, seed: &str, length: usize, temperature: f64) -> String {
        if self.trainer.is_none() {
            println!("Model not trained yet!");
            return String::new();
        }
        
        let mut generated = seed.to_string();
        let mut current_sequence: Vec<char> = seed.chars().collect();
        
        // Ensure we have enough characters to start
        while current_sequence.len() < self.sequence_length {
            current_sequence.insert(0, ' '); // Pad with spaces
        }
        
        let network = if let Some(ref trainer) = self.trainer {
            &trainer.network
        } else {
            println!("Trainer not available");
            return generated;
        };
        
        let mut inference_network = network.clone();
        inference_network.eval();
        
        for _ in 0..length {
            // Prepare input sequence
            let start_idx = current_sequence.len().saturating_sub(self.sequence_length);
            let input_chars = &current_sequence[start_idx..];
            
            let inputs: Vec<Array2<f64>> = input_chars.iter()
                .map(|&ch| self.char_to_embedding(ch))
                .collect();

            let (outputs, _) = inference_network.forward_sequence_with_cache(&inputs);
            
            if let Some((last_output, _)) = outputs.last() {
                let predicted_embedding = self.project_to_embedding(last_output);
                
                let next_char = self.sample_next_char(&predicted_embedding, temperature);
                
                generated.push(next_char);
                current_sequence.push(next_char);
                
                if current_sequence.len() > self.sequence_length * 2 {
                    current_sequence.drain(0..self.sequence_length);
                }
            } else {
                println!("No prediction generated, stopping text generation");
                break;
            }
        }
        
        generated
    }

    /// Project LSTM output to embedding space
    fn project_to_embedding(&self, lstm_output: &Array2<f64>) -> Array2<f64> {
        // Simple projection: take first embedding_size elements of hidden state
        // In practice, this would be a learned linear layer
        let hidden_size = lstm_output.nrows();
        let mut embedding = vec![0.0; self.embedding_size];
        
        for i in 0..self.embedding_size.min(hidden_size) {
            embedding[i] = lstm_output[[i, 0]];
        }
        
        Array2::from_shape_vec((self.embedding_size, 1), embedding).unwrap()
    }

    /// Sample next character using temperature-based sampling
    fn sample_next_char(&self, predicted_embedding: &Array2<f64>, temperature: f64) -> char {
        let mut similarities = Vec::new();
        
        for (&ch, &_idx) in &self.char_to_idx {
            let char_embedding = self.char_to_embedding(ch);
            
            let dot_product = predicted_embedding.iter()
                .zip(char_embedding.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
            
            let pred_norm = predicted_embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            let char_norm = char_embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            let similarity = if pred_norm > 0.0 && char_norm > 0.0 {
                dot_product / (pred_norm * char_norm)
            } else {
                0.0
            };
            
            similarities.push((ch, similarity));
        }
        
        let max_similarity = similarities.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities = Vec::new();
        let mut total_prob = 0.0;
        
        for (ch, similarity) in &similarities {
            let scaled_similarity = (similarity - max_similarity) / temperature;
            let prob = scaled_similarity.exp();
            probabilities.push((*ch, prob));
            total_prob += prob;
        }
        
        for (_, prob) in &mut probabilities {
            *prob /= total_prob;
        }
        
        let random_value: f64 = rand::random();
        let mut cumulative_prob = 0.0;
        
        for (ch, prob) in probabilities {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return ch;
            }
        }
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities[0].0
    }
}

/// Sample training texts for different domains
fn get_sample_texts() -> HashMap<&'static str, &'static str> {
    let mut texts = HashMap::new();
    
    texts.insert("poetry", 
        "The woods are lovely, dark and deep, But I have promises to keep, And miles to go before I sleep, And miles to go before I sleep. Two roads diverged in a yellow wood, And sorry I could not travel both And be one traveler, long I stood And looked down one as far as I could To where it bent in the undergrowth.");
    
    texts.insert("code", 
        "fn main() { println!(\"Hello, world!\"); let x = 42; if x > 10 { println!(\"x is greater than 10\"); } for i in 0..5 { println!(\"i = {}\", i); } }");
    
    texts.insert("prose",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.");
    
    texts
}

fn main() {
    println!("Advanced Text Generation with Character-Level LSTM");
    println!("===================================================\n");
    
    let sample_texts = get_sample_texts();
    
    for (domain, text) in &sample_texts {
        println!("Training {} model...", domain);
        println!("Training text preview: {}...\n", &text[..text.len().min(100)]);
        
        // Create and train model with embedding
        let mut model = CharacterLSTM::new(text, 8, 32, 16); // 8-char sequences, 32 hidden, 16 embedding
        model.train(text, 8, 0.1); // 8 epochs for quick demo, 10% validation
        
        println!("\nGenerating text samples:");
        
        // Generate with different temperatures
        let temperatures = [0.8, 1.2];
        for &temp in &temperatures {
            let seed = text.chars().take(5).collect::<String>();
            let generated = model.generate_text(&seed, 60, temp);
            println!("\nTemperature {:.1}: {}", temp, generated);
        }
        
        println!("\n{}\n", "=".repeat(60));
    }
} 