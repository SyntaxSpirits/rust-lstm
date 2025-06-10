use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::Adam;
use std::collections::HashMap;

/// Simple character-level text generator using character indices
struct SimpleTextGenerator {
    network: LSTMNetwork,
    trainer: Option<LSTMTrainer<MSELoss, Adam>>,
    char_to_idx: HashMap<char, usize>,
    idx_to_char: HashMap<usize, char>,
    vocab_size: usize,
    sequence_length: usize,
}

impl SimpleTextGenerator {
    fn new(text: &str, sequence_length: usize, hidden_size: usize) -> Self {
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
        
        // Create network: 1 input (character index) -> hidden -> 1 output (next character index)
        let network = LSTMNetwork::new(1, hidden_size, 1);
        
        println!("📚 Built vocabulary: {} unique characters", vocab_size);
        println!("Characters: {:?}", chars.iter().take(20).collect::<Vec<_>>());
        
        Self {
            network,
            trainer: None,
            char_to_idx,
            idx_to_char,
            vocab_size,
            sequence_length,
        }
    }

    /// Convert character to normalized index
    fn char_to_input(&self, ch: char) -> Array2<f64> {
        let idx = self.char_to_idx.get(&ch).copied().unwrap_or(0);
        let normalized = idx as f64 / (self.vocab_size - 1) as f64; // Normalize to [0, 1]
        Array2::from_shape_vec((1, 1), vec![normalized]).unwrap()
    }

    /// Convert normalized index back to character
    fn output_to_char(&self, output: f64) -> char {
        let idx = ((output * (self.vocab_size - 1) as f64).round() as usize).min(self.vocab_size - 1);
        self.idx_to_char.get(&idx).copied().unwrap_or(' ')
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
                inputs.push(self.char_to_input(chars[j]));
                
                // Target is the next character
                if j + 1 < chars.len() {
                    targets.push(self.char_to_input(chars[j + 1]));
                }
            }
            
            if inputs.len() == targets.len() && !inputs.is_empty() {
                sequences.push((inputs, targets));
            }
        }
        
        sequences
    }

    /// Train the text generator
    fn train(&mut self, text: &str, epochs: usize, validation_split: f64) {
        println!("🔤 Creating character sequences from text...");
        let sequences = self.create_sequences(text);
        
        if sequences.is_empty() {
            println!("❌ No training sequences created!");
            return;
        }
        
        let split_idx = ((sequences.len() as f64) * (1.0 - validation_split)) as usize;
        let (train_data, val_data) = sequences.split_at(split_idx);
        
        println!("📖 Training on {} sequences, validating on {} sequences",
                train_data.len(), val_data.len());
        
        // Create trainer with MSE loss for regression
        let loss_function = MSELoss;
        let optimizer = Adam::new(0.001);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_function, optimizer);
        
        // Configure training
        let mut config = rust_lstm::training::TrainingConfig::default();
        config.epochs = epochs;
        config.print_every = 2; // Print frequently for quick demo
        config.clip_gradient = Some(1.0);
        
        trainer = trainer.with_config(config);
        
        // Train the model
        trainer.train(train_data, if val_data.is_empty() { None } else { Some(val_data) });
        
        self.trainer = Some(trainer);
        println!("✅ Text generation training completed!");
    }

    /// Generate text starting with a seed string
    fn generate_text(&self, seed: &str, length: usize) -> String {
        let trainer = match &self.trainer {
            Some(trainer) => trainer,
            None => {
                println!("❌ Model not trained yet!");
                return String::new();
            }
        };
        
        let mut generated = seed.to_string();
        let mut current_sequence: Vec<char> = seed.chars().collect();
        
        // Ensure we have enough characters to start
        while current_sequence.len() < self.sequence_length {
            current_sequence.insert(0, ' '); // Pad with spaces
        }
        
        for _ in 0..length {
            // Prepare input sequence
            let start_idx = current_sequence.len().saturating_sub(self.sequence_length);
            let input_chars = &current_sequence[start_idx..];
            
            let inputs: Vec<Array2<f64>> = input_chars.iter()
                .map(|&ch| self.char_to_input(ch))
                .collect();
            
            // Get prediction
            let predictions = trainer.predict(&inputs);
            
            if let Some(prediction) = predictions.last() {
                // Convert prediction to character
                let next_char = self.output_to_char(prediction[[0, 0]]);
                generated.push(next_char);
                current_sequence.push(next_char);
                
                // Keep sequence length manageable
                if current_sequence.len() > self.sequence_length * 2 {
                    current_sequence.remove(0);
                }
            } else {
                break;
            }
        }
        
        generated
    }
}

fn main() {
    println!("📝 Simple Text Generation with LSTM");
    println!("===================================\n");
    
    // Simple training text
    let text = "hello world this is a simple test for text generation with lstm neural networks";
    
    println!("🎭 Training text generator...");
    println!("Training text: {}\n", text);
    
    // Create and train model
    let mut generator = SimpleTextGenerator::new(text, 5, 32); // 5-char sequences, 32 hidden units
    generator.train(text, 5, 0.1); // 5 epochs for quick demo
    
    println!("\n🎲 Generating text samples:");
    
    // Generate text with different starting seeds
    let seeds = ["hello", "world", "this", "test"];
    for seed in &seeds {
        let generated = generator.generate_text(seed, 30);
        println!("Seed '{}': {}", seed, generated);
    }
} 