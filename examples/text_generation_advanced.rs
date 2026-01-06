use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::layers::linear::LinearLayer;
use rust_lstm::text::{TextVocabulary, CharacterEmbedding, sample_with_temperature};
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::CrossEntropyLoss;
use rust_lstm::optimizers::Adam;
use std::collections::HashMap;

struct CharacterLSTM {
    vocab: TextVocabulary,
    embedding: CharacterEmbedding,
    network: LSTMNetwork,
    output_layer: LinearLayer,
    trainer: Option<LSTMTrainer<CrossEntropyLoss, Adam>>,
    hidden_size: usize,
    sequence_length: usize,
}

impl CharacterLSTM {
    fn new(text: &str, sequence_length: usize, hidden_size: usize, embed_dim: usize) -> Self {
        let vocab = TextVocabulary::from_text(text);
        let embedding = CharacterEmbedding::new(vocab.size(), embed_dim);
        let network = LSTMNetwork::new(embed_dim, hidden_size, 1);
        let output_layer = LinearLayer::new(hidden_size, vocab.size());

        println!("Vocabulary size: {}", vocab.size());
        println!("Network: embed({}) -> LSTM({}) -> Linear({})", embed_dim, hidden_size, vocab.size());

        Self {
            vocab,
            embedding,
            network,
            output_layer,
            trainer: None,
            hidden_size,
            sequence_length,
        }
    }

    fn create_sequences(&self, text: &str) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
        let chars: Vec<char> = text.chars().collect();
        let mut sequences = Vec::new();

        for i in 0..chars.len().saturating_sub(self.sequence_length + 1) {
            let mut inputs = Vec::new();
            let mut targets = Vec::new();

            for j in 0..self.sequence_length {
                let char_idx = self.vocab.char_to_index(chars[i + j]).unwrap_or(0);
                let next_idx = self.vocab.char_to_index(chars[i + j + 1]).unwrap_or(0);

                let emb = self.embedding.lookup(char_idx);
                let input = Array2::from_shape_vec((emb.len(), 1), emb.to_vec()).unwrap();
                inputs.push(input);

                let mut target = Array2::zeros((self.hidden_size, 1));
                target[[next_idx % self.hidden_size, 0]] = 1.0;
                targets.push(target);
            }

            sequences.push((inputs, targets));
        }

        sequences
    }

    fn train(&mut self, text: &str, epochs: usize) {
        println!("Creating sequences...");
        let sequences = self.create_sequences(text);

        if sequences.is_empty() {
            println!("No sequences created!");
            return;
        }

        let split = (sequences.len() as f64 * 0.9) as usize;
        let (train, val) = sequences.split_at(split);

        println!("Training on {} sequences, validating on {}", train.len(), val.len());

        let loss_fn = CrossEntropyLoss;
        let optimizer = Adam::new(0.002);
        let mut trainer = LSTMTrainer::new(self.network.clone(), loss_fn, optimizer);

        let mut config = rust_lstm::training::TrainingConfig::default();
        config.epochs = epochs;
        config.print_every = epochs / 5;
        config.clip_gradient = Some(5.0);
        trainer = trainer.with_config(config);

        trainer.train(train, if val.is_empty() { None } else { Some(val) });
        self.trainer = Some(trainer);

        println!("Training complete!");
    }

    fn generate(&mut self, seed: &str, length: usize, temperature: f64) -> String {
        if self.trainer.is_none() {
            return seed.to_string();
        }

        let mut generated = seed.to_string();
        let mut chars: Vec<char> = seed.chars().collect();

        while chars.len() < self.sequence_length {
            chars.insert(0, ' ');
        }

        let network = &self.trainer.as_ref().unwrap().network;
        let mut inference_net = network.clone();
        inference_net.eval();

        for _ in 0..length {
            let start = chars.len().saturating_sub(self.sequence_length);
            let window: Vec<char> = chars[start..].to_vec();

            let mut h = Array2::zeros((self.hidden_size, 1));
            let mut c = Array2::zeros((self.hidden_size, 1));

            for ch in &window {
                let idx = self.vocab.char_to_index(*ch).unwrap_or(0);
                let emb = self.embedding.lookup(idx);
                let input = Array2::from_shape_vec((emb.len(), 1), emb.to_vec()).unwrap();
                let (new_h, new_c) = inference_net.forward(&input, &h, &c);
                h = new_h;
                c = new_c;
            }

            let logits_2d = self.output_layer.forward(&h);
            let logits = logits_2d.column(0).to_owned();

            let next_idx = sample_with_temperature(&logits, temperature);
            if let Some(next_char) = self.vocab.index_to_char(next_idx) {
                generated.push(next_char);
                chars.push(next_char);
            }
        }

        generated
    }
}

fn get_sample_texts() -> HashMap<&'static str, &'static str> {
    let mut texts = HashMap::new();

    texts.insert("poetry",
        "The woods are lovely, dark and deep, But I have promises to keep, \
         And miles to go before I sleep, And miles to go before I sleep. \
         Two roads diverged in a yellow wood, And sorry I could not travel both.");

    texts.insert("code",
        "fn main() { println!(\"Hello, world!\"); let x = 42; \
         if x > 10 { println!(\"x is greater\"); } for i in 0..5 { println!(\"i = {}\", i); } }");

    texts.insert("prose",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, \
         filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole.");

    texts
}

fn main() {
    println!("Character-Level LSTM with Text Utilities");
    println!("=========================================\n");

    let texts = get_sample_texts();

    for (domain, text) in &texts {
        println!("Domain: {}", domain);
        println!("Text: {}...\n", &text[..text.len().min(60)]);

        let mut model = CharacterLSTM::new(text, 8, 32, 16);
        model.train(text, 10);

        println!("\nGenerated samples:");
        for temp in [0.5, 1.0, 1.5] {
            let seed: String = text.chars().take(5).collect();
            let output = model.generate(&seed, 50, temp);
            println!("  temp={:.1}: {}", temp, output);
        }

        println!("\n{}\n", "=".repeat(50));
    }
}
