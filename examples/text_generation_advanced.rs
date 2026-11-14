//! Character-level language model trained end to end.
//!
//! Embedding, LSTM and output layer are trained together with softmax
//! cross-entropy: every parameter receives a gradient on every step.

use ndarray::{s, Array1, Array2};
use rust_lstm::{
    sample_with_temperature, Adam, CharacterEmbedding, CrossEntropyLoss, LSTMNetwork, LinearLayer,
    LossFunction, TextVocabulary,
};

const CORPUS: &str = "the cat sat on the mat. the dog sat on the log. \
                      the cat saw the dog and the dog saw the cat. ";
const SEQ_LEN: usize = 16;
const EMBED_DIM: usize = 8;
const HIDDEN_SIZE: usize = 32;
const EPOCHS: usize = 100;

struct CharModel {
    embedding: CharacterEmbedding,
    lstm: LSTMNetwork,
    head: LinearLayer,
}

impl CharModel {
    fn new(vocab_size: usize) -> Self {
        Self {
            embedding: CharacterEmbedding::new(vocab_size, EMBED_DIM),
            lstm: LSTMNetwork::new(EMBED_DIM, HIDDEN_SIZE, 1),
            head: LinearLayer::new(HIDDEN_SIZE, vocab_size),
        }
    }

    /// Embeds `steps` time steps of a batch of windows; one (embed_dim, batch) matrix per step.
    fn embed(&mut self, windows: &[&[usize]], steps: usize) -> Vec<Array2<f64>> {
        let batch = windows.len();
        let indices: Vec<usize> = (0..steps)
            .flat_map(|t| windows.iter().map(move |w| w[t]))
            .collect();
        let embedded = self.embedding.forward(&indices);
        (0..steps)
            .map(|t| {
                embedded
                    .slice(s![t * batch..(t + 1) * batch, ..])
                    .t()
                    .to_owned()
            })
            .collect()
    }

    /// One optimisation step on a batch of windows of length `SEQ_LEN + 1`.
    /// Returns the mean cross-entropy per character.
    fn train_step(&mut self, windows: &[&[usize]], optimizer: &mut Adam) -> f64 {
        let batch = windows.len();
        let vocab_size = self.head.output_size;
        let xs = self.embed(windows, SEQ_LEN);
        let (outputs, caches) = self.lstm.forward_sequence_with_cache(&xs);

        let mut loss = 0.0;
        let mut head_grads = self.head.zero_gradients();
        let mut d_outputs = Vec::with_capacity(SEQ_LEN);
        for (t, (h, _)) in outputs.iter().enumerate() {
            let mut targets = Array2::zeros((vocab_size, batch));
            for (b, w) in windows.iter().enumerate() {
                targets[[w[t + 1], b]] = 1.0;
            }
            let logits = self.head.forward(h);
            loss += CrossEntropyLoss.compute_loss(&logits, &targets) / SEQ_LEN as f64;
            let d_logits = CrossEntropyLoss.compute_gradient(&logits, &targets) / SEQ_LEN as f64;
            let (g, dh) = self.head.backward(&d_logits);
            head_grads.weight += &g.weight;
            head_grads.bias += &g.bias;
            d_outputs.push(dh);
        }

        let (lstm_grads, d_inputs) = self.lstm.backward_sequence(&d_outputs, &caches);
        let mut d_embedded = Array2::zeros((SEQ_LEN * batch, EMBED_DIM));
        for (t, d) in d_inputs.iter().enumerate() {
            d_embedded
                .slice_mut(s![t * batch..(t + 1) * batch, ..])
                .assign(&d.t());
        }
        let embedding_grads = self.embedding.backward(&d_embedded);

        self.lstm.update_parameters(&lstm_grads, optimizer);
        self.head.update_parameters(&head_grads, optimizer, "head");
        self.embedding
            .update_parameters(&embedding_grads, optimizer, "embedding");
        loss
    }

    fn next_logits(&mut self, context: &[usize]) -> Array1<f64> {
        let xs = self.embed(&[context], context.len());
        let h = self.lstm.forward_sequence(&xs).pop().unwrap();
        self.head.forward(&h).column(0).to_owned()
    }

    fn generate(
        &mut self,
        vocab: &TextVocabulary,
        prompt: &str,
        length: usize,
        temp: f64,
    ) -> String {
        let mut tokens = vocab.encode(prompt);
        for _ in 0..length {
            let start = tokens.len().saturating_sub(SEQ_LEN);
            let logits = self.next_logits(&tokens[start..]);
            tokens.push(sample_with_temperature(&logits, temp));
        }
        vocab.decode(&tokens)
    }
}

fn main() {
    rust_lstm::seed(42);

    let vocab = TextVocabulary::from_text(CORPUS);
    let text = vocab.encode(&CORPUS.repeat(2));
    let windows: Vec<&[usize]> = (0..CORPUS.chars().count())
        .map(|i| &text[i..i + SEQ_LEN + 1])
        .collect();

    println!("Character-level LSTM language model");
    println!("Corpus: \"{}\"", CORPUS);
    println!(
        "Vocabulary: {} characters, {} training windows of {} steps",
        vocab.size(),
        windows.len(),
        SEQ_LEN
    );
    println!(
        "Model: embedding({}) -> LSTM({}) -> linear({}) -> softmax",
        EMBED_DIM,
        HIDDEN_SIZE,
        vocab.size()
    );
    println!(
        "Loss of a uniform guess: {:.3}\n",
        (vocab.size() as f64).ln()
    );

    let mut model = CharModel::new(vocab.size());
    let mut optimizer = Adam::new(0.03);
    for epoch in 1..=EPOCHS {
        let loss = model.train_step(&windows, &mut optimizer);
        if epoch == 1 || epoch % 20 == 0 {
            println!("epoch {:>3}: loss {:.4}", epoch, loss);
        }
    }

    model.lstm.eval();
    println!();
    for temp in [0.2, 0.5, 1.0] {
        let sample = model.generate(&vocab, "the ", 80, temp);
        println!("temperature {:.1}: {}", temp, sample);
    }
}
