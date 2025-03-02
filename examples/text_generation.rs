use ndarray::{Array2, Array3, s};
use rust_lstm::models::lstm_network::LSTMNetwork;
use std::collections::HashMap;

fn main() {
    // Define the text and the set of characters
    let text = "hello world";
    let chars: Vec<char> = text.chars().collect();
    let unique_chars: Vec<char> = chars.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
    let char_to_idx: HashMap<char, usize> = unique_chars.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
    let idx_to_char: HashMap<usize, char> = char_to_idx.iter().map(|(&c, &i)| (i, c)).collect();
    let input_size = unique_chars.len();
    let hidden_size = 10;
    let num_layers = 2;
    let sequence_length = 5;

    // Create an LSTM network
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Prepare the input sequence
    let input_text = "hello";
    let input_indices: Vec<usize> = input_text.chars().map(|c| *char_to_idx.get(&c).unwrap()).collect();
    let mut input = Array3::zeros((sequence_length, input_size, 1));
    for (t, &idx) in input_indices.iter().enumerate() {
        input.slice_mut(s![t, idx, ..]).fill(1.0);
    }

    // Initialize the hidden state and cell state
    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));

    // Process the sequence through the LSTM network
    for t in 0..sequence_length {
        let input_t = input.slice(s![t, .., ..]).to_owned();
        let (new_hx, new_cx) = network.forward(&input_t, &hx, &cx);
        hx = new_hx;
        cx = new_cx;
    }

    // Generate the next characters
    let mut generated_text = input_text.to_string();
    let num_generated_chars = 20;
    for _ in 0..num_generated_chars {
        // Create a one-hot encoded input from the current hidden state
        let mut input_t = Array2::zeros((input_size, 1));
        let output_idx = hx.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        input_t.slice_mut(s![output_idx, ..]).fill(1.0);

        // Perform the forward pass
        let (new_hx, new_cx) = network.forward(&input_t, &hx, &cx);
        hx = new_hx;
        cx = new_cx;

        // Get the predicted character
        let output_idx = hx.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        let output_char = idx_to_char[&output_idx];
        generated_text.push(output_char);
    }

    // Print the generated text
    println!("Generated text: {}", generated_text);
}
