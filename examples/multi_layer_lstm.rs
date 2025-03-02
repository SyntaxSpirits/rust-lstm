use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    let input_size = 3;
    let hidden_size = 2;
    let num_layers = 3;
    let sequence_length = 5;

    // Create a multi-layer LSTM network
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Create a sequence of example input data
    let sequence = (0..sequence_length).map(|i| {
        Array2::from_shape_vec((input_size, 1), vec![i as f64 * 0.1, i as f64 * 0.2, i as f64 * 0.3]).unwrap()
    }).collect::<Vec<_>>();

    // Initialize the hidden state and cell state
    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));

    // Process the sequence through the LSTM network
    let mut outputs = Vec::new();
    for input in sequence {
        let (new_hx, new_cx) = network.forward(&input, &hx, &cx);
        outputs.push(new_hx.clone());
        hx = new_hx;
        cx = new_cx;
    }

    // Print the outputs
    for (i, output) in outputs.iter().enumerate() {
        println!("Output for sequence step {}: {:?}", i, output);
    }
}
