use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    let input_size = 1; // The input size should match the feature dimension of the input data
    let hidden_size = 10;
    let num_layers = 2;
    let sequence_length = 10;

    // Create an LSTM network
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Generate a simple sine wave as input data
    let sequence = (0..sequence_length).map(|i| {
        Array2::from_shape_vec((input_size, 1), vec![(i as f64 * 0.1).sin()]).unwrap()
    }).collect::<Vec<_>>();

    // Process the sequence through the LSTM network
    let mut outputs = Vec::new();
    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));
    for input in sequence {
        let (new_hx, new_cx) = network.forward(&input, &hx, &cx);
        outputs.push(new_hx.clone());
        hx = new_hx;
        cx = new_cx;
    }

    // Print the outputs
    for (i, output) in outputs.iter().enumerate() {
        println!("Predicted value at step {}: {:?}", i, output);
    }
}
