use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    let input_size = 3;
    let hidden_size = 2;
    let num_layers = 2;

    // Create an LSTM network
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Create some example input data
    let input = Array2::from_shape_vec((input_size, 1), vec![0.5, 0.1, -0.3]).unwrap();

    // Initialize the hidden state and cell state
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    // Perform a forward pass
    let (output, _) = network.forward(&input, &hx, &cx);

    // Print the output
    println!("Output: {:?}", output);
}
