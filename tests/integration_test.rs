use rust_lstm::models::lstm_network::LSTMNetwork;
use ndarray::arr2;

#[test]
fn test_lstm_network_integration() {
    let input_size = 3;
    let hidden_size = 2;
    let num_layers = 2;
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    let input = arr2(&[[0.5], [0.1], [-0.3]]);

    // Initialize the hidden state and cell state
    let hx = arr2(&[[0.0], [0.0]]);
    let cx = arr2(&[[0.0], [0.0]]);

    // Perform a forward pass
    let (output, _) = network.forward(&input, &hx, &cx);

    assert_eq!(output.shape(), &[hidden_size, 1]);
}
