use rust_lstm::*;
use ndarray::arr2;

#[test]
fn test_network_forward() {
    let input_size = 2;
    let hidden_size = 3;
    let num_layers = 1;
    
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    
    let input = arr2(&[[1.0], [0.5]]);
    let hx = arr2(&[[0.0], [0.0], [0.0]]);
    let cx = arr2(&[[0.0], [0.0], [0.0]]);
    
    let (output, _) = network.forward(&input, &hx, &cx);
    
    assert_eq!(output.shape(), &[3, 1]);
}
