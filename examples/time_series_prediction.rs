use ndarray::{Array2, arr2};
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
    println!("Time Series Prediction Example");
    
    let input_size = 1;
    let hidden_size = 10;
    let num_layers = 2;
    
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    
    // Create a simple sine wave sequence
    let sequence_len = 10;
    let mut sequence = Vec::new();
    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));
    
    for i in 0..sequence_len {
        let t = i as f64 * 0.1;
        let input = arr2(&[[(t).sin()]]);
        
        let (new_hx, new_cx) = network.forward(&input, &hx, &cx);
        
        sequence.push((input, new_hx.clone()));
        hx = new_hx;
        cx = new_cx;
    }
    
    println!("Generated {} time steps", sequence.len());
    
    // Print some outputs
    for (i, (input, output)) in sequence.iter().take(5).enumerate() {
        println!("Step {}: Input={:.3}, Output[0]={:.3}", 
                 i, input[[0, 0]], output[[0, 0]]);
    }
}
