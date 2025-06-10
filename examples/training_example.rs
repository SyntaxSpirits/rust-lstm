use ndarray::{Array2, arr2};
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::LSTMTrainer;
use rust_lstm::loss::MSELoss;
use rust_lstm::optimizers::{SGD, Adam};

/// Generate sine wave training data for sequence prediction
fn generate_sine_data(num_sequences: usize, sequence_length: usize) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();
    
    for i in 0..num_sequences {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        let start = (i as f64) * 0.1;
        
        for j in 0..sequence_length {
            let t = start + (j as f64) * 0.1;
            let x = (t * 2.0 * std::f64::consts::PI).sin();
            let y = ((t + 0.1) * 2.0 * std::f64::consts::PI).sin(); // Next value in sequence
            
            inputs.push(arr2(&[[x]]));
            targets.push(arr2(&[[y]]));
        }
        
        data.push((inputs, targets));
    }
    
    data
}

/// Evaluate prediction accuracy on sine wave data
fn evaluate_predictions(network: &mut LSTMNetwork, test_data: &[(Vec<Array2<f64>>, Vec<Array2<f64>>)]) -> f64 {
    let mut total_error = 0.0;
    let mut count = 0;
    
    for (inputs, targets) in test_data {
        let predictions = network.forward_sequence_with_cache(inputs).0;
        
        for ((pred, _), target) in predictions.iter().zip(targets.iter()) {
            let error = (pred[[0, 0]] - target[[0, 0]]).abs();
            total_error += error;
            count += 1;
        }
    }
    
    total_error / count as f64
}

fn main() {
    println!("=== LSTM Training Demonstration ===\n");
    
    // Generate training and validation data
    let train_data = generate_sine_data(50, 10);
    let val_data = generate_sine_data(10, 10);
    
    println!("Generated {} training sequences and {} validation sequences", 
             train_data.len(), val_data.len());
    
    // Network configuration
    let input_size = 1;
    let hidden_size = 10;
    let num_layers = 1;
    
    println!("Network: {} input -> {} hidden units -> {} layers\n", 
             input_size, hidden_size, num_layers);
    
    // Training with SGD
    println!("Training with SGD optimizer:");
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer_sgd = LSTMTrainer::new(network, MSELoss, SGD::new(0.01));
    
    trainer_sgd.train(&train_data, Some(&val_data));
    
    let final_metrics_sgd = trainer_sgd.get_latest_metrics().unwrap();
    println!("SGD - Final training loss: {:.6}", final_metrics_sgd.train_loss);
    if let Some(val_loss) = final_metrics_sgd.validation_loss {
        println!("SGD - Final validation loss: {:.6}", val_loss);
    }
    
    let prediction_error_sgd = evaluate_predictions(&mut trainer_sgd.network, &val_data);
    println!("SGD - Average prediction error: {:.6}\n", prediction_error_sgd);
    
    // Training with Adam
    println!("Training with Adam optimizer:");
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers);
    let mut trainer_adam = LSTMTrainer::new(network, MSELoss, Adam::new(0.001));
    
    trainer_adam.train(&train_data, Some(&val_data));
    
    let final_metrics_adam = trainer_adam.get_latest_metrics().unwrap();
    println!("Adam - Final training loss: {:.6}", final_metrics_adam.train_loss);
    if let Some(val_loss) = final_metrics_adam.validation_loss {
        println!("Adam - Final validation loss: {:.6}", val_loss);
    }
    
    let prediction_error_adam = evaluate_predictions(&mut trainer_adam.network, &val_data);
    println!("Adam - Average prediction error: {:.6}\n", prediction_error_adam);
    
    // Compare results
    println!("=== Comparison ===");
    println!("SGD  - Prediction Error: {:.6}", prediction_error_sgd);
    println!("Adam - Prediction Error: {:.6}", prediction_error_adam);
    
    if prediction_error_adam < prediction_error_sgd {
        println!("Adam achieved better accuracy!");
    } else {
        println!("SGD achieved better accuracy!");
    }
    
    // Demonstrate prediction on a new sequence
    println!("\n=== Sample Prediction ===");
    let test_sequence = vec![
        arr2(&[[0.0]]),     // sin(0) = 0
        arr2(&[[0.841]]),   // sin(π/2) ≈ 0.841  
        arr2(&[[0.909]]),   // sin(π) ≈ 0.909
    ];
    
    let predictions = trainer_adam.network.forward_sequence_with_cache(&test_sequence).0;
    
    println!("Input sequence: {:?}", 
             test_sequence.iter().map(|x| x[[0, 0]]).collect::<Vec<_>>());
    println!("Predicted next values: {:?}",
             predictions.iter().map(|(pred, _)| pred[[0, 0]]).collect::<Vec<_>>());
} 