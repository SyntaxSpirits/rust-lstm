use ndarray::{Array2, arr2};
use rust_lstm::{
    LSTMNetwork, LayerDropoutConfig,
    training::{LSTMTrainer, TrainingConfig},
    optimizers::Adam,
    loss::MSELoss,
};

fn main() {
    println!("Rust LSTM Dropout Example");
    println!("=========================\n");

    // Demonstrate different dropout configurations
    demonstrate_basic_dropout();
    demonstrate_variational_dropout();
    demonstrate_layer_specific_dropout();
    demonstrate_zoneout();
    demonstrate_training_with_dropout();
}

fn demonstrate_basic_dropout() {
    println!("1. Basic Dropout Configuration");
    println!("------------------------------");

    let input_size = 4;
    let hidden_size = 6;
    let num_layers = 2;

    // Create network with standard dropout
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.2, false)
        .with_recurrent_dropout(0.3, false)
        .with_output_dropout(0.1);

    let input = arr2(&[[1.0], [0.5], [-0.2], [0.8]]);
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    // Test training mode
    network.train();
    println!("Training mode:");
    let (hy_train, _) = network.forward(&input, &hx, &cx);
    println!("  Output shape: {:?}", hy_train.shape());
    println!("  Sample output values: [{:.4}, {:.4}, {:.4}]", 
             hy_train[[0, 0]], hy_train[[1, 0]], hy_train[[2, 0]]);

    // Test evaluation mode
    network.eval();
    println!("Evaluation mode:");
    let (hy_eval, _) = network.forward(&input, &hx, &cx);
    println!("  Output shape: {:?}", hy_eval.shape());
    println!("  Sample output values: [{:.4}, {:.4}, {:.4}]", 
             hy_eval[[0, 0]], hy_eval[[1, 0]], hy_eval[[2, 0]]);

    println!();
}

fn demonstrate_variational_dropout() {
    println!("2. Variational Dropout Configuration");
    println!("------------------------------------");

    let input_size = 3;
    let hidden_size = 4;
    let num_layers = 2;

    // Create network with variational dropout (same mask across time steps)
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.25, true)
        .with_recurrent_dropout(0.2, true);

    let sequence = vec![
        arr2(&[[1.0], [0.0], [0.5]]),
        arr2(&[[0.5], [1.0], [0.0]]),
        arr2(&[[-0.2], [0.8], [0.3]]),
    ];

    network.train();
    println!("Processing sequence with variational dropout:");
    
    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));

    for (i, input) in sequence.iter().enumerate() {
        let (new_hx, new_cx) = network.forward(input, &hx, &cx);
        println!("  Step {}: Output sum = {:.4}", i, new_hx.sum());
        hx = new_hx;
        cx = new_cx;
    }

    println!();
}

fn demonstrate_layer_specific_dropout() {
    println!("3. Layer-Specific Dropout Configuration");
    println!("----------------------------------------");

    let input_size = 3;
    let hidden_size = 4;
    let num_layers = 3;

    // Configure different dropout for each layer
    let layer_configs = vec![
        // Layer 0: Input layer with moderate input dropout
        LayerDropoutConfig::new()
            .with_input_dropout(0.1, false),
        
        // Layer 1: Hidden layer with recurrent dropout and zoneout
        LayerDropoutConfig::new()
            .with_recurrent_dropout(0.2, true)
            .with_zoneout(0.05, 0.1),
        
        // Layer 2: Output layer with light output dropout
        LayerDropoutConfig::new()
            .with_output_dropout(0.1),
    ];

    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_layer_dropout(layer_configs);

    let input = arr2(&[[0.5], [1.0], [-0.3]]);
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    network.train();
    let (hy, _) = network.forward(&input, &hx, &cx);
    
    println!("Network with layer-specific dropout:");
    println!("  Input size: {}, Hidden size: {}, Layers: {}", 
             input_size, hidden_size, num_layers);
    println!("  Output: {:?}", hy.shape());
    println!("  Output mean: {:.4}", hy.mean().unwrap());

    println!();
}

fn demonstrate_zoneout() {
    println!("4. Zoneout Regularization");
    println!("-------------------------");

    let input_size = 2;
    let hidden_size = 3;
    let num_layers = 1;

    // Create network with zoneout
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_zoneout(0.1, 0.15); // 10% cell zoneout, 15% hidden zoneout

    let sequence = vec![
        arr2(&[[1.0], [0.0]]),
        arr2(&[[0.0], [1.0]]),
        arr2(&[[0.5], [0.5]]),
    ];

    network.train();
    println!("Sequence processing with zoneout:");

    let mut hx = Array2::zeros((hidden_size, 1));
    let mut cx = Array2::zeros((hidden_size, 1));

    for (i, input) in sequence.iter().enumerate() {
        let (new_hx, new_cx) = network.forward(input, &hx, &cx);
        println!("  Step {}: Hidden state norm = {:.4}, Cell state norm = {:.4}", 
                 i, (new_hx.mapv(|x| x * x).sum()).sqrt(), (new_cx.mapv(|x| x * x).sum()).sqrt());
        hx = new_hx;
        cx = new_cx;
    }

    println!();
}

fn demonstrate_training_with_dropout() {
    println!("5. Training with Dropout");
    println!("------------------------");

    let input_size = 2;
    let hidden_size = 4;
    let num_layers = 2;

    // Create network with comprehensive dropout
    let network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.2, true)       // Variational input dropout
        .with_recurrent_dropout(0.3, true)   // Variational recurrent dropout
        .with_output_dropout(0.1)            // Standard output dropout
        .with_zoneout(0.05, 0.1);           // Light zoneout

    // Create trainer
    let loss_function = MSELoss;
    let optimizer = Adam::new(0.001);
    let mut trainer = LSTMTrainer::new(network, loss_function, optimizer);

    // Configure training
    let config = TrainingConfig {
        epochs: 20,
        print_every: 5,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    };
    trainer = trainer.with_config(config);

    // Generate simple training data (sine wave prediction)
    let train_data = generate_sine_wave_data(10, 5);

    println!("Training LSTM with dropout regularization...");
    println!("Dataset: {} sequences of length {}", train_data.len(), train_data[0].0.len());
    
    // Train the model
    trainer.train(&train_data, None);

    // Test prediction
    let test_input = vec![
        arr2(&[[1.0], [0.0]]),
        arr2(&[[0.0], [1.0]]),
        arr2(&[[-1.0], [0.0]]),
    ];

    println!("\nMaking predictions:");
    let predictions = trainer.predict(&test_input);
    for (i, pred) in predictions.iter().enumerate() {
        println!("  Prediction {}: [{:.4}, {:.4}, {:.4}, {:.4}]", 
                 i, pred[[0, 0]], pred[[1, 0]], pred[[2, 0]], pred[[3, 0]]);
    }

    println!("\nTraining completed with dropout regularization!");
}

fn generate_sine_wave_data(num_sequences: usize, sequence_length: usize) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();
    
    for seq_idx in 0..num_sequences {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        let phase = seq_idx as f64 * 0.1;
        
        for t in 0..sequence_length {
            let time = t as f64 * 0.1 + phase;
            let input_val = (time).sin();
            let target_val = (time + 0.1).sin();
            
            // Create 2D input and 4D target (matching network architecture)
            let input = arr2(&[[input_val], [input_val * 0.5]]);
            let target = arr2(&[[target_val], [target_val * 0.8], [target_val * 0.6], [target_val * 0.3]]);
            
            inputs.push(input);
            targets.push(target);
        }
        
        data.push((inputs, targets));
    }
    
    data
} 