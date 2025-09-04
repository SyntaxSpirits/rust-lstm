use ndarray::Array2;
use rust_lstm::{
    LSTMNetwork, LayerDropoutConfig, LSTMTrainer, TrainingConfig,
    layers::dropout::{Dropout, Zoneout},
    layers::peephole_lstm_cell::PeepholeLSTMCell,
    optimizers::{SGD, Adam, RMSprop},
    loss::{MSELoss, MAELoss, CrossEntropyLoss},
    training::create_basic_trainer,
};

#[test]
fn test_basic_forward_pass_example() {
    let input_size = 3;
    let hidden_size = 2;
    let num_layers = 2;

    // Create an LSTM network
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers);

    // Create some example input data
    let input = Array2::from_shape_vec((input_size, 1), vec![0.5, 0.1, -0.3]).unwrap();

    // Initialize the hidden state and cell state
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    // Perform a forward pass
    let (output, _) = network.forward(&input, &hx, &cx);

    // Verify output shape
    assert_eq!(output.shape(), &[hidden_size, 1]);
}

#[test]
fn test_dropout_regularization_example() {
    let input_size = 10;
    let hidden_size = 20;
    let num_layers = 3;

    // Create network with uniform dropout across all layers
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.2, true)      // 20% variational input dropout
        .with_recurrent_dropout(0.3, true)  // 30% variational recurrent dropout
        .with_output_dropout(0.1)           // 10% output dropout
        .with_zoneout(0.05, 0.1);          // 5% cell, 10% hidden zoneout

    // Configure dropout per layer for fine-grained control
    let layer_configs = vec![
        LayerDropoutConfig::new()
            .with_input_dropout(0.1, false),
        LayerDropoutConfig::new()
            .with_recurrent_dropout(0.2, true)
            .with_zoneout(0.05, 0.1),
        LayerDropoutConfig::new()
            .with_output_dropout(0.1),
    ];
    
    let mut custom_network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_layer_dropout(layer_configs);

    // Set training mode (enables dropout)
    network.train();
    
    // Set evaluation mode (disables dropout)
    network.eval();

    // Test with sample input
    let input = Array2::zeros((input_size, 1));
    let hx = Array2::zeros((hidden_size, 1));
    let cx = Array2::zeros((hidden_size, 1));

    let (output1, _) = network.forward(&input, &hx, &cx);
    let (output2, _) = custom_network.forward(&input, &hx, &cx);

    assert_eq!(output1.shape(), &[hidden_size, 1]);
    assert_eq!(output2.shape(), &[hidden_size, 1]);
}

#[test]
fn test_training_example() {
    // Create network with dropout regularization
    let network = LSTMNetwork::new(1, 4, 1) // Smaller for faster test
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, true)
        .with_output_dropout(0.1);
    
    // Setup training with Adam optimizer
    let loss_function = MSELoss;
    let optimizer = Adam::new(0.001);
    let mut trainer = LSTMTrainer::new(network, loss_function, optimizer);
    
    // Configure training
    let config = TrainingConfig {
        epochs: 2, // Small number for test
        print_every: 1,
        clip_gradient: Some(1.0),
        log_lr_changes: false,
        early_stopping: None,
    };
    trainer = trainer.with_config(config);
    
    // Generate some training data
    let train_data = generate_test_data();
    
    // Train the model (automatically handles train/eval modes)
    trainer.train(&train_data, None);
    
    // Make predictions (automatically sets eval mode)
    let input_sequence = vec![Array2::zeros((1, 1)), Array2::ones((1, 1))];
    let predictions = trainer.predict(&input_sequence);
    
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0].shape(), &[4, 1]);
}

#[test]
fn test_dropout_types_example() {
    // Standard dropout
    let mut dropout = Dropout::new(0.3);

    // Variational dropout (same mask across time steps)
    let mut variational_dropout = Dropout::variational(0.3);

    // Zoneout for RNN hidden/cell states
    let zoneout = Zoneout::new(0.1, 0.15); // cell_rate, hidden_rate

    // Test that they can be created without panicking
    let input = Array2::ones((3, 1));
    
    dropout.train();
    let _output1 = dropout.forward(&input);
    
    variational_dropout.train();
    let _output2 = variational_dropout.forward(&input);
    
    let prev_state = Array2::zeros((3, 1));
    let _output3 = zoneout.apply_cell_zoneout(&input, &prev_state);
}

#[test]
fn test_optimizers_example() {
    // SGD optimizer
    let _sgd = SGD::new(0.01);

    // Adam optimizer with custom parameters
    let _adam = Adam::with_params(0.001, 0.9, 0.999, 1e-8);

    // RMSprop optimizer
    let _rmsprop = RMSprop::new(0.01);

    // Test that they can be created without panicking
    assert!(true);
}

#[test]
fn test_loss_functions_example() {
    // Mean Squared Error for regression
    let _mse_loss = MSELoss;

    // Mean Absolute Error for robust regression
    let _mae_loss = MAELoss;

    // Cross-Entropy for classification
    let _ce_loss = CrossEntropyLoss;

    // Test that they can be created without panicking
    assert!(true);
}

#[test]
fn test_peephole_lstm_example() {
    let input_size = 3;
    let hidden_size = 4;
    
    let cell = PeepholeLSTMCell::new(input_size, hidden_size);
    
    let input = Array2::ones((input_size, 1));
    let h_prev = Array2::zeros((hidden_size, 1));
    let c_prev = Array2::zeros((hidden_size, 1));
    
    let (h_t, c_t) = cell.forward(&input, &h_prev, &c_prev);
    
    assert_eq!(h_t.shape(), &[hidden_size, 1]);
    assert_eq!(c_t.shape(), &[hidden_size, 1]);
}

#[test]
fn test_create_basic_trainer() {
    let network = LSTMNetwork::new(2, 3, 1);
    let _trainer = create_basic_trainer(network, 0.01);
    
    // Test that trainer can be created without panicking
    assert!(true);
}

// Helper function to generate test data
fn generate_test_data() -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut data = Vec::new();
    
    for _seq_idx in 0..3 { // Small dataset for test
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        
        for _t in 0..2 { // Short sequences for test
            let input = Array2::ones((1, 1));
            let target = Array2::ones((4, 1)) * 0.5;
            
            inputs.push(input);
            targets.push(target);
        }
        
        data.push((inputs, targets));
    }
    
    data
}
