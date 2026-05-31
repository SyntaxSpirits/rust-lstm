#![allow(clippy::type_complexity)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::absurd_extreme_comparisons)]
#![allow(unused_comparisons)]

use ndarray::Array2;
use rust_lstm::{
    layers::bilstm_network::BiLSTMNetwork,
    layers::dropout::{Dropout, Zoneout},
    layers::linear::LinearLayer,
    layers::peephole_lstm_cell::PeepholeLSTMCell,
    loss::{CrossEntropyLoss, MAELoss, MSELoss},
    models::gru_network::GRUNetwork,
    optimizers::{Adam, RMSprop, ScheduledOptimizer, SGD},
    schedulers::{CyclicalLR, LRScheduleVisualizer, PolynomialLR, WarmupScheduler},
    training::{
        create_basic_trainer, create_cosine_annealing_trainer, create_one_cycle_trainer,
        create_step_lr_trainer,
    },
    EarlyStoppingConfig, EarlyStoppingMetric, LSTMNetwork, LSTMTrainer, LayerDropoutConfig,
    TrainingConfig,
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
        .with_input_dropout(0.2, true) // 20% variational input dropout
        .with_recurrent_dropout(0.3, true) // 30% variational recurrent dropout
        .with_output_dropout(0.1) // 10% output dropout
        .with_zoneout(0.05, 0.1); // 5% cell, 10% hidden zoneout

    // Configure dropout per layer for fine-grained control
    let layer_configs = vec![
        LayerDropoutConfig::new().with_input_dropout(0.1, false),
        LayerDropoutConfig::new()
            .with_recurrent_dropout(0.2, true)
            .with_zoneout(0.05, 0.1),
        LayerDropoutConfig::new().with_output_dropout(0.1),
    ];

    let mut custom_network =
        LSTMNetwork::new(input_size, hidden_size, num_layers).with_layer_dropout(layer_configs);

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
fn test_readme_early_stopping_example() {
    let network = LSTMNetwork::new(1, 4, 1);

    // Configure early stopping
    let early_stopping = EarlyStoppingConfig {
        patience: 2,
        min_delta: 1e-4,
        restore_best_weights: true,
        monitor: EarlyStoppingMetric::ValidationLoss,
    };

    let config = TrainingConfig {
        epochs: 2,
        early_stopping: Some(early_stopping),
        ..Default::default()
    };

    let mut trainer = create_basic_trainer(network, 0.001).with_config(config);
    let train_data = generate_test_data();
    let validation_data = generate_test_data();

    trainer.train(&train_data, Some(&validation_data));

    assert_eq!(trainer.config.early_stopping.as_ref().unwrap().patience, 2);
}

#[test]
fn test_readme_bilstm_example() {
    let input_size = 3;
    let hidden_size = 5;
    let num_layers = 1;

    // BiLSTM with concatenated outputs (output_size = 2 * hidden_size)
    let mut bilstm = BiLSTMNetwork::new_concat(input_size, hidden_size, num_layers);

    // Process sequence with both past and future context
    let sequence = vec![
        Array2::from_shape_vec((input_size, 1), vec![0.5, 0.1, -0.3]).unwrap(),
        Array2::from_shape_vec((input_size, 1), vec![0.2, -0.4, 0.7]).unwrap(),
    ];
    let outputs = bilstm.forward_sequence(&sequence);

    assert_eq!(outputs.len(), sequence.len());
    for output in outputs {
        assert_eq!(output.shape(), &[2 * hidden_size, 1]);
    }
}

#[test]
fn test_readme_gru_example() {
    let input_size = 3;
    let hidden_size = 5;
    let num_layers = 2;

    // Create GRU network (alternative to LSTM)
    let mut gru = GRUNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, true);

    let input = Array2::from_shape_vec((input_size, 1), vec![0.5, 0.1, -0.3]).unwrap();
    let hidden_states = vec![Array2::zeros((hidden_size, 1)); num_layers];

    // Forward pass returns one hidden state per layer
    let outputs = gru.forward(&input, &hidden_states);
    let output = outputs.last().unwrap();

    assert_eq!(outputs.len(), num_layers);
    assert_eq!(output.shape(), &[hidden_size, 1]);
}

#[test]
fn test_readme_linear_layer_example() {
    let hidden_size = 4;
    let num_classes = 3;

    // Create linear layer for classification: hidden_size -> num_classes
    let mut classifier = LinearLayer::new(hidden_size, num_classes);
    let mut optimizer = Adam::new(0.001);

    // Forward pass
    let lstm_output = Array2::ones((hidden_size, 1));
    let logits = classifier.forward(&lstm_output);

    // Backward pass
    let grad_output = Array2::ones((num_classes, 1));
    let (gradients, input_grad) = classifier.backward(&grad_output);
    classifier.update_parameters(&gradients, &mut optimizer, "classifier");

    assert_eq!(logits.shape(), &[num_classes, 1]);
    assert_eq!(input_grad.shape(), &[hidden_size, 1]);
}

#[test]
fn test_readme_advanced_learning_rate_scheduling_example() {
    // Create a network
    let network = LSTMNetwork::new(1, 4, 1);

    // Step decay: reduce LR by 50% every 10 epochs
    let mut step_trainer = create_step_lr_trainer(network.clone(), 0.01, 10, 0.5);

    // OneCycle policy for modern deep learning
    let mut one_cycle_trainer = create_one_cycle_trainer(network.clone(), 0.1, 100);

    // Cosine annealing with warm restarts
    let mut cosine_trainer = create_cosine_annealing_trainer(network.clone(), 0.01, 20, 1e-6);

    // Advanced combinations - Warmup + Cyclical scheduling
    let base_scheduler = CyclicalLR::new(0.001, 0.01, 10);
    let warmup_scheduler = WarmupScheduler::new(5, base_scheduler, 0.0001);
    let mut optimizer = ScheduledOptimizer::new(Adam::new(0.01), warmup_scheduler, 0.01);

    // Polynomial decay with visualization
    let poly_scheduler = PolynomialLR::new(100, 2.0, 0.001);
    let schedule = LRScheduleVisualizer::generate_schedule(poly_scheduler, 0.01, 100);

    step_trainer.optimizer.step();
    one_cycle_trainer.optimizer.step();
    cosine_trainer.optimizer.step();
    optimizer.step();

    assert_eq!(schedule.len(), 100);
    assert!(optimizer.get_current_lr() > 0.0);
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

    for _seq_idx in 0..3 {
        // Small dataset for test
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _t in 0..2 {
            // Short sequences for test
            let input = Array2::ones((1, 1));
            let target = Array2::ones((4, 1)) * 0.5;

            inputs.push(input);
            targets.push(target);
        }

        data.push((inputs, targets));
    }

    data
}
