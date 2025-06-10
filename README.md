# Rust-LSTM

A comprehensive LSTM (Long Short-Term Memory) neural network library implemented in Rust. This library provides complete functionalities to create, train, and use LSTM networks for various sequence modeling tasks with advanced dropout regularization.

## Features

- **LSTM cell implementation** with forward and backward propagation
- **Peephole LSTM variant** for enhanced performance
- **Multi-layer LSTM networks** with configurable architecture
- **Complete training system** with backpropagation through time (BPTT)
- **Multiple optimizers**: SGD, Adam, RMSprop
- **Loss functions**: MSE, MAE, Cross-entropy with softmax
- **Dropout regularization**: Input, recurrent, output dropout and zoneout
- **Training utilities**: gradient clipping, validation, metrics tracking
- **Random initialization** of weights and biases
- **Model persistence**: Save/load models in JSON or binary format

## Dropout Features

The library provides comprehensive dropout support for regularization:

- **Input Dropout**: Applied to inputs before computing gates
- **Recurrent Dropout**: Applied to hidden states (supports variational dropout)
- **Output Dropout**: Applied to LSTM layer outputs
- **Variational Dropout**: Uses same mask across time steps for RNNs
- **Zoneout**: Randomly preserves hidden/cell state values from previous timestep
- **Layer-specific Configuration**: Different dropout settings per layer

## Getting Started

### Prerequisites

Ensure you have Rust installed on your machine. If Rust is not already installed, you can install it by following the instructions on the official Rust website: https://www.rust-lang.org/tools/install.

### Installing

To use Rust-LSTM in your project, add the following to your Cargo.toml:

```toml
[dependencies]
rust-lstm = "0.2.0"
```

Then, run the following command to build your project and download the Rust-LSTM crate:

```bash
cargo build
```

## Usage

### Basic Forward Pass

Here's a simple example demonstrating basic LSTM usage:

```rust
use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;

fn main() {
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

    // Print the output
    println!("Output: {:?}", output);
}
```

### LSTM with Dropout Regularization

Here's how to create an LSTM network with comprehensive dropout:

```rust
use ndarray::Array2;
use rust_lstm::{LSTMNetwork, LayerDropoutConfig};

fn main() {
    let input_size = 10;
    let hidden_size = 20;
    let num_layers = 3;

    // Create network with uniform dropout across all layers
    let mut network = LSTMNetwork::new(input_size, hidden_size, num_layers)
        .with_input_dropout(0.2, true)      // 20% variational input dropout
        .with_recurrent_dropout(0.3, true)  // 30% variational recurrent dropout
        .with_output_dropout(0.1)           // 10% output dropout
        .with_zoneout(0.05, 0.1);          // 5% cell, 10% hidden zoneout

    // Or configure dropout per layer for fine-grained control
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
}
```

### Training an LSTM Network

Here's how to train an LSTM for time series prediction:

```rust
use ndarray::Array2;
use rust_lstm::models::lstm_network::LSTMNetwork;
use rust_lstm::training::{create_basic_trainer, TrainingConfig};
use rust_lstm::optimizers::Adam;
use rust_lstm::loss::MSELoss;

fn main() {
    // Create network with dropout regularization
    let network = LSTMNetwork::new(1, 10, 2)
        .with_input_dropout(0.2, true)
        .with_recurrent_dropout(0.3, true)
        .with_output_dropout(0.1);
    
    // Setup training with Adam optimizer
    let loss_function = MSELoss;
    let optimizer = Adam::new(0.001);
    let mut trainer = LSTMTrainer::new(network, loss_function, optimizer);
    
    // Configure training
    let config = TrainingConfig {
        epochs: 100,
        print_every: 10,
        clip_gradient: Some(1.0),
    };
    trainer = trainer.with_config(config);
    
    // Generate some training data (sine wave prediction)
    let train_data = generate_sine_wave_data();
    
    // Train the model (automatically handles train/eval modes)
    trainer.train(&train_data, None);
    
    // Make predictions (automatically sets eval mode)
    let predictions = trainer.predict(&input_sequence);
}
```

### Advanced Features

#### Dropout Types

```rust
use rust_lstm::layers::dropout::{Dropout, Zoneout};

// Standard dropout
let mut dropout = Dropout::new(0.3);

// Variational dropout (same mask across time steps)
let mut variational_dropout = Dropout::variational(0.3);

// Zoneout for RNN hidden/cell states
let zoneout = Zoneout::new(0.1, 0.15); // cell_rate, hidden_rate
```

#### Different Optimizers

```rust
use rust_lstm::optimizers::{SGD, Adam, RMSprop};

// SGD optimizer
let sgd = SGD::new(0.01);

// Adam optimizer with custom parameters
let adam = Adam::with_params(0.001, 0.9, 0.999, 1e-8);

// RMSprop optimizer
let rmsprop = RMSprop::new(0.01);
```

#### Different Loss Functions

```rust
use rust_lstm::loss::{MSELoss, MAELoss, CrossEntropyLoss};

// Mean Squared Error for regression
let mse_loss = MSELoss;

// Mean Absolute Error for robust regression
let mae_loss = MAELoss;

// Cross-Entropy for classification
let ce_loss = CrossEntropyLoss;
```

#### Peephole LSTM

```rust
use rust_lstm::layers::peephole_lstm_cell::PeepholeLSTMCell;

let mut cell = PeepholeLSTMCell::new(input_size, hidden_size);
let (h_t, c_t) = cell.forward(&input, &h_prev, &c_prev);
```

To run this example, save it as main.rs, and run:

```bash
cargo run
```

## Examples

The library includes several examples demonstrating different use cases:

- `basic_usage.rs` - Simple forward pass example
- `training_example.rs` - Complete training workflow with multiple optimizers
- `dropout_example.rs` - Comprehensive dropout regularization demo
- `time_series_prediction.rs` - Time series forecasting
- `text_generation_advanced.rs` - Character-level text generation
- `multi_layer_lstm.rs` - Multi-layer network usage
- `model_inspection.rs` - Model analysis and debugging utilities
- `stock_prediction.rs` - Financial time series prediction
- `weather_prediction.rs` - Weather forecasting with multiple features
- `real_data_example.rs` - Real-world data loading and processing

Run examples with:

```bash
cargo run --example training_example
cargo run --example dropout_example
cargo run --example time_series_prediction
cargo run --example stock_prediction
cargo run --example weather_prediction
```

## Architecture

The library is organized into several modules:

- **`layers`**: LSTM cell implementations (standard and peephole variants) with dropout support
- **`models`**: High-level network architectures with configurable dropout
- **`loss`**: Loss functions for training
- **`optimizers`**: Optimization algorithms
- **`training`**: Training utilities and trainer struct with automatic train/eval mode handling
- **`utils`**: Common utility functions
- **`persistence`**: Model saving and loading functionality

## Training Features

- **Backpropagation Through Time (BPTT)**: Complete gradient computation for sequence modeling
- **Gradient Clipping**: Prevents exploding gradients during training
- **Multiple Optimizers**: SGD, Adam, RMSprop with configurable parameters
- **Validation Support**: Track validation metrics during training
- **Metrics Tracking**: Loss curves and training progress monitoring
- **Flexible Training Loop**: Configurable epochs, learning rates, and logging
- **Automatic Mode Switching**: Training and evaluation modes for dropout

## Dropout and Regularization

- **Input Dropout**: Regularizes input features
- **Recurrent Dropout**: Regularizes hidden state connections
- **Output Dropout**: Regularizes layer outputs
- **Variational Dropout**: Consistent masks across time steps
- **Zoneout**: RNN-specific regularization technique
- **Layer-specific Configuration**: Fine-grained dropout control
- **Automatic Train/Eval Switching**: Seamless mode management

## Running the Tests

To run the tests included with Rust-LSTM, execute:

```bash
cargo test
```

## Version History

- **v0.1.0**: Initial LSTM implementation with forward pass
- **v0.2.0**: Complete training system with BPTT, optimizers, and comprehensive dropout support

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
