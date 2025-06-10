# Rust-LSTM

A comprehensive LSTM (Long Short-Term Memory) neural network library implemented in Rust. This library provides complete functionalities to create, train, and use LSTM networks for various sequence modeling tasks.

## Features

- **LSTM cell implementation** with forward and backward propagation
- **Peephole LSTM variant** for enhanced performance
- **Multi-layer LSTM networks** with configurable architecture
- **Complete training system** with backpropagation through time (BPTT)
- **Multiple optimizers**: SGD, Adam, RMSprop
- **Loss functions**: MSE, MAE, Cross-entropy with softmax
- **Training utilities**: gradient clipping, validation, metrics tracking
- **Random initialization** of weights and biases

## Getting Started

### Prerequisites

Ensure you have Rust installed on your machine. If Rust is not already installed, you can install it by following the instructions on the official Rust website: https://www.rust-lang.org/tools/install.

### Installing

To use Rust-LSTM in your project, add the following to your Cargo.toml:

```toml
[dependencies]
rust-lstm = "0.1.0"
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
    // Create network
    let network = LSTMNetwork::new(1, 10, 2);
    
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
    
    // Train the model
    trainer.train(&train_data, None);
    
    // Make predictions
    let predictions = trainer.predict(&input_sequence);
}
```

### Advanced Features

#### Using Different Optimizers

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

let cell = PeepholeLSTMCell::new(input_size, hidden_size);
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
- `time_series_prediction.rs` - Time series forecasting
- `text_generation.rs` - Character-level text generation
- `multi_layer_lstm.rs` - Multi-layer network usage
- `peephole.rs` - Peephole LSTM variant

Run examples with:

```bash
cargo run --example training_example
cargo run --example time_series_prediction
```

## Architecture

The library is organized into several modules:

- **`layers`**: LSTM cell implementations (standard and peephole variants)
- **`models`**: High-level network architectures
- **`loss`**: Loss functions for training
- **`optimizers`**: Optimization algorithms
- **`training`**: Training utilities and trainer struct
- **`utils`**: Common utility functions

## Training Features

- **Backpropagation Through Time (BPTT)**: Complete gradient computation for sequence modeling
- **Gradient Clipping**: Prevents exploding gradients during training
- **Multiple Optimizers**: SGD, Adam, RMSprop with configurable parameters
- **Validation Support**: Track validation metrics during training
- **Metrics Tracking**: Loss curves and training progress monitoring
- **Flexible Training Loop**: Configurable epochs, learning rates, and logging

## Running the Tests

To run the tests included with Rust-LSTM, execute:

```bash
cargo test
```

This will run all the unit and integration tests defined in the library.

## Contributing

Contributions to Rust-LSTM are welcome! Here are a few ways you can help:

- Report bugs and issues
- Suggest new features or improvements
- Open a pull request with improvements to code or documentation
- Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
